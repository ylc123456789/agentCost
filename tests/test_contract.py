from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from capability_eval.analysis import analyze, crossing, verify
from capability_eval.analysis.resources import summarize
from capability_eval.contracts import PublicTask
from capability_eval.controller import run
from capability_eval.integrations.importer import import_events
from capability_eval.io import ContractError, hash_file, read_json, rows, run_lock, write_bytes, write_json
from capability_eval.ledger import Recorder
from capability_eval.plan import build_plan
from capability_eval.state import prepare
from capability_eval.tasks import BigCodeBenchComplete, DS1000
from capability_eval.tasks.scoring import _sandbox, score

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples"


class TestClock:
    domain = "clock-fixture"
    t = 0

    def now(self):
        return self.t * 10**9

    def utc(self):
        return "2026-11-18T00:00:00+00:00"


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.root = self.dir / "run"

    def config(self, overrides=None):
        cfg = read_json(EXAMPLE / "fixture.json")
        cfg["analysis"].pop("workload_file")
        cfg.update(overrides or {})
        file = self.dir / "config.json"
        write_json(file, cfg)
        return file

    def capture(self, name):
        target = os.getenv("CE_TEST_RECORDS")
        if target and self.root.is_dir():
            shutil.copytree(self.root, Path(target) / name)

    def external_config(self, body, *, timeout=2, count=1, sdk=False):
        script = self.dir / "external.py"
        write_bytes(script, body.encode())
        req = {"dispatch": 1}
        if sdk:
            req["call"] = 1
        return self.config({"scoring": {"kind": "none", "version": "none"},
                            "tasks": [{"id": f"fixture/{i}", "prompt": "Go", "dataset": "synthetic"} for i in range(count)],
                            "methods": [{"id": "baseline", "driver": "external", "command": [sys.executable, "-B", str(script)],
                                         "requests": req}],
                            "budget": {"max_attempts": count, "max_requests": {k: count * v for k, v in req.items()},
                                       "attempt_timeout_seconds": timeout}})

    def test_fixture_white_list_score_no_new_generation_and_recompute(self):
        cfg = self.config()
        payload = read_json(cfg)
        payload["tasks"][0]["reference"] = "PRIVATE-CANARY"
        write_json(cfg, payload)
        plan, _ = prepare(cfg, self.root)
        self.assertNotIn("PRIVATE-CANARY", (self.root / "plan.json").read_text())
        self.assertEqual(set(run(self.root)["executed"].values()), {"completed"})
        for p in (self.root / "raw").rglob("request.json"):
            self.assertNotIn("PRIVATE-CANARY", p.read_text())
        before = {p: hash_file(p) for p in (self.root / "raw").rglob("candidate.txt")}
        score(self.root, EXAMPLE / "fixture-private.json")
        score(self.root, EXAMPLE / "fixture-private.json", rescore="repeat correctness check")
        self.assertEqual(before, {p: hash_file(p) for p in before})
        events = list(rows(self.root / "events.jsonl"))
        self.assertEqual(sum(e["kind"] == "budget" and e["metric"] == "dispatch" for e in events), 6)
        analyze(self.root, workload=EXAMPLE / "workload.json")
        first = read_json(self.root / "analysis.json")
        self.assertEqual(first["score_version"], 2)
        self.assertEqual(first["quality"]["variant"]["pass"], 2)
        self.assertEqual(first["quality"]["baseline"]["pass"], 2)
        self.assertEqual((first["pairs"]["variant"]["added"], first["pairs"]["variant"]["lost"]), (1, 1))
        self.assertIsNone(first["workload"]["scenarios"][2]["methods"]["variant"]["wall_seconds"]["current_total"])
        rebuilt = self.dir / "rebuilt"
        analyze(self.root, workload=EXAMPLE / "workload.json", output=rebuilt)
        for name in ("analysis.json", "resource_summary.json", "report.md"):
            self.assertEqual(hash_file(self.root / name), hash_file(rebuilt / name))
        self.assertGreater(verify(self.root), 10)
        p = next(iter(before))
        p.write_text("corrupted")
        with self.assertRaisesRegex(ContractError, "changed"):
            analyze(self.root)
        self.assertEqual(len(plan["attempts"]), 6)

    def test_budget_preserves_unstarted_denominator(self):
        cfg = self.config({"budget": {"max_attempts": 99, "max_requests": {"dispatch": 2}, "attempt_timeout_seconds": 3}})
        prepare(cfg, self.root)
        self.assertEqual(len(run(self.root)["executed"]), 2)
        self.assertEqual(len(run(self.root)["executed"]), 0)
        q = analyze(self.root)["quality"]
        self.assertEqual(sum(v["denominator"] for v in q.values()), 6)
        self.assertEqual(sum(v["not_started"] for v in q.values()), 4)
        self.capture("budget-stopped")

    def test_timeout_kills_reparented_session(self):
        cfg = self.external_config('''import os, pathlib, subprocess, sys, time
p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
pathlib.Path("child.pid").write_text(str(p.pid))
time.sleep(30)
''', timeout=0.7)
        prepare(cfg, self.root)
        self.assertEqual(run(self.root)["executed"], {"a000001": "timeout"})
        work = self.root / "raw/a000001"
        self.assertFalse(Path(f"/proc/{(work / 'child.pid').read_text()}").exists())
        self.assertTrue(read_json(work / "receipt.json")["cleanup_confirmed"])
        self.assertEqual(analyze(self.root)["quality"]["baseline"]["execution_failed"], 1)
        self.capture("timeout-cleaned")

    def test_controller_sigkill_then_resume_without_redispatch(self):
        cfg = self.external_config('''import pathlib, time
pathlib.Path("started.marker").write_text("one")
time.sleep(1.1)
print("returned")
''', timeout=4, count=2)
        prepare(cfg, self.root)
        ctrl = subprocess.Popen([sys.executable, "-B", "-m", "capability_eval", "run", "--run-dir", str(self.root)],
                                stdout=subprocess.DEVNULL)
        self.addCleanup(lambda: ctrl.kill() if ctrl.poll() is None else None)
        work = self.root / "raw/a000001"
        deadline = time.monotonic() + 8
        while not (work / "started.marker").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue((work / "started.marker").exists())
        ctrl.send_signal(signal.SIGKILL)
        ctrl.wait(timeout=3)
        with self.assertRaisesRegex(ContractError, "locked"):
            run(self.root)
        while not (work / "receipt.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        # The receipt is published before the supervisor exits and releases its
        # copy of the lock. Wait for both conditions; a visible receipt alone is not a lease.
        while True:
            try:
                restarted = run(self.root)
                break
            except ContractError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        self.assertEqual(restarted["executed"], {"a000002": "completed"})
        self.assertEqual(restarted["reserved_requests"], {"dispatch": 2})
        statuses = [r["status"] for r in rows(self.root / "attempts.jsonl") if r["attempt_id"] == "a000001"]
        self.assertEqual(statuses, ["reserved", "interrupted"])
        analyze(self.root)
        self.capture("controller-killed-and-resumed")

    def test_wrap_exact_command_and_sdk_events(self):
        cfg = self.external_config('''from capability_eval import Recorder
from pathlib import Path
rec = Recorder.from_environment()
with rec.span("library call", role="tool"):
    rec.reserve("call")
    rec.measure("sdk_count", 0, "calls")
    Path("note.txt").write_text("retained")
    rec.attach_artifact("note.txt")
print("ok")
''', sdk=True)
        plan, _ = prepare(cfg, self.root)
        with self.assertRaisesRegex(ContractError, "frozen argv"):
            run(self.root, only="a000001", command=["no"])
        cmd = plan["methods"][0]["command"]
        self.assertEqual(run(self.root, only="a000001", command=cmd)["executed"], {"a000001": "completed"})
        with self.assertRaisesRegex(ContractError, "already started"):
            run(self.root, only="a000001", command=cmd)
        events = list(rows(self.root / "events.jsonl"))
        artifact = next(e for e in events if e["kind"] == "artifact")
        self.assertTrue((self.root / artifact["artifact_refs"][0]["path"]).is_file())
        self.assertIsNone(next(e for e in events if e["metric"] == "training_tokens")["value"])

    def test_freezes_and_unsupported_combo_fail_before_dispatch(self):
        cfg = self.config()
        prepare(cfg, self.root)
        plan = self.root / "plan.json"
        plan.write_bytes(plan.read_bytes() + b" ")
        with self.assertRaisesRegex(ContractError, "identity"):
            run(self.root)
        src = read_json(EXAMPLE / "tiny-cpu.json")
        src["methods"] = [{"id": "invalid", "driver": "transformer", "carriers": ["actor_lora", "layer_kv"]}]
        invalid = self.dir / "unsupported.json"
        src["materials"] = []
        write_json(invalid, src)
        with self.assertRaisesRegex(ContractError, "unsupported transformer"):
            build_plan(invalid)
        self.assertEqual(list(rows(self.root / "attempts.jsonl")), [])

    def test_dependency_edit_stops_before_reservation(self):
        cfg = self.external_config('print("old")')
        prepare(cfg, self.root)
        (self.dir / "external.py").write_text('print("new")')
        with self.assertRaisesRegex(ContractError, "source changed"):
            run(self.root)
        self.assertEqual(list(rows(self.root / "attempts.jsonl")), [])

    def test_official_entrypoint_fails_before_touching_data(self):
        from capability_eval.tasks.official_contract import official_contract

        with self.assertRaisesRegex(ContractError, "official data is disabled"):
            official_contract(self.root, self.dir / "not-present", "ds1000",
                              bwrap=self.dir / "not-present-either", allow_official=False)
        self.assertFalse(self.root.exists())

    def test_metrology_nested_hold_and_peak(self):
        clock = TestClock()
        rec = Recorder(self.dir / "events", run_id="metered", method_id="m", attempt_id="a1", clock=clock, device_id="GPU-fixture")
        with rec.span("allocated", phase="device_hold", data={"attribution": "exclusive"}):
            with rec.span("serve", phase="serve", accounting="phase", data={"charge_gpu_allocation": True}):
                clock.t = 2
                with rec.span("nested"):
                    clock.t = 6
                rec.measure("torch_allocated_peak_bytes", 10, "B")
                rec.measure("torch_allocated_peak_bytes", 20, "B")
                clock.t = 10
            with rec.span("evaluation", phase="evaluation", accounting="phase"):
                clock.t = 15
            clock.t = 20
        sample = {"run_id": "metered", "methods": [{"id": "m", "driver": "fixture"}],
                  "attempts": [{"id": "a1", "method_id": "m"}]}
        fr = {"components": {}, "states": {"m": {"status": "completed"}}}
        metrics = summarize(sample, fr, list(rows(rec.journal)), {"a1": {"cleanup_confirmed": True}})
        self.assertEqual(metrics["all_phase_wall_seconds"]["serve"], 10)
        self.assertEqual(metrics["gpu_hold"]["value"], 20)
        device = metrics["device_windows"][0]
        self.assertEqual((device["overlap_with_offline_evaluation_seconds"], device["unattributed_seconds"]), (5, 5))
        self.assertEqual(metrics["by_method"]["m"]["resources"]["gpu_allocation_seconds"]["phases"]["serve"]["value"], 10)
        self.assertEqual(metrics["by_method"]["m"]["metrics"][0]["value"], 20)
        self.assertEqual(crossing(60, 0, 7, 10, eligible=True)["tasks"], 20)
        self.assertIsNone(crossing(60, 0, 10, 10, eligible=True)["tasks"])
        self.assertIsNone(crossing(60, 0, 7, 10, eligible=False)["tasks"])

    def test_clock_domains_and_open_spans(self):
        a, b = TestClock(), TestClock()
        b.domain = "other-boot"
        for clock, end in ((a, 8), (b, 5)):
            rec = Recorder(self.dir / "events", run_id="clock", clock=clock)
            with rec.span("parallel-looking but different domain"):
                clock.t = end
        rec.emit("start", span_id="unknown-span", status="started", measurement_status="observed")
        summary = summarize({"run_id": "clock", "methods": [], "attempts": []}, {"components": {}}, list(rows(rec.journal)), {})
        self.assertEqual(summary["all_phase_wall_seconds"]["serve"], 13)
        self.assertEqual(len(summary["open_spans"]), 1)

    def test_observed_only_import_never_pairs_and_preserves_null(self):
        original = self.dir / "source.jsonl"
        write_bytes(original, b'{"task":"same","method":"left","result":"ok","passed":true,"seconds":2}\n'
                              b'{"task":"same","method":"right","result":"timeout","passed":null}\n')
        import_events(original, EXAMPLE / "flat-mapping.json", self.root)
        analyze(self.root)
        self.assertEqual(read_json(self.root / "analysis.json")["pairs"], {})
        self.assertEqual(read_json(self.root / "analysis.json")["quality"]["right"]["missing_scores"], 1)
        events = list(rows(self.root / "events.jsonl"))
        self.assertEqual(events[0]["measurement_status"], "estimated")
        self.assertEqual(events[1]["measurement_status"], "unavailable")
        self.assertEqual(hash_file(original), hash_file(self.root / "raw/import/original.jsonl"))

    def test_official_adapters_public_shape_only(self):
        ds = DS1000().public(dict(prompt="Q", metadata={"problem_id": 1, "library": "Pandas", "origin_id": 1},
                                  code_context="PRIVATE-CANARY", reference_code="PRIVATE-CANARY"))
        bcb = BigCodeBenchComplete().public(dict(task_id="synthetic/1", complete_prompt="def f():",
                                                  test="PRIVATE-CANARY", canonical_solution="PRIVATE-CANARY"))
        self.assertIsInstance(ds, PublicTask)
        self.assertEqual(ds.family, 1)
        self.assertNotIn("PRIVATE-CANARY", str((ds.record(), bcb.record())))
        self.assertFalse(hasattr(bcb, "test"))

    @unittest.skipUnless(os.getenv("CE_TEST_BWRAP"), "bwrap test needs CE_TEST_BWRAP")
    def test_synthetic_sandbox_positive_negative_timeout(self):
        self.root.mkdir()
        context = '''def test_execution(code):
    scope = {}
    exec("def f():\\n    [insert]".replace("[insert]", code), scope)
    assert scope["f"]() == 2
def test_string(code):
    assert "return" in code and "+" not in code
'''
        case = {"kind": "synthetic_code", "code_context": context}
        bwrap = os.environ["CE_TEST_BWRAP"]
        with run_lock(self.root) as fd:
            for index, (candidate, expected) in enumerate((("return 2", True), ("return 1+1", False),
                                                          ("while True: pass", None))):
                p = self.dir / f"candidate{index}.txt"
                p.write_text(candidate)
                work = self.dir / f"sandbox{index}"
                work.mkdir()
                result = _sandbox(work, case, p, bwrap, 0.7 if expected is None else 3, fd)
                self.assertIs(result["passed"], expected, (result, (work / "stderr.log").read_text()))
                if expected is None:
                    self.assertEqual(result["reason"], "grader_timeout")


if __name__ == "__main__":
    unittest.main()
