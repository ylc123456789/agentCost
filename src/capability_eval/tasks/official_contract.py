"""Bounded grader-only controls. This module is not used by local smoke."""
from __future__ import annotations

import gzip
import hashlib
import sys
from pathlib import Path

from ..analysis import write_manifest
from ..io import ContractError, append, digest, hash_file, loads, ref, run_lock, write_bytes, write_json
from ..ledger import Recorder
from ..plan import environment
from ..supervisor import launch
from . import BigCodeBenchComplete, DS1000
from .scoring import _sandbox

DS = {10: "Pandas", 315: "Numpy", 846: "Sklearn"}
DS_REV = "b39aab71da6d23ef8d3cac59a7c5f834516ab334"
DS_HASH = "e8c6daa9d7223976bce0296644f3933f78d7f47830669ff05cd61da62c6ba9b3"
BCB_ID = "BigCodeBench/206"  # historically exposed; dev-only
BCB_REV = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
BCB_HASH = "d9a4965821c9507ebdfb551c288656b2d5fe553234f5183044333ca8a4018267"
BCB_CODE_REV = "09dd993f46c3fbf3a799465bb96d524edcb0b199"
BCB_CODE_HASH = {
    "bigcodebench/__init__.py": "da556cf4bc44f5922a7b610aaa9e875111f4bb670a7ce8d2755b45fad532c9e0",
    "bigcodebench/eval/__init__.py": "d5fd553559ac1b76659ebc32ae30e3e779449ecd31c5201f2301319ceeee01fe",
    "bigcodebench/eval/utils.py": "9061f74fe937c4eb7a1b2bc423f7acab547ae01804d5e25933e2aa8a3cc2d685",
    "bigcodebench/eval/_special_oracle.py": "0cf930163987d30f455547aec6cbc500a154bc58a2eb109aa247ef4e0962448b",
}


def _cases(dataset, source, evaluator):
    actual = hash_file(source)
    expected = DS_HASH if dataset == "ds1000" else BCB_HASH
    if actual != expected:
        raise ContractError(f"official dataset identity mismatch: expected {expected}")
    if dataset == "ds1000":
        adapter = DS1000()
        chosen = {}
        with gzip.open(source, "rb") as data:
            for line in data:
                row = loads(line)
                ident = int(row["metadata"]["problem_id"])
                if ident in DS:
                    if ident in chosen or row["metadata"]["library"].casefold() != DS[ident].casefold():
                        raise ContractError("DS-1000 fixed development case identity mismatch")
                    chosen[ident] = row
        if set(chosen) != DS.keys():
            raise ContractError("DS-1000 fixed development row missing")
        return DS_REV, [(adapter.public(chosen[i]), adapter.private(chosen[i])) for i in DS]
    if evaluator is None:
        raise ContractError("BCB needs --evaluator-root at the pinned upstream revision")
    for path, sha in BCB_CODE_HASH.items():
        if hash_file(evaluator / path) != sha:
            raise ContractError(f"BigCodeBench evaluator file mismatch: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ContractError("BCB parquet requires pyarrow==19.0.1 in the dedicated grading environment") from exc
    columns = ["task_id", "entry_point", "complete_prompt", "canonical_solution", "test", "code_prompt"]
    rows = pq.read_table(source, columns=columns, filters=[("task_id", "=", BCB_ID)]).to_pylist()
    if len(rows) != 1:
        raise ContractError("fixed BigCodeBench Complete development row missing or ambiguous")
    adapter = BigCodeBenchComplete()
    code = {"root": str(evaluator.resolve()), "revision": BCB_CODE_REV, "sha256": BCB_CODE_HASH}
    return BCB_REV, [(adapter.public(rows[0]), adapter.private(rows[0], code, calibrated=True))]


def official_contract(root, source, dataset, *, bwrap, allow_official, evaluator=None):
    if not allow_official:
        raise ContractError("official data is disabled; requires authorized server run and --allow-official")
    if sys.platform != "linux":
        raise ContractError("official controls require isolated Linux grading")
    revision, tasks = _cases(dataset, source, evaluator)
    cases = []
    for index, (task, case) in enumerate(tasks):
        body = ("def " + case["entry_point"] + "(*args, **kwargs):\n    "
                if case["kind"] == "bcb_complete" else "")
        cases += [(task, label, f"case-{index + 1}-{label}", code, {**case, "reference_control": label == "reference"})
                  for label, code in (("reference", case["reference"]), ("negative", body + "raise AssertionError('negative control')"),
                                      ("timeout", body + "while True: pass"))]
    from uuid import uuid4

    root.mkdir(parents=True, exist_ok=False)
    with run_lock(root) as fd:
        plan = {"schema_version": "1.0", "run_id": uuid4().hex, "kind": "grader_controls", "dataset": dataset,
                "dataset_revision": revision, "dataset_sha256": hash_file(source), "bwrap_sha256": hash_file(bwrap),
                "grading_worker_sha256": hash_file(Path(__file__).with_name("grade_worker.py")),
                "tasks": [task.record() for task, _ in tasks], "requests": len(cases), "model_requests": 0,
                "per_case_wall_limit_seconds": 10, "retry": 0,
                "cases": [{"task_id": task.id, "label": label, "attempt_id": aid, "private_case_sha256": digest(case),
                           "candidate_sha256": hashlib.sha256(code.encode()).hexdigest()} for task, label, aid, code, case in cases]}
        write_json(root / "plan.json", plan, create=True)
        write_json(root / "environment.json", environment(), create=True)
        preflight = root / "raw/preflight"
        preflight.mkdir(parents=True)
        preflight_receipt = launch(preflight, fd, [str(bwrap.resolve()), "--unshare-all", "--die-with-parent", "--new-session",
                                  "--cap-drop", "ALL", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
                                  "--symlink", "usr/lib64", "/lib64", "/usr/bin/true"], 3)
        if preflight_receipt["status"] != "completed" or not preflight_receipt["cleanup_confirmed"]:
            raise ContractError("sandbox preflight failed; no official candidate launched")
        freeze = {"plan_sha256": hash_file(root / "plan.json"), "environment_sha256": hash_file(root / "environment.json"),
                  "preflight": ref(root, preflight / "receipt.json")}
        write_json(root / "artifact_freeze.json", {"payload_sha256": digest(freeze), **freeze}, create=True)
        output = []
        for task, label, aid, code, case in cases:
            work = root / "raw" / aid
            work.mkdir()
            candidate = work / "candidate.txt"
            write_bytes(candidate, code.encode("utf-8"), create=True)
            candidate.chmod(0o444)
            identity = {"run_id": plan["run_id"], "task_id": task.id, "attempt_id": aid}
            rec = Recorder(root / "events.jsonl", **identity, phase="evaluation", role="grader", limits={"grade": 1})
            append(root / "attempts.jsonl", {**identity, "status": "reserved", "control": label})
            with rec.span("official_grader_control", accounting="phase", data={"control": label}):
                rec.reserve("grade")
                result = _sandbox(work, case, candidate, bwrap, plan["per_case_wall_limit_seconds"], fd)
            expected = {"reference": True, "negative": False, "timeout": None}[label]
            qualified = result["passed"] is expected and (label != "timeout" or result["reason"] == "grader_timeout")
            row = {**identity, **result, "label": label, "qualified": qualified, "candidate": ref(root, candidate),
                   "receipt": ref(root, work / "receipt.json"), "score_version": 1}
            append(root / "attempts.jsonl", {**identity, "status": "completed", "control": label})
            append(root / "scores.v0001.jsonl", row)
            output.append(row)
        write_bytes(root / "scores.jsonl", (root / "scores.v0001.jsonl").read_bytes(), create=True)
        verdict = all(row["qualified"] for row in output)
        result = {"qualified": verdict, "dataset": dataset, "controls": output, "model_requests": 0}
        write_json(root / "analysis.json", result, create=True)
        write_json(root / "resource_summary.json", {"device": "cpu", "gpu_seconds": None,
                   "gpu_reason": "grader-only; no GPU lease", "receipts": [row["receipt"] for row in output]}, create=True)
        write_bytes(root / "report.md", (f"# Official grader controls: {dataset}\n\nQualified: {verdict}. "
                   f"{len(output)} planned controls, zero model calls, dev only. See `analysis.json`.\n").encode(), create=True)
        write_manifest(root)
        return result
