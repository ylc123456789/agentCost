from __future__ import annotations

import re
import sys
from pathlib import Path

from ..contracts import VERSION
from ..controller import assert_score_cleanup, attempt_index
from ..io import (ContractError, append, hash_file, loads, read_json, ref, run_lock, safe_file,
                 verify_ref, write_bytes, write_json)
from ..ledger import Recorder
from ..metering import cpu_window
from ..state import read_run
from ..supervisor import launch

_UPSTREAM_FILES = {"bigcodebench/__init__.py", "bigcodebench/eval/__init__.py",
                   "bigcodebench/eval/utils.py", "bigcodebench/eval/_special_oracle.py"}


def _sandbox(work, case, candidate, bwrap, timeout, fd):
    spec = {k: v for k, v in case.items() if k not in {"evaluator", "reference"}}
    code = work / "case.json"
    write_json(code, spec, create=True)
    grader = Path(__file__).with_name("grade_worker.py")
    prefix = Path(sys.prefix).resolve()
    python = Path(sys.executable).relative_to(prefix).as_posix()
    if prefix in {Path("/usr"), Path("/")}:
        raise ContractError("use a dedicated Python environment for isolated scoring")
    argv = [str(Path(bwrap).resolve(strict=True)), "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin", "--proc", "/proc",
            "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/work", "--chdir", "/work",
            "--ro-bind", str(prefix), "/env", "--ro-bind", str(grader), "/grade_worker.py",
            "--ro-bind", str(candidate), "/candidate.txt", "--ro-bind", str(code), "/case.json"]
    for key, value in {"HOME": "/work", "PATH": "/env/bin:/usr/bin", "TMPDIR": "/tmp", "LANG": "C.UTF-8",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": "/tmp"}.items():
        argv += ["--setenv", key, value]
    if case["kind"] == "bcb_complete":
        evaluator = case["evaluator"]
        if set(evaluator["sha256"]) != _UPSTREAM_FILES:
            raise ContractError("pin all four upstream BigCodeBench evaluator source hashes")
        for path, expected in evaluator["sha256"].items():
            src = safe_file(Path(evaluator["root"]).resolve(), path)
            if hash_file(src) != expected:
                raise ContractError("upstream BigCodeBench evaluator identity mismatch")
            argv += ["--ro-bind", str(src), f"/official/{path}"]
    argv += [f"/env/{python}", "-I", "-B", "/grade_worker.py", str(timeout)]
    receipt = launch(work, fd, argv, timeout)
    if not receipt["cleanup_confirmed"]:
        raise ContractError("grader process tree cleanup not confirmed; stop")
    if receipt["status"] != "completed":
        return {"score": None, "passed": None, "reason": f"grader_{receipt['status']}", "detail": None}
    try:
        result = read_json(work / "stdout.log")
    except ValueError:
        return {"score": None, "passed": None, "reason": "invalid_grader_output", "detail": None}
    passed = result.get("passed") if isinstance(result, dict) else None
    if isinstance(result, dict) and passed is None and result.get("reason"):
        return {"score": None, "passed": None, "reason": result["reason"], "detail": result}
    if type(passed) is not bool:
        raise ContractError("sandbox did not deliver a binary official result")
    return {"score": int(passed), "passed": passed, "reason": None, "detail": result}


def next_version(root):
    return 1 + max((int(path.name.split(".")[1][1:]) for path in root.glob("scoreplan.v*.json")), default=0)


def score(root, private, *, rescore=None, revised_spec=False, bwrap=None, allow_official=False):
    with run_lock(root) as fd:
        plan, freeze = read_run(root)
        assert_score_cleanup(root)
        scoring = plan["scoring"]
        if plan["provenance"] != "current" or scoring["kind"] == "none":
            raise ContractError("no local grader frozen for this run")
        version = next_version(root)
        if version > 1 and not rescore:
            raise ContractError("scores already exist or an earlier scoring phase began; use --rescore REASON")
        if revised_spec and (version < 2 or not rescore):
            raise ContractError("a revised private spec needs a new version and reason")
        if scoring["kind"] == "official" and (not allow_official or bwrap is None):
            raise ContractError("official scoring requires separately authorized --allow-official and --bwrap")
        import hashlib

        raw_private = private.read_bytes()
        sha = hashlib.sha256(raw_private).hexdigest()
        if sha != scoring["private_sha256"] and not revised_spec:
            raise ContractError("private grader spec identity mismatch")
        cases = loads(raw_private)
        last, _ = attempt_index(root, plan)
        if any(r["status"] in {"reserved", "unconfirmed"} for r in last.values()):
            raise ContractError("finish process recovery before grading")
        candidates = {key: row["candidate"] for key, row in last.items() if row["status"] == "completed"}
        for item in candidates.values():
            verify_ref(root, item)
        tag = f"v{version:04}"
        worker = Path(__file__).with_name("grade_worker.py")
        journal = root / f"scores.{tag}.jsonl"
        write_json(root / f"scoreplan.{tag}.json", {"score_version": version, "private_sha256": sha,
                   "candidates": candidates, "reason": rescore, "grader_version": scoring["version"],
                   "scorer_sha256": hash_file(Path(__file__)), "worker_sha256": hash_file(worker),
                   "bwrap_sha256": hash_file(Path(bwrap)) if bwrap else None}, create=True)
        write_bytes(journal, b"", create=True)
        for slot in plan["attempts"]:
            aid, tid, mid = slot["id"], slot["task_id"], slot["method_id"]
            if aid not in candidates:
                continue
            path = verify_ref(root, candidates[aid])
            rec = Recorder(root / "events.jsonl", run_id=plan["run_id"], method_id=mid,
                           state_id=freeze["states"][mid]["state_id"], task_id=tid, attempt_id=aid,
                           phase="evaluation", role="grader")
            with rec.span("offline_scoring", accounting="phase", data={"score_version": version}), cpu_window(rec):
                case = cases.get(tid)
                result = {"score": None, "passed": None, "reason": "private_spec_missing", "detail": None}
                if case is not None and scoring["kind"] == "exact":
                    passed = path.read_text(encoding="utf-8") == case["expected"]
                    result = {"score": int(passed), "passed": passed, "reason": None, "detail": None}
                if case is not None and scoring["kind"] == "official":
                    if case["kind"] not in {"ds1000", "bcb_complete"}:
                        raise ContractError("unknown official contract")
                    work = root / "raw" / f"scoring-{tag}" / aid
                    work.mkdir(parents=True)
                    result = _sandbox(work, case, path, bwrap, scoring["timeout_seconds"], fd)
                verify_ref(root, candidates[aid])
                append(journal, {"schema_version": VERSION, "run_id": plan["run_id"], "source": "current",
                                 "attempt_id": aid, "score_version": version, "candidate": candidates[aid], **result})
        write_json(root / f"scorecomplete.{tag}.json", {"score_version": version,
                   "journal": ref(root, journal)}, create=True)
        if version == 1:
            write_bytes(root / "scores.jsonl", journal.read_bytes())
        return {"score_version": version, "candidate_count": len(candidates)}


def selected_scores(root):
    completed = [p for p in root.glob("scorecomplete.v*.json") if re.fullmatch(r"scorecomplete\.v\d+\.json", p.name)]
    latest = max(completed, key=lambda p: int(p.name.split(".")[1][1:]), default=None)
    if latest is None:
        return root / "scores.jsonl", None
    receipt = read_json(latest)
    return verify_ref(root, receipt["journal"]), receipt["score_version"]
