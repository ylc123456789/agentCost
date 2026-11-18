from __future__ import annotations

import sys
from collections import Counter

from .io import ContractError, append, digest, read_json, ref, rows, run_lock, verify_ref, write_bytes, write_json
from .ledger import Recorder, validate_event
from .state import read_run, verify_method
from .supervisor import isolated_env, launch

TERMINAL = {"completed", "failed", "timeout", "interrupted", "unconfirmed"}


def attempt_index(root, plan):
    last, starts = {}, {}
    planned = {a["id"]: a for a in plan["attempts"]}
    for row in rows(root / "attempts.jsonl"):
        aid, status = row["attempt_id"], row["status"]
        if row.get("run_id") != plan["run_id"] or aid not in planned:
            raise ContractError("attempt journal record is outside its frozen plan")
        slot = planned[aid]
        if any(row[k] != slot[k] for k in ("method_id", "task_id", "replicate")):
            raise ContractError("attempt journal row does not match the frozen slot")
        if status == "reserved":
            if aid in last or row["requests"] != planned[aid]["requests"]:
                raise ContractError("attempt replay or reservation mismatch")
            starts[aid] = row
        elif status not in TERMINAL or aid not in starts or last[aid]["status"] != "reserved":
            raise ContractError("invalid attempt transition")
        last[aid] = row
    return last, starts


def record_attempt(root, plan, slot, status, **extras):
    row = {"schema_version": plan["schema_version"], "run_id": plan["run_id"],
           "attempt_id": slot["id"], "method_id": slot["method_id"], "task_id": slot["task_id"],
           "replicate": slot["replicate"], "status": status, **extras}
    append(root / "attempts.jsonl", row)
    return row


def merge_child(root, work, identity, *, wrapper=False):
    existing = {e["event_id"] for e in rows(root / "events.jsonl")}
    counts = Counter()
    seen = set()
    child = []
    for event in rows(work / "child.events.jsonl"):
        validate_event(event)
        for key in ("run_id", "method_id", "state_id", "task_id", "attempt_id"):
            if event[key] != identity[key]:
                raise ContractError(f"child event scope mismatch: {key}")
        if event["source"] != "current" or event["event_id"] in seen:
            raise ContractError("child event source or event id invalid")
        seen.add(event["event_id"])
        if event["kind"] == "budget":
            if type(event["value"]) is not int or event["value"] < 1:
                raise ContractError("invalid child request reservation")
            counts[event["metric"]] += event["value"]
            if counts[event["metric"]] > identity["limits"].get(event["metric"], 0):
                raise ContractError("child exceeded frozen request envelope")
        if event["artifact_refs"]:
            event["artifact_refs"] = [ref(root, verify_ref(work, item)) for item in event["artifact_refs"]]
        if wrapper and event["kind"] in {"start", "end"} and event["data"].get("accounting") == "phase":
            event["data"] = {**event["data"], "accounting": "sdk_detail", "bounded_by_wrapper": True}
        child.append(event)
    for event in child:
        if event["event_id"] not in existing:
            append(root / "events.jsonl", event)
    return dict(counts)


def _process_receipt(rec, receipt, *, external):
    cleanup = receipt["cleanup_confirmed"]
    phase = "serve" if external else "deployment_overhead"
    if receipt.get("clock_domain") is None:
        rec.emit("note", phase=phase, status="unconfirmed", reason=receipt.get("reason"))
        return
    start = receipt["start_ns"]
    span_id = f"process-{rec.identity['attempt_id']}"
    existing = {e["event_id"] for e in rows(rec.journal) if e["span_id"] == span_id}
    def once(writer, kind, **kwargs):
        event_id = digest([rec.identity["run_id"], span_id, kind])
        if event_id not in existing:
            writer(event_id=event_id, **kwargs)
    scope = "phase" if external else "process_envelope"
    fields = {"span_id": span_id, "phase": phase, "role": "runtime", "process_id": receipt["process_id"],
              "clock_domain": receipt["clock_domain"], "utc": receipt["utc"],
              "data": {"name": "subprocess_tree", "accounting": scope}}
    once(lambda **kw: rec.emit("start", **kw), "start", status="started", monotonic_ns=start,
         measurement_status="observed", **fields)
    # Without a cleanup receipt this interval is only a lower bound, never a settled total.
    once(lambda **kw: rec.emit("end", **kw), "end", status=receipt["status"], monotonic_ns=receipt["end_ns"],
         measurement_status="observed", **fields)
    metrics = receipt.get("metrics") or {}
    for name in ("cpu_user_seconds", "cpu_system_seconds", "process_max_rss_bytes"):
        once(rec.measure, name, metric=name, value=metrics.get(name), unit="s" if name.endswith("seconds") else "B",
             phase=phase, span_id=span_id, data={"scope": "process_tree_wait4" if name.endswith("seconds") else
                                               "max_reaped_process_ru_maxrss", "partial": not cleanup},
             reason=None if name in metrics else "process tree wait4 not observed")
    if external:
        for name, unit in (("torch_allocated_peak_bytes", "B"), ("torch_reserved_peak_bytes", "B"),
                           ("training_tokens", "tokens"), ("gpu_hold_seconds", "s")):
            once(rec.measure, name, metric=name, value=None, unit=unit, phase="serve", span_id=span_id,
                 reason="not observable from wrapper; SDK data separate", data={"scope": "wrapper_only"})


def _recover(root, plan, freeze):
    last, starts = attempt_index(root, plan)
    methods = {m["id"]: m for m in plan["methods"]}
    for aid, row in last.items():
        if row["status"] == "unconfirmed":
            raise ContractError(f"{aid}: previous cleanup unconfirmed; cannot dispatch safely")
        if row["status"] != "reserved":
            continue
        work = root / "raw" / aid
        slot = next(a for a in plan["attempts"] if a["id"] == aid)
        method = methods[slot["method_id"]]
        receipt_path = work / "receipt.json"
        if (work / "job.json").exists() and not receipt_path.exists():
            record_attempt(root, plan, slot, "unconfirmed", reason="no supervisor receipt after controller interruption",
                           cleanup_confirmed=False, candidate=None)
            raise ContractError(f"{aid}: interrupted with no cleanup receipt; preserve run and inspect")
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        clean = receipt is None or receipt["cleanup_confirmed"]
        if receipt is not None:
            identity = read_json(work / "recorder.json")
            rec = Recorder(root / "events.jsonl", **identity)
            _process_receipt(rec, receipt, external=method["driver"] == "external")
            try:
                merge_child(root, work, identity, wrapper=method["driver"] == "external")
            except ContractError as exc:
                rec.emit("note", status="failed", reason=f"child events not merged on recovery: {exc}")
        record_attempt(root, plan, slot, "interrupted" if clean else "unconfirmed",
                       reason="controller interrupted after reservation; no redispatch", cleanup_confirmed=clean,
                       receipt=ref(root, receipt_path) if receipt is not None else None, candidate=None)
        if not clean:
            raise ContractError(f"{aid}: descendant cleanup unconfirmed; stopped")
        if clean and receipt is not None:
            verify_method(root, plan, freeze, method)
    return attempt_index(root, plan)


def _execute(root, fd, plan, freeze, slot, method, task, state):
    aid = slot["id"]
    work = root / "raw" / aid
    work.mkdir(exist_ok=False)
    external = method["driver"] == "external"
    identity = {"run_id": plan["run_id"], "method_id": method["id"], "state_id": state["state_id"],
                "task_id": task["id"], "attempt_id": aid, "limits": {k: v for k, v in slot["requests"].items()
                                                                   if k != "dispatch"}}
    recorder = Recorder(root / "events.jsonl", **identity)
    record_attempt(root, plan, slot, "reserved", requests=slot["requests"], state_id=state["state_id"],
                   utc=recorder.clock.utc())
    recorder.emit("budget", role="controller", status="reserved", metric="dispatch", value=1, unit="calls",
                  measurement_status="observed", data={"request_envelope": slot["requests"]})
    write_json(work / "recorder.json", identity, create=True)
    request = {"task": task, "method": {k: method[k] for k in ("id", "driver", "entrypoint", "identity_files")},
               "state": {"method_id": method["id"], "carrier": state["carrier"], "metadata": state["metadata"],
                         "files": {name: {**item, "path": str(root / item["path"])}
                                   for name, item in state["files"].items()}}, "runtime": plan["runtime"]}
    command = method["command"] if external else [sys.executable, "-B", "-m", "capability_eval.worker"]
    write_json(work / "request.json", request, create=True)
    env = isolated_env(work, threads=plan["runtime"]["threads"])
    env.update(CAPABILITY_EVAL_RECORDER=str(work / "recorder.json"), CAPABILITY_EVAL_REQUEST=str(work / "request.json"))
    receipt = launch(work, fd, command, plan["budget"]["attempt_timeout_seconds"], env)
    status = receipt["status"]
    reason = receipt["reason"]
    counts = {}
    try:
        counts = merge_child(root, work, identity, wrapper=external)
    except ContractError as exc:
        if receipt["cleanup_confirmed"]:
            status, reason = "failed", str(exc)
    _process_receipt(recorder, receipt, external=external)
    if receipt["cleanup_confirmed"]:
        try:
            verify_method(root, plan, freeze, method)
        except (OSError, ContractError) as exc:
            record_attempt(root, plan, slot, "failed", reason=str(exc), cleanup_confirmed=True,
                           receipt=ref(root, work / "receipt.json"), candidate=None)
            raise ContractError(f"{aid}: state changed while in use; stopped") from exc
    if status == "completed" and method["driver"] == "transformer" and counts != {
            key: value for key, value in identity["limits"].items() if value}:
        status, reason = "failed", "incomplete model reservation evidence"
    if status == "completed" and external:
        data = (work / "stdout.log").read_bytes()
        write_bytes(work / "raw.txt", data, create=True)
        write_bytes(work / "candidate.txt", data, create=True)
    if status == "completed" and not (work / "candidate.txt").is_file():
        status, reason = "failed", "candidate was not delivered"
    return record_attempt(root, plan, slot, status, reason=reason, cleanup_confirmed=receipt["cleanup_confirmed"],
                          receipt=ref(root, work / "receipt.json"), actual_sdk_requests=counts,
                          candidate=ref(root, work / "candidate.txt") if status == "completed" else None)


def run(root, *, only=None, command=None):
    with run_lock(root) as fd:
        plan, freeze = read_run(root, check_environment=True)
        assert_score_cleanup(root)
        if plan["provenance"] != "current":
            raise ContractError("historical runs cannot be dispatched")
        methods = {m["id"]: m for m in plan["methods"]}
        tasks = {t["id"]: t for t in plan["tasks"]}
        selected = [a for a in plan["attempts"] if only is None or a["id"] == only]
        if only is not None:
            if not selected or methods[selected[0]["method_id"]]["driver"] != "external":
                raise ContractError("wrap requires a pre-planned external attempt id")
            if command != methods[selected[0]["method_id"]]["command"]:
                raise ContractError("command does not match exact frozen argv")
        last, starts = _recover(root, plan, freeze)
        if only in last:
            raise ContractError("attempt was already started; will not dispatch again")
        used = Counter()
        for row in starts.values():
            used.update(row["requests"])
        sent = len(starts)
        blocked, executed = {}, {}
        for slot in selected:
            aid = slot["id"]
            if aid in starts:
                continue
            method = methods[slot["method_id"]]
            if freeze["states"][method["id"]]["status"] != "completed":
                blocked[aid] = "prepare_failed"
                continue
            if sent >= plan["budget"]["max_attempts"] or any(used[k] + n > plan["budget"]["max_requests"].get(k, 0)
                                                              for k, n in slot["requests"].items()):
                blocked[aid] = "request_budget"
                continue
            state = verify_method(root, plan, freeze, method)
            row = _execute(root, fd, plan, freeze, slot, method, tasks[slot["task_id"]], state)
            used.update(slot["requests"])
            sent += 1
            executed[aid] = row["status"]
            if not row["cleanup_confirmed"]:
                raise ContractError(f"{aid}: descendant cleanup unconfirmed; stopped without settling resources")
        return {"planned": len(plan["attempts"]), "previously_started": len(last), "executed": executed,
                "not_started_reasons": blocked, "reserved_requests": dict(used)}


def assert_score_cleanup(root):
    for job in (root / "raw").glob("scoring-*/*/job.json"):
        receipt = job.parent / "receipt.json"
        if not receipt.is_file() or read_json(receipt).get("cleanup_confirmed") is not True:
            raise ContractError(f"grader cleanup unconfirmed: {job.parent.name}")
