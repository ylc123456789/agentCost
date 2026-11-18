from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

from ..contracts import VERSION
from ..controller import record_attempt
from ..io import (ContractError, append, digest, hash_file, read_json, ref, rows, safe_file,
                  write_bytes, write_json)
from ..ledger import Recorder
from ..plan import environment


def _copy(root, relative, source):
    target = safe_file(root, f"raw/import/{relative}")
    write_bytes(target, source.read_bytes(), create=True)
    item = ref(root, target)
    if item["sha256"] != hash_file(source):
        raise ContractError(f"source changed during import: {source}")
    return item


def _v20(root, source):
    files = ["ATTEMPT_LEDGER.json", "BATCH_FREEZE.json", "runtime_events.jsonl", "GPU_BUDGET.json",
             "resource_samples.jsonl", "COMPLETION.json"]
    receipts = {name: _copy(root, name, source / name) for name in files if (source / name).exists()}
    batch = read_json(root / receipts["BATCH_FREEZE.json"]["path"])
    ledger = read_json(root / receipts["ATTEMPT_LEDGER.json"]["path"])
    if len(set(batch["task_ids"])) != len(batch["task_ids"]) or len(set(batch["conditions"])) != len(batch["conditions"]):
        raise ContractError("duplicate task or condition in historical freeze")
    if [row["task_id"] for row in batch["sequence"]] != batch["task_ids"]:
        raise ContractError("historical sequence does not match frozen task list")
    if any(len(row["order"]) != len(batch["conditions"]) or set(row["order"]) != set(batch["conditions"])
           for row in batch["sequence"]):
        raise ContractError("historical sequence does not match frozen condition set")
    sequence = [(row["task_id"], mid) for row in batch["sequence"] for mid in row["order"]]
    ledger_rows = {(row["task_id"], row["condition"]): row for row in ledger["items"]}
    if (len(sequence) != ledger["planned_actor_count"] or len(ledger_rows) != len(ledger["items"])
            or set(ledger_rows) - set(sequence)):
        raise ContractError("historical ledger and pre-generation denominator do not agree")
    run_id = uuid.uuid4().hex
    attempts = []
    for i, (tid, mid) in enumerate(sequence, 1):
        if not re.fullmatch(r"BigCodeBench/\d+", tid) or not re.fullmatch(r"[A-Za-z0-9-]+", mid):
            raise ContractError("unexpected v20 task or method id")
        attempts.append({"id": f"a{i:06}", "task_id": tid, "method_id": mid, "replicate": 0, "requests": {}})
    plan = _plan(run_id, [dict(id=tid, dataset="BigCodeBench/Complete", library=None, family=None)
                          for tid in batch["task_ids"]], batch["conditions"], attempts, "planned", "v20-historical")
    for slot in attempts:
        tid, mid = slot["task_id"], slot["method_id"]
        row = ledger_rows.get((tid, mid))
        if row is None or row["status"] in {"pending", "planned", "not_started", "unreached"}:
            continue
        prefix = f"actors/{tid.replace('/', '_')}/{mid}"
        item = {name: _copy(root, f"{prefix}/{name}", source / prefix / name) for name in
                ("status.json", "score.json", "final.txt", "extracted-source.py", "score.stderr")
                if (source / prefix / name).is_file()}
        if bool(row.get("raw_output_exists")) != ("final.txt" in item) or bool(row.get("score_record_exists")) != ("score.json" in item):
            raise ContractError(f"v20 declared raw/score file mismatch: {tid} {mid}")
        record_attempt(root, plan, slot, "reserved", requests={}, source="imported", inferred_reservation=True)
        terminal = row["status"] if row["status"] in {"completed", "failed", "timeout", "interrupted"} else "interrupted"
        candidate = item.get("final.txt")
        record_attempt(root, plan, slot, terminal, candidate=candidate, source="imported", cleanup_confirmed=None,
                       original_status=row["status"], original_files=item)
        rec = Recorder(root / "events.jsonl", run_id=run_id, method_id=mid, task_id=tid, attempt_id=slot["id"], device_id=None)
        status = read_json(root / item["status.json"]["path"]) if "status.json" in item else {}
        for key, label, phase in (("seconds", "legacy_actor_seconds", "serve"), ("scorer", "legacy_scorer_seconds", "evaluation")):
            value = status.get(key)
            if key == "scorer":
                value = (value or {}).get("seconds")
            rec.measure(label, value, "s", source="imported", phase=phase, clock_domain=None, monotonic_ns=None,
                        utc=None, process_id=None, artifact_refs=[item["status.json"]] if "status.json" in item else [],
                        reason="not recorded in v20" if value is None else None,
                        data={"scope": "v20_actor_including_offline_scoring" if key == "seconds" else "v20_nested_scorer",
                              "comparable_to_live_phase": False})
        if "score.json" not in item:
            continue
        grade = read_json(root / item["score.json"]["path"])
        if grade.get("task_id") != tid:
            raise ContractError("v20 score identity mismatch")
        valid = grade.get("valid_official_result") is True and type(grade.get("passed")) is bool and not grade.get("engineering_error")
        passed = grade["passed"] if valid else None
        _grade(root, plan, slot, passed, candidate, item["score.json"], grade.get("score"))
    rec = Recorder(root / "events.jsonl", run_id=run_id, device_id=None, role="runtime")
    if "runtime_events.jsonl" in receipts:
        for i, row in enumerate(rows(root / receipts["runtime_events.jsonl"]["path"]), 1):
            for field in ("allocated_peak_bytes", "reserved_peak_bytes"):
                if row.get(field) is not None:
                    rec.measure(f"legacy_torch_{field}", row[field], "B", source="imported", clock_domain=None,
                                monotonic_ns=None, process_id=None, artifact_refs=[receipts["runtime_events.jsonl"]],
                                utc=(datetime.fromtimestamp(row["epoch"], timezone.utc).isoformat() if "epoch" in row else None),
                                data={"scope": "v20_actor_or_retriever_process_cumulative", "original_line": i,
                                      "original_kind": row["kind"], "comparable_to_per_attempt_peak": False})
    for field, unit in (("gpu_hold_seconds", "s"), ("historical_training_tokens", "tokens")):
        rec.measure(field, None, unit, source="imported", clock_domain=None, monotonic_ns=None, utc=None,
                    process_id=None, reason="no compatible attribution reconstructed from this mapping")
    return plan, receipts, {"mapping_kind": "v20", "legacy_budget": ledger.get("limits"),
                            "actor_seconds": "includes offline scoring; not pure serve", "torch_peaks": "cumulative process peak",
                            "planned_denominator_source": receipts["BATCH_FREEZE.json"]}


def _plan(run_id, tasks, methods, attempts, denominator, label):
    return {"schema_version": VERSION, "run_id": run_id, "experiment_id": label, "kind": "plan_freeze",
            "provenance": "imported", "denominator_kind": denominator, "tasks": tasks,
            "methods": [dict(id=m, driver="historical") for m in methods], "attempts": attempts,
            "analysis": {"baseline": methods[0] if methods else None, "quality_floor": None, "workload_sha256": None},
            "runtime": {"session_policy": "historical_unknown"}, "scoring": {"kind": "historical"}}


def _grade(root, plan, slot, passed, candidate, original, raw_score):
    append(root / "scores.v0001.jsonl", {"schema_version": VERSION, "run_id": plan["run_id"], "source": "imported",
           "attempt_id": slot["id"], "score_version": 1, "candidate": candidate, "original_score_ref": original,
           "score": None if passed is None else int(passed), "passed": passed, "original_score": raw_score,
           "reason": "original score validity unavailable" if passed is None else None})


def _flat(root, source, mapping):
    original = _copy(root, "original.jsonl", source)
    records = list(rows(root / original["path"]))
    fields = mapping["fields"]
    observed = [{"id": f"a{i:06}", "task_id": str(r[fields["task_id"]]), "method_id": str(r[fields["method_id"]]),
                 "replicate": i, "requests": {}} for i, r in enumerate(records, 1)]
    methods = list(dict.fromkeys(r["method_id"] for r in observed))
    tasks = [dict(id=tid, family=None, library=None) for tid in dict.fromkeys(r["task_id"] for r in observed)]
    plan = _plan(uuid.uuid4().hex, tasks, methods, observed, "observed_only", "offline-jsonl-import")
    for i, (event, slot) in enumerate(zip(records, observed), 1):
        terminal = mapping["statuses"].get(str(event.get(fields["status"])), "interrupted")
        if terminal not in {"completed", "failed", "timeout", "interrupted"}:
            raise ContractError("flat mapping contains an unsupported attempt status")
        record_attempt(root, plan, slot, "reserved", requests={}, source="imported", inferred_reservation=True)
        record_attempt(root, plan, slot, terminal, candidate=None, source="imported", cleanup_confirmed=None, original_line=i)
        if "passed" in fields:
            value = event.get(fields["passed"])
            _grade(root, plan, slot, value if type(value) is bool else None, None, original, value)
        rec = Recorder(root / "events.jsonl", run_id=plan["run_id"], task_id=slot["task_id"],
                       method_id=slot["method_id"], attempt_id=slot["id"], device_id=None)
        for metric in mapping.get("metrics", []):
            value = event.get(metric["field"])
            rec.measure(metric["label"], value, metric["unit"], source="imported", phase=metric["phase"],
                        measurement_status=(metric.get("measurement_status", "observed") if value is not None else "unavailable"),
                        reason="missing original observation" if value is None else None,
                        clock_domain=None, monotonic_ns=None, utc=None, process_id=None, artifact_refs=[original],
                        data={"scope": metric["boundary"], "original_line": i})
    return plan, {"original": original}, {"mapping_kind": "flat-jsonl", "unobserved_plan_denominator": None}


def import_events(source, mapping_path, root):
    mapping = read_json(mapping_path)
    if mapping["schema_version"] != VERSION or not mapping["mapping_version"] or mapping["kind"] not in {"v20", "flat-jsonl"}:
        raise ContractError("unsupported event import mapping")
    root.mkdir(parents=True, exist_ok=False)
    for name in ("attempts.jsonl", "events.jsonl", "scores.v0001.jsonl"):
        write_bytes(root / name, b"", create=True)
    mref = _copy(root, "mapping.json", mapping_path)
    plan, sources, conventions = _v20(root, source) if mapping["kind"] == "v20" else _flat(root, source, mapping)
    plan["import"] = {"mapping_version": mapping["mapping_version"], "mapping": mref, "original_records": sources, **conventions}
    write_json(root / "plan.json", plan, create=True)
    write_json(root / "environment.json", {"importer_environment": environment(), "historical_environment": None}, create=True)
    state = {"schema_version": VERSION, "run_id": plan["run_id"], "plan_sha256": hash_file(root / "plan.json"),
             "environment_sha256": hash_file(root / "environment.json"), "components": {},
             "states": {m["id"]: {"status": "imported", "historical": "unknown", "prepare_ids": []} for m in plan["methods"]}}
    write_json(root / "artifact_freeze.json", {"payload_sha256": digest(state), **state}, create=True)
    scores = root / "scores.v0001.jsonl"
    write_json(root / "scorecomplete.v0001.json", {"score_version": 1, "journal": ref(root, scores)}, create=True)
    write_bytes(root / "scores.jsonl", scores.read_bytes(), create=True)
    return {"run_id": plan["run_id"], "denominator": len(plan["attempts"]), "denominator_kind": plan["denominator_kind"],
            "source": "imported", "mapping_version": mapping["mapping_version"]}
