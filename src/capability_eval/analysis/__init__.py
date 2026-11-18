from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..controller import attempt_index
from ..io import ContractError, hash_file, read_json, ref, rows, run_lock, verify_ref, write_bytes, write_json
from ..state import read_run
from ..tasks.scoring import selected_scores
from .resources import summarize


def crossing(candidate_fixed, baseline_fixed, candidate_use, baseline_use, *, eligible, reason=None):
    if not eligible or any(item is None for item in (candidate_fixed, baseline_fixed, candidate_use, baseline_use)):
        return {"tasks": None, "reason": reason or "quality or comparable cost not established"}
    saving, extra = baseline_use - candidate_use, candidate_fixed - baseline_fixed
    if saving <= 0:
        return {"tasks": None, "reason": "no positive per-task saving"}
    if extra <= 0:
        return {"tasks": None, "reason": "no positive up-front premium"}
    return {"tasks": extra / saving, "reason": None, "fixed_difference": extra, "saving_per_task": saving}


def quality(plan, attempts, scores):
    qualities, by_pair = {}, {}
    for method in plan["methods"]:
        mid = method["id"]
        slots = [a for a in plan["attempts"] if a["method_id"] == mid]
        counts = Counter(attempts.get(a["id"], {}).get("status", "not_started") for a in slots)
        values = {a["id"]: scores.get(a["id"], {}).get("passed") for a in slots}
        valid = sum(type(value) is bool for value in values.values())
        passed = sum(value is True for value in values.values())
        denominator = len(slots)
        qualities[mid] = {"pass": passed, "denominator": denominator, "denominator_kind": plan["denominator_kind"],
                          "confirmed_pass_rate": passed / denominator if denominator else None,
                          "valid": valid, "valid_fail": valid - passed, "missing_scores": denominator - valid,
                          "not_started": counts["not_started"], "pending": counts["reserved"],
                          "execution_failed": counts["failed"] + counts["timeout"] + counts["interrupted"] + counts["unconfirmed"],
                          "statuses": dict(counts)}
        by_pair[mid] = {(a["task_id"], a["replicate"]): values[a["id"]] for a in slots}
    paired = {}
    baseline = plan["analysis"]["baseline"]
    if plan["denominator_kind"] == "planned" and baseline in by_pair:
        for mid, ours in by_pair.items():
            if mid == baseline:
                continue
            theirs = by_pair[baseline]
            match = set(ours) & set(theirs)
            valid = sorted(key for key in match if type(ours[key]) is bool and type(theirs[key]) is bool)
            added = [list(key) for key in valid if ours[key] and not theirs[key]]
            lost = [list(key) for key in valid if not ours[key] and theirs[key]]
            # Display family/library without assuming unrecorded families or a statistical sample.
            info = {t["id"]: t for t in plan["tasks"]}
            by_library = Counter(info.get(key[0], {}).get("library") or "unavailable" for key in valid)
            paired[mid] = {"baseline": baseline, "valid_pairs": len(valid), "planned_pairs": len(match),
                           "pair_coverage": len(valid) / len(match) if match else None,
                           "added": len(added), "lost": len(lost), "added_ids": added, "lost_ids": lost,
                           "valid_by_library": dict(by_library), "inference": "descriptive_only"}
    return qualities, paired


def _workload(root, path, plan, res, qualities):
    if path is None:
        if plan["analysis"]["workload_sha256"] is not None:
            raise ContractError("analysis requires the frozen --workload file")
        return {"sha256": None, "scenarios": [], "break_even": []}
    manifest = read_json(path)
    sha = hash_file(path)
    planned_sha = plan["analysis"]["workload_sha256"]
    if planned_sha is not None and planned_sha != sha:
        raise ContractError("workload was changed after PlanFreeze")
    if manifest["schema_version"] != "1.0":
        raise ContractError("unsupported workload schema")
    scenarios = []
    method_results = res["by_method"]
    threshold = plan["analysis"]["quality_floor"]
    baseline = plan["analysis"]["baseline"]
    def floor(mid):
        q = qualities[mid]
        return (threshold is not None and q["denominator_kind"] == "planned" and
                q["valid"] == q["denominator"] and q["denominator"] > 0 and q["confirmed_pass_rate"] >= threshold)

    for scenario in manifest["scenarios"]:
        n, count = scenario["tasks"], scenario["sessions"]
        if type(n) is not int or type(count) is not int or n < 1 or not 1 <= count <= n:
            raise ContractError("scenario needs positive tasks and 1 <= sessions <= tasks")
        record = {"id": scenario["id"], "tasks": n, "sessions": count, "methods": {}}
        for mid, summary in method_results.items():
            observed_count = qualities[mid]["denominator"]
            profiles = {}
            for resource, raw in summary["resources"].items():
                phases = raw["phases"]
                fields = [phases[phase]["value"] for phase in ("load", "warmup", "serve", "deployment_overhead")]
                compatible = plan.get("runtime", {}).get("session_policy") == "per_attempt_cold" and count == n
                known_history = plan["analysis"].get("known_historical", {}).get(mid, {}).get(resource)
                base = {"unit": "s", "known_historical": known_history,
                        "historical_status": "known" if known_history is not None else summary["history"],
                        "current_fixed": raw["fixed_current"], "current_total": None,
                        "total_including_history": None, "quality_floor_observed": floor(mid)}
                if raw["fixed_current"] is None or None in fields or not compatible or observed_count < 1:
                    profiles[resource] = {**base, "reason": "incomplete observations or incompatible cold-session policy"}
                    continue
                load, warm, serve, other = (v / observed_count for v in fields)
                current = raw["fixed_current"] + count * (load + warm) + n * serve + count * other
                profiles[resource] = {**base, "basis": "observed cold mean projected to declared count",
                                      "current_total": current, "load_per_session": load + warm,
                                      "serve_per_task": serve, "other_per_session": other,
                                      "total_including_history": (current + known_history if known_history is not None
                                       else current if summary["history"] == "not_applicable" else None), "reason": None}
            record["methods"][mid] = profiles
        scenarios.append(record)
    matches = []
    if scenarios and baseline in method_results:
        # At one cold session per request, per-session costs move with n, so are variable, not fixed.
        sample = next((s for s in scenarios if s["tasks"] == s["sessions"]), scenarios[0])["methods"]
        for mid, profiles in sample.items():
            if mid == baseline:
                continue
            for key, item in profiles.items():
                base = sample[baseline][key]
                comparable = floor(mid) and floor(baseline) and item["current_total"] is not None and base["current_total"] is not None
                per_task = lambda v: v.get("load_per_session", 0) + v.get("serve_per_task", 0) + v.get("other_per_session", 0)
                current = crossing(item["current_fixed"], base["current_fixed"], per_task(item), per_task(base), eligible=comparable)
                hist = lambda v: v["known_historical"] if v["historical_status"] == "known" else 0
                if item["total_including_history"] is not None and base["total_including_history"] is not None:
                    total = crossing(item["current_fixed"] + hist(item), base["current_fixed"] + hist(base),
                                     per_task(item), per_task(base), eligible=comparable)
                else:
                    total = {"tasks": None, "reason": "history unknown or comparable cost unavailable"}
                matches.append({"method": mid, "baseline": baseline, "resource": key, "policy": "one cold session per task",
                                "current": current, "including_history": total})
    return {"sha256": sha, "frozen": planned_sha is not None, "scenarios": scenarios, "break_even": matches,
            "updates": {"status": "unsupported", "cost": None}}


def write_manifest(root):
    entries = [ref(root, p) for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
               and p.name not in {"checksums.json", ".run.lock"}]
    write_json(root / "checksums.json", {"algorithm": "sha256", "files": entries})
    return len(entries)


def verify(root):
    with run_lock(root):
        read_run(root)
        entries = read_json(root / "checksums.json")["files"]
        for item in entries:
            verify_ref(root, item)
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink()
                  and p.name not in {"checksums.json", ".run.lock"}}
        if actual != {e["path"] for e in entries}:
            raise ContractError("run directory contents differ from checksum manifest")
        return len(entries)


def _report(plan, res, result):
    show = lambda n: "NA" if n is None else f"{n:.6g}"
    text = [f"# Capability Eval | {plan['experiment_id']}", "", f"Run `{plan['run_id']}`. Source: `{plan['provenance']}`.",
            "", "## Quality and execution", "",
            "|Method|Pass/N|Valid|Failed execution|Scores missing|Not started|",
            "|---|---:|---:|---:|---:|---:|"]
    for mid, q in result["quality"].items():
        text.append(f"|{mid}|{q['pass']}/{q['denominator']}|{q['valid']}|{q['execution_failed']}|{q['missing_scores']}|{q['not_started']}|")
    for mid, paired in result["pairs"].items():
        text.append(f"\n{mid} vs {paired['baseline']}: +{paired['added']}/-{paired['lost']}; "
                    f"paired {paired['valid_pairs']}/{paired['planned_pairs']} (descriptive only).")
    text += ["", "## Measurement boundaries", "",
             "|Method|Prep/import s¹|Load/warmup s²|Serve observed s|Grade observed s|Deployment phase union s²|",
             "|---|---:|---:|---:|---:|---:|"]
    for mid, row in res["by_method"].items():
        phase = row["phase_wall_observed_seconds"]
        wall = row["resources"]["wall_seconds"]
        load = wall["phases"]["load"]["value"]
        warm = wall["phases"]["warmup"]["value"]
        text.append(f"|{mid}|{show(wall['fixed_current'])}|{show(load + warm if load is not None and warm is not None else None)}|"
                    f"{show(phase.get('serve'))}|{show(phase.get('evaluation'))}|{show(row['deployment_end_to_end_seconds']['value'])}|")
    text += ["", "¹ Per-method view; shared artifacts counted only once in the run preparation total. "
             "² Complete comparable phase windows only; excludes separately noted process residuals.", "",
             "|Method|Observed serve / planned|Outer process CPU user s|Outer process CPU system s|Charged GPU serve s|Torch alloc./reserved peak B|",
             "|---|---:|---:|---:|---:|---:|"]
    for mid, row in res["by_method"].items():
        indexed = {(m["metric"], m["scope"]): m["value"] for m in row["metrics"]
                   if m["source"] == "current" and m["measurement_status"] == "observed"}
        tracked = row["resources"]["wall_seconds"]["phases"]["serve"]
        alloc = row["resources"]["gpu_allocation_seconds"]["phases"]["serve"]["value"]
        user = indexed.get(("cpu_user_seconds", "process_tree_wait4"))
        system = indexed.get(("cpu_system_seconds", "process_tree_wait4"))
        pa = indexed.get(("torch_allocated_peak_bytes", "torch_allocator"))
        pr = indexed.get(("torch_reserved_peak_bytes", "torch_allocator"))
        text.append(f"|{mid}|{tracked['attempts_observed']}/{tracked['planned_attempts']}|{show(user)}|{show(system)}|"
                    f"{show(alloc)}|{show(pa)} / {show(pr)}|")
    text += ["", f"Device hold: {show(res['gpu_hold']['value'])} s. "
             "Offline grading is not deployment serve. A device lease, when measured, includes any overlapping grading.",
             f"Shared preparation observed once: {show(res['shared_preparation_wall_seconds_once'])} s.",
             f"Open spans: {len(res['open_spans'])}.", "",
             "See `resource_summary.json` for CPU user/system, peaks by scope, imported measurement boundaries, coverage and missing reasons.",
             "", "## Workload (descriptive projection)"]
    for s in result["workload"]["scenarios"]:
        text.append(f"\n- {s['id']}: {s['tasks']} tasks / {s['sessions']} sessions; wall seconds current / incl. history: " +
                    ", ".join(f"{mid} {show(cost['wall_seconds']['current_total'])} / {show(cost['wall_seconds']['total_including_history'])}"
                              for mid, cost in s["methods"].items()))
    if plan.get("import", {}).get("mapping_kind") == "v20":
        text += ["", "Legacy actor seconds include scoring; legacy Torch peaks are process cumulative, not per-attempt. "
                 "No compatible live deployment cost or historical preparation cost is inferred."]
    text += ["", "Rebuild from frozen inputs and versioned scores with `capability-eval analyze`; JSON records include source hashes.\n"]
    return "\n".join(text).encode()


def analyze(root, *, workload=None, output=None, ledger_only=False):
    if ledger_only and output is None:
        raise ContractError("--ledger-only requires a separate --output and does not certify raw files")
    if output is not None and Path(output).resolve().is_relative_to(root):
        raise ContractError("rebuild output must be outside the original run directory")
    with run_lock(root):
        plan, freeze = read_run(root)
        events = list(rows(root / "events.jsonl"))
        last, _ = attempt_index(root, plan)
        grade_path, version = selected_scores(root)
        grades = {}
        for row in rows(grade_path):
            aid = row["attempt_id"]
            if aid in grades or row["run_id"] != plan["run_id"] or aid not in last or row["candidate"] != last[aid]["candidate"]:
                raise ContractError("score does not match frozen attempt and candidate")
            if row["passed"] is not None and (type(row["passed"]) is not bool or row["score"] != int(row["passed"])):
                raise ContractError("invalid binary score")
            if row["passed"] is None and row["score"] is not None:
                raise ContractError("missing binary score must remain null")
            grades[aid] = row
        if not ledger_only:
            for row in last.values():
                if row.get("candidate") is not None:
                    verify_ref(root, row["candidate"])
                if row.get("receipt") is not None:
                    verify_ref(root, row["receipt"])
        resources = summarize(plan, freeze, events, last)
        result, paired = quality(plan, last, grades)
        inputs = {path.relative_to(root).as_posix(): hash_file(path) for path in
                  (root / "plan.json", root / "artifact_freeze.json", root / "environment.json", root / "events.jsonl",
                   root / "attempts.jsonl", grade_path)}
        reasons = {a["id"]: ("prepare_failed" if freeze["states"][a["method_id"]]["status"] == "failed" else
                             "not_dispatched; see frozen budget and controller receipt")
                   for a in plan["attempts"] if a["id"] not in last}
        analysis = {"schema_version": plan["schema_version"], "run_id": plan["run_id"], "inputs_sha256": inputs,
                    "quality": result, "pairs": paired, "score_version": version,
                    "not_started_reasons": reasons,
                    "workload": _workload(root, workload, plan, resources, result),
                    "stage_status": {"scoring": "completed" if version is not None else "not_run_or_no_grader",
                                     "official": "unverified" if plan["provenance"] == "current" else "imported_not_revalidated"}}
        target = Path(output) if output else root
        if output:
            target.mkdir(parents=True, exist_ok=False)
        write_json(target / "resource_summary.json", resources)
        write_json(target / "analysis.json", analysis)
        write_bytes(target / "report.md", _report(plan, resources, analysis))
        return {"output": str(target), "quality": result, "raw_verified": not ledger_only,
                "files_checked": None if output else write_manifest(root)}
