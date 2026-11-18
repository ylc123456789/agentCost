from __future__ import annotations

from collections import defaultdict

from ..contracts import PHASES
from ..io import ContractError
from ..ledger import validate_event

DEPLOYMENT = {"load", "warmup", "serve", "deployment_overhead"}
CPU = ("cpu_user_seconds", "cpu_system_seconds")
GPU = "gpu_allocation_seconds"


def union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if hi < lo:
            raise ContractError("negative interval")
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def length(intervals):
    return sum(hi - lo for lo, hi in union(intervals)) / 1e9


def overlap(a, b):
    return length([(max(lo, start), min(hi, end)) for lo, hi in union(a) for start, end in union(b)
                   if max(lo, start) < min(hi, end)])


def intervals(events, run_id):
    started, closed, event_ids = {}, {}, set()
    for e in events:
        validate_event(e)
        if e["run_id"] != run_id or e["event_id"] in event_ids:
            raise ContractError("event outside run or duplicate event id")
        event_ids.add(e["event_id"])
        sid = e["span_id"]
        if e["kind"] == "start":
            if sid in started:
                raise ContractError("duplicate span start")
            started[sid] = e
        elif e["kind"] == "end":
            start = started.get(sid)
            fields = ("phase", "clock_domain", "method_id", "task_id", "attempt_id", "state_id", "device_id",
                      "process_id", "parent_span_id")
            if start is None or sid in closed or any(start[k] != e[k] for k in fields):
                raise ContractError("span end without matching start identity")
            if start["monotonic_ns"] > e["monotonic_ns"]:
                raise ContractError("span clock stepped backward")
            closed[sid] = {**start, "lo": start["monotonic_ns"], "hi": e["monotonic_ns"],
                           "status": e["status"]}
    return list(closed.values()), [e for sid, e in started.items() if sid not in closed]


def _groups(spans):
    clocks = defaultdict(list)
    for s in spans:
        clocks[s["clock_domain"]].append((s["lo"], s["hi"]))
    return clocks


def _wall(spans):
    return sum(length(v) for v in _groups(spans).values())


def coverage(value, complete, reason=None, **details):
    return {"value": value if complete else None, "observed_lower_bound": value,
            "complete": complete, "reason": reason if not complete else None, **details}


def _eligible(s):
    return s["status"] != "unconfirmed" and s["measurement_status"] == "observed"


def peak_series(events):
    groups = defaultdict(list)
    for e in events:
        if e["kind"] not in {"measurement", "budget"}:
            continue
        groups[(e["metric"], e["data"].get("scope", "request_reservations" if e["kind"] == "budget" else "unspecified"), e["unit"], e["device_id"],
                e["source"], e["measurement_status"])].append(e)
    result = []
    for (name, scope, unit, dev, source, status), es in sorted(groups.items(), key=lambda pair: str(pair[0])):
        values = [e["value"] for e in es if e["value"] is not None]
        aggregate = "max" if any(word in name for word in ("peak", "rss", "baseline")) else "sum_observations"
        # These are observation series, not deployment totals. Nested scopes are kept apart.
        value = (max(values) if aggregate == "max" else sum(values)) if values else None
        result.append(dict(metric=name, scope=scope, unit=unit, device_id=dev, source=source,
                           measurement_status=status, aggregate=aggregate, value=value,
                           measurements=len(values), records=len(es), partial=any(e["data"].get("partial") for e in es),
                           reasons=sorted({e["reason"] for e in es if e["reason"]})))
    return result


def _devices(spans, opened):
    evaluation = _groups(s for s in spans if s["phase"] == "evaluation" and _eligible(s))
    windows = defaultdict(list)
    for s in spans:
        if s["phase"] == "device_hold" and s["device_id"] not in {None, "cpu"}:
            windows[(s["device_id"], s["clock_domain"])].append(s)
    result = []
    for dev, domain in sorted(set(windows) | {(s["device_id"], s["clock_domain"]) for s in opened
                                             if s["phase"] == "device_hold" and s["device_id"] not in {None, "cpu"}}):
        records = windows.get((dev, domain), [])
        hold = [(s["lo"], s["hi"]) for s in records]
        total = length(hold)
        grading = evaluation.get(domain, [])
        method_activity = [(s["lo"], s["hi"]) for s in spans if s["clock_domain"] == domain
                           and s["device_id"] == dev and s["phase"] in DEPLOYMENT | {"prepare", "import"} and _eligible(s)]
        complete = all(_eligible(s) for s in records) and not any(s["phase"] == "device_hold" and
                       s["device_id"] == dev and s["clock_domain"] == domain for s in opened)
        result.append({"device_id": dev, "clock_domain": domain,
                       "actual_hold_seconds": coverage(total if records else None, complete,
                                                       "hold did not reach verified end"),
                       "overlap_with_offline_evaluation_seconds": overlap(hold, grading),
                       "unattributed_seconds": max(0, total - overlap(hold, grading + method_activity)),
                       "attribution": sorted({s["data"].get("attribution", "unassigned") for s in records}),
                       "kernel_busy_seconds": None, "kernel_busy_reason": "unsupported"})
    return result


def summarize(plan, freeze, events, attempts):
    spans, opened = intervals(events, plan["run_id"])
    owned = [s for s in spans if s["data"].get("accounting") == "phase"]
    # Charge only explicit phase windows inside a matching exclusive lease.
    claims = [s for s in owned if s["phase"] in DEPLOYMENT | {"prepare", "import"} and
              s["device_id"] not in {None, "cpu"} and s["data"].get("charge_gpu_allocation") is True]
    leases = [s for s in spans if s["phase"] == "device_hold" and _eligible(s)]
    gpu_charge = {}
    for claim in claims:
        colocated = [lease for lease in leases if all(lease[k] == claim[k] for k in ("device_id", "clock_domain"))]
        identity = [(lease["method_id"], lease["data"].get("prepare_id")) for lease in colocated
                    if lease["data"].get("attribution") == "exclusive" and lease["lo"] <= claim["lo"] and lease["hi"] >= claim["hi"]]
        conflict = any(s is not claim and s["clock_domain"] == claim["clock_domain"] and
                       (s["device_id"] == claim["device_id"] or (s["phase"] == "evaluation" and s["device_id"] in {None, "cpu"})) and
                       min(s["hi"], claim["hi"]) > max(s["lo"], claim["lo"])
                       for s in claims + [other for other in spans if other["phase"] == "evaluation"])
        if identity == [(claim["method_id"], claim["data"].get("prepare_id"))] and _eligible(claim) and not conflict:
            gpu_charge[claim["span_id"]] = (claim["hi"] - claim["lo"]) / 1e9
    resource_keys = ("wall_seconds", *CPU, GPU)
    cpu = defaultdict(dict)
    for e in events:
        if (e["kind"] == "measurement" and e["metric"] in CPU and e["measurement_status"] == "observed"
                and e["data"].get("scope") in {"process_tree_wait4", "calling_process_window"}
                and not e["data"].get("partial")):
            cpu[e["span_id"]][e["metric"]] = e["value"]

    def calc(records, complete, metric="wall_seconds"):
        if metric == "wall_seconds":
            value = _wall(records) if records else None
        elif metric == GPU:
            values = [gpu_charge[s["span_id"]] for s in records if s["span_id"] in gpu_charge]
            complete &= len(values) == len(records)
            value = sum(values) if values else None
        else:
            values = [cpu[s["span_id"]][metric] for s in records if metric in cpu[s["span_id"]]]
            complete &= len(values) == len(records)
            value = sum(values) if values else None
        return coverage(value, complete, "missing or incomplete comparable outer observations")

    preparations = {}
    for prep_id, part in freeze["components"].items():
        contributing = [s for s in owned if s["data"].get("prepare_id") == prep_id]
        complete = (bool(contributing) and all(_eligible(s) for s in contributing)
                    and part["status"] == "completed")
        preparations[prep_id] = {"phase": part["phase"], "status": part["status"], "historical": part["historical"],
                                  "metrics": {key: calc(contributing, complete, key) for key in resource_keys}}
    by_method = {}
    audit_open = {s["attempt_id"] for s in opened if s["data"].get("accounting") == "phase"}
    for method in plan["methods"]:
        mid = method["id"]
        slots = [s for s in plan["attempts"] if s["method_id"] == mid]
        aids = {s["id"] for s in slots}
        ended = {aid for aid in aids if attempts.get(aid, {}).get("cleanup_confirmed") is True}
        relevant = [s for s in owned if s["method_id"] == mid]
        nested = [s for s in spans if s["method_id"] == mid and s["data"].get("accounting") not in
                  {"process_envelope", "device_hold"}]
        by_phase = {phase: [s for s in relevant if s["phase"] == phase] for phase in PHASES}
        profile = {}
        prepare_ids = freeze["states"].get(mid, {}).get("prepare_ids", [])
        proper = freeze["states"].get(mid, {}).get("status") == "completed"
        for key in resource_keys:
            phase_costs = {}
            for phase, records in by_phase.items():
                ids = {s["attempt_id"] for s in records if _eligible(s)}
                complete = (bool(aids) and ids == aids and ended == aids and not (aids & audit_open))
                data = calc(records, complete, key)
                if phase == "warmup" and not records and method["driver"] in {"fixture", "transformer"}:
                    data = coverage(0, True)
                data["attempts_observed"] = len(ids - {None})
                data["planned_attempts"] = len(aids)
                phase_costs[phase] = data
            parts = [preparations[p]["metrics"][key] for p in sorted(set(prepare_ids))]
            prep_ok = proper and all(item["complete"] for item in parts)
            prep_value = sum(item["value"] for item in parts) if prep_ok and (key != GPU or gpu_charge) else None
            profile[key] = {"fixed_current": prep_value, "phases": phase_costs}
        observed = {phase: _wall([s for s in nested if s["phase"] == phase]) for phase in PHASES
                    if any(s["phase"] == phase for s in nested)}
        # Deployment aggregate is computed from intervals, never summed nested scopes.
        deployment = [s for s in nested if s["phase"] in DEPLOYMENT]
        e2e_complete = all(profile["wall_seconds"]["phases"][p]["complete"] for p in DEPLOYMENT)
        process = [s for s in spans if s["method_id"] == mid and s["data"].get("accounting") == "process_envelope"]
        residual = 0.0
        for s in process:
            measured = [(child["lo"], child["hi"]) for child in relevant
                        if child["attempt_id"] == s["attempt_id"] and child["clock_domain"] == s["clock_domain"]
                        and child["phase"] != "evaluation"]
            residual += max(0, (s["hi"] - s["lo"]) / 1e9 - overlap([(s["lo"], s["hi"])], measured))
        by_method[mid] = {"phase_wall_observed_seconds": observed, "resources": profile,
                          "deployment_end_to_end_seconds": coverage(_wall(deployment) if deployment else None,
                                                                     e2e_complete, "not all method phases observed"),
                          "supervised_envelope_seconds_including_diagnostics": _wall(process) if process else None,
                          "supervised_residual_seconds_unallocated": residual if process else None,
                          "prepare_ids": sorted(set(prepare_ids)), "history": freeze["states"].get(mid, {}).get("historical"),
                          "metrics": peak_series([e for e in events if e["method_id"] == mid])}
    devices = _devices(spans, opened)
    return {"by_method": by_method, "preparation": preparations, "shared_preparation_wall_seconds_once":
            _wall([s for s in owned if s["phase"] in {"prepare", "import"}]) if preparations else None,
            "all_phase_wall_seconds": {phase: _wall([s for s in spans if s["phase"] == phase and
                   s["data"].get("accounting") != "process_envelope"]) for phase in PHASES
                   if any(s["phase"] == phase and s["data"].get("accounting") != "process_envelope" for s in spans)},
            "device_windows": devices, "gpu_hold": {"value": sum(d["actual_hold_seconds"]["value"] for d in devices)
                    if devices and all(d["actual_hold_seconds"]["complete"] for d in devices) else None,
                    "reason": ("no device lease observed" if not devices else "one or more device leases are incomplete"
                               if any(not d["actual_hold_seconds"]["complete"] for d in devices) else None)},
            "open_spans": [{k: s[k] for k in ("phase", "span_id", "attempt_id", "clock_domain", "monotonic_ns")}
                           for s in opened], "run_level_metrics": peak_series([e for e in events if e["method_id"] is None])}
