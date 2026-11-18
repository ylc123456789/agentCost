from __future__ import annotations

import math
import os
import platform
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contracts import PHASES, ROLES, VERSION
from .io import ContractError, append, hash_file, ref, write_bytes


class Clock:
    def __init__(self):
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            boot = uuid.uuid4().hex
        self.domain = f"monotonic:{platform.node()}:{boot}"

    def now(self):
        return time.monotonic_ns()

    def utc(self):
        return datetime.now(timezone.utc).isoformat()


EVENT_FIELDS = frozenset("""
schema_version run_id event_id kind method_id state_id task_id attempt_id span_id
parent_span_id phase component_role clock_domain monotonic_ns utc process_id device_id
status source measurement_status metric value unit reason artifact_refs data
""".split())


def validate_event(event):
    if set(event) != EVENT_FIELDS or event["schema_version"] != VERSION:
        raise ContractError("event fields or schema version do not match v1")
    if event["phase"] not in PHASES or event["component_role"] not in ROLES:
        raise ContractError("unknown event phase or role")
    if event["kind"] not in {"start", "end", "measurement", "budget", "artifact", "note"}:
        raise ContractError("unknown event kind")
    if event["status"] not in {"started", "completed", "failed", "interrupted", "timeout",
                               "unconfirmed", "reserved", "not_applicable", "unsupported"}:
        raise ContractError("unknown event status")
    for key in ("event_id", "run_id"):
        if not isinstance(event[key], str) or not event[key]:
            raise ContractError(f"empty event {key}")
    for key in ("method_id", "state_id", "task_id", "attempt_id", "span_id", "parent_span_id",
                "clock_domain", "utc", "device_id", "metric", "unit", "reason"):
        if event[key] is not None and not isinstance(event[key], str):
            raise ContractError(f"invalid event {key}")
    for key in ("monotonic_ns", "process_id"):
        if event[key] is not None and (type(event[key]) is not int or event[key] < 0):
            raise ContractError(f"invalid event {key}")
    if event["kind"] in {"start", "end"} and any(event[k] is None for k in
                                                  ("span_id", "clock_domain", "monotonic_ns")):
        raise ContractError("timed span requires a clock, a moment and a span id")
    if event["source"] not in {"current", "imported"}:
        raise ContractError("unknown event source")
    status = event["measurement_status"]
    if status not in {"observed", "estimated", "unavailable", "not_applicable"}:
        raise ContractError("unknown measurement status")
    val = event["value"]
    if val is not None and (type(val) not in (float, int) or not math.isfinite(val) or val < 0):
        raise ContractError("metric must be finite, nonnegative or null")
    if event["kind"] == "measurement":
        if not event["metric"] or not event["unit"]:
            raise ContractError("measurement requires metric and unit")
        if (val is None) != (status in {"unavailable", "not_applicable"}):
            raise ContractError("measurement value does not match its status")
    if status == "unavailable" and (val is not None or not event["reason"]):
        raise ContractError("missing measurement requires null and a reason")
    if not isinstance(event["data"], dict) or not isinstance(event["artifact_refs"], list):
        raise ContractError("invalid data or artifact_refs")
    return event


class Recorder:
    """Process-local SDK recorder. One writer per JSONL; merge only after exit."""

    def __init__(self, journal: Path, *, run_id: str, method_id=None, state_id=None,
                 task_id=None, attempt_id=None, phase="serve", role="actor", device_id="cpu",
                 parent_span_id=None, clock=None, limits=None, evidence_dir=None, budget_used=None):
        self.journal = Path(journal)
        self.clock = clock or Clock()
        self.identity = dict(run_id=run_id, method_id=method_id, state_id=state_id,
                             task_id=task_id, attempt_id=attempt_id, device_id=device_id)
        self.phase = phase
        self.role = role
        self.parent_span_id = parent_span_id
        self.stack = []
        self.limits = dict(limits or {})
        self.used = {} if budget_used is None else budget_used
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None

    @classmethod
    def from_environment(cls):
        from .io import read_json

        config = Path(os.environ["CAPABILITY_EVAL_RECORDER"])
        identity = read_json(config)
        return cls(config.parent / "child.events.jsonl", evidence_dir=config.parent / "attachments",
                   **identity)

    def emit(self, kind, *, phase=None, role=None, span_id=None, status="completed",
             source="current", measurement_status="not_applicable", value=None, unit=None,
             metric=None, reason=None, artifact_refs=(), data=None, **overrides):
        span, parent, current_phase, current_role = self.stack[-1] if self.stack else (
            None, self.parent_span_id, self.phase, self.role)
        event = dict(self.identity, schema_version=VERSION, event_id=uuid.uuid4().hex,
                     kind=kind, span_id=span_id or span, parent_span_id=parent,
                     phase=phase or current_phase, component_role=role or current_role,
                     clock_domain=self.clock.domain, monotonic_ns=self.clock.now(),
                     utc=self.clock.utc(), process_id=os.getpid(), status=status, source=source,
                     measurement_status=measurement_status, metric=metric, value=value, unit=unit,
                     reason=reason, artifact_refs=list(artifact_refs), data=data or {})
        event.update(overrides)
        append(self.journal, validate_event(event))
        return event

    @contextmanager
    def span(self, name, *, phase=None, role=None, accounting=None, data=None):
        parent = self.stack[-1][0] if self.stack else self.parent_span_id
        phase = phase or (self.stack[-1][2] if self.stack else self.phase)
        role = role or (self.stack[-1][3] if self.stack else self.role)
        sid = uuid.uuid4().hex
        details = {"name": name, **(data or {})}
        if accounting is not None:
            details["accounting"] = accounting
        self.stack.append((sid, parent, phase, role))
        status = "completed"
        try:
            self.emit("start", status="started", data=details, measurement_status="observed")
            yield sid
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            details = {**details, "exception_type": type(exc).__name__}
            raise
        finally:
            try:
                self.emit("end", status=status, data=details, measurement_status="observed")
            finally:
                self.stack.pop()

    def measure(self, metric, value, unit, *, reason=None, measurement_status=None, **kwargs):
        return self.emit("measurement", metric=metric, value=value, unit=unit, reason=reason,
                         measurement_status=measurement_status or ("observed" if value is not None
                                                                    else "unavailable"), **kwargs)

    def reserve(self, kind, amount=1):
        if type(amount) is not int or amount < 1:
            raise ContractError("request reservation must be a positive integer")
        used = self.used.get(kind, 0) + amount
        if used > self.limits.get(kind, 0):
            raise ContractError(f"request budget exhausted or not frozen: {kind}")
        self.emit("budget", status="reserved", metric=kind, value=amount, unit="calls",
                  measurement_status="observed", data={"cumulative": used})
        self.used[kind] = used

    def attach_artifact(self, path, *, role=None, label=None):
        if self.evidence_dir is None:
            raise ContractError("recorder has no configured evidence directory")
        source = Path(path)
        sha = hash_file(source)
        target = self.evidence_dir / sha
        if not target.exists():
            write_bytes(target, source.read_bytes(), create=True)
        item = ref(self.evidence_dir.parent, target)
        if item["sha256"] != sha:
            raise ContractError("attachment changed during copy")
        self.emit("artifact", role=role, artifact_refs=[item], data={"label": label or source.name})
        return item
