from __future__ import annotations

import importlib.metadata
import math
import platform
import re
import shutil
import sys
import uuid
from pathlib import Path

from .contracts import CARRIERS, PublicTask, VERSION
from .io import ContractError, digest, hash_file, read_json

SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CPU_CARRIERS = {(), ("text",), ("input_embeds",), ("actor_lora",), ("text", "actor_lora")}


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value, label, minimum=0):
    if type(value) not in (float, int) or not math.isfinite(value) or value < minimum:
        raise ContractError(f"{label} must be a finite number >= {minimum}")
    return value


def _id(value, label):
    if not isinstance(value, str) or not SLUG.fullmatch(value):
        raise ContractError(f"invalid {label}: {value!r}")
    return value


def _sha(value, label):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{label} needs a SHA-256 hex digest")
    return value


def _distinct(records, label):
    if len({record["id"] for record in records}) != len(records):
        raise ContractError(f"duplicate {label} id")


def package_identity():
    home = Path(__file__).parent
    return digest({p.relative_to(home).as_posix(): hash_file(p) for p in sorted(home.rglob("*.py"))})


def environment():
    return {"python": platform.python_version(), "implementation": sys.implementation.name,
            "platform": platform.platform(), "architecture": platform.machine(),
            "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                         if dist.metadata["Name"]},
            "package_source_sha256": package_identity()}


def _command(argv, home):
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and "\0" not in a for a in argv):
        raise ContractError("command must be a nonempty argv array")
    argv = [arg.replace("${CONFIG_DIR}", str(home)).replace("${PYTHON}", sys.executable) for arg in argv]
    executable = shutil.which(argv[0])
    if executable is None:
        raise ContractError(f"frozen command executable not found: {argv[0]}")
    argv[0] = str(Path(executable).absolute())
    return argv


def _envelope(value):
    if not isinstance(value, dict):
        raise ContractError("requests must be an object of named limits")
    return {_id(key, "request kind"): _integer(count, f"requests.{key}") for key, count in value.items()}


def build_plan(config_path, *, run_id=None):
    path = Path(config_path).resolve()
    src = read_json(path)
    if src["schema_version"] != VERSION:
        raise ContractError("only config schema 1.0 is supported")
    home = path.parent
    if ("tasks" in src) == ("tasks_file" in src):
        raise ContractError("specify exactly one of tasks and tasks_file")
    tasks_src = (read_json(home / src["tasks_file"]) if "tasks_file" in src else src["tasks"])
    tasks = [PublicTask.from_record(row).record() for row in tasks_src]
    _distinct(tasks, "task")
    if not tasks or len({t["split"] for t in tasks}) != 1:
        raise ContractError("a run needs nonempty tasks from one split")
    if len({t["dataset"] for t in tasks}) != 1:
        raise ContractError("do not combine dataset denominators in one v1 run")
    exposed_ids = src.get("exposed_task_ids", [])
    exposed_families = src.get("exposed_families", [])
    if tasks[0]["split"] == "heldout" and any(t["id"] in exposed_ids or (t["family"] is not None and
                                    t["family"] in exposed_families) for t in tasks):
        raise ContractError("heldout task or family appears in exposure manifest")
    runtime = src.get("runtime", {})
    if set(runtime) - {"device", "seed", "threads", "max_input", "max_new_tokens"}:
        raise ContractError("unrecognized v1 runtime settings; do not silently change generation policy")
    if runtime.get("device", "cpu") != "cpu":
        raise ContractError("unsupported: built-in v0.1 runner only launches on CPU")
    runtime = {"device": "cpu", "seed": _integer(runtime.get("seed", 17), "seed"),
               "threads": _integer(runtime.get("threads", 1), "threads", 1),
               "max_input": _integer(runtime.get("max_input", 192), "max_input", 1),
               "max_new_tokens": _integer(runtime.get("max_new_tokens", 4), "max_new_tokens", 1),
               "template": "byte-chat-v1", "generation": "greedy", "session_policy": "per_attempt_cold"}
    if runtime["max_input"] + runtime["max_new_tokens"] > 4096:
        raise ContractError("tiny fixture context plus output must not exceed 4096")
    cache = src.get("cache", {})
    if (set(cache) - {"answer_cache", "request_kv_cache", "cross_task_capability_cache"}
            or cache.get("answer_cache", False) is not False
            or cache.get("cross_task_capability_cache", False) is not False
            or cache.get("request_kv_cache", True) is not True):
        raise ContractError("unsupported cache policy: answers/cross-task off, request KV on")
    cache = {"answer_cache": False, "request_kv_cache": True, "cross_task_capability_cache": False}
    materials = []
    for m in src.get("materials", []):
        ident = _id(m["id"], "material id")
        asset = (home / m["path"]).resolve(strict=True)
        sha = hash_file(asset)
        if sha != _sha(m["sha256"], "material"):
            raise ContractError(f"material hash mismatch: {ident}")
        if m.get("split") not in {"train", "dev", "public"}:
            raise ContractError(f"material not authorized for preparation: {ident}")
        if not isinstance(m.get("source"), str) or not m["source"]:
            raise ContractError(f"material provenance is missing: {ident}")
        materials.append({"id": ident, "path": str(asset), "sha256": sha, "bytes": asset.stat().st_size,
                          "source": m["source"], "split": m["split"], "version": m.get("version"),
                          "historical": m.get("historical", "unknown")})
        if materials[-1]["historical"] not in {"unknown", "not_applicable"}:
            raise ContractError("use analysis.known_historical for known historical costs")
    _distinct(materials, "material")
    material_ids = {m["id"] for m in materials}
    methods = []
    for m in src["methods"]:
        ident, driver = _id(m["id"], "method id"), m["driver"]
        carriers = tuple(sorted(m.get("carriers", []), key=lambda item: (item != "text", item)))
        if set(carriers) - CARRIERS or len(set(carriers)) != len(carriers):
            raise ContractError(f"unknown or duplicate carrier: {ident}")
        if driver not in {"fixture", "transformer", "external", "plugin"}:
            raise ContractError(f"unsupported method driver: {driver}")
        if driver == "transformer" and carriers not in CPU_CARRIERS:
            raise ContractError(f"unsupported transformer carrier combination: {carriers}")
        if driver != "transformer" and driver != "plugin" and carriers:
            raise ContractError(f"unsupported carrier for {driver}: {carriers}")
        allowed = m.get("materials", [])
        if set(allowed) - material_ids or len(set(allowed)) != len(allowed):
            raise ContractError(f"undeclared or duplicate material: {ident}")
        if driver == "transformer" and len(allowed) != int("text" in carriers):
            raise ContractError(f"text carrier needs exactly one frozen skill; other carriers need none: {ident}")
        command = _command(m["command"], home) if driver == "external" else None
        dependencies = []
        inputs = (m.get("identity_files") or []) + ([command[0]] if command else [])
        if command:
            inputs += [arg for arg in command[1:] if Path(arg).is_absolute() and Path(arg).is_file()]
        if driver == "plugin" and (not m.get("entrypoint") or not inputs):
            raise ContractError("a trusted plugin needs an entrypoint and frozen identity_files")
        for file in sorted(set(inputs)):
            asset = (home / file).absolute()
            dependencies.append({"path": str(asset), "sha256": hash_file(asset)})
        per_attempt = _envelope(m.get("requests", {"dispatch": 1}))
        if per_attempt.get("dispatch") != 1:
            raise ContractError("every method must reserve exactly one dispatch per planned attempt")
        if driver == "transformer" and per_attempt.get("generate") != 1:
            raise ContractError("transformer method needs exactly one generate reservation")
        if driver == "transformer":
            required_probes = 3 * int("input_embeds" in carriers) + 3 * int("actor_lora" in carriers)
            if per_attempt.get("probe", 0) != required_probes:
                raise ContractError(f"{ident} needs exactly {required_probes} forward probes")
        methods.append(dict(id=ident, driver=driver, carriers=list(carriers), materials=allowed,
                            recipe=m.get("recipe", {}), command=command, requests=per_attempt,
                            identity_files=dependencies, entrypoint=m.get("entrypoint")))
    _distinct(methods, "method")
    if not methods:
        raise ContractError("no methods")
    devices = {m["driver"] == "transformer" for m in methods}
    if len(devices) > 1:
        raise ContractError("the tiny transformer comparison must use a common core model")
    limits = src["budget"]
    timeout = _number(limits["attempt_timeout_seconds"], "attempt timeout", 0.01)
    budget = {"max_attempts": _integer(limits["max_attempts"], "max_attempts"),
              "max_requests": _envelope(limits["max_requests"]), "attempt_timeout_seconds": timeout,
              "max_retries": _integer(limits.get("max_retries", 0), "max_retries"),
              "prepare_requests": _envelope(limits.get("prepare_requests", {}))}
    if budget["max_retries"]:
        raise ContractError("retry/resubmission is unsupported in v0.1")
    grader = src["scoring"]
    if grader["kind"] not in {"none", "exact", "official"}:
        raise ContractError("unsupported grader kind")
    scoring = {"kind": grader["kind"], "version": grader["version"],
               "private_sha256": (_sha(grader["private_sha256"], "private spec")
                                  if grader["kind"] != "none" else None),
               "timeout_seconds": _number(grader.get("timeout_seconds", 10), "score timeout", 0.01)}
    policy = src.get("analysis", {})
    threshold = policy.get("quality_floor")
    if threshold is not None and _number(threshold, "quality_floor") > 1:
        raise ContractError("quality_floor must be in [0,1]")
    baseline = policy.get("baseline", methods[0]["id"])
    if baseline not in {m["id"] for m in methods}:
        raise ContractError("analysis baseline is not a registered method")
    known = policy.get("known_historical", {})
    for mid, metrics in known.items():
        if mid not in {m["id"] for m in methods}:
            raise ContractError("known_historical references an unregistered method")
        for metric, number in metrics.items():
            if metric not in {"wall_seconds", "cpu_user_seconds", "cpu_system_seconds", "gpu_allocation_seconds"}:
                raise ContractError("unsupported historical measurement unit")
            _number(number, f"known_historical.{mid}.{metric}")
    attempts = []
    for task_index, task in enumerate(tasks):
        for rep in range(_integer(src.get("replicates", 1), "replicates", 1)):
            pivot = (task_index + rep) % len(methods)
            for method in methods[pivot:] + methods[:pivot]:
                attempts.append({"id": f"a{len(attempts)+1:06}", "task_id": task["id"], "method_id": method["id"],
                                 "replicate": rep, "requests": method["requests"]})
    work = (home / policy["workload_file"]) if "workload_file" in policy else None
    rules = {"quality_floor": threshold, "baseline": baseline,
             "comparison": "observed_floor_full_coverage_descriptive_only", "shared_prep": "once_per_identity_per_workload",
             "workload_sha256": hash_file(work) if work is not None else None,
             "known_historical": known}
    task_source = {"tasks_file_sha256": hash_file(home / src["tasks_file"]) if "tasks_file" in src else None,
                   "public_tasks_sha256": digest(tasks), "provenance": src.get("task_source", {})}
    return {"schema_version": VERSION, "kind": "plan_freeze", "run_id": run_id or uuid.uuid4().hex,
            "experiment_id": _id(src["experiment_id"], "experiment_id"), "provenance": "current",
            "source_config_sha256": hash_file(path), "task_source": task_source,
            "tasks": tasks, "methods": methods, "attempts": attempts, "denominator_kind": "planned",
            "runtime": runtime, "materials": materials, "cache": cache, "budget": budget, "scoring": scoring,
            "exposure": {"task_ids": exposed_ids, "families": exposed_families}, "analysis": rules,
            "model_or_api_download": False, "training": "not_run", "continuous_update": "unsupported"}
