from __future__ import annotations

import sys
from pathlib import Path

from .io import ContractError, digest, hash_file, read_json, ref, verify_ref, write_bytes, write_json
from .ledger import Recorder
from .metering import cpu_window
from .methods import resolve_method
from .plan import build_plan, environment


class Preparation:
    def __init__(self, plan, root):
        self.plan, self.root = plan, root
        self.components = {}
        self.allowed = {m["id"]: m for m in plan["materials"]}
        self.used = {}
        self.method_components = {}
        self.preparing = None

    def component(self, key, build, *, phase="prepare", role="runtime", method_id=None, history="not_applicable"):
        prep_id = digest(key)
        if self.preparing is not None:
            self.method_components.setdefault(self.preparing, set()).add(prep_id)
        if prep_id in self.components:
            component = self.components[prep_id]
            if component["status"] != "completed":
                raise ContractError(f"shared preparation previously failed: {prep_id}")
            return prep_id, self.root / component["directory"]
        target = self.root / "state" / prep_id
        target.mkdir(parents=True)
        recorder = Recorder(self.root / "events.jsonl", run_id=self.plan["run_id"], method_id=method_id,
                            phase=phase, role=role, limits=self.plan["budget"]["prepare_requests"],
                            evidence_dir=self.root / "prepare-attachments", budget_used=self.used)
        component = {"directory": target.relative_to(self.root).as_posix(), "phase": phase,
                     "identity": key, "historical": history, "status": "failed", "files": []}
        self.components[prep_id] = component
        try:
            with recorder.span("prepare_component", accounting="phase", data={"prepare_id": prep_id}), cpu_window(recorder):
                build(target, recorder)
            for file in sorted(target.rglob("*")):
                if file.is_symlink():
                    raise ContractError("no symlinks in frozen artifacts")
                if file.is_file():
                    component["files"].append(ref(self.root, file))
                    file.chmod(0o444)
            component["status"] = "completed"
        except Exception as exc:
            component["error"] = f"{type(exc).__name__}: {exc}"
            raise
        return prep_id, target

    def _copy(self, key):
        item = self.allowed[key]

        def copy(target, recorder):
            data = Path(item["path"]).read_bytes()
            write_bytes(target / "material", data, create=True)
            if hash_file(target / "material") != item["sha256"]:
                raise ContractError("material changed after PlanFreeze")
            recorder.measure("imported_material_bytes", len(data), "B")

        return self.component(["material", item["sha256"]], copy, phase="import", history=item["historical"])

    def method(self, method):
        mid = method["id"]
        files, used, allowed = {}, [], {}
        for key in method["materials"]:
            prep_id, target = self._copy(key)
            used.append(prep_id)
            allowed[key] = target / "material"
        metadata = {"recipe": method["recipe"]}
        if method["driver"] == "transformer":
            from .methods.tiny_transformer import save_adapter, save_base, save_prefix

            runtime = self.plan["runtime"]
            base_id, base = self.component(["synthetic_gpt2", runtime], lambda d, r: save_base(d, runtime))
            used.append(base_id)
            files["base_model"] = ref(self.root, base / "model.safetensors")
            metadata |= {"fixture": "tiny_gpt2", "training_steps": 0, "training_tokens": 0,
                         "generation": runtime["generation"], "runtime_sha256": digest(runtime)}
            if "text" in method["carriers"]:
                files["text"] = ref(self.root, next(iter(allowed.values())))
            if "input_embeds" in method["carriers"]:
                prep_id, prefix = self.component(["synthetic_prefix", base_id], lambda d, r: save_prefix(d, runtime))
                used.append(prep_id)
                files["prefix"] = ref(self.root, prefix / "prefix.safetensors")
            if "actor_lora" in method["carriers"]:
                prep_id, lora = self.component(["synthetic_peft_adapter", base_id],
                                               lambda d, r: save_adapter(d, base, runtime))
                used.append(prep_id)
                files["lora"] = ref(self.root, lora / "adapter_model.safetensors")
        elif method["driver"] == "plugin":
            obj = resolve_method(method)
            prepared = None

            def build(target, recorder):
                nonlocal prepared
                prepared = obj.prepare(allowed, {**method["recipe"], "output_directory": str(target)},
                                       self.plan["budget"]["prepare_requests"], recorder)
                if prepared.method_id != mid or list(prepared.carrier) != method["carriers"]:
                    raise ContractError("prepared plugin method or carrier identity mismatch")
                for item in prepared.files.values():
                    path = Path(item["path"]).resolve(strict=True)
                    if not path.is_relative_to(target):
                        raise ContractError("plugin output must be inside its allocated preparation directory")

            prep_id, _ = self.component(["plugin", mid, method["identity_files"], method["recipe"], used], build,
                                        method_id=mid, history="unknown")
            used.append(prep_id)
            files |= {key: ref(self.root, Path(item["path"]).resolve()) for key, item in prepared.files.items()}
            metadata |= prepared.metadata
        else:
            metadata["fixture_only"] = method["driver"] == "fixture"
            if method["driver"] == "external":
                metadata["external_history"] = "unknown"
        material_history = {self.allowed[key]["historical"] for key in allowed}
        ident = {"method_id": mid, "carrier": method["carriers"], "files": files, "metadata": metadata,
                 "prepare_ids": sorted(set(used)), "runtime_sha256": digest(self.plan["runtime"]),
                 "method_sha256": digest(method), "historical": "unknown" if "unknown" in material_history or
                 method["driver"] in {"plugin", "external"} else "not_applicable", "status": "completed"}
        return {"state_id": digest(ident), **ident}


def prepare(config_path, root):
    if sys.platform != "linux":
        raise ContractError("prepare requires Linux; validation and analysis can run without a model")
    plan, env = build_plan(config_path), environment()
    root.mkdir(parents=True, exist_ok=False)
    (root / "raw").mkdir()
    write_json(root / "plan.json", plan, create=True)
    write_json(root / "environment.json", env, create=True)
    for name in ("events.jsonl", "attempts.jsonl", "scores.jsonl"):
        write_bytes(root / name, b"", create=True)
    prep = Preparation(plan, root)
    states = {}
    for method in plan["methods"]:
        prep.preparing = method["id"]
        try:
            states[method["id"]] = prep.method(method)
        except Exception as exc:
            states[method["id"]] = {"status": "failed", "state_id": None,
                                    "prepare_ids": sorted(prep.method_components.get(method["id"], set()) & prep.components.keys()),
                                    "reason": f"{type(exc).__name__}: {exc}"}
    frozen = {"schema_version": plan["schema_version"], "run_id": plan["run_id"],
              "plan_sha256": hash_file(root / "plan.json"), "environment_sha256": hash_file(root / "environment.json"),
              "states": states, "components": prep.components}
    write_json(root / "artifact_freeze.json", {"payload_sha256": digest(frozen), **frozen}, create=True)
    return plan, frozen


def read_run(root, *, check_environment=False):
    freeze = read_json(root / "artifact_freeze.json")
    payload_sha = freeze.pop("payload_sha256")
    if digest(freeze) != payload_sha or any(hash_file(root / filename) != freeze[key] for filename, key in
             (("plan.json", "plan_sha256"), ("environment.json", "environment_sha256"))):
        raise ContractError("PlanFreeze, ArtifactFreeze or environment identity mismatch")
    plan = read_json(root / "plan.json")
    if check_environment and environment() != read_json(root / "environment.json"):
        raise ContractError("runtime/package environment changed since freeze")
    return plan, freeze


def verify_method(root, plan, freeze, method):
    state = freeze["states"][method["id"]]
    if state["status"] != "completed":
        raise ContractError("method preparation unsuccessful")
    if digest({k: v for k, v in state.items() if k != "state_id"}) != state["state_id"]:
        raise ContractError("prepared state identity mismatch")
    if state["carrier"] != method["carriers"] or state["runtime_sha256"] != digest(plan["runtime"]) or \
            state["method_sha256"] != digest(method):
        raise ContractError("incompatible frozen state, method or runtime")
    for dependency in method["identity_files"]:
        if hash_file(Path(dependency["path"])) != dependency["sha256"]:
            raise ContractError(f"method source changed: {dependency['path']}")
    for prep_id in state["prepare_ids"]:
        component = freeze["components"][prep_id]
        expected = {item["path"] for item in component["files"]}
        actual = {file.relative_to(root).as_posix() for file in (root / component["directory"]).rglob("*")
                  if file.is_file() or file.is_symlink()}
        if expected != actual:
            raise ContractError("frozen component membership changed")
        for item in component["files"]:
            verify_ref(root, item)
    for item in state["files"].values():
        verify_ref(root, item)
    return state
