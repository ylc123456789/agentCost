from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .io import ContractError

if TYPE_CHECKING:
    from .ledger import Recorder

VERSION = "1.0"
PHASES = {"prepare", "import", "load", "warmup", "serve", "evaluation",
          "research_overhead", "deployment_overhead", "device_hold"}
ROLES = {"controller", "actor", "retriever", "encoder", "composer", "teacher",
         "grader", "tool", "runtime", "device"}
CARRIERS = {"text", "input_embeds", "layer_kv", "actor_lora"}
KNOWN = "observed"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class PublicTask:
    id: str
    prompt: str
    dataset: str
    split: str = "dev"
    source_id: str | int | None = None
    library: str | None = None
    family: str | int | None = None
    perturbation_type: str | None = None

    @classmethod
    def from_record(cls, row):
        keys = ("id", "prompt", "dataset", "split", "source_id", "library", "family", "perturbation_type")
        if any(k not in row or not isinstance(row[k], str) or not row[k]
               for k in ("id", "prompt", "dataset")):
            raise ContractError("public task needs nonempty id, prompt and dataset")
        task = cls(**{key: row[key] for key in keys if key in row})
        if task.split not in {"dev", "heldout"}:
            raise ContractError("task split must be dev or heldout")
        if any(v is not None and not isinstance(v, str) for v in (task.library, task.perturbation_type)) or any(
            v is not None and type(v) not in (str, int) for v in (task.source_id, task.family)
        ):
            raise ContractError("task identity metadata must be scalar")
        return task

    def record(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedState:
    method_id: str
    carrier: tuple[str, ...]
    files: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SolveResult:
    raw: str
    candidate: str
    evidence: dict = field(default_factory=dict)


class CapabilityMethod(Protocol):
    def prepare(self, allowed_materials: dict[str, Path], recipe: dict,
                budget: dict, recorder: Recorder) -> PreparedState: ...
    def load(self, state: PreparedState, runtime_spec: dict, recorder: Recorder) -> None: ...
    def solve(self, public_task: PublicTask, context: dict, recorder: Recorder) -> SolveResult: ...
    def close(self) -> None: ...


class TaskAdapter(Protocol):
    def public(self, row: dict, *, split: str = "dev") -> PublicTask: ...


class Grader(Protocol):
    def score(self, candidate: Path, private_spec: dict) -> dict: ...
