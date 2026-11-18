from __future__ import annotations

import importlib
import time

from ..contracts import PreparedState, SolveResult
from ..io import ContractError


class Fixture:
    def __init__(self, method):
        self.method = method

    def prepare(self, allowed_materials, recipe, budget, recorder):
        return PreparedState(self.method["id"], (), metadata={"fixture_only": True})

    def load(self, state, runtime_spec, recorder):
        self.recipe = state.metadata["recipe"]

    def solve(self, public_task, context, recorder):
        with recorder.span("fixture_compute", role="actor"):
            if self.recipe.get("request"):
                recorder.reserve(self.recipe["request"])
            if self.recipe.get("delay_seconds"):
                time.sleep(self.recipe["delay_seconds"])
            if self.recipe.get("fail"):
                raise RuntimeError("configured fixture failure")
            answers = self.recipe.get("answers", {})
            result = answers.get(public_task.id, self.recipe.get("answer", public_task.prompt))
        return SolveResult(result, result, {"fixture_only": True, "public_task": public_task.record()})

    def close(self):
        pass


def resolve_method(method):
    driver = method["driver"]
    if driver in {"fixture", "external"}:
        return Fixture(method)
    if driver == "transformer":
        from .tiny_transformer import TinyTransformer

        return TinyTransformer()
    if driver != "plugin":
        raise ContractError(f"unsupported method: {driver}")
    entry = method["entrypoint"]
    module, symbol = entry.split(":", 1)
    loaded = importlib.import_module(module)
    from pathlib import Path

    if Path(loaded.__file__).resolve() not in {Path(item["path"]).resolve() for item in method["identity_files"]}:
        raise ContractError("plugin implementation not present among frozen identity_files")
    return getattr(loaded, symbol)(method)
