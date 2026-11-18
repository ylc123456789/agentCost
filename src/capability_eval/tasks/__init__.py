from __future__ import annotations

from ..contracts import PublicTask
from ..io import ContractError


class DS1000:
    def public(self, row, *, split="dev"):
        meta = row["metadata"]
        return PublicTask.from_record({"id": f"DS-1000/{meta['problem_id']}", "prompt": row["prompt"],
                                       "dataset": "DS-1000", "split": split, "source_id": meta["problem_id"],
                                       "library": meta["library"], "family": meta.get("origin_id"),
                                       "perturbation_type": meta.get("perturbation_type")})

    def private(self, row):
        return {"kind": "ds1000", "code_context": row["code_context"], "reference": row["reference_code"]}


class BigCodeBenchComplete:
    def public(self, row, *, split="dev"):
        return PublicTask.from_record({"id": row["task_id"], "dataset": "BigCodeBench/Complete",
                                       "prompt": row["complete_prompt"], "source_id": row["task_id"], "split": split})

    def private(self, row, evaluator, *, calibrated=False):
        if calibrated and not row.get("code_prompt"):
            raise ContractError("calibrated BigCodeBench needs the official code_prompt")
        return {"kind": "bcb_complete", "entry_point": row["entry_point"], "test": row["test"],
                "code_prompt": row.get("code_prompt"), "calibrated": calibrated, "evaluator": evaluator,
                "reference": row["complete_prompt"] + "\n" + row["canonical_solution"]}
