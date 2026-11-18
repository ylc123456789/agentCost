"""Assert rather than infer CPU carrier success from different generated text."""
from __future__ import annotations

import json
import sys
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator

from capability_eval.io import hash_file, read_json, rows


def main(root: Path):
    tiny = root / "tiny-cpu"
    methods = read_json(tiny / "plan.json")["attempts"]
    terminal = {r["attempt_id"]: r for r in rows(tiny / "attempts.jsonl") if r["status"] != "reserved"}
    assert len(terminal) == 5 and all(r["status"] == "completed" and r["cleanup_confirmed"] for r in terminal.values())
    evidence = {a["method_id"]: read_json(tiny / "raw" / a["id"] / "model_evidence.json") for a in methods}
    probes = {a["method_id"]: read_json(tiny / "raw" / a["id"] / "probe_evidence.json") for a in methods}
    assert len({e["base_model_sha256"] for e in evidence.values()}) == 1
    assert evidence["T"]["input_text_tokens"] > evidence["B0"]["input_text_tokens"]
    assert evidence["T"]["rendered_prompt_sha256"] == evidence["T-L"]["rendered_prompt_sha256"]
    assert evidence["Z"]["prefix"]["nonzero_values"] > 0
    assert evidence["Z"]["prefix"]["shape"] == [1, 3, 32]
    assert evidence["Z"]["input_model_positions"] - evidence["Z"]["input_text_tokens"] == 3
    baseline_input = evidence["B0"]["rendered_prompt_sha256"]
    assert evidence["L"]["rendered_prompt_sha256"] == evidence["Z"]["rendered_prompt_sha256"] == baseline_input
    z = probes["Z"]["prefix_logits"]
    assert min(z["perturbation_max_abs"], z["removed_max_abs"]) > z["threshold"]
    for arm in ("L", "T-L"):
        lora = probes[arm]["lora_logits"]
        assert evidence[arm]["actor_lora"]["enabled"] and lora["reactivated"]
        assert lora["enabled_vs_disabled_max_abs"] > lora["threshold"] >= lora["disabled_vs_base_max_abs"]
    assert evidence["T-L"]["text_sha256"] == evidence["T"]["text_sha256"]
    assert evidence["T-L"]["actor_lora"]["sha256"] == evidence["L"]["actor_lora"]["sha256"]
    freeze = read_json(tiny / "artifact_freeze.json")
    assert len(freeze["components"]) == 4  # base, text, Z prefix, L adapter; T+L reuses both
    assert len(set(freeze["states"]["T"]["prepare_ids"]) & set(freeze["states"]["T-L"]["prepare_ids"])) == 2
    assert all(e["new_token_start"] == e["input_model_positions"] and
               0 < len(e["output_token_ids"]) <= 4 for e in evidence.values())

    fixture = root / "fixture"
    v1, v2 = [list(rows(fixture / f"scores.v{i:04}.jsonl")) for i in (1, 2)]
    assert {rec["attempt_id"]: rec["candidate"]["sha256"] for rec in v1} == {
            rec["attempt_id"]: rec["candidate"]["sha256"] for rec in v2}
    assert len(v1) == len(v2) == 6
    assert hash_file(fixture / "analysis.json") == hash_file(root / "ledger-only-rebuild/analysis.json")
    inspect_run = root / "inspect-wrapped"
    assert (inspect_run / "raw/a000001/stdout.log").read_bytes() == (root / "inspect-original/result.json").read_bytes()
    assert any(e["source"] == "current" and e["kind"] == "budget" and e["metric"] == "inspect_generate"
               for e in rows(inspect_run / "events.jsonl"))

    schemas = files("capability_eval") / "schemas"
    event_schema = json.loads((schemas / "event-v1.schema.json").read_text())
    journals_schema = json.loads((schemas / "journals-v1.schema.json").read_text())
    Draft202012Validator.check_schema(event_schema)
    Draft202012Validator.check_schema(journals_schema)
    legacy = root / "v20-import"
    targets = [tiny, fixture, inspect_run] + ([legacy] if legacy.exists() else [])
    validated = 0
    for run in targets:
        for row in rows(run / "events.jsonl"):
            Draft202012Validator(event_schema).validate(row)
            validated += 1
        for row in [read_json(run / "plan.json"), *rows(run / "attempts.jsonl"), *rows(run / "scores.jsonl")]:
            Draft202012Validator(journals_schema).validate(row)
            validated += 1
    imported = None
    if legacy.exists():
        q = read_json(legacy / "analysis.json")["quality"]
        expected = {"N": 15, "T-full": 21, "T-dedup-match": 20, "T-dedup-swap": 17, "Z-match": 15, "Z-swap": 16}
        assert {mid: info["pass"] for mid, info in q.items()} == expected and all(info["denominator"] == 40 for info in q.values())
        imported = expected
    return {"tiny_cpu_completed": len(terminal), "tiny_model_probes": probes, "tiny_base_sha256": evidence["B0"]["base_model_sha256"],
            "tiny_token_ids": {arm: item["output_token_ids"] for arm, item in evidence.items()},
            "paired": read_json(fixture / "analysis.json")["pairs"], "unchanged_candidate_count": len(v1),
            "ledger_only_byte_identical": True, "inspect_unwrapped_equals_wrapped": True,
            "validated_schema_records": validated, "v20_import_passes": imported,
            "official_candidates_executed": 0, "commercial_api_calls": 0, "training_calls": 0, "gpu_device_windows": 0}


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2, sort_keys=True))
