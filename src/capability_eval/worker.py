from __future__ import annotations

from pathlib import Path

from .contracts import PreparedState, PublicTask, SolveResult
from .io import ContractError, read_json, write_bytes, write_json
from .ledger import Recorder
from .metering import cpu_window
from .methods import resolve_method


def main():
    job = read_json(Path("request.json"))
    recorder = Recorder.from_environment()
    task = PublicTask.from_record(job["task"])
    state = PreparedState(**job["state"])
    method = resolve_method(job["method"])
    try:
        with recorder.span("cold_load", phase="load", role="runtime", accounting="phase"), cpu_window(recorder):
            method.load(state, job["runtime"], recorder)
        warmup = getattr(method, "warmup", None)
        if warmup is not None:
            with recorder.span("warmup", phase="warmup", role="runtime", accounting="phase"), cpu_window(recorder):
                warmup(recorder)
        with recorder.span("complete_use", phase="serve", role="actor", accounting="phase"), cpu_window(recorder):
            result = method.solve(task, {"attempt_id": recorder.identity["attempt_id"]}, recorder)
            if not isinstance(result, SolveResult):
                raise ContractError("solve must return SolveResult")
            write_bytes(Path("raw.txt"), result.raw.encode("utf-8"), create=True)
            write_bytes(Path("candidate.txt"), result.candidate.encode("utf-8"), create=True)
            write_json(Path("model_evidence.json"), result.evidence, create=True)
        probe = getattr(method, "probe", None)
        if probe is not None:
            with recorder.span("diagnostic_probes", phase="research_overhead", role="runtime",
                               accounting="phase"), cpu_window(recorder):
                evidence = probe(recorder)
                write_json(Path("probe_evidence.json"), evidence, create=True)
    finally:
        with recorder.span("close", phase="deployment_overhead", role="runtime", accounting="phase"), cpu_window(recorder):
            method.close()


if __name__ == "__main__":
    main()
