from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze, verify
from .controller import run
from .integrations.importer import import_events
from .plan import build_plan
from .state import prepare
from .tasks.scoring import score


def main(argv=None):
    parser = argparse.ArgumentParser(prog="capability-eval")
    sub = parser.add_subparsers(dest="action", required=True)
    for command in ("validate", "prepare", "run", "wrap", "score", "analyze", "import-events", "verify", "official-contract"):
        child = sub.add_parser(command)
        if command in {"validate", "prepare"}:
            child.add_argument("--config", type=Path, required=True)
        if command != "validate":
            child.add_argument("--run-dir", type=Path, required=True)
        if command == "wrap":
            child.add_argument("--phase", choices=["serve"], required=True)
            child.add_argument("--attempt-id", required=True)
            child.add_argument("command", nargs=argparse.REMAINDER)
        if command == "score":
            child.add_argument("--private", type=Path, required=True)
            child.add_argument("--rescore", metavar="REASON")
            child.add_argument("--revised-spec", action="store_true")
            child.add_argument("--allow-official", action="store_true")
            child.add_argument("--bwrap", type=Path)
        if command == "analyze":
            child.add_argument("--workload", type=Path)
            child.add_argument("--output", type=Path)
            child.add_argument("--ledger-only", action="store_true")
        if command == "import-events":
            child.add_argument("--input", type=Path, required=True)
            child.add_argument("--mapping", type=Path, required=True)
        if command == "official-contract":
            child.add_argument("--input", type=Path, required=True)
            child.add_argument("--dataset", choices=["ds1000", "bcb"], required=True)
            child.add_argument("--bwrap", type=Path, required=True)
            child.add_argument("--evaluator-root", type=Path)
            child.add_argument("--allow-official", action="store_true")
    a = parser.parse_args(argv)
    code = 0
    try:
        if a.action != "validate":
            root = a.run_dir.resolve()
        match a.action:
            case "validate":
                plan = build_plan(a.config, run_id="validation-only")
                result = {"valid": True, "attempts": len(plan["attempts"]), "methods": [m["id"] for m in plan["methods"]]}
            case "prepare":
                plan, freeze = prepare(a.config, root)
                result = {"run_id": plan["run_id"], "planned": len(plan["attempts"]),
                          "states": {mid: {k: s[k] for k in ("state_id", "status", "reason") if k in s}
                                     for mid, s in freeze["states"].items()}}
                code = int(any(s["status"] != "completed" for s in freeze["states"].values()))
            case "run":
                result = run(root)
            case "wrap":
                command = a.command[1:] if a.command[:1] == ["--"] else a.command
                result = run(root, only=a.attempt_id, command=command)
                code = int(result["executed"].get(a.attempt_id) != "completed")
            case "score":
                result = score(root, a.private, rescore=a.rescore, revised_spec=a.revised_spec,
                               bwrap=a.bwrap, allow_official=a.allow_official)
            case "analyze":
                result = analyze(root, workload=a.workload, output=a.output, ledger_only=a.ledger_only)
            case "import-events":
                result = import_events(a.input.resolve(), a.mapping.resolve(), root)
            case "verify":
                result = {"files_checked": verify(root)}
            case "official-contract":
                from .tasks.official_contract import official_contract

                result = official_contract(root, a.input, a.dataset, bwrap=a.bwrap,
                                           allow_official=a.allow_official, evaluator=a.evaluator_root)
                code = int(not result["qualified"])
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"capability-eval: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("capability-eval: interrupted; already reserved attempts won't be dispatched again", file=sys.stderr)
        return 130
