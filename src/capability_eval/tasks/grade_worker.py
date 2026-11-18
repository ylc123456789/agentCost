"""Mounted as a standalone script in a Linux scoring sandbox."""
import ast
import contextlib
import ctypes
import json
import os
import resource
import socket
import sys
import types
from pathlib import Path


def verify_isolation():
    if sys.platform != "linux" or Path("/mnt").exists() or Path("/home").exists():
        raise RuntimeError("unexpected filesystem isolation")
    flags = ctypes.CDLL(None).prctl(39, 0, 0, 0, 0)  # PR_GET_NO_NEW_PRIVS
    if flags != 1 or set(name for _, name in socket.if_nameindex()) - {"lo"}:
        raise RuntimeError("unexpected network or privilege boundary")
    lines = Path("/proc/self/status").read_text().splitlines()
    if int(next(line.split()[1] for line in lines if line.startswith("CapEff:")), 16):
        raise RuntimeError("unexpected effective capabilities")


def main():
    verify_isolation()
    timeout = float(sys.argv[1])
    hard = max(1, int(timeout) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    case = json.loads(Path("/case.json").read_text(encoding="utf-8"))
    candidate = Path("/candidate.txt").read_text(encoding="utf-8")
    kind = case["kind"]
    if kind in {"ds1000", "synthetic_code"}:
        context = case["code_context"]
        has_string = any(isinstance(node, ast.FunctionDef) and node.name == "test_string"
                         for node in ast.parse(context).body)
        scope = {"__name__": "__main__"}
        with contextlib.redirect_stdout(sys.stderr):
            # The pinned DS context owns its [insert] replacement, preserving its original indentation.
            exec(compile(context, "<official test context>", "exec"), scope)
            test_execution = scope["test_execution"]
            try:
                test_execution(candidate)
                if has_string:
                    scope["test_string"](candidate)
            except (ImportError, ModuleNotFoundError):
                raise
            except Exception as exc:
                return {"passed": False, "detail": type(exc).__name__, "test_string_present": has_string}
        return {"passed": True, "test_string_present": has_string}
    if kind == "bcb_complete":
        try:
            ast.parse(candidate)
        except SyntaxError:
            return {"passed": False, "detail": "candidate is not a complete Python module"}
        sys.path.insert(0, "/official")
        from bigcodebench import eval as official

        # Bubblewrap replaces the upstream Python-only environment wrapper; grading logic is upstream.
        official.safe_environment = contextlib.nullcontext
        stat, details = types.SimpleNamespace(value=3), {}
        # A successful return with no discovered official tests is not a pass.
        count = []
        loader = official.unittest.TestLoader.loadTestsFromTestCase

        def count_tests(self, test_case):
            suite = loader(self, test_case)
            count.append(suite.countTestCases())
            return suite

        official.unittest.TestLoader.loadTestsFromTestCase = count_tests
        solution = candidate
        if case["calibrated"] and not case.get("reference_control", False):
            solution = case["code_prompt"] + "\n    pass\n" + candidate
        with contextlib.redirect_stdout(sys.stderr):
            official.unsafe_execute(case["entry_point"], solution, case["test"], timeout,
                                    None, None, None, stat, details)
        timed_out = stat.value == 2 or any("timed out" in str(value).lower() for value in details.values())
        if timed_out or stat.value == 3 or (stat.value == 0 and not sum(count) and not details):
            return {"passed": None, "reason": "grader_timeout" if timed_out else "official_tests_not_confirmed",
                    "official_stat": stat.value, "called_tests": sum(count)}
        return {"passed": stat.value == 0 and not details, "official_stat": stat.value,
                "failed_tests": len(details), "called_tests": sum(count), "calibrated": case["calibrated"]}
    raise RuntimeError(f"unsupported isolated grader: {kind}")


if __name__ == "__main__":
    print(json.dumps(main(), allow_nan=False))
