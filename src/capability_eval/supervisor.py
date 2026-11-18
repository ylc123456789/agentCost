"""Linux subprocess supervision, including adopted descendants that left the session."""
from __future__ import annotations

import ctypes
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

from .io import ContractError, read_json, write_json
from .ledger import Clock


def isolated_env(work: Path, threads=1):
    home, tmp = work / "home", work / "tmp"
    home.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    return {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "LANG": "C.UTF-8",
            "HOME": str(home), "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads), "TOKENIZERS_PARALLELISM": "false"}


def _children(pid):
    tasks = Path(f"/proc/{pid}/task")
    try:
        tids = list(tasks.iterdir())
    except FileNotFoundError:
        return set()
    children = set()
    for tid in tids:
        try:
            children.update(map(int, (tid / "children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return children


def _terminate_descendants():
    # Only signal our own unreaped children; their PIDs cannot be reused until
    # this single wait4 loop has reaped them. Orphaned descendants are adopted
    # by the subreaper and handled on a subsequent pass, regardless of setsid.
    for pid in _children(os.getpid()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _supervise(job, work):
    clock = Clock()
    start = clock.now()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        return {"status": "unconfirmed", "cleanup_confirmed": False,
                "reason": f"subreaper unavailable: errno {ctypes.get_errno()}"}
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    limit = job.get("file_limit_bytes", 64 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    timed_out, no_children, child_status, error = False, False, None, None
    cleanup_failed, cleaned_descendants = False, False
    usage = {"cpu_user_seconds": 0.0, "cpu_system_seconds": 0.0, "process_max_rss_bytes": 0}
    reaped = 0
    proc = None
    with (work / "stdout.log").open("xb", buffering=0) as out, (work / "stderr.log").open("xb", buffering=0) as err:
        try:
            proc = subprocess.Popen(job["command"], cwd=work, env=job["env"], stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, close_fds=True, start_new_session=True)
        except OSError as exc:
            error = f"exec failed: {exc}"
            no_children = True
        if proc is not None:
            deadline = clock.now() + int(job["timeout"] * 1e9)
            cleanup_deadline = None
            while not no_children:
                try:
                    pid, status, metrics = os.wait4(-1, os.WNOHANG)
                except ChildProcessError:
                    no_children = True
                    break
                if pid:
                    reaped += 1
                    usage["cpu_user_seconds"] += metrics.ru_utime
                    usage["cpu_system_seconds"] += metrics.ru_stime
                    usage["process_max_rss_bytes"] = max(usage["process_max_rss_bytes"], metrics.ru_maxrss * 1024)
                    if pid == proc.pid:
                        child_status = proc.returncode = os.waitstatus_to_exitcode(status)
                if child_status is None and clock.now() >= deadline:
                    timed_out = True
                if (timed_out or child_status is not None) and not pid:
                    cleaned_descendants |= child_status is not None
                    if cleanup_deadline is None:
                        cleanup_deadline = clock.now() + 3_000_000_000
                    try:
                        _terminate_descendants()
                    except OSError as exc:
                        cleanup_failed, error = True, f"descendant cleanup: {exc}"
                        break
                    if clock.now() >= cleanup_deadline:
                        cleanup_failed, error = True, "descendant exit not confirmed before safety deadline"
                        break
                if not pid:
                    time.sleep(0.01)
        os.fsync(out.fileno())
        os.fsync(err.fileno())
    status = ("unconfirmed" if not no_children or cleanup_failed else "timeout" if timed_out else
              "completed" if child_status == 0 and not cleaned_descendants else "failed")
    return {"status": status, "cleanup_confirmed": no_children and not cleanup_failed,
            "returncode": child_status, "timed_out": timed_out, "forced_descendant_cleanup": cleaned_descendants,
            "reason": error or ("descendants outlived primary process" if cleaned_descendants else None),
            "process_id": proc.pid if proc else None, "supervisor_pid": os.getpid(), "reaped_processes": reaped,
            "clock_domain": clock.domain, "start_ns": start, "end_ns": clock.now(), "utc": clock.utc(),
            "metrics": usage if reaped else None}


def launch(work: Path, fd: int, command, timeout, env=None):
    if sys.platform != "linux":
        raise ContractError("execution requires Linux subprocess supervision")
    child_env = isolated_env(work) if env is None else env
    write_json(work / "job.json", {"command": command, "timeout": timeout, "env": child_env}, create=True)
    controller_env = isolated_env(work)
    proc = subprocess.Popen([sys.executable, "-B", "-m", "capability_eval.supervisor", str(work), str(fd)],
                            stdin=subprocess.DEVNULL, env=controller_env, pass_fds=(fd,), start_new_session=True)
    proc.wait()
    if proc.returncode != 0 or not (work / "receipt.json").is_file():
        raise ContractError("supervisor stopped without cleanup confirmation; preserve run for review")
    return read_json(work / "receipt.json")


def main():
    work = Path(sys.argv[1])
    lease = int(sys.argv[2])
    os.set_inheritable(lease, False)  # keeper survives controller death; workload never inherits it
    receipt = _supervise(read_json(work / "job.json"), work)
    write_json(work / "receipt.json", receipt, create=True)


if __name__ == "__main__":
    main()
