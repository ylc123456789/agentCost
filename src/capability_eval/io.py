from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


class ContractError(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _bad_number(value):
    raise ContractError(f"non-finite JSON number: {value}")


def loads(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_bad_number)


def read_json(path: Path):
    return loads(path.read_bytes())


def write_json(path: Path, value, *, create=False):
    write_bytes(path, canonical(value) + b"\n", create=create)


def write_bytes(path: Path, payload: bytes, *, create=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            os.link(staging, path)
        else:
            os.replace(staging, path)
        if os.name == "posix":
            parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    finally:
        Path(staging).unlink(missing_ok=True)


def append(path: Path, value):
    raw = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        size = os.fstat(fd).st_size
        if size:
            os.lseek(fd, -1, os.SEEK_END)
            if os.read(fd, 1) != b"\n":
                raise ContractError(f"truncated journal; preserve and investigate: {path}")
        if os.write(fd, raw) != len(raw):
            raise OSError(f"short journal write: {path}")
        os.fsync(fd)
    finally:
        os.close(fd)


def rows(path: Path):
    if not path.exists():
        return
    with path.open("rb") as stream:
        for i, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ContractError(f"truncated journal: {path}:{i}")
            try:
                row = loads(line)
                if not isinstance(row, dict):
                    raise ContractError("journal record must be an object")
                yield row
            except (ValueError, UnicodeError) as exc:
                raise ContractError(f"invalid journal: {path}:{i}: {exc}") from exc


@contextmanager
def run_lock(root: Path):
    import fcntl

    with (root / ".run.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("run is locked by an active controller or supervisor") from exc
        # Do not explicitly unlock: the supervisor may still own a duplicate of this fd.
        yield stream.fileno()


def safe_file(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ContractError(f"path escapes run: {relative}")
    return path


def ref(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": hash_file(path),
            "bytes": path.stat().st_size}


def verify_ref(root: Path, item: dict) -> Path:
    path = safe_file(root, item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or hash_file(path) != item["sha256"]:
        raise ContractError(f"artifact changed or missing: {item['path']}")
    return path
