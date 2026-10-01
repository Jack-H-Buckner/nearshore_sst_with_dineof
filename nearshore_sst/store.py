"""Atomic zarr writes and scratch cleanup.

Extracted from coastal_sst_data.store — only the two functions this pipeline uses:
  * sweep_scratch(dest)  -- discard dead scratch beside a destination path
  * atomic(dest)         -- context manager: write to a temp path, rename on success
"""

from __future__ import annotations

import contextlib
import glob as globlib
import logging
import os
import shutil
import socket
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

PART_SUFFIX = ".part-"
OLD_SUFFIX = ".old-"
SCRATCH_SUFFIXES = (PART_SUFFIX, OLD_SUFFIX)

# How long scratch owned by a run we cannot interrogate must sit untouched
# before we call it dead.
STALE_SCRATCH_S = 6 * 3600

_HOST = socket.gethostname().split(".")[0] or "unknown"

# Scratch this process has open right now: {scratch path -> its destination}.
_ACTIVE: dict[Path, Path] = {}
_ACTIVE_LOCK = threading.Lock()


def _tag() -> str:
    return f"{_HOST}-{os.getpid()}-{threading.get_ident()}-{int(time.time() * 1000)}"


def _owner_of(path: Path) -> tuple[str, int] | None:
    for suffix in SCRATCH_SUFFIXES:
        _, sep, tail = path.name.partition(suffix)
        if not sep:
            continue
        parts = tail.rsplit("-", 3)
        if len(parts) != 4 or not parts[0] or not all(p.isdigit() for p in parts[1:]):
            return None
        return parts[0], int(parts[1])
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _newest_mtime(path: Path, *, at_least: float = float("inf")) -> float:
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    if newest >= at_least or not path.is_dir():
        return newest
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in (dirpath, *(os.path.join(dirpath, f) for f in filenames)):
            try:
                newest = max(newest, os.stat(name).st_mtime)
            except OSError:
                continue
            if newest >= at_least:
                return newest
    return newest


def _register(tmp: Path, dest: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE[Path(tmp)] = Path(dest)


def _unregister(tmp: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(Path(tmp), None)


def _is_scratch_name(name: str) -> bool:
    return any(s in name for s in SCRATCH_SUFFIXES)


def scratch_owner(path: Path) -> str:
    owner = _owner_of(Path(path))
    return f"{owner[0]}:{owner[1]}" if owner else "an unidentified run"


def _why_dead(path: Path) -> str:
    owner = _owner_of(Path(path))
    if owner is None:
        return "no host/pid in its name (written before this version)"
    if owner[0] != _HOST:
        return f"{owner[0]}:{owner[1]} is on another host and it has been untouched too long"
    if owner[1] == os.getpid():
        return "we started it ourselves and are no longer writing it"
    return f"its process ({owner[0]}:{owner[1]}) is gone"


def is_live_scratch(path: Path, *, max_age_s: float | None = None) -> bool:
    path = Path(path)
    with _ACTIVE_LOCK:
        if path in _ACTIVE:
            return True
    owner = _owner_of(path)
    if owner is not None and owner[0] == _HOST:
        if owner[1] == os.getpid():
            return False
        return _pid_alive(owner[1])
    cutoff = time.time() - (STALE_SCRATCH_S if max_age_s is None else max_age_s)
    return _newest_mtime(path, at_least=cutoff) > cutoff


def scratch_beside(dest: Path) -> list[Path]:
    dest = Path(dest)
    pattern = globlib.escape(dest.name)
    return sorted(p for s in SCRATCH_SUFFIXES for p in dest.parent.glob(f"{pattern}{s}*"))


def _rm(path: Path) -> None:
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        log.warning("  could not remove %s (%s); delete it by hand", path.name, exc)


def sweep_scratch(dest: Path, *, max_age_s: float | None = None) -> list[Path]:
    """Discard scratch beside `dest` left by a run that died — and only that."""
    live: list[Path] = []
    for stale in scratch_beside(dest):
        if is_live_scratch(stale, max_age_s=max_age_s):
            live.append(stale)
            log.warning("  leaving %s alone -- %s may still be writing it",
                        stale.name, scratch_owner(stale))
            continue
        log.warning("  discarding %s left by an unfinished run -- %s",
                    stale.name, _why_dead(stale))
        _rm(stale)
    return live


def _swap(tmp: Path, dest: Path) -> None:
    if not tmp.is_dir():
        os.replace(tmp, dest)
        return
    stash = None
    if dest.exists():
        stash = dest.with_name(f"{dest.name}{OLD_SUFFIX}{_tag()}")
        dest.rename(stash)
        _register(stash, dest)
    try:
        os.replace(tmp, dest)
    except OSError:
        if stash is not None:
            stash.rename(dest)
        raise
    finally:
        if stash is not None:
            _unregister(stash)
    if stash is not None:
        _rm(stash)


@contextlib.contextmanager
def atomic(dest: Path, *, placeholder: bool = False):
    """Yield a scratch path to write to; swap it onto `dest` only if the write returns."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    live = sweep_scratch(dest)
    if live:
        log.warning("  %s is being written by more than one run at once (%s); both writes "
                    "complete and whichever finishes LAST wins -- check for overlapping "
                    "--aoi lists or date ranges", dest.name,
                    ", ".join(sorted({scratch_owner(p) for p in live})))
    tmp = dest.with_name(f"{dest.name}{PART_SUFFIX}{_tag()}")
    _register(tmp, dest)
    if placeholder:
        tmp.touch()
    try:
        yield tmp
        if not tmp.exists() or (placeholder and tmp.is_file() and tmp.stat().st_size == 0):
            raise RuntimeError(f"nothing was written to {tmp.name}")
        _swap(tmp, dest)
    except BaseException:
        log.info("  removing %s after a failed write", tmp.name)
        _rm(tmp)
        raise
    finally:
        _unregister(tmp)
