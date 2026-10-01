"""Provenance stamps for output cubes: version strings and UTC timestamps."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def package_version() -> str:
    from importlib.metadata import version
    return version("nearshore-sst-dineof")


@lru_cache(maxsize=1)
def code_version() -> str:
    """Git commit SHA of the running code, or 'unknown' outside a checkout."""
    repo = Path(__file__).resolve().parent

    def git(*args) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if not sha:
        return "unknown"
    dirty = git("status", "--porcelain")
    return f"{sha}-dirty" if dirty else sha
