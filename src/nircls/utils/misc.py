
from __future__ import annotations
from datetime import datetime
import os
import random
import hashlib
import subprocess
from pathlib import Path

import numpy as np


from ..core.settings import settings


# Default directories for raw data and results
RAW_DIR = settings.RAW_DIR
RESULTS_DIR = settings.RESULTS_DIR

def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not already exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    """Return current UTC timestamp in ``YYYYMMDD_HHMMSS`` format."""

    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def make_run_id(prefix: str) -> str:
    """Generate a run identifier using ``prefix`` and the current timestamp."""

    return f"{prefix}_{now_ts()}"


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and hash-based random generators."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def file_hash(path: Path | str) -> str:
    """Return SHA256 hash of file at ``path``.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    str
        Hexadecimal representation of the file's SHA256 hash.
    """

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str | None:
    """Return the current Git commit hash if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None
