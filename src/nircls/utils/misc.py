
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
import random

import numpy as np


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
