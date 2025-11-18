"""Utilities for model hyperparameter search."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

from sklearn.model_selection import GridSearchCV, ParameterGrid

logger = logging.getLogger(__name__)

_WINDOWS_WAIT_HANDLE_LIMIT = 63
_WINDOWS_MAX_LOKY_WORKERS = max(1, (_WINDOWS_WAIT_HANDLE_LIMIT - 1) // 2)


def _sanitize_n_jobs(n_jobs: Optional[int]) -> int:
    """Return a safe ``n_jobs`` value for ``GridSearchCV`` workers."""

    if not n_jobs:
        return 1

    if n_jobs < 0:
        cpu_count = os.cpu_count() or 1
        n_jobs = max(1, cpu_count + n_jobs + 1)

    n_jobs = max(1, n_jobs)

    if sys.platform.startswith("win") and n_jobs > _WINDOWS_MAX_LOKY_WORKERS:
        logger.warning(
            "Reducing n_jobs from %d to %d to avoid Windows handle limits",
            n_jobs,
            _WINDOWS_MAX_LOKY_WORKERS,
        )
        return _WINDOWS_MAX_LOKY_WORKERS

    return n_jobs


class LoggingGridSearchCV(GridSearchCV):
    """GridSearchCV with optional debug logging for progress."""

    def _run_search(self, evaluate_candidates):  # type: ignore[override]
        grid = list(ParameterGrid(self.param_grid))
        total = len(grid)
        for i, params in enumerate(grid, 1):
            if self.verbose < 1:
                logger.debug("Grid search progress: %d/%d parameters", i, total)
            evaluate_candidates([params])


def _get_n_components(best_params: Dict[str, Any]) -> Optional[int]:
    """Return the best ``n_components`` value from GridSearchCV parameters."""

    for k, v in best_params.items():
        if k.endswith("n_components"):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


__all__ = ["_sanitize_n_jobs", "LoggingGridSearchCV", "_get_n_components"]
