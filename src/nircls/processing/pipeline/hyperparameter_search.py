"""Utilities for model hyperparameter search."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score

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


def compute_pls_lv_cv_curve(
    base_estimator,
    X,
    y,
    cv,
    candidates: List[int],
    groups=None,
) -> Dict[int, float]:
    """
    Calcula RMSECV para cada n_components em ``candidates``.
    Retorna um dict {n_components: rmsecv}.
    NÃO escolhe LV ótima; apenas calcula a curva.
    """

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rmsecv_map: Dict[int, float] = {}

    params = getattr(base_estimator, "get_params", None)
    param_key = "n_components"
    if callable(params):
        for key in params().keys():
            if key.endswith("n_components"):
                param_key = key
                break

    for n_comp in candidates:
        est = clone(base_estimator)
        try:
            est.set_params(**{param_key: n_comp})
        except ValueError:
            continue
        scores = cross_val_score(est, X, y, cv=cv, scoring=scorer, groups=groups)
        mse_vals = -scores  # porque greater_is_better=False
        rmse_vals = np.sqrt(mse_vals)
        rmsecv_map[n_comp] = float(np.mean(rmse_vals))

    return rmsecv_map


__all__ = [
    "_sanitize_n_jobs",
    "LoggingGridSearchCV",
    "_get_n_components",
    "compute_pls_lv_cv_curve",
]
