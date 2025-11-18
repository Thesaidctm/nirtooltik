"""Utility helpers for metric computation and aggregation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from ..results.metrics.regression import compute_rmse


def _rmse(
    y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray
) -> float:
    """Compute root mean squared error compatible with older sklearn versions."""

    return float(compute_rmse(y_true, y_pred))


def _compute_ci95(mean: float, std: float, n: int) -> Optional[tuple[float, float]]:
    """Return 95% confidence interval bounds for ``mean`` when ``n`` >= 2."""

    if n < 2:
        return None
    if not np.isfinite(mean) or not np.isfinite(std):
        return None
    se = std / np.sqrt(n)
    if not np.isfinite(se):
        return None
    try:
        t_multiplier = stats.t.ppf(0.975, df=n - 1)
    except Exception:  # pragma: no cover - defensive
        return None
    if not np.isfinite(t_multiplier):
        return None
    margin = t_multiplier * se
    if not np.isfinite(margin):
        return None
    return mean - margin, mean + margin


def _summarize_metric_values(values: Sequence[float]) -> Optional[Dict[str, float]]:
    """Compute mean, std and 95% CI summary statistics for ``values``."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    mask = np.isfinite(arr)
    if not np.any(mask):
        return None
    arr = arr[mask]
    mean_val = float(np.mean(arr))
    summary: Dict[str, float] = {"mean": mean_val}
    if arr.size > 1:
        std_val = float(np.std(arr, ddof=1))
        summary["std"] = std_val
        ci_bounds = _compute_ci95(mean_val, std_val, arr.size)
        if ci_bounds is not None:
            low, high = ci_bounds
            summary["ci95_low"] = float(low)
            summary["ci95_high"] = float(high)
    return summary


def _summarize_scores(
    scores: Mapping[str, Sequence[float]], *, prefix: str = ""
) -> Dict[str, float]:
    """Return flattened summary statistics for a metric mapping."""

    summary: Dict[str, float] = {}
    for metric, values in scores.items():
        stats_dict = _summarize_metric_values(values)
        if not stats_dict:
            continue
        base = f"{prefix}{metric}"
        summary[f"{base}_mean"] = stats_dict["mean"]
        if "std" in stats_dict:
            summary[f"{base}_std"] = stats_dict["std"]
        if "ci95_low" in stats_dict and "ci95_high" in stats_dict:
            summary[f"{base}_ci95_low"] = stats_dict["ci95_low"]
            summary[f"{base}_ci95_high"] = stats_dict["ci95_high"]
    return summary


def _sanitize_metrics_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``row`` with parameter values coerced for metrics persistence."""

    return {key: _coerce_param_value(value) for key, value in row.items()}


# Late import to avoid circular dependency with ``model_factory``
from .model_factory import _coerce_param_value  # noqa: E402  pylint: disable=C0413

__all__ = [
    "_rmse",
    "_compute_ci95",
    "_summarize_metric_values",
    "_summarize_scores",
    "_sanitize_metrics_row",
]
