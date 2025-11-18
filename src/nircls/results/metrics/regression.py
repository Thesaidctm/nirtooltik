"""Utility functions for regression metrics used across the project."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import numpy as np


def _to_1d_array(values: Sequence[float] | np.ndarray | Iterable[float]) -> np.ndarray:
    """Convert an input sequence to a 1-D ``numpy`` array of floats."""

    if isinstance(values, np.ndarray):
        arr = values.astype(float, copy=False)
    elif isinstance(values, Iterable):
        arr = np.asarray(list(values), dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.ravel()
    return arr


def compute_rmse(
    y_true: Sequence[float] | np.ndarray | Iterable[float],
    y_pred: Sequence[float] | np.ndarray | Iterable[float],
) -> float:
    """Return the root mean squared error between ``y_true`` and ``y_pred``."""

    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_rmsecv(
    y_true: Sequence[float] | np.ndarray | Iterable[float],
    y_pred: Sequence[float] | np.ndarray | Iterable[float],
) -> float:
    """Compute the RMSE of cross-validation predictions.

    Parameters
    ----------
    y_true, y_pred:
        Sequences containing the concatenated ground-truth and predicted
        responses from every validation fold.
    """

    return compute_rmse(y_true, y_pred)


def compute_rmsep(
    y_true: Sequence[float] | np.ndarray | Iterable[float],
    y_pred: Sequence[float] | np.ndarray | Iterable[float],
) -> float:
    """Compute the root mean squared error of prediction (RMSEP)."""

    return compute_rmse(y_true, y_pred)


def rmse_with_ci(
    y_true: Sequence[float] | np.ndarray | Iterable[float],
    y_pred: Sequence[float] | np.ndarray | Iterable[float],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Return RMSE with non-parametric bootstrap confidence interval."""

    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        return float("nan"), float("nan"), float("nan")

    rmse = compute_rmse(yt, yp)
    if not np.isfinite(rmse):
        return rmse, float("nan"), float("nan")
    if yt.size == 1 or n_bootstrap <= 0:
        return rmse, rmse, rmse

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, yt.size, size=(n_bootstrap, yt.size))
    residuals = yt[indices] - yp[indices]
    rmse_samples = np.sqrt(np.mean(residuals**2, axis=1))
    lower, upper = np.quantile(rmse_samples, [alpha / 2, 1 - alpha / 2])
    return rmse, float(lower), float(upper)


def compute_mae_percent(
    y_true: Sequence[float] | np.ndarray | Iterable[float],
    y_pred: Sequence[float] | np.ndarray | Iterable[float],
) -> float:
    """Return the mean absolute error normalised by ``mean(|y_true|)`` in percent."""

    yt = _to_1d_array(y_true)
    yp = _to_1d_array(y_pred)
    if yt.size == 0:
        return float("nan")
    denom = np.mean(np.abs(yt))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)) / denom * 100.0)


def compute_rpd(
    y_true: Sequence[float] | np.ndarray | Iterable[float],
    y_pred: Sequence[float] | np.ndarray | Iterable[float],
) -> float:
    """Compute the Residual Predictive Deviation (RPD)."""

    yt = _to_1d_array(y_true)
    if yt.size <= 1:
        return float("nan")
    rmsep = compute_rmsep(yt, y_pred)
    if not np.isfinite(rmsep) or rmsep == 0:
        return float("nan")
    std = np.std(yt, ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(std / rmsep)
