import math

import numpy as np

from nircls.metrics.regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmsecv,
    compute_rmsep,
)


def test_compute_rmsecv_matches_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.5, 4.0])
    expected = math.sqrt(((y_true - y_pred) ** 2).mean())
    assert math.isclose(compute_rmsecv(y_true, y_pred), expected)


def test_compute_rmsep_and_rpd():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    rmsep = compute_rmsep(y_true, y_pred)
    assert rmsep > 0
    std = np.std(y_true, ddof=1)
    expected_rpd = std / rmsep
    assert math.isclose(compute_rpd(y_true, y_pred), expected_rpd)


def test_mae_percent_scales_with_amplitude():
    base_true = np.array([1.0, -2.0, 3.0])
    base_pred = np.array([0.0, -1.0, 2.5])
    base_mae_pct = compute_mae_percent(base_true, base_pred)
    scaled_true = base_true * 10
    scaled_pred = base_pred * 10
    scaled_mae_pct = compute_mae_percent(scaled_true, scaled_pred)
    assert math.isclose(base_mae_pct, scaled_mae_pct)


def test_mae_percent_handles_zero_reference():
    y_true = np.zeros(3)
    y_pred = np.array([0.0, 0.0, 1.0])
    assert math.isnan(compute_mae_percent(y_true, y_pred))
