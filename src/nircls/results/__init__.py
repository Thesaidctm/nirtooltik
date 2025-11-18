"""Reporting and evaluation helpers for nircls results."""

from .classification import expected_calibration_error
from .regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmsecv,
    compute_rmsep,
    compute_rmse,
    rmse_with_ci,
)
from .report import generate_scientific_report

__all__ = [
    "compute_mae_percent",
    "compute_rpd",
    "compute_rmsecv",
    "compute_rmsep",
    "compute_rmse",
    "expected_calibration_error",
    "generate_scientific_report",
    "rmse_with_ci",
]
