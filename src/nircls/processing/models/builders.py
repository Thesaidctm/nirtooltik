from __future__ import annotations
from typing import Dict, Any
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


def _maybe_balance(estimator, grid: Dict[str, Any]):
    """Set ``class_weight='balanced'`` when not provided in grid."""
    if "class_weight" not in (grid or {}):
        try:
            estimator.set_params(class_weight="balanced")
        except ValueError:  # pragma: no cover - estimator may not support
            pass
    return estimator


def _build_svm(grid: Dict[str, Any], *, kernel: str = "rbf") -> SVC:
    est = SVC(probability=True, kernel=kernel)
    return _maybe_balance(est, grid)


def _build_rf(grid: Dict[str, Any]) -> RandomForestClassifier:
    est = RandomForestClassifier()
    return _maybe_balance(est, grid)


def _build_etc(grid: Dict[str, Any]) -> ExtraTreesClassifier:
    est = ExtraTreesClassifier()
    return _maybe_balance(est, grid)


def _build_logreg(grid: Dict[str, Any]) -> LogisticRegression:
    est = LogisticRegression(max_iter=5000)
    return _maybe_balance(est, grid)
