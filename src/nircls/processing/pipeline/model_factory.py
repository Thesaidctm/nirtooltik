"""Helpers for building models and formatting model-related metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from .preprocessing import (
    _call_build_pipeline,
    _format_preprocess_step,
    _format_preprocess_steps,
    _get_model_step_name,
    _get_preprocess_steps_from_pipeline,
    _infer_task_type,
    _make_preprocess,
    _normalize_preprocess_value,
)


def _coerce_param_value(value: Any) -> Any:
    """Convert parameter ``value`` to a JSON-friendly representation."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_coerce_param_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _coerce_param_value(v) for k, v in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_metrics_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``row`` with parameter values coerced for metrics persistence."""

    return {key: _coerce_param_value(value) for key, value in row.items()}


def _update_row_with_best_params(
    row: Dict[str, Any], best_params: Optional[Mapping[str, Any]]
) -> None:
    """Add ``best_params`` entries to ``row`` using ``best_<param>`` keys."""

    if not best_params:
        return
    for key, value in best_params.items():
        row[f"best_{key}"] = _coerce_param_value(value)


def _extract_plsda_method_from_params(
    best_params: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Return the PLS-DA classifier method recorded in ``best_params`` if any."""

    if not best_params:
        return None
    for key in ("clf__method", "clf__estimator__method"):
        value = best_params.get(key)
        if value is not None:
            return str(value)
    return None


def _resolve_plsda_estimator(estimator: Any) -> Any:
    """Unwrap nested estimators and return the underlying PLS-DA instance."""

    current = estimator
    while isinstance(current, CalibratedClassifierCV):
        current = current.estimator
    if isinstance(current, Pipeline):
        step_name = _get_model_step_name(current)
        clf = current.named_steps.get(step_name)
        if clf is None:
            return current
        return _resolve_plsda_estimator(clf)
    return current


def _get_plsda_method_details(
    name: str, estimator: Any, best_params: Optional[Mapping[str, Any]]
) -> Optional[Tuple[str, str]]:
    """Return normalized slug and display label for a PLS-DA classifier method."""

    if not name:
        return None
    normalized = re.sub(r"[^A-Za-z]", "", name).upper()
    if "PLSDA" not in normalized:
        return None
    method = _extract_plsda_method_from_params(best_params)
    if method is None:
        core = _resolve_plsda_estimator(estimator)
        method = getattr(core, "method", None)
    if not method:
        return None
    method_str = str(method).strip()
    if not method_str:
        return None
    method_slug = re.sub(r"[^0-9A-Za-z]+", "_", method_str.upper()).strip("_")
    if not method_slug:
        return None
    display_core = method_slug.replace("_", "-")
    if display_core.startswith("PLS"):
        display_label = display_core
    else:
        display_label = f"PLS-{display_core}"
    return method_slug, display_label


def _format_model_label(
    name: str,
    estimator: Any,
    best_params: Optional[Mapping[str, Any]],
    *,
    method_details: Optional[Tuple[str, str]] = None,
) -> str:
    """Include the PLS-DA classifier method in the model label when available."""

    if not name:
        return name
    parts = name.split("__", 1)
    base = parts[0]
    suffix = f"__{parts[1]}" if len(parts) > 1 else ""
    normalized = re.sub(r"[^A-Za-z]", "", base).upper()
    if "PLSDA" not in normalized:
        return name
    if method_details is None:
        method_details = _get_plsda_method_details(name, estimator, best_params)
    if not method_details:
        return name
    method_slug, display_label = method_details
    base_clean = base.rstrip("_")
    pattern = re.compile(r"pls\s*[-_]*da", re.IGNORECASE)
    if display_label.upper() in base_clean.upper():
        new_base = base_clean
    else:
        replaced = pattern.sub(display_label, base_clean, count=1)
        if replaced != base_clean:
            new_base = replaced
        elif base_clean.upper().endswith(f"_{method_slug}"):
            new_base = base_clean
        else:
            new_base = f"{display_label}{('_' + base_clean) if base_clean else ''}".rstrip("_")
    return f"{new_base}{suffix}"


def _format_model_and_method_labels(
    name: str, estimator: Any, best_params: Optional[Mapping[str, Any]]
) -> Tuple[str, Optional[str]]:
    """Return the model label and human-readable classifier method label."""

    method_details = _get_plsda_method_details(name, estimator, best_params)
    model_label = _format_model_label(
        name, estimator, best_params, method_details=method_details
    )
    method_label = method_details[1] if method_details else None
    return model_label, method_label


__all__ = [
    "_call_build_pipeline",
    "_coerce_param_value",
    "_format_model_and_method_labels",
    "_format_model_label",
    "_format_preprocess_step",
    "_format_preprocess_steps",
    "_get_model_step_name",
    "_get_plsda_method_details",
    "_get_preprocess_steps_from_pipeline",
    "_infer_task_type",
    "_make_preprocess",
    "_normalize_preprocess_value",
    "_sanitize_metrics_row",
    "_update_row_with_best_params",
]
