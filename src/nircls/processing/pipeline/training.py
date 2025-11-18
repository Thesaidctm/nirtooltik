"""High-level training interface built from segmented helpers."""

from __future__ import annotations

from .cv_runner import (
    _run_cv,
    _run_cv_classification,
    _run_cv_regression,
    _run_holdout,
    _run_holdout_classification,
    _run_holdout_regression,
)
from .hyperparameter_search import _get_n_components, _sanitize_n_jobs, LoggingGridSearchCV
from .metrics_utils import (
    _compute_ci95,
    _rmse,
    _sanitize_metrics_row,
    _summarize_metric_values,
    _summarize_scores,
)
from .model_factory import (
    _call_build_pipeline,
    _coerce_param_value,
    _format_model_and_method_labels,
    _format_model_label,
    _format_preprocess_step,
    _format_preprocess_steps,
    _get_model_step_name,
    _get_plsda_method_details,
    _get_preprocess_steps_from_pipeline,
    _infer_task_type,
    _make_preprocess,
    _normalize_preprocess_value,
    _sanitize_metrics_row as _model_sanitize_metrics_row,
    _update_row_with_best_params,
)
from .orchestration import _train_and_save_classification, _train_and_save_regression

__all__ = [
    "_call_build_pipeline",
    "_coerce_param_value",
    "_compute_ci95",
    "_format_model_and_method_labels",
    "_format_model_label",
    "_format_preprocess_step",
    "_format_preprocess_steps",
    "_get_model_step_name",
    "_get_n_components",
    "_get_plsda_method_details",
    "_get_preprocess_steps_from_pipeline",
    "_infer_task_type",
    "_make_preprocess",
    "_normalize_preprocess_value",
    "_rmse",
    "_run_cv",
    "_run_cv_classification",
    "_run_cv_regression",
    "_run_holdout",
    "_run_holdout_classification",
    "_run_holdout_regression",
    "_sanitize_metrics_row",
    "_sanitize_n_jobs",
    "_summarize_metric_values",
    "_summarize_scores",
    "_train_and_save_classification",
    "_train_and_save_regression",
    "LoggingGridSearchCV",
]

# Backwards compatibility: reuse the metrics row sanitizer from the model factory
# to ensure identical behaviour to the legacy training implementation.
_sanitize_metrics_row = _model_sanitize_metrics_row

# Legacy aliases retained for external callers expecting training-only helpers.
train_single_model = _train_and_save_classification
train_models = _train_and_save_regression
