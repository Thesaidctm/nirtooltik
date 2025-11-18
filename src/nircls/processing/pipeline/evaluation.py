from __future__ import annotations

import logging
import gc
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import traceback
import warnings
from functools import lru_cache
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
import psutil
import sklearn
import yaml  # type: ignore[import-untyped]
from joblib import dump, load
from rich.console import Console
from rich.progress import Progress
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneOut,
    ParameterGrid,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

from ..config_schema import ConfigModel, ValidationError
from .feature_selection.vip import compute_vip
from .io import export_xy_xlsx
from .io import prepare_inputs as _prepare_inputs
from .io import read_tabular, read_xlsx_xy, summarize_dataset
from ..results.metrics.classification import expected_calibration_error
from ..results.metrics.regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmsecv,
    compute_rmsep,
    compute_rmse,
    rmse_with_ci,
)
from .models.factory import make_models_from_config
from .preprocessors.spectro import (
    EMSC,
    EPO,
    MSC,
    SNV,
    AdaptiveQuantileSegmentNormalizer,
    Detrend,
    SavitzkyGolayTransformer,
)
from ..results.report import generate_scientific_report
from ..utils.compliance import check_astm_classification
from ..utils.email import enviar_email
from ..utils.errors import DataError, ModelError
from ..utils.logging import log_versions
from ..utils.misc import RESULTS_DIR, file_hash, get_git_commit
from ..utils.palettes import Palette
from ..utils.plots import (
    plot_confusion,
    plot_pls_loadings,
    plot_pls_weights,
    plot_pls_y_loadings,
    plot_reliability,
    plot_roc_ovr,
    plot_vip,
)
from ..utils.plots import plot_confusion, plot_pca_scores, plot_reliability, plot_roc_ovr
from ..utils.qc import detect_identical_spectra, find_duplicates, pca_outliers
from ..utils.summarize import summarize_runs
from ..utils.split import spxy_split
from ..utils.y_scramble import y_scramble_scores

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)


_WINDOWS_WAIT_HANDLE_LIMIT = 63
_WINDOWS_MAX_LOKY_WORKERS = max(1, (_WINDOWS_WAIT_HANDLE_LIMIT - 1) // 2)


def _rmse(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute root mean squared error compatible with older sklearn versions."""

    return float(compute_rmse(y_true, y_pred))



def _compute_ci95(mean: float, std: float, n: int) -> Optional[Tuple[float, float]]:
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



def _save_metrics(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    *,
    append: bool = False,
) -> None:
    """Persist metrics to disk.

    When ``append`` is ``True``, metrics are appended to CSV files allowing
    incremental saving. Otherwise, a full summary in Excel/JSON format is
    written, overwriting any existing files.
    """

    for row in rows:
        row["preprocess_steps"] = _normalize_preprocess_value(
            row.get("preprocess_steps")
        )
        if row.get("preprocess_steps") == "RAW":
            row["result"] = "RAW"
        else:
            row.setdefault("result", None)
        row.setdefault("overfit_warning", False)
    sanitized_rows = [_sanitize_metrics_row(row) for row in rows]
    df_new = pd.DataFrame(sanitized_rows)
    metrics_path = out_dir / "metrics.json"
    existing_df = pd.DataFrame()
    if metrics_path.exists():
        try:
            existing_data = json.loads(metrics_path.read_text())
            sanitized_existing = [
                _sanitize_metrics_row(row) for row in existing_data
            ]
            existing_df = pd.DataFrame(sanitized_existing)
        except json.JSONDecodeError:
            existing_df = pd.DataFrame()

    frames: List[pd.DataFrame] = []
    if not existing_df.empty:
        frames.append(existing_df)
    if not df_new.empty:
        frames.append(df_new)
    if frames:
        combined_df = pd.concat(frames, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    if combined_df.empty:
        return

    combined_df = combined_df.drop_duplicates(ignore_index=True)

    if "result" not in combined_df.columns:
        combined_df["result"] = None

    id_cols = [
        "file",
        "target",
        "model",
        "preprocess_steps",
        "result",
        "calibrated",
    ]

    if append:
        if not df_new.empty:
            val_cols_new = id_cols + [
                c for c in df_new.columns if c not in id_cols and not c.startswith("train_")
            ]
            val_df_new = df_new[val_cols_new]

            train_cols_new = [c for c in df_new.columns if c.startswith("train_")]
            calib_df_new = df_new[id_cols + train_cols_new].copy()
            if train_cols_new:
                calib_df_new.rename(
                    columns={c: c.removeprefix("train_") for c in train_cols_new},
                    inplace=True,
                )

            val_csv = out_dir / "metrics_summary.csv"
            calib_csv = out_dir / "metrics_calibration.csv"
            val_df_new.to_csv(
                val_csv, mode="a", header=not val_csv.exists(), index=False
            )
            calib_df_new.to_csv(
                calib_csv, mode="a", header=not calib_csv.exists(), index=False
            )

        json_rows = combined_df.where(pd.notnull(combined_df), None).to_dict(
            orient="records"
        )
        with open(metrics_path, "w") as f:
            json.dump(json_rows, f, indent=2)
        return

    export_df = combined_df
    val_cols = id_cols + [
        c for c in export_df.columns if c not in id_cols and not c.startswith("train_")
    ]
    val_df = export_df[val_cols]

    train_cols = [c for c in export_df.columns if c.startswith("train_")]
    calib_df = export_df[id_cols + train_cols].copy()
    if train_cols:
        calib_df.rename(
            columns={c: c.removeprefix("train_") for c in train_cols}, inplace=True
        )

    with pd.ExcelWriter(out_dir / "metrics_summary.xlsx") as writer:
        val_df.to_excel(writer, sheet_name="validation", index=False)
        calib_df.to_excel(writer, sheet_name="calibration", index=False)

    json_rows = export_df.where(pd.notnull(export_df), None).to_dict(orient="records")
    with open(metrics_path, "w") as f:
        json.dump(json_rows, f, indent=2)



def _maybe_run_qc(
    cfg: Dict[str, Any], data: List[Dict[str, Any]], out_dir: Path
) -> None:
    """Run optional quality control checks and persist a report."""

    data_cfg = cfg.get("data", {}) or {}
    qc_enabled = data_cfg.get("qc")
    if qc_enabled is None:
        qc_enabled = cfg.get("qc")
    if not qc_enabled:
        return

    pre_cfg = cfg.get("preprocess", {}) or {}
    steps_cfg = list(pre_cfg.get("steps") or [])

    def _normalize_step(value: str) -> str:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", value or "")
        slug = slug.strip("_").lower()
        return slug or "step"

    def _sanitize_sheet_name(name: str) -> str:
        cleaned = re.sub(r"[:\\/?*\[\]]", "_", name or "")
        cleaned = cleaned.strip()
        return cleaned or "Sheet"

    def _make_sheet_name(base: str, slug: str, used: Dict[str, Set[str]]) -> str:
        sanitized = _sanitize_sheet_name(base)
        choices = used.setdefault(slug, set())
        candidate = sanitized[:31]
        counter = 1
        while not candidate or candidate in choices:
            suffix = f"_{counter}"
            counter += 1
            available = max(0, 31 - len(suffix))
            trimmed = sanitized[:available]
            if not trimmed:
                trimmed = "Sheet"
            candidate = f"{trimmed}{suffix}"[:31]
        choices.add(candidate)
        return candidate

    sheet_names: Dict[str, Set[str]] = {}
    per_step_frames: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
    interactive = bool(cfg.get("interactive", False))

    out_dir.mkdir(parents=True, exist_ok=True)
    report: List[Dict[str, Any]] = []
    for idx, ds in enumerate(data, start=1):
        X, y = ds["X"], ds["Y"]
        df = X.copy()
        df[ds["target"]] = y
        if isinstance(y, pd.Series):
            labels = y.reindex(X.index)
        else:
            labels = pd.Series(y, index=X.index)

        def _record_pca(step_slug: str, scores: pd.DataFrame, *, key_suffix: str) -> None:
            scores_export = scores.copy()
            scores_export["target"] = labels.reindex(scores.index)

            key = f"pca_{key_suffix}"
            entry[key] = scores_export.to_dict(orient="index")

            per_step_frames.setdefault(step_slug, []).append(
                (
                    _make_sheet_name(base_sheet, step_slug, sheet_names),
                    scores_export,
                )
            )

            step_dir = Path(step_slug)
            excel_rel = step_dir / f"qc_pca_{step_slug}.xlsx"
            entry[f"{key}_file"] = excel_rel.as_posix()

            fig_stem = f"{Path(ds['file']).stem}__{ds['target']}__pca_{step_slug}"
            fig_rel = step_dir / "figures" / fig_stem
            entry[f"{key}_figure"] = fig_rel.with_suffix(".png").as_posix()
            fig_path = out_dir / fig_rel
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                plot_pca_scores(scores, labels, fig_path, interactive=interactive)
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                logger.warning("Skipping PCA plot export for %s: %s", step_slug, exc)

        raw_scores = pca_outliers(X, n_components=2)
        raw_export = raw_scores.copy()
        raw_export["target"] = labels

        entry: Dict[str, Any] = {
            "file": ds["file"],
            "target": ds["target"],
            "duplicates": find_duplicates(df).index.tolist(),
            "identical_spectra": detect_identical_spectra(X),
            "pca_outliers": raw_export.to_dict(orient="index"),
        }

        base_sheet = f"{idx}_{Path(ds['file']).stem}_{ds['target']}"

        _record_pca("raw", raw_scores, key_suffix="raw")

        for step in steps_cfg:
            step_slug = _normalize_step(step)
            pre_steps, _ = _make_preprocess([step], pre_cfg)
            if pre_steps:
                try:
                    Xp = Pipeline(pre_steps).fit_transform(X)
                except Exception as exc:  # pragma: no cover - best effort logging
                    logger.warning(
                        "QC preprocessing '%s' failed for %s: %s",
                        step,
                        ds["file"],
                        exc,
                    )
                    continue
                if isinstance(Xp, pd.DataFrame):
                    Xp_df = Xp
                else:
                    Xp_df = pd.DataFrame(Xp, index=X.index)
                    if Xp_df.shape[1] == X.shape[1]:
                        Xp_df.columns = X.columns
            else:
                Xp_df = X

            scores = pca_outliers(Xp_df, n_components=2)
            _record_pca(step_slug, scores, key_suffix=step)

        report.append(entry)

    for slug, frames in per_step_frames.items():
        if not frames:
            continue
        step_dir = Path(slug)
        excel_rel = step_dir / f"qc_pca_{slug}.xlsx"
        xlsx_path = out_dir / excel_rel
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with pd.ExcelWriter(xlsx_path) as writer:
                for sheet_name, frame in frames:
                    frame.to_excel(writer, sheet_name=sheet_name, index=True)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            logger.warning("Skipping PCA Excel export for %s: %s", slug, exc)

    with open(out_dir / "qc_report.json", "w") as f:
        json.dump(report, f, indent=2)



def _save_plots(
    file_path: str,
    target: str,
    model: str,
    classes: Optional[np.ndarray],
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    preprocess_label: Optional[str] = None,
    *,
    interactive: bool = False,
    palette: Palette = Palette.BLUES,
    task_type: str = "classification",
) -> None:
    """Save confusion matrix and optional ROC/reliability plots.

    When ``interactive`` is ``True`` additional HTML versions of the confusion
    matrix and ROC curve are generated using Plotly. When provided,
    ``preprocess_label`` is appended to the output stem so that figures created
    for different preprocessing pipelines do not overwrite one another.
    """

    out_fig = out_dir / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    if preprocess_label is not None:
        label = _normalize_preprocess_value(preprocess_label)
        if label:
            stem = f"{stem}__{label}"
    if str(task_type).lower() != "regression":
        if classes is None:
            raise ValueError("classes must be provided for classification plots")
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion(
            cm,
            classes,
            out_fig / f"{stem}__{target}__{model}__confusion",
            interactive=interactive,
            palette=palette,
        )
        if y_proba is not None:
            plot_roc_ovr(
                y_true,
                y_proba,
                classes,
                out_fig / f"{stem}__{target}__{model}__roc",
                interactive=interactive,
                palette=palette,
            )
            plot_reliability(
                y_true,
                y_proba,
                out_fig / f"{stem}__{target}__{model}__reliability",
                interactive=interactive,
                palette=palette,
            )



def _append_table(df: pd.DataFrame, path: Path) -> None:
    """Append rows from ``df`` to ``path`` preserving headers."""

    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    mode = "a" if exists else "w"
    df.to_csv(path, mode=mode, header=not exists, index=False)



def _build_pls_scores_df(
    scores: Optional[np.ndarray], index: pd.Index, labels: np.ndarray
) -> pd.DataFrame:
    """Return a tidy DataFrame with PLS scores per sample and component."""

    columns = [
        "sample",
        "sample_index",
        "y_true",
        "component",
        "component_index",
        "value",
    ]
    if scores is None:
        return pd.DataFrame(columns=columns)
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2:
        raise ValueError("scores must be a 2-D array")
    n_samples, n_components = arr.shape
    if n_samples == 0 or n_components == 0:
        return pd.DataFrame(columns=columns)
    idx = pd.Index(index)
    samples = [str(v) for v in idx.tolist()]
    if len(samples) != n_samples:
        raise ValueError("Number of samples does not match scores shape")
    lab_arr = np.asarray(labels)
    if lab_arr.shape[0] != n_samples:
        raise ValueError("Number of labels does not match scores shape")
    records: list[dict[str, object]] = []
    for sample_idx, (sample, label, row) in enumerate(
        zip(samples, lab_arr, arr, strict=False)
    ):
        for comp_idx, value in enumerate(row, start=1):
            records.append(
                {
                    "sample": sample,
                    "sample_index": sample_idx,
                    "y_true": label,
                    "component": f"Comp {comp_idx}",
                    "component_index": comp_idx,
                    "value": float(value),
                }
            )
    return pd.DataFrame.from_records(records, columns=columns)



def _save_pls_diagnostics(
    *,
    template_pipe: Pipeline,
    best_params: Optional[Mapping[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    classes: Optional[np.ndarray],
    task_type: str = "classification",
    file_path: str,
    target: str,
    model: str,
    preprocess_steps: Optional[str],
    out_dir: Path,
    interactive: bool,
    palette: Palette,
) -> Dict[str, str]:
    """Export PLS loadings, Y loadings, weights and VIP tables when available."""

    if not hasattr(template_pipe, "named_steps"):
        return {}
    final_step = _get_model_step_name(template_pipe)
    clf_step = template_pipe.named_steps.get(final_step)
    if clf_step is None or not hasattr(clf_step, "pls"):
        return {}

    try:
        pipe = clone(template_pipe)
    except Exception:  # pragma: no cover - fallback to deepcopy
        pipe = deepcopy(template_pipe)
    if best_params:
        try:
            pipe.set_params(**best_params)
        except Exception:  # pragma: no cover - log and continue
            logger.debug("Could not apply best params for PLS diagnostics", exc_info=True)
    try:
        pipe.fit(X.values, y)
    except Exception:  # pragma: no cover - unable to fit pipeline
        logger.debug("Could not fit pipeline for PLS diagnostics", exc_info=True)
        return {}

    clf = pipe.named_steps.get(_get_model_step_name(pipe))
    pls = getattr(clf, "pls", None)
    if pls is None:
        return {}

    loadings = getattr(pls, "x_loadings_", None)
    weights = getattr(pls, "x_weights_", None)
    y_loadings = getattr(pls, "y_loadings_", None)
    x_scores = getattr(pls, "x_scores_", None)
    y_scores = getattr(pls, "y_scores_", None)
    if loadings is None or weights is None:
        return {}

    try:
        Xt = pipe[:-1].transform(X.values) if len(pipe.steps) > 1 else X.values
    except Exception:  # pragma: no cover - preprocessing transform failed
        Xt = X.values

    task_type_normalized = str(task_type or "classification").lower()
    y_array = np.asarray(y)
    if task_type_normalized == "regression":
        Y = np.asarray(y_array, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        class_labels = None
    else:
        if classes is None:
            return {}
        classes = np.asarray(classes)
        y_indices = y_array.astype(int, copy=False)
        Y = np.eye(len(classes))[y_indices]
        class_labels = classes
    try:
        vip_scores = compute_vip(pls, Xt, Y)
    except Exception:  # pragma: no cover - VIP computation failed
        logger.debug("Could not compute VIP scores", exc_info=True)
        return {}

    figures_dir = out_dir / "figures"
    stem = Path(file_path).stem
    if preprocess_steps is not None:
        label = _normalize_preprocess_value(preprocess_steps)
        if label:
            stem = f"{stem}__{label}"
    base = f"{stem}__{target}__{model}"
    loadings_base = figures_dir / f"{base}__pls_loadings"
    weights_base = figures_dir / f"{base}__pls_weights"
    vip_base = figures_dir / f"{base}__vip"
    y_loadings_base: Optional[Path] = None
    if task_type_normalized == "classification" and y_loadings is not None:
        y_loadings_base = figures_dir / f"{base}__pls_y_loadings"

    loadings_df = plot_pls_loadings(
        loadings,
        X.columns,
        loadings_base,
        interactive=interactive,
        palette=palette,
    )
    weights_df = plot_pls_weights(
        weights,
        X.columns,
        weights_base,
        interactive=interactive,
        palette=palette,
    )
    vip_df = plot_vip(
        vip_scores,
        X.columns,
        vip_base,
        interactive=interactive,
        palette=palette,
    )
    y_loadings_df: Optional[pd.DataFrame] = None
    if y_loadings_base is not None and task_type_normalized == "classification":
        y_loadings_df = plot_pls_y_loadings(
            y_loadings,
            class_labels if class_labels is not None else np.array([]),
            y_loadings_base,
            interactive=interactive,
            palette=palette,
        )

    preprocess_value = preprocess_steps if preprocess_steps else "None"

    y_arr = np.asarray(y)
    if task_type_normalized == "classification" and class_labels is not None:
        y_int = y_arr.astype(int, copy=False)
        try:
            y_labels = class_labels[y_int]
        except Exception:
            y_labels = y_arr
    else:
        y_labels = y_arr
    index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(y_arr))
    x_scores_df = _build_pls_scores_df(x_scores, index, y_labels)
    y_scores_df = _build_pls_scores_df(y_scores, index, y_labels)

    loadings_df.insert(0, "file", file_path)
    loadings_df.insert(1, "target", target)
    loadings_df.insert(2, "model", model)
    loadings_df.insert(3, "preprocess_steps", preprocess_value)
    weights_df.insert(0, "file", file_path)
    weights_df.insert(1, "target", target)
    weights_df.insert(2, "model", model)
    weights_df.insert(3, "preprocess_steps", preprocess_value)
    vip_df.insert(0, "file", file_path)
    vip_df.insert(1, "target", target)
    vip_df.insert(2, "model", model)
    vip_df.insert(3, "preprocess_steps", preprocess_value)
    x_scores_df.insert(0, "file", file_path)
    x_scores_df.insert(1, "target", target)
    x_scores_df.insert(2, "model", model)
    x_scores_df.insert(3, "preprocess_steps", preprocess_value)
    y_scores_df.insert(0, "file", file_path)
    y_scores_df.insert(1, "target", target)
    y_scores_df.insert(2, "model", model)
    y_scores_df.insert(3, "preprocess_steps", preprocess_value)
    if y_loadings_df is not None:
        y_loadings_df.insert(0, "file", file_path)
        y_loadings_df.insert(1, "target", target)
        y_loadings_df.insert(2, "model", model)
        y_loadings_df.insert(3, "preprocess_steps", preprocess_value)

    loadings_cols = [
        "file",
        "target",
        "model",
        "preprocess_steps",
        "feature",
        "component",
        "component_index",
        "value",
    ]
    vip_cols = [
        "file",
        "target",
        "model",
        "preprocess_steps",
        "feature",
        "vip",
    ]
    score_cols = [
        "file",
        "target",
        "model",
        "preprocess_steps",
        "sample",
        "sample_index",
        "y_true",
        "component",
        "component_index",
        "value",
    ]

    results_dir = out_dir / "results"
    y_loadings_csv = results_dir / "pls_y_loadings.csv"

    if y_loadings_df is not None:
        _append_table(y_loadings_df[loadings_cols], y_loadings_csv)
    _append_table(loadings_df[loadings_cols], results_dir / "pls_loadings.csv")
    _append_table(weights_df[loadings_cols], results_dir / "pls_weights.csv")
    _append_table(vip_df[vip_cols], results_dir / "vip.csv")
    _append_table(x_scores_df[score_cols], results_dir / "pls_x_scores.csv")
    _append_table(y_scores_df[score_cols], results_dir / "pls_y_scores.csv")

    def _rel(path: Path) -> str:
        try:
            return path.relative_to(out_dir).as_posix()
        except ValueError:
            return path.as_posix()

    artifacts: Dict[str, str] = {
        "pls_loadings_csv": _rel(results_dir / "pls_loadings.csv"),
        "pls_weights_csv": _rel(results_dir / "pls_weights.csv"),
        "vip_csv": _rel(results_dir / "vip.csv"),
        "pls_loadings_png": _rel(loadings_base.with_suffix(".png")),
        "pls_weights_png": _rel(weights_base.with_suffix(".png")),
        "vip_png": _rel(vip_base.with_suffix(".png")),
    }
    if y_loadings_df is not None:
        if not y_loadings_df.empty and y_loadings_base is not None:
            artifacts["pls_y_loadings_png"] = _rel(
                y_loadings_base.with_suffix(".png")
            )
        if y_loadings_csv.exists():
            artifacts["pls_y_loadings_csv"] = _rel(y_loadings_csv)
    if (results_dir / "pls_x_scores.csv").exists():
        artifacts["pls_x_scores_csv"] = _rel(results_dir / "pls_x_scores.csv")
    if (results_dir / "pls_y_scores.csv").exists():
        artifacts["pls_y_scores_csv"] = _rel(results_dir / "pls_y_scores.csv")
    if interactive:
        html_targets = [
            ("pls_loadings_html", loadings_base),
            ("pls_weights_html", weights_base),
            ("vip_html", vip_base),
        ]
        if y_loadings_base is not None:
            html_targets.append(("pls_y_loadings_html", y_loadings_base))
        for key, base in html_targets:
            html_path = base.with_suffix(".html")
            if html_path.exists():
                artifacts[key] = _rel(html_path)

    return artifacts
