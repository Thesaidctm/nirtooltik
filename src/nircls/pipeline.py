from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
import traceback
import warnings
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
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

from .config_schema import ConfigModel, ValidationError
from .feature_selection.vip import compute_vip
from .io import export_xy_xlsx
from .io import prepare_inputs as _prepare_inputs
from .io import read_tabular, read_xlsx_xy, summarize_dataset
from .metrics.calibration import expected_calibration_error
from .metrics.regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmsecv,
    compute_rmsep,
    compute_rmse,
)
from .models.factory import make_models_from_config
from .preprocessors.spectro import (
    EMSC,
    EPO,
    MSC,
    SNV,
    Detrend,
    SavitzkyGolayTransformer,
)
from .report import generate_scientific_report
from .utils.compliance import check_astm_classification
from .utils.email import enviar_email
from .utils.errors import DataError, ModelError
from .utils.logging import log_versions, logger
from .utils.misc import RESULTS_DIR, file_hash
from .utils.palettes import Palette
from .utils.plots import (
    plot_confusion,
    plot_pls_loadings,
    plot_pls_weights,
    plot_pls_y_loadings,
    plot_reliability,
    plot_roc_ovr,
    plot_vip,
)
from .utils.plots import plot_confusion, plot_pca_scores, plot_reliability, plot_roc_ovr
from .utils.qc import detect_identical_spectra, find_duplicates, pca_outliers
from .utils.split import spxy_split
from .utils.y_scramble import y_scramble_scores

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)


def _rmse(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute root mean squared error compatible with older sklearn versions."""

    return float(compute_rmse(y_true, y_pred))


def _validate_split_capacity(allowed: int, *, kind: str) -> None:
    """Raise a :class:`DataError` when ``allowed`` splits are insufficient."""

    if allowed >= 2:
        return
    if kind == "groups":
        raise DataError(
            "Número insuficiente de grupos distintos para validação cruzada. "
            "Forneça pelo menos dois grupos no conjunto de treino."
        )
    raise DataError(
        "Número insuficiente de amostras para validação cruzada. "
        "Forneça pelo menos duas amostras no conjunto de treino."
    )


class LoggingGridSearchCV(GridSearchCV):
    """GridSearchCV with optional debug logging for progress."""

    def _run_search(self, evaluate_candidates):  # type: ignore[override]
        grid = list(ParameterGrid(self.param_grid))
        total = len(grid)
        for i, params in enumerate(grid, 1):
            if self.verbose < 1:
                logger.debug("Grid search progress: %d/%d parameters", i, total)
            evaluate_candidates([params])


def _validate_cfg(
    cfg: Dict[str, Any],
    *,
    need_data: bool = False,
    need_cv: bool = False,
    need_models: bool = True,
) -> ConfigModel:
    """Validate configuration dictionary using :class:`ConfigModel`."""

    try:
        model = ConfigModel(**cfg)
    except ValidationError as exc:
        raise DataError(
            f"Configuração inválida: {exc}. Revise os campos informados."
        ) from exc
    if need_data and model.data is None:
        raise DataError("Seção 'data' é obrigatória. Informe o caminho dos dados.")
    if need_models and (not model.models):
        raise ModelError("Seção 'models' é obrigatória. Adicione pelo menos um modelo.")
    if need_cv and (model.cv is None or not model.cv.strategies):
        raise ModelError(
            "Seção 'cv' é obrigatória. Defina ao menos uma estratégia de validação."
        )
    return model


def save_run_metadata(cfg: Dict[str, Any], out_dir: Path) -> None:
    """Persist experiment configuration and environment details."""

    out_dir.mkdir(parents=True, exist_ok=True)

    try:  # pragma: no cover - best effort
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
    except Exception:  # pragma: no cover
        logger.debug("Could not save config.yaml", exc_info=True)

    log_versions()
    try:  # pragma: no cover - best effort
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        (out_dir / "env.txt").write_text(result.stdout)
    except Exception:  # pragma: no cover
        logger.debug("Could not save env.txt", exc_info=True)


def prepare_inputs(
    cfg: Dict[str, Any], *, use_cache: bool = True
) -> List[Dict[str, Any]]:
    """Load datasets, using cached features when available."""

    if not use_cache:
        return _prepare_inputs(cfg, use_cache=False)

    files = [Path(p) for p in glob(cfg["input_path"])]
    sp = cfg.get("spectral_range", None)
    sp = (float(sp[0]), float(sp[1])) if sp else None
    out_dir = Path(cfg.get("out_dir", RESULTS_DIR))
    cache_dir = out_dir / "cache"
    target_type = str(cfg.get("target_type", "classification")).lower()
    if target_type not in {"classification", "regression"}:
        raise DataError(f"Tipo de alvo inválido: {cfg.get('target_type')}")

    cached_items: List[Tuple[Path, Dict[str, Any]]] = []
    for f in files:
        cpath = cache_dir / f"{f.stem}.joblib"
        fingerprint = {"mtime": f.stat().st_mtime, "spectral_range": sp}
        if not cpath.exists():
            cached_items = []
            break
        try:
            cached = load(cpath)
        except Exception:
            cached_items = []
            break
        if cached.get("fingerprint") != fingerprint:
            cached_items = []
            break
        cached_items.append((f, cached))

    if not cached_items:
        return _prepare_inputs(cfg, use_cache=use_cache)

    tgt = cfg.get("target_cols")
    if tgt is None and "target_col" in cfg:
        tgt = cfg["target_col"]
    grp = cfg.get("group_col")

    out: List[Dict[str, Any]] = []
    for f, cached in cached_items:
        keep_idx = cached["keep_idx"]
        X = cached["X"]
        n_nans = cached.get("n_nans", 0)
        if f.suffix.lower() in [".xlsx", ".xls"] and (
            "x_sheet" in cfg or "y_sheet" in cfg
        ):
            _, y_list, groups = read_xlsx_xy(
                f,
                cfg.get("x_sheet", "X"),
                cfg.get("y_sheet", "Y"),
                tgt,
                grp,
                target_type=target_type,
            )
        else:
            _, y_list, groups = read_tabular(
                f, tgt, grp, target_type=target_type
            )
        groups_reset = None
        if groups is not None:
            groups_reset = groups.loc[keep_idx].reset_index(drop=True)
        for name, y in y_list:
            y = y.loc[keep_idx].reset_index(drop=True)
            ds = {
                "file": f.as_posix(),
                "target": name,
                "X": X,
                "Y": y,
                "n_nans": n_nans,
                "target_type": target_type,
                "task_type": target_type,
            }
            if groups_reset is not None:
                ds["groups"] = groups_reset
            out.append(ds)
    return out


def _make_preprocess(
    steps_cfg: Optional[Sequence[str]], cfg: Dict[str, Any]
) -> Tuple[List[Tuple[str, Any]], Dict[str, Any]]:
    """Create preprocessing steps and optional parameter grid.

    For the Savitzky-Golay (SG) step it is possible to provide a list of
    parameter values (e.g. window length) which will be added to the grid for
    hyper-parameter search.  This allows testing multiple window sizes during
    cross-validation while keeping the feature optional – supplying a single
    value keeps the behaviour unchanged.
    """

    steps = []
    grid = {}
    for name in steps_cfg or []:
        if name is None:
            continue
        if isinstance(name, str):
            cleaned = name.strip()
            if not cleaned:
                continue
            if cleaned.lower() in {"none", "raw"}:
                continue
            n = cleaned.upper()
        else:
            n = str(name).upper()
        if n == "SNV":
            steps.append(("snv", SNV()))
        elif n == "MSC":
            steps.append(("msc", MSC()))
        elif n == "DETREND":
            steps.append(("detrend", Detrend()))
        elif n == "SG":
            sg = cfg.get("sg", {"window": 15, "polyorder": 2, "deriv": 1})
            wl = sg.get("window", 15)
            po = sg.get("polyorder", 2)
            dv = sg.get("deriv", 1)
            # allow lists/tuples to trigger grid search over SG parameters
            if isinstance(wl, (list, tuple)):
                grid["sg__window_length"] = [int(w) for w in wl]
                wl = wl[0]
            if isinstance(po, (list, tuple)):
                grid["sg__polyorder"] = [int(p) for p in po]
                po = po[0]
            if isinstance(dv, (list, tuple)):
                grid["sg__deriv"] = [int(d) for d in dv]
                dv = dv[0]
            steps.append(("sg", SavitzkyGolayTransformer(wl, po, dv)))
        elif n == "EMSC":
            steps.append(("emsc", EMSC()))
        elif n == "EPO":
            steps.append(("epo", EPO()))
        elif n == "SCALE":
            steps.append(("scaler", StandardScaler()))
        else:
            raise ModelError(
                f"Pré-processo desconhecido: {name}. Verifique a lista de etapas suportadas."
            )
    return steps, grid


def _infer_task_type(cfg: Mapping[str, Any]) -> str:
    """Infer the task type (classification or regression) from ``cfg``."""

    task_type = cfg.get("task_type")
    if not task_type:
        data_cfg = cfg.get("data") if isinstance(cfg, Mapping) else None
        if isinstance(data_cfg, Mapping):
            task_type = data_cfg.get("task_type")
        if not task_type and isinstance(data_cfg, Mapping):
            task_type = data_cfg.get("target_type")
    normalized = str(task_type or "classification").lower()
    if normalized not in {"classification", "regression"}:
        raise DataError(f"Tipo de tarefa inválido: {task_type}")
    return normalized


def _get_model_step_name(pipe: Pipeline) -> str:
    """Return the name of the final estimator step in ``pipe``."""

    if hasattr(pipe, "named_steps") and pipe.named_steps:
        for candidate in ("clf", "reg"):
            if candidate in pipe.named_steps:
                return candidate
        # fall back to the last named step if available
        try:
            return next(reversed(pipe.named_steps))  # type: ignore[arg-type]
        except StopIteration:  # pragma: no cover - empty pipeline
            pass
    if getattr(pipe, "steps", None):
        return pipe.steps[-1][0]
    return "clf"


def _get_preprocess_steps_from_pipeline(pipe: Pipeline) -> List[Any]:
    """Return preprocessing steps stored in ``pipe`` excluding the estimator."""

    if not hasattr(pipe, "named_steps"):
        return []
    final_name = _get_model_step_name(pipe)
    return [v for k, v in pipe.named_steps.items() if k != final_name]


def build_pipeline_from_config(
    cfg: Dict[str, Any],
) -> List[Tuple[str, Pipeline, Dict[str, Any]]]:
    _validate_cfg(cfg, need_models=True)
    task_type = _infer_task_type(cfg)
    final_step_name = "clf" if task_type == "classification" else "reg"
    pre_cfg = cfg.get("preprocess", {})
    steps_cfg = pre_cfg.get("steps", [])
    evaluate_flag = pre_cfg.get("evaluate_individual")
    if evaluate_flag is None:
        evaluate_ind = len(steps_cfg) > 1
    else:
        evaluate_ind = bool(evaluate_flag)
    out: List[Tuple[str, Pipeline, Dict[str, Any]]] = []

    def _build_models(
        pre_steps: List[Tuple[str, Any]], pre_grid: Dict[str, Any], suffix: str = ""
    ) -> None:
        try:
            models = make_models_from_config(cfg["models"], task_type=task_type)
        except TypeError:
            models = make_models_from_config(cfg["models"])
        for m in models:
            est = m["estimator"]
            grid = pre_grid.copy()
            model_grid = m.get("grid", {}) or {}
            if task_type == "classification" and not hasattr(est, "predict_proba"):
                calib_cfg = m.get("calibration") or {}
                method = calib_cfg.get("method", "sigmoid")
                cv = calib_cfg.get("cv", 5)
                est = CalibratedClassifierCV(estimator=est, method=method, cv=cv)
                grid.update(
                    {
                        f"{final_step_name}__estimator__{k}": v
                        for k, v in model_grid.items()
                    }
                )
            else:
                grid.update({f"{final_step_name}__{k}": v for k, v in model_grid.items()})
            name = m["name"] + suffix
            pipe = Pipeline(pre_steps + [(final_step_name, est)])
            out.append((name, pipe, grid))

    if evaluate_ind and len(steps_cfg) > 1:
        for step in steps_cfg:
            pre_steps, pre_grid = _make_preprocess([step], pre_cfg)
            _build_models(pre_steps, pre_grid, suffix=f"__{step.lower()}")
    else:
        pre_steps, pre_grid = _make_preprocess(steps_cfg, pre_cfg)
        _build_models(pre_steps, pre_grid)

    return out


def _format_preprocess_step(step: Any) -> str:
    """Return a concise string representation for a preprocessing ``step``."""

    if step is None:
        return "None"
    if isinstance(step, str):
        cleaned = step.strip()
        if not cleaned:
            return ""
        if cleaned.endswith("()"):
            cleaned = cleaned[:-2]
        upper = cleaned.upper()
        mapping = {
            "SNV": "SNV",
            "MSC": "MSC",
            "SG": "SG",
            "EMSC": "EMSC",
            "EPO": "EPO",
            "SCALE": "SCALE",
        }
        if upper in mapping:
            return mapping[upper]
        if cleaned.lower().startswith("detrend"):
            suffix = cleaned[len("Detrend") :]
            return f"DETREND{suffix}"
        if cleaned.startswith("SavitzkyGolayTransformer") or cleaned.startswith("SG("):
            cleaned = cleaned.replace("SavitzkyGolayTransformer", "SG")
            cleaned = cleaned.replace("window_length", "window")
            cleaned = cleaned.replace("polyorder", "poly")
            return cleaned
        if cleaned.lower().startswith("standardscaler"):
            return "SCALE"
        return cleaned

    if isinstance(step, SNV):
        return "SNV"
    if isinstance(step, MSC):
        return "MSC"
    if isinstance(step, SavitzkyGolayTransformer):
        return (
            f"SG(window={step.window_length},poly={step.polyorder},deriv={step.deriv})"
        )
    if isinstance(step, Detrend):
        return f"DETREND(order={step.order})"
    if isinstance(step, EMSC):
        return f"EMSC(degree={step.degree})"
    if isinstance(step, EPO):
        return "EPO"
    if isinstance(step, StandardScaler):
        return "SCALE"
    return step.__class__.__name__


def _format_preprocess_steps(steps: Iterable[Any]) -> str:
    """Format a sequence of preprocessing ``steps`` into a readable string."""

    formatted: List[str] = []
    for step in steps:
        desc = _format_preprocess_step(step)
        if desc:
            formatted.append(desc)
    result = ", ".join(formatted)
    return result if result else "None"


def _normalize_preprocess_value(value: Any) -> str:
    """Normalize any representation of preprocessing steps to a string."""

    if value is None:
        return "None"
    if isinstance(value, str):
        formatted = _format_preprocess_step(value)
        return formatted if formatted else "None"
    if isinstance(value, (list, tuple)):
        return _format_preprocess_steps(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return _format_preprocess_steps(list(value))
    if value:
        return str(value)
    return "None"


def _coerce_param_value(value: Any) -> Any:
    """Convert parameter ``value`` to a JSON-friendly representation."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_coerce_param_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_param_value(v) for k, v in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    df_new = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics.json"
    existing_df = pd.DataFrame()
    if metrics_path.exists():
        try:
            existing_data = json.loads(metrics_path.read_text())
            existing_df = pd.DataFrame(existing_data)
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

    if not cfg.get("qc"):
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
            scores_export = scores.copy()
            scores_export["target"] = labels.reindex(scores.index)

            key = f"pca_{step}"
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
            plot_pca_scores(scores, labels, fig_path, interactive=interactive)

        report.append(entry)

    for slug, frames in per_step_frames.items():
        if not frames:
            continue
        step_dir = Path(slug)
        excel_rel = step_dir / f"qc_pca_{slug}.xlsx"
        xlsx_path = out_dir / excel_rel
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx_path) as writer:
            for sheet_name, frame in frames:
                frame.to_excel(writer, sheet_name=sheet_name, index=True)

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


def _get_n_components(best_params: Dict[str, Any]) -> Optional[int]:
    """Return the best ``n_components`` value from GridSearchCV parameters."""

    for k, v in best_params.items():
        if k.endswith("n_components"):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def _run_cv_classification(
    cfg: Dict[str, Any],
    out_dir: Path,
    outer: Any,
    progress: Progress,
    *,
    random_state: int,
    interactive: bool = False,
    n_jobs: int = 1,
    log_memory: bool = False,
    step_label: Optional[str] = None,
) -> None:
    """Run cross-validation for configured datasets and models (classification).

    Parameters
    ----------
    cfg:
        Experiment configuration.
    out_dir:
        Directory used to persist metrics and plots.
    outer:
        External cross-validation splitter.
    progress:
        Progress bar instance for user feedback.
    random_state:
        Seed applied to inner cross-validation splits.
    n_jobs:
        Number of parallel jobs for ``GridSearchCV``.
    step_label:
        Optional label to use when no preprocessing steps are selected.
    """

    n_jobs = max(1, n_jobs)
    logger.info("Starting cross-validation")
    try:
        data = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))
        _maybe_run_qc(cfg, data, out_dir)
    except DataError as e:
        logger.error("Data preparation failed: %s", e)
        raise DataError(
            "Arquivo de dados não encontrado. Verifique o caminho informado."
        ) from e
    summary = summarize_dataset(data)
    is_loo = isinstance(outer, LeaveOneOut)
    rows: List[Dict[str, Any]] = []
    tgt_task = progress.add_task("Target", total=len(data))
    for ds in data:
        progress.update(tgt_task, description=f"Target: {ds['target']}")
        X, y = ds["X"], ds["Y"]
        groups = ds.get("groups")
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        counts = np.bincount(y_enc)
        cv_cfg = cfg.get("cv", {})
        outer_folds = int(
            cv_cfg.get(
                "outer_folds",
                getattr(outer, "n_splits", outer.get_n_splits(X, y_enc, groups)),
            )
        )
        n_splits = int(min(outer_folds, counts.min()))
        n_splits = max(n_splits, 2)
        if n_splits != outer_folds:
            logger.warning(
                "Adjusted number of outer folds from %d to %d", outer_folds, n_splits
            )
        if isinstance(outer, RepeatedStratifiedKFold):
            outer = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=outer.n_repeats,
                random_state=random_state,
            )
        elif isinstance(outer, StratifiedKFold):
            outer = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        elif isinstance(outer, GroupKFold):
            outer = GroupKFold(n_splits=n_splits)
        cv_method = cv_cfg.get("cv_method", "stratified").lower()
        inner_folds = cv_cfg.get("inner_folds", 3)
        if inner_folds in (None, False):
            inner_folds = 3
        try:
            inner_folds = int(inner_folds)
        except (TypeError, ValueError) as exc:  # pragma: no cover - sanity check
            raise ModelError("'inner_folds' must be an integer >= 2") from exc
        if inner_folds < 2:
            raise ModelError("'inner_folds' must be at least 2")
        models = build_pipeline_from_config(cfg)
        model_cfgs = cfg.get("models", [])
        model_task = progress.add_task("Model", total=len(models))
        for idx, (name, pipe, grid) in enumerate(models):
            if not model_cfgs:
                continue
            m_cfg = model_cfgs[idx % len(model_cfgs)]
            progress.update(model_task, description=f"Model: {name}")
            steps_desc = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(pipe)
            )
            logger.info("Running %s with steps: %s", name, steps_desc)
            mem_start = None
            if log_memory:
                mem_start = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory start %.3f MB", mem_start)
            yscramble_iter = int(m_cfg.get("y_scramble_n_iter", 20))
            scores_val: Dict[str, List[float]] = {
                "F1-macro": [],
                "BalancedAcc": [],
                "Accuracy": [],
                "Brier": [],
                "ECE": [],
            }
            scores_train: Dict[str, List[float]] = {
                "F1-macro": [],
                "BalancedAcc": [],
                "Accuracy": [],
                "Brier": [],
                "ECE": [],
            }
            y_true_all: List[np.ndarray] = []
            y_pred_all: List[np.ndarray] = []
            y_proba_all: List[np.ndarray] = []
            n_comp_list: List[int] = []
            best_steps_repr: Optional[str] = None
            best_params_example: Optional[Dict[str, Any]] = None
            if is_loo:
                inner_splits = int(min(inner_folds, counts.min()))
                inner_splits = max(inner_splits, 2)
                inner = StratifiedKFold(
                    n_splits=inner_splits,
                    shuffle=True,
                    random_state=random_state,
                )
                gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=1)
                gs.fit(X.values, y_enc)
                best_params = gs.best_params_
                n_comp = _get_n_components(best_params)
                if n_comp is not None:
                    n_comp_list.append(n_comp)
                best = pipe.set_params(**best_params)
                if best_steps_repr is None:
                    best_steps_repr = _format_preprocess_steps(
                        _get_preprocess_steps_from_pipeline(best)
                    )
                calib_cfg = m_cfg.get("calibration")
                if calib_cfg:
                    best = CalibratedClassifierCV(
                        estimator=best,
                        method=calib_cfg.get("method", "sigmoid"),
                        cv=calib_cfg.get("cv", 5),
                    )
                proba = cross_val_predict(
                    best,
                    X.values,
                    y_enc,
                    cv=LeaveOneOut(),
                    method="predict_proba",
                    n_jobs=1,
                )
                yhat = np.argmax(proba, axis=1)
                scores_val["F1-macro"].append(f1_score(y_enc, yhat, average="macro"))
                scores_val["BalancedAcc"].append(balanced_accuracy_score(y_enc, yhat))
                scores_val["Accuracy"].append(accuracy_score(y_enc, yhat))
                y_proba_all.append(proba)
                brier = np.mean(
                    [
                        brier_score_loss((y_enc == k).astype(int), proba[:, k])
                        for k in range(len(classes))
                    ]
                )
                scores_val["Brier"].append(float(brier))
                scores_val["ECE"].append(
                    expected_calibration_error(y_enc, proba, n_bins=15)
                )
                best.fit(X.values, y_enc)
                yhat_tr = best.predict(X.values)
                scores_train["F1-macro"].append(
                    f1_score(y_enc, yhat_tr, average="macro")
                )
                scores_train["BalancedAcc"].append(
                    balanced_accuracy_score(y_enc, yhat_tr)
                )
                scores_train["Accuracy"].append(accuracy_score(y_enc, yhat_tr))
                proba_tr = best.predict_proba(X.values)
                brier_tr = np.mean(
                    [
                        brier_score_loss((y_enc == k).astype(int), proba_tr[:, k])
                        for k in range(len(classes))
                    ]
                )
                scores_train["Brier"].append(float(brier_tr))
                scores_train["ECE"].append(
                    expected_calibration_error(y_enc, proba_tr, n_bins=15)
                )
                y_true_all.append(y_enc)
                y_pred_all.append(yhat)
                model_label, method_label = _format_model_and_method_labels(
                    name, pipe, best_params
                )
                row = {
                    "file": ds["file"],
                    "target": ds["target"],
                    "model": model_label,
                    "calibrated": bool(m_cfg.get("calibration")),
                    "method": method_label,
                }
                row.update(
                    {f"{k}_mean": float(np.mean(v)) for k, v in scores_val.items() if v}
                )
                row.update(
                    {
                        f"{k}_std": float(np.std(v, ddof=1))
                        for k, v in scores_val.items()
                        if v and len(v) > 1
                    }
                )
                row.update(
                    {
                        f"train_{k}_mean": float(np.mean(v))
                        for k, v in scores_train.items()
                        if v
                    }
                )
                row.update(
                    {
                        f"train_{k}_std": float(np.std(v, ddof=1))
                        for k, v in scores_train.items()
                        if v and len(v) > 1
                    }
                )
                for m in ["Brier", "ECE"]:
                    row.setdefault(f"{m}_mean", None)
                    row.setdefault(f"{m}_std", None)
                    row.setdefault(f"train_{m}_mean", None)
                    row.setdefault(f"train_{m}_std", None)
                _update_row_with_best_params(row, best_params)
                if n_comp_list:
                    row["n_components"] = int(np.median(n_comp_list))
                row["preprocess_steps"] = best_steps_repr or "None"
                if step_label and row["preprocess_steps"] == "None":
                    row["preprocess_steps"] = step_label
                if row.get("preprocess_steps") == "RAW":
                    row["result"] = "RAW"
                else:
                    row.setdefault("result", None)
                pls_artifacts = _save_pls_diagnostics(
                    template_pipe=pipe,
                    best_params=best_params,
                    X=X,
                    y=y_enc,
                    classes=classes,
                    task_type="classification",
                    file_path=ds["file"],
                    target=ds["target"],
                    model=model_label,
                    preprocess_steps=row.get("preprocess_steps"),
                    out_dir=out_dir,
                    interactive=interactive,
                    palette=Palette(cfg.get("palette", Palette.BLUES)),
                )
                if pls_artifacts:
                    row.update(pls_artifacts)
                if log_memory:
                    mem_end = psutil.Process().memory_info().rss / 1e6
                    logger.debug("memory end %.3f MB", mem_end)
                    if mem_start is not None:
                        row["memory_mb_start"] = float(mem_start)
                    row["memory_mb_end"] = float(mem_end)
                rows.append(row)
                _save_metrics(
                    [row],
                    out_dir,
                    append=True,
                )
                y_true = np.concatenate(y_true_all)
                y_pred = np.concatenate(y_pred_all)
                y_proba = np.vstack(y_proba_all) if y_proba_all else None
                _save_plots(
                    ds["file"],
                    ds["target"],
                    model_label,
                    classes,
                    out_dir,
                    y_true,
                    y_pred,
                    y_proba,
                    preprocess_label=row.get("preprocess_steps"),
                    interactive=interactive,
                    palette=Palette(cfg.get("palette", Palette.BLUES)),
                    task_type="classification",
                )
                progress.advance(model_task)
                del (
                    gs,
                    best,
                    proba,
                    proba_tr,
                    yhat,
                    yhat_tr,
                    y_true_all,
                    y_pred_all,
                    y_proba_all,
                )
                gc.collect()
                continue
            if cv_method == "spxy":
                outer_splits = spxy_split(X.values, y_enc, outer.get_n_splits(X, y_enc))
            elif cv_method == "group":
                if groups is None:
                    raise DataError(
                        "group_col deve ser informado para cv_method 'group'"
                    )
                outer_splits = list(outer.split(X, y_enc, groups))
            else:
                outer_splits = list(outer.split(X, y_enc))
            fold_task = progress.add_task("Fold", total=len(outer_splits))
            proba = None
            proba_tr = None
            yhat = None
            yhat_tr = None
            for tr, te in outer_splits:
                if log_memory:
                    mem_fold_start = psutil.Process().memory_info().rss / 1e6
                    logger.debug("fold start %.3f MB", mem_fold_start)
                Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
                ytr, yte = y_enc[tr], y_enc[te]
                fit_kwargs = {}
                if cv_method == "spxy":
                    inner = spxy_split(Xtr, ytr, inner_folds)
                elif cv_method == "group":
                    inner = GroupKFold(n_splits=inner_folds)
                    assert groups is not None
                    fit_kwargs["groups"] = groups.iloc[tr]
                else:
                    inner = StratifiedKFold(
                        n_splits=inner_folds,
                        shuffle=True,
                        random_state=random_state,
                    )
                gs = GridSearchCV(
                    pipe, grid, scoring="f1_macro", cv=inner, n_jobs=n_jobs
                )
                gs.fit(Xtr, ytr, **fit_kwargs)
                if best_params_example is None:
                    best_params_example = dict(gs.best_params_)
                n_comp = _get_n_components(gs.best_params_)
                if n_comp is not None:
                    n_comp_list.append(n_comp)
                best = gs.best_estimator_
                if best_steps_repr is None:
                    best_steps_repr = _format_preprocess_steps(
                        _get_preprocess_steps_from_pipeline(best)
                    )
                calib_cfg = m_cfg.get("calibration")
                if calib_cfg:
                    best = CalibratedClassifierCV(
                        estimator=best,
                        method=calib_cfg.get("method", "sigmoid"),
                        cv=calib_cfg.get("cv", 5),
                    )
                    best.fit(Xtr, ytr)

                # validation metrics
                yhat = best.predict(Xte)
                scores_val["F1-macro"].append(f1_score(yte, yhat, average="macro"))
                scores_val["BalancedAcc"].append(balanced_accuracy_score(yte, yhat))
                scores_val["Accuracy"].append(accuracy_score(yte, yhat))
                proba_fn = getattr(best, "predict_proba", None)
                if proba_fn is not None:
                    proba = proba_fn(Xte)
                    y_proba_all.append(proba)
                    brier = np.mean(
                        [
                            brier_score_loss((yte == k).astype(int), proba[:, k])
                            for k in range(len(classes))
                        ]
                    )
                    scores_val["Brier"].append(float(brier))
                    scores_val["ECE"].append(
                        expected_calibration_error(yte, proba, n_bins=15)
                    )

                # training metrics
                yhat_tr = best.predict(Xtr)
                scores_train["F1-macro"].append(f1_score(ytr, yhat_tr, average="macro"))
                scores_train["BalancedAcc"].append(
                    balanced_accuracy_score(ytr, yhat_tr)
                )
                scores_train["Accuracy"].append(accuracy_score(ytr, yhat_tr))
                if proba_fn is not None:
                    proba_tr = proba_fn(Xtr)
                    brier_tr = np.mean(
                        [
                            brier_score_loss((ytr == k).astype(int), proba_tr[:, k])
                            for k in range(len(classes))
                        ]
                    )
                    scores_train["Brier"].append(float(brier_tr))
                    scores_train["ECE"].append(
                        expected_calibration_error(ytr, proba_tr, n_bins=15)
                    )
                y_true_all.append(yte)
                y_pred_all.append(yhat)
                progress.advance(fold_task)
                if log_memory:
                    mem_fold_end = psutil.Process().memory_info().rss / 1e6
                    logger.debug("fold end %.3f MB", mem_fold_end)
            model_label, method_label = _format_model_and_method_labels(
                name, pipe, best_params_example
            )
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": bool(m_cfg.get("calibration")),
                "method": method_label,
            }
            row.update(
                {f"{k}_mean": float(np.mean(v)) for k, v in scores_val.items() if v}
            )
            row.update(
                {
                    f"{k}_std": float(np.std(v, ddof=1))
                    for k, v in scores_val.items()
                    if v and len(v) > 1
                }
            )
            row.update(
                {
                    f"train_{k}_mean": float(np.mean(v))
                    for k, v in scores_train.items()
                    if v
                }
            )
            row.update(
                {
                    f"train_{k}_std": float(np.std(v, ddof=1))
                    for k, v in scores_train.items()
                    if v and len(v) > 1
                }
            )
            for m in ["Brier", "ECE"]:
                row.setdefault(f"{m}_mean", None)
                row.setdefault(f"{m}_std", None)
                row.setdefault(f"train_{m}_mean", None)
                row.setdefault(f"train_{m}_std", None)
            _update_row_with_best_params(row, best_params_example)
            if m_cfg.get("overfit_check") and yscramble_iter > 0:
                cv_scramble = outer
                if cv_method == "spxy":
                    cv_scramble = StratifiedKFold(
                        n_splits=inner_folds,
                        shuffle=True,
                        random_state=random_state,
                    )
                elif cv_method == "group":
                    cv_scramble = GroupKFold(n_splits=inner_folds)
                scramble_scores = y_scramble_scores(
                    best,
                    X.values,
                    y_enc,
                    cv=cv_scramble,
                    scoring="balanced_accuracy",
                    n_iter=yscramble_iter,
                    random_state=random_state,
                    groups=groups.values if groups is not None else None,
                )
                if scramble_scores.size:
                    row["Yscramble_mean"] = float(np.mean(scramble_scores))
                    row["Yscramble_p95"] = float(np.percentile(scramble_scores, 95))
                else:
                    row["Yscramble_mean"] = None
                    row["Yscramble_p95"] = None
            if n_comp_list:
                row["n_components"] = int(np.median(n_comp_list))
            row["preprocess_steps"] = best_steps_repr or "None"
            if step_label and row["preprocess_steps"] == "None":
                row["preprocess_steps"] = step_label
            if row.get("preprocess_steps") == "RAW":
                row["result"] = "RAW"
            else:
                row.setdefault("result", None)
            pls_artifacts = _save_pls_diagnostics(
                template_pipe=pipe,
                best_params=best_params_example,
                X=X,
                y=y_enc,
                classes=classes,
                file_path=ds["file"],
                target=ds["target"],
                model=model_label,
                preprocess_steps=row.get("preprocess_steps"),
                out_dir=out_dir,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
            )
            if pls_artifacts:
                row.update(pls_artifacts)
            if log_memory:
                mem_end = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory end %.3f MB", mem_end)
                if mem_start is not None:
                    row["memory_mb_start"] = float(mem_start)
                row["memory_mb_end"] = float(mem_end)
            rows.append(row)
            _save_metrics(
                [row],
                out_dir,
                append=True,
            )
            y_true = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)
            y_proba = np.vstack(y_proba_all) if y_proba_all else None
            _save_plots(
                ds["file"],
                ds["target"],
                model_label,
                classes,
                out_dir,
                y_true,
                y_pred,
                y_proba,
                preprocess_label=row.get("preprocess_steps"),
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
                task_type="classification",
            )
            progress.advance(model_task)
            del (
                gs,
                best,
                proba,
                proba_tr,
                yhat,
                yhat_tr,
                y_true_all,
                y_pred_all,
                y_proba_all,
            )
            gc.collect()
        progress.advance(tgt_task)
    _save_metrics(rows, out_dir)
    if cfg.get("astm"):
        flags = check_astm_classification(
            summary, {"external_validation": isinstance(outer, LeaveOneOut)}
        )
        with open(out_dir / "astm_classification.json", "w") as f:
            json.dump(flags, f, indent=2)
    logger.info("Cross-validation finished")


def _run_cv_regression(
    cfg: Dict[str, Any],
    out_dir: Path,
    outer: Any,
    progress: Progress,
    *,
    random_state: int,
    interactive: bool = False,
    n_jobs: int = 1,
    log_memory: bool = False,
    step_label: Optional[str] = None,
) -> None:
    """Run cross-validation for regression tasks."""

    n_jobs = max(1, n_jobs)
    logger.info("Starting cross-validation (regression)")
    try:
        data = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))
        _maybe_run_qc(cfg, data, out_dir)
    except DataError as e:
        logger.error("Data preparation failed: %s", e)
        raise DataError(
            "Arquivo de dados não encontrado. Verifique o caminho informado."
        ) from e

    rows: List[Dict[str, Any]] = []
    tgt_task = progress.add_task("Target", total=len(data))
    for ds in data:
        progress.update(tgt_task, description=f"Target: {ds['target']}")
        X: pd.DataFrame = ds["X"]
        y_series: pd.Series = ds["Y"]
        y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise DataError("Valores alvo inválidos ou não numéricos detectados")
        groups = ds.get("groups")
        if groups is not None and not isinstance(groups, pd.Series):
            groups = pd.Series(groups, index=X.index)
        cv_cfg = cfg.get("cv", {})
        inner_folds_cfg = cv_cfg.get("inner_folds", 3)
        if inner_folds_cfg in (None, False):
            inner_folds_cfg = 3
        inner_folds = int(inner_folds_cfg)
        inner_folds = max(inner_folds, 2)
        cv_method = cv_cfg.get("cv_method", "kfold").lower()
        model_cfgs = cfg.get("models", [])
        pipelines = build_pipeline_from_config(cfg)
        model_task = progress.add_task("Model", total=len(pipelines))
        mem_start = None
        if log_memory:
            mem_start = psutil.Process().memory_info().rss / 1e6
            logger.debug("memory start %.3f MB", mem_start)

        for idx, (name, pipe, grid) in enumerate(pipelines):
            if not model_cfgs:
                continue
            _ = model_cfgs[idx % len(model_cfgs)]
            logger.info("Running %s", name)
            scores_val: Dict[str, List[float]] = {
                "RMSE": [],
                "MAE": [],
                "MAE_percent": [],
                "R2": [],
            }
            scores_train: Dict[str, List[float]] = {
                "RMSE": [],
                "MAE": [],
                "MAE_percent": [],
                "R2": [],
            }
            y_true_val_all: List[np.ndarray] = []
            y_pred_val_all: List[np.ndarray] = []
            y_true_train_all: List[np.ndarray] = []
            y_pred_train_all: List[np.ndarray] = []
            best_steps_repr: Optional[str] = None
            best_params_example: Optional[Dict[str, Any]] = None
            n_comp_list: List[int] = []

            if groups is not None:
                outer_splits = list(outer.split(X, y, groups))
            else:
                outer_splits = list(outer.split(X, y))
            fold_task = progress.add_task("Fold", total=len(outer_splits))
            for tr, te in outer_splits:
                Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
                ytr, yte = y[tr], y[te]
                fit_kwargs: Dict[str, Any] = {}
                if cv_method == "group" and groups is not None:
                    train_groups = groups.iloc[tr]
                    allowed = len(pd.Series(train_groups).dropna().unique())
                    _validate_split_capacity(allowed, kind="groups")
                    n_splits = min(inner_folds, allowed)
                    inner = GroupKFold(n_splits=n_splits)
                    fit_kwargs["groups"] = train_groups
                elif cv_method == "spxy":
                    allowed = len(Xtr)
                    _validate_split_capacity(allowed, kind="samples")
                    n_splits = min(inner_folds, allowed)
                    inner = spxy_split(Xtr, ytr, n_splits)
                else:
                    allowed = len(Xtr)
                    _validate_split_capacity(allowed, kind="samples")
                    n_splits = min(inner_folds, allowed)
                    inner = KFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=random_state,
                    )
                gs = GridSearchCV(
                    pipe,
                    grid,
                    scoring="neg_root_mean_squared_error",
                    cv=inner,
                    n_jobs=n_jobs,
                )
                gs.fit(Xtr, ytr, **fit_kwargs)
                if best_params_example is None:
                    best_params_example = dict(gs.best_params_)
                n_comp = _get_n_components(gs.best_params_)
                if n_comp is not None:
                    n_comp_list.append(n_comp)
                best = gs.best_estimator_
                if best_steps_repr is None:
                    best_steps_repr = _format_preprocess_steps(
                        _get_preprocess_steps_from_pipeline(best)
                    )

                yhat = best.predict(Xte)
                yhat_tr = best.predict(Xtr)
                y_true_val_all.append(np.asarray(yte, dtype=float))
                y_pred_val_all.append(np.asarray(yhat, dtype=float))
                y_true_train_all.append(np.asarray(ytr, dtype=float))
                y_pred_train_all.append(np.asarray(yhat_tr, dtype=float))
                scores_val["RMSE"].append(_rmse(yte, yhat))
                scores_val["MAE"].append(float(mean_absolute_error(yte, yhat)))
                scores_val["MAE_percent"].append(compute_mae_percent(yte, yhat))
                scores_val["R2"].append(float(r2_score(yte, yhat)))
                scores_train["RMSE"].append(_rmse(ytr, yhat_tr))
                scores_train["MAE"].append(float(mean_absolute_error(ytr, yhat_tr)))
                scores_train["MAE_percent"].append(compute_mae_percent(ytr, yhat_tr))
                scores_train["R2"].append(float(r2_score(ytr, yhat_tr)))
                progress.advance(fold_task)
            progress.remove_task(fold_task)

            model_label, _ = _format_model_and_method_labels(name, pipe, best_params_example)
            row: Dict[str, Any] = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": False,
            }
            for metric in ["RMSE", "MAE", "MAE_percent", "R2"]:
                values = np.asarray(scores_val[metric], dtype=float)
                if values.size:
                    valid_mask = np.isfinite(values)
                    if np.any(valid_mask):
                        mean_val = float(np.mean(values[valid_mask]))
                        row[f"{metric}_mean"] = mean_val
                        if np.sum(valid_mask) > 1:
                            row[f"{metric}_std"] = float(
                                np.std(values[valid_mask], ddof=1)
                            )
                train_values = np.asarray(scores_train[metric], dtype=float)
                if train_values.size:
                    valid_train = np.isfinite(train_values)
                    if np.any(valid_train):
                        train_mean = float(np.mean(train_values[valid_train]))
                        row[f"train_{metric}_mean"] = train_mean
                        if np.sum(valid_train) > 1:
                            row[f"train_{metric}_std"] = float(
                                np.std(train_values[valid_train], ddof=1)
                            )
            if "RMSE_std" in row:
                row["RMSECV_std"] = row["RMSE_std"]
            if "train_RMSE_std" in row:
                row["train_RMSECV_std"] = row["train_RMSE_std"]

            if y_true_val_all and y_pred_val_all:
                y_true_val = np.concatenate(y_true_val_all)
                y_pred_val = np.concatenate(y_pred_val_all)
                rmsecv = compute_rmsecv(y_true_val, y_pred_val)
                row["RMSECV_mean"] = float(rmsecv) if np.isfinite(rmsecv) else None
                rmsep = compute_rmsep(y_true_val, y_pred_val)
                if np.isfinite(rmsep):
                    row["RMSEP"] = float(rmsep)
                rpd = compute_rpd(y_true_val, y_pred_val)
                if np.isfinite(rpd):
                    row["RPD"] = float(rpd)
            if y_true_train_all and y_pred_train_all:
                y_true_train = np.concatenate(y_true_train_all)
                y_pred_train = np.concatenate(y_pred_train_all)
                train_rmsecv = compute_rmsecv(y_true_train, y_pred_train)
                row["train_RMSECV_mean"] = (
                    float(train_rmsecv) if np.isfinite(train_rmsecv) else None
                )
                train_rmsep = compute_rmsep(y_true_train, y_pred_train)
                if np.isfinite(train_rmsep):
                    row["train_RMSEP"] = float(train_rmsep)
                train_rpd = compute_rpd(y_true_train, y_pred_train)
                if np.isfinite(train_rpd):
                    row["train_RPD"] = float(train_rpd)
            if n_comp_list:
                row["n_components"] = int(np.median(n_comp_list))
            row["preprocess_steps"] = best_steps_repr or "None"
            if step_label and row["preprocess_steps"] == "None":
                row["preprocess_steps"] = step_label
            _update_row_with_best_params(row, best_params_example)
            pls_artifacts = _save_pls_diagnostics(
                template_pipe=pipe,
                best_params=best_params_example,
                X=X,
                y=y,
                classes=None,
                task_type="regression",
                file_path=ds["file"],
                target=ds["target"],
                model=model_label,
                preprocess_steps=row.get("preprocess_steps"),
                out_dir=out_dir,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
            )
            if pls_artifacts:
                row.update(pls_artifacts)
            if log_memory:
                mem_end = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory end %.3f MB", mem_end)
                if mem_start is not None:
                    row["memory_mb_start"] = float(mem_start)
                row["memory_mb_end"] = float(mem_end)
            rows.append(row)
            _save_metrics([row], out_dir, append=True)
            progress.advance(model_task)
        progress.remove_task(model_task)
        progress.advance(tgt_task)
    _save_metrics(rows, out_dir)
    logger.info("Cross-validation finished (regression)")


def _run_cv(
    cfg: Dict[str, Any],
    out_dir: Path,
    outer: Any,
    progress: Progress,
    *,
    random_state: int,
    interactive: bool = False,
    n_jobs: int = 1,
    log_memory: bool = False,
    step_label: Optional[str] = None,
) -> None:
    """Dispatch cross-validation to the appropriate routine."""

    task_type = _infer_task_type(cfg)
    if task_type == "regression":
        _run_cv_regression(
            cfg,
            out_dir,
            outer,
            progress,
            random_state=random_state,
            interactive=interactive,
            n_jobs=n_jobs,
            log_memory=log_memory,
            step_label=step_label,
        )
    else:
        _run_cv_classification(
            cfg,
            out_dir,
            outer,
            progress,
            random_state=random_state,
            interactive=interactive,
            n_jobs=n_jobs,
            log_memory=log_memory,
            step_label=step_label,
        )


def _run_holdout_classification(
    cfg: Dict[str, Any],
    out_dir: Path,
    test_size: float,
    random_state: int,
    *,
    interactive: bool = False,
    n_jobs: int = 1,
) -> None:
    """Evaluate classification models using a single train/test split.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    out_dir:
        Output folder for generated artefacts.
    test_size:
        Proportion of the dataset used for testing.
    random_state:
        Seed applied to the train/test split and inner cross-validation.
    interactive:
        If ``True``, also generate interactive Plotly HTML figures.
    n_jobs:
        Number of parallel jobs for ``GridSearchCV``.
    """

    n_jobs = max(1, n_jobs)
    try:
        data = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))
        _maybe_run_qc(cfg, data, out_dir)
    except DataError as e:
        raise DataError(
            "Arquivo de dados não encontrado. Verifique o caminho informado."
        ) from e
    summary = summarize_dataset(data)
    rows: List[Dict[str, Any]] = []
    for ds in data:
        X, y = ds["X"], ds["Y"]
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        cv_cfg = cfg.get("cv", {})
        cv_method = cv_cfg.get("cv_method", "stratified").lower()
        outer_folds = int(cv_cfg.get("outer_folds", cv_cfg.get("inner_folds", 3)))
        model_cfgs = cfg.get("models", [])
        pipelines = build_pipeline_from_config(cfg)
        for idx, (name, pipe, grid) in enumerate(pipelines):
            if not model_cfgs:
                continue
            m_cfg = model_cfgs[idx % len(model_cfgs)]
            Xtr, Xte, ytr, yte = train_test_split(
                X.values,
                y_enc,
                test_size=test_size,
                random_state=random_state,
                stratify=y_enc,
            )
            counts = np.bincount(ytr)
            n_splits = int(min(outer_folds, counts.min()))
            n_splits = max(n_splits, 2)
            if n_splits != outer_folds:
                logger.warning(
                    "Adjusted number of CV folds from %d to %d", outer_folds, n_splits
                )
            steps_desc = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(pipe)
            )
            logger.info("Running %s with steps: %s", name, steps_desc)
            yscramble_iter = int(m_cfg.get("y_scramble_n_iter", 20))
            if cv_method == "spxy":
                inner = spxy_split(Xtr, ytr, n_splits)
            else:
                inner = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state,
                )
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=n_jobs)
            gs.fit(Xtr, ytr)
            n_comp = _get_n_components(gs.best_params_)
            best = gs.best_estimator_
            preprocess_steps = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(best)
            )
            calib_cfg = m_cfg.get("calibration")
            if calib_cfg:
                best = CalibratedClassifierCV(
                    estimator=best,
                    method=calib_cfg.get("method", "sigmoid"),
                    cv=calib_cfg.get("cv", 5),
                )
                best.fit(Xtr, ytr)
            yhat = best.predict(Xte)
            yhat_tr = best.predict(Xtr)
            model_label, method_label = _format_model_and_method_labels(
                name, pipe, gs.best_params_
            )
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": bool(calib_cfg),
                "method": method_label,
                "F1-macro_mean": float(f1_score(yte, yhat, average="macro")),
                "BalancedAcc_mean": float(balanced_accuracy_score(yte, yhat)),
                "Accuracy_mean": float(accuracy_score(yte, yhat)),
                "train_F1-macro_mean": float(f1_score(ytr, yhat_tr, average="macro")),
                "train_BalancedAcc_mean": float(balanced_accuracy_score(ytr, yhat_tr)),
                "train_Accuracy_mean": float(accuracy_score(ytr, yhat_tr)),
            }
            diff = row["train_BalancedAcc_mean"] - row["BalancedAcc_mean"]
            row["overfit_warning"] = diff > 0.05
            row["n_components"] = n_comp
            _update_row_with_best_params(row, gs.best_params_)
            proba_fn = getattr(best, "predict_proba", None)
            proba = None
            proba_tr = None
            if proba_fn is not None:
                proba = proba_fn(Xte)
                brier = np.mean(
                    [
                        brier_score_loss((yte == k).astype(int), proba[:, k])
                        for k in range(len(classes))
                    ]
                )
                row["Brier_mean"] = float(brier)
                row["ECE_mean"] = float(
                    expected_calibration_error(yte, proba, n_bins=15)
                )
                proba_tr = proba_fn(Xtr)
                brier_tr = np.mean(
                    [
                        brier_score_loss((ytr == k).astype(int), proba_tr[:, k])
                        for k in range(len(classes))
                    ]
                )
                row["train_Brier_mean"] = float(brier_tr)
                row["train_ECE_mean"] = float(
                    expected_calibration_error(ytr, proba_tr, n_bins=15)
                )
            for m in ["Brier", "ECE"]:
                row.setdefault(f"{m}_mean", None)
                row.setdefault(f"train_{m}_mean", None)
            if m_cfg.get("overfit_check") and yscramble_iter > 0:
                cv_scramble = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state,
                )
                scramble_scores = y_scramble_scores(
                    best,
                    X.values,
                    y_enc,
                    cv=cv_scramble,
                    scoring="balanced_accuracy",
                    n_iter=yscramble_iter,
                    random_state=random_state,
                )
                if scramble_scores.size:
                    row["Yscramble_mean"] = float(np.mean(scramble_scores))
                    row["Yscramble_p95"] = float(np.percentile(scramble_scores, 95))
                else:
                    row["Yscramble_mean"] = None
                    row["Yscramble_p95"] = None
            row["preprocess_steps"] = preprocess_steps
            if row.get("preprocess_steps") == "RAW":
                row["result"] = "RAW"
            else:
                row.setdefault("result", None)
            pls_artifacts = _save_pls_diagnostics(
                template_pipe=pipe,
                best_params=gs.best_params_,
                X=X,
                y=y_enc,
                classes=classes,
                task_type="classification",
                file_path=ds["file"],
                target=ds["target"],
                model=model_label,
                preprocess_steps=row.get("preprocess_steps"),
                out_dir=out_dir,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
            )
            if pls_artifacts:
                row.update(pls_artifacts)
            rows.append(row)
            _save_metrics(
                [row],
                out_dir,
                append=True,
            )
            _save_plots(
                ds["file"],
                ds["target"],
                model_label,
                classes,
                out_dir,
                yte,
                yhat,
                proba,
                preprocess_label=row.get("preprocess_steps"),
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
                task_type="classification",
            )
            del best, proba, proba_tr, yhat, yhat_tr
            gc.collect()
    _save_metrics(rows, out_dir)
    if cfg.get("astm"):
        flags = check_astm_classification(summary, {"external_validation": True})
        with open(out_dir / "astm_classification.json", "w") as f:
            json.dump(flags, f, indent=2)


def _run_holdout_regression(
    cfg: Dict[str, Any],
    out_dir: Path,
    test_size: float,
    random_state: int,
    *,
    interactive: bool = False,
    n_jobs: int = 1,
) -> None:
    """Evaluate regression models using a single train/test split."""

    n_jobs = max(1, n_jobs)
    try:
        data = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))
        _maybe_run_qc(cfg, data, out_dir)
    except DataError as e:
        raise DataError(
            "Arquivo de dados não encontrado. Verifique o caminho informado."
        ) from e
    rows: List[Dict[str, Any]] = []
    for ds in data:
        X, y_series = ds["X"], ds["Y"]
        y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            raise DataError("Valores alvo inválidos ou não numéricos detectados")
        cv_cfg = cfg.get("cv", {})
        cv_method = cv_cfg.get("cv_method", "kfold").lower()
        if cv_method == "group":
            raise ModelError("cv_method 'group' não suportado com estratégia holdout")
        model_cfgs = cfg.get("models", [])
        pipelines = build_pipeline_from_config(cfg)
        Xtr, Xte, ytr, yte = train_test_split(
            X.values,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        rows_ds: List[Dict[str, Any]] = []
        for idx, (name, pipe, grid) in enumerate(pipelines):
            if not model_cfgs:
                continue
            logger.info("Running %s", name)
            n_splits_req = int(cv_cfg.get("inner_folds", 3) or 3)
            n_splits_req = max(2, n_splits_req)
            allowed = len(Xtr)
            _validate_split_capacity(allowed, kind="samples")
            n_splits = min(n_splits_req, allowed)
            if cv_method == "spxy":
                inner = spxy_split(Xtr, ytr, n_splits)
            else:
                inner = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state,
                )
            gs = GridSearchCV(
                pipe,
                grid,
                scoring="neg_root_mean_squared_error",
                cv=inner,
                n_jobs=n_jobs,
            )
            gs.fit(Xtr, ytr)
            best = gs.best_estimator_
            yhat = best.predict(Xte)
            yhat_tr = best.predict(Xtr)
            rmse = _rmse(yte, yhat)
            mae = mean_absolute_error(yte, yhat)
            r2 = r2_score(yte, yhat)
            mae_pct = compute_mae_percent(yte, yhat)
            rmsep = compute_rmsep(yte, yhat)
            rpd = compute_rpd(yte, yhat)
            rmse_tr = _rmse(ytr, yhat_tr)
            mae_tr = mean_absolute_error(ytr, yhat_tr)
            r2_tr = r2_score(ytr, yhat_tr)
            mae_tr_pct = compute_mae_percent(ytr, yhat_tr)
            rmsep_tr = compute_rmsep(ytr, yhat_tr)
            rpd_tr = compute_rpd(ytr, yhat_tr)
            model_label, _ = _format_model_and_method_labels(name, pipe, gs.best_params_)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": False,
                "RMSE_mean": float(rmse),
                "RMSECV_mean": None,
                "MAE_mean": float(mae),
                "MAE_percent_mean": float(mae_pct) if np.isfinite(mae_pct) else None,
                "R2_mean": float(r2),
                "RMSEP": float(rmsep) if np.isfinite(rmsep) else None,
                "RPD": float(rpd) if np.isfinite(rpd) else None,
                "train_RMSE_mean": float(rmse_tr),
                "train_MAE_mean": float(mae_tr),
                "train_MAE_percent_mean": float(mae_tr_pct)
                if np.isfinite(mae_tr_pct)
                else None,
                "train_R2_mean": float(r2_tr),
                "train_RMSEP": float(rmsep_tr) if np.isfinite(rmsep_tr) else None,
                "train_RPD": float(rpd_tr) if np.isfinite(rpd_tr) else None,
            }
            n_comp = _get_n_components(gs.best_params_)
            if n_comp is not None:
                row["n_components"] = int(n_comp)
            preprocess_steps = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(best)
            )
            row["preprocess_steps"] = preprocess_steps
            _update_row_with_best_params(row, gs.best_params_)
            pls_artifacts = _save_pls_diagnostics(
                template_pipe=pipe,
                best_params=gs.best_params_,
                X=X,
                y=y,
                classes=None,
                task_type="regression",
                file_path=ds["file"],
                target=ds["target"],
                model=model_label,
                preprocess_steps=preprocess_steps,
                out_dir=out_dir,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
            )
            if pls_artifacts:
                row.update(pls_artifacts)
            rows.append(row)
            rows_ds.append(row)
            _save_metrics([row], out_dir, append=True)
        del rows_ds
    _save_metrics(rows, out_dir)


def _run_holdout(
    cfg: Dict[str, Any],
    out_dir: Path,
    test_size: float,
    random_state: int,
    *,
    interactive: bool = False,
    n_jobs: int = 1,
) -> None:
    """Dispatch holdout evaluation based on task type."""

    task_type = _infer_task_type(cfg)
    if task_type == "regression":
        _run_holdout_regression(
            cfg,
            out_dir,
            test_size,
            random_state,
            interactive=interactive,
            n_jobs=n_jobs,
        )
    else:
        _run_holdout_classification(
            cfg,
            out_dir,
            test_size,
            random_state,
            interactive=interactive,
            n_jobs=n_jobs,
        )


def run_multiple_eval(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    random_state: int = 42,
    interactive: bool = False,
    n_jobs: int = 1,
    mlflow_uri: Optional[str] = None,
    log_memory: bool = False,
) -> List[Tuple[Path, Path]]:
    """Run evaluation for multiple cross-validation strategies.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    out_dir:
        Directory that receives a subfolder for each strategy.
    interactive:
        If ``True``, also generate interactive Plotly HTML figures.
    n_jobs:
        Number of parallel jobs for ``GridSearchCV``.
    Returns
    -------
    list of tuple(Path, Path)
        Pairs with the paths to ``metrics.json`` and the ``figures`` directory
        for each evaluated strategy.
    """

    n_jobs = max(1, n_jobs)
    _validate_cfg(cfg, need_data=True, need_cv=True, need_models=True)
    task_type = _infer_task_type(cfg)
    pre_cfg = cfg.get("preprocess", {})
    steps_cfg = pre_cfg.get("steps", [])
    evaluate_flag = pre_cfg.get("evaluate_individual")
    if evaluate_flag is None:
        evaluate_individual = len(steps_cfg) > 1
    else:
        evaluate_individual = bool(evaluate_flag)

    step_variants: List[Tuple[Optional[str], Dict[str, Any]]] = []
    if evaluate_individual and len(steps_cfg) > 1:
        for step in steps_cfg:
            step_cfg = deepcopy(cfg)
            step_cfg.setdefault("preprocess", {})["steps"] = [step]
            step_cfg["preprocess"]["evaluate_individual"] = False
            step_label = _format_preprocess_step(step) or str(step)
            if not step_label:
                step_label = str(step)
            step_variants.append((step_label, step_cfg))
    else:
        step_variants.append((None, cfg))

    strategies = cfg.get("cv", {}).get("strategies", [])
    random_state = cfg.get("cv", {}).get("random_state", random_state)
    cv_method = cfg.get("cv", {}).get("cv_method", "stratified").lower()
    if task_type == "regression" and cv_method == "stratified":
        cv_method = "kfold"
    mlflow_enabled = bool(mlflow_uri)
    mlflow_cm = nullcontext()
    if mlflow_enabled:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow_cm = mlflow.start_run(run_name="crossval")
    model_names = [m.get("name", "") for m in cfg.get("models", [])]
    steps_cfg = cfg.get("preprocess", {}).get("steps", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    email = cfg.get("notification_email")
    if email:
        enviar_email(
            subject="Validação iniciada",
            body="O processo de validação foi iniciado.",
            to_addr=email,
        )

    with mlflow_cm:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "models": ",".join(model_names),
                    "preprocess_steps": _format_preprocess_steps(list(steps_cfg)),
                    "cv_method": cv_method,
                }
            )
        save_run_metadata(cfg, out_dir)
        files = [Path(p) for p in glob(cfg["data"]["input_path"])]
        input_hashes = {f.as_posix(): file_hash(f) for f in files}
        metadata = {
            "sys_version": sys.version,
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
            },
            "seed": random_state,
            "input_hashes": input_hashes,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        outputs: List[Tuple[Path, Path]] = []
        seen_outputs: Set[Tuple[Path, Path]] = set()
        logger.info("Running evaluation for %d CV strategies", len(strategies))
        if len(step_variants) > 1:
            logger.info(
                "Evaluating %d preprocessing variants individually", len(step_variants)
            )

        with Progress(console=Console(stderr=True)) as progress:
            total_iterations = max(1, len(strategies) * len(step_variants))
            strat_task = progress.add_task("CV strategy", total=total_iterations)
            for step_label, cfg_variant in step_variants:
                variant_pre_cfg = cfg_variant.get("preprocess", {})
                raw_steps = variant_pre_cfg.get("steps", []) or []
                if isinstance(raw_steps, (list, tuple)):
                    variant_steps = list(raw_steps)
                else:
                    variant_steps = [raw_steps]
                variant_steps_repr = _format_preprocess_steps(variant_steps)
                variant_step_label = step_label
                if variant_steps_repr == "None":
                    variant_step_label = variant_step_label or "RAW"
                step_desc = step_label or variant_steps_repr or "None"
                safe_step = None
                if step_label:
                    safe_step = re.sub(r"[^\w.-]+", "_", step_label.lower()).strip("_")
                    if not safe_step:
                        fallback = variant_steps_repr or step_label
                        safe_step = (
                            re.sub(r"[^\w.-]+", "_", fallback.lower()).strip("_")
                            or "step"
                        )
                for strat in strategies:
                    stype = strat.get("type", "").lower()
                    desc = f"Strategy: {stype}"
                    if len(step_variants) > 1:
                        desc = f"{desc} ({step_desc})"
                    logger.info(
                        "Evaluating strategy %s%s",
                        stype,
                        f" with preprocess {step_desc}" if len(step_variants) > 1 else "",
                    )
                    progress.update(strat_task, description=desc)
                    if stype == "kfold":
                        n_repeats = strat.get("n_repeats", 1)
                        for k in strat.get("n_splits", [5]):
                            if cv_method == "group":
                                if n_repeats > 1:
                                    raise ModelError(
                                        "cv_method 'group' não suporta repetições"
                                    )
                                outer = GroupKFold(n_splits=int(k))
                            else:
                                if task_type == "regression":
                                    if n_repeats > 1:
                                        outer = RepeatedKFold(
                                            n_splits=int(k),
                                            n_repeats=int(n_repeats),
                                            random_state=random_state,
                                        )
                                    else:
                                        outer = KFold(
                                            n_splits=int(k),
                                            shuffle=True,
                                            random_state=random_state,
                                        )
                                else:
                                    if n_repeats > 1:
                                        outer = RepeatedStratifiedKFold(
                                            n_splits=int(k),
                                            n_repeats=int(n_repeats),
                                            random_state=random_state,
                                        )
                                    else:
                                        outer = StratifiedKFold(
                                            n_splits=int(k),
                                            shuffle=True,
                                            random_state=random_state,
                                        )
                            if safe_step:
                                sub = out_dir / f"kfold_{k}__{safe_step}"
                            else:
                                sub = out_dir / f"kfold_{k}"
                            sub.mkdir(parents=True, exist_ok=True)
                            _run_cv(
                                cfg_variant,
                                sub,
                                outer,
                                progress,
                                random_state=random_state,
                                interactive=interactive,
                                n_jobs=n_jobs,
                                log_memory=log_memory,
                                step_label=variant_step_label,
                            )
                            metrics = sub / "metrics.json"
                            figs = sub / "figures"
                            if task_type == "classification":
                                with open(metrics) as f:
                                    metrics_data = json.load(f)
                                generate_scientific_report(metrics_data, cfg_variant, sub)
                            pair = (metrics, figs)
                            if pair not in seen_outputs:
                                outputs.append(pair)
                                seen_outputs.add(pair)
                            if mlflow_enabled:
                                params = {
                                    "strategy": "kfold",
                                    "n_splits": int(k),
                                    "n_repeats": int(n_repeats),
                                    "preprocess_steps": variant_steps_repr or "None",
                                }
                                run_name = sub.name
                                with mlflow.start_run(run_name=run_name, nested=True):
                                    mlflow.log_params(params)
                                    try:
                                        df = pd.read_json(metrics)
                                        if not df.empty:
                                            mlflow.log_metrics(
                                                {
                                                    k: float(v)
                                                    for k, v in df.iloc[0].items()
                                                    if isinstance(v, (int, float))
                                                }
                                            )
                                    except Exception:
                                        pass
                                    mlflow.log_artifact(metrics)
                                    mlflow.log_artifacts(
                                        figs, artifact_path="figures"
                                    )
                    elif stype == "loo":
                        if cv_method == "group":
                            raise ModelError(
                                "cv_method 'group' não suportado com estratégia LOO"
                            )
                        if safe_step:
                            sub = out_dir / f"loo__{safe_step}"
                        else:
                            sub = out_dir / "loo"
                        sub.mkdir(parents=True, exist_ok=True)
                        _run_cv(
                            cfg_variant,
                            sub,
                            LeaveOneOut(),
                            progress,
                            random_state=random_state,
                            interactive=interactive,
                            n_jobs=n_jobs,
                            log_memory=log_memory,
                            step_label=variant_step_label,
                        )
                        metrics = sub / "metrics.json"
                        figs = sub / "figures"
                        if task_type == "classification":
                            with open(metrics) as f:
                                metrics_data = json.load(f)
                            generate_scientific_report(metrics_data, cfg_variant, sub)
                        pair = (metrics, figs)
                        if pair not in seen_outputs:
                            outputs.append(pair)
                            seen_outputs.add(pair)
                        if mlflow_enabled:
                            params = {
                                "strategy": "loo",
                                "preprocess_steps": variant_steps_repr or "None",
                            }
                            run_name = sub.name
                            with mlflow.start_run(run_name=run_name, nested=True):
                                mlflow.log_params(params)
                                try:
                                    df = pd.read_json(metrics)
                                    if not df.empty:
                                        mlflow.log_metrics(
                                            {
                                                k: float(v)
                                                for k, v in df.iloc[0].items()
                                                if isinstance(v, (int, float))
                                            }
                                        )
                                except Exception:
                                    pass
                                mlflow.log_artifact(metrics)
                                mlflow.log_artifacts(
                                    figs, artifact_path="figures"
                                )
                    elif stype == "holdout":
                        if cv_method == "group":
                            raise ModelError(
                                "cv_method 'group' não suportado com estratégia holdout"
                            )
                        if safe_step:
                            sub = out_dir / f"holdout__{safe_step}"
                        else:
                            sub = out_dir / "holdout"
                        sub.mkdir(parents=True, exist_ok=True)
                        test_size = float(strat.get("test_size", 0.3))
                        _run_holdout(
                            cfg_variant,
                            sub,
                            test_size,
                            random_state,
                            interactive=interactive,
                            n_jobs=n_jobs,
                        )
                        metrics = sub / "metrics.json"
                        figs = sub / "figures"
                        if task_type == "classification":
                            with open(metrics) as f:
                                metrics_data = json.load(f)
                            generate_scientific_report(metrics_data, cfg_variant, sub)
                        pair = (metrics, figs)
                        if pair not in seen_outputs:
                            outputs.append(pair)
                            seen_outputs.add(pair)
                        if mlflow_enabled:
                            params = {
                                "strategy": "holdout",
                                "test_size": float(test_size),
                                "preprocess_steps": variant_steps_repr or "None",
                            }
                            run_name = sub.name
                            with mlflow.start_run(run_name=run_name, nested=True):
                                mlflow.log_params(params)
                                try:
                                    df = pd.read_json(metrics)
                                    if not df.empty:
                                        mlflow.log_metrics(
                                            {
                                                k: float(v)
                                                for k, v in df.iloc[0].items()
                                                if isinstance(v, (int, float))
                                            }
                                        )
                                except Exception:
                                    pass
                                mlflow.log_artifact(metrics)
                                mlflow.log_artifacts(
                                    figs, artifact_path="figures"
                                )
                    else:
                        raise ModelError(
                            f"Estratégia de validação desconhecida: {stype}. Use 'kfold', 'loo' ou 'holdout'."
                        )
                    progress.advance(strat_task)

    email = cfg.get("notification_email")
    if email:
        enviar_email(
            subject="Validação concluída",
            body="A validação foi concluída.",
            to_addr=email,
        )
    return outputs


def _train_and_save_classification(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    random_state: int = 42,
    interactive: bool = False,
    n_jobs: int = 1,
    mlflow_uri: Optional[str] = None,
    gs_verbose: int = 0,
    show_progress: bool = True,
    log_memory: bool = False,
) -> Tuple[Path, Path]:
    """Train final model and persist artefacts for classification."""

    n_jobs = max(1, n_jobs)
    logger.info("Starting training pipeline")
    email = cfg.get("notification_email")
    try:
        mlflow_cm = nullcontext()
        if mlflow_uri:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow_cm = mlflow.start_run(run_name="train")
        with mlflow_cm:
            with Progress(
                console=Console(stderr=True), disable=not show_progress
            ) as progress:
                load_task = progress.add_task("Carregando dados", total=None)
                _validate_cfg(cfg, need_data=True, need_models=True)
                ds_list = prepare_inputs(
                    cfg["data"], use_cache=cfg.get("use_cache", True)
                )
                _maybe_run_qc(cfg, ds_list, out_dir)
                ds = ds_list[0]
            summary = summarize_dataset(ds_list)
            X, y = ds["X"], ds["Y"]
            groups = ds.get("groups")
            le = LabelEncoder().fit(y)
            y_enc = le.transform(y)
            # persist class labels for later use in predictions
            out_dir.mkdir(parents=True, exist_ok=True)
            save_run_metadata(cfg, out_dir)
            with open(out_dir / "classes.json", "w") as f:
                json.dump(le.classes_.tolist(), f)
            progress.update(load_task, completed=1)

            # persist environment metadata
            input_hashes = {}
            if Path(ds["file"]).exists():
                input_hashes[ds["file"]] = file_hash(ds["file"])
            metadata = {
                "sys_version": sys.version,
                "versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "sklearn": sklearn.__version__,
                },
                "seed": random_state,
                "input_hashes": input_hashes,
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            name, pipe, grid = build_pipeline_from_config(cfg)[0]
            model_cfg = cfg.get("models", [])[0] if cfg.get("models") else {}
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
            mem_start = None
            if log_memory:
                mem_start = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory start %.3f MB", mem_start)
            cv_method = cfg.get("cv", {}).get("cv_method", "stratified").lower()
            train_task = progress.add_task("Treinando", total=None)
            fit_kwargs = {}
            if cv_method == "spxy":
                cv = spxy_split(X.values, y_enc, 5)
            elif cv_method == "group":
                if groups is None:
                    raise DataError(
                        "group_col deve ser informado para cv_method 'group'"
                    )
                cv = GroupKFold(n_splits=5)
                fit_kwargs["groups"] = groups
            else:
                cv = RepeatedStratifiedKFold(
                    n_splits=5, n_repeats=1, random_state=random_state
                )
            gs = LoggingGridSearchCV(
                pipe,
                grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=n_jobs,
                verbose=gs_verbose,
            )
            start_ts = datetime.now().isoformat()
            logger.info("Grid search start: %s", start_ts)
            gs.fit(X.values, y_enc, **fit_kwargs)
            logger.info("Grid search end: %s", datetime.now().isoformat())
            progress.update(train_task, completed=1)

            n_comp = _get_n_components(gs.best_params_)
            best = gs.best_estimator_
            preprocess_steps = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(best)
            )
            if preprocess_steps == "None":
                preprocess_steps = "RAW"
            model_label, method_label = _format_model_and_method_labels(
                name, pipe, gs.best_params_
            )
            calib_cfg = model_cfg.get("calibration") if model_cfg else None
            if calib_cfg:
                best = CalibratedClassifierCV(
                    estimator=best,
                    method=calib_cfg.get("method", "sigmoid"),
                    cv=calib_cfg.get("cv", 5),
                )
                best.fit(X.values, y_enc)

            save_task = progress.add_task("Salvando", total=None)
            dump(best, out_dir / "model.joblib")
            export_xy_xlsx(out_dir / "prepared_XY.xlsx", X, y)
            (out_dir / "model_card.md").write_text(
                f"# Model Card\nBest model: {model_label}\nBest params: {gs.best_params_}\n",
            )

            yhat = best.predict(X.values)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": bool(calib_cfg),
                "method": method_label,
                "F1-macro_mean": float(f1_score(y_enc, yhat, average="macro")),
                "BalancedAcc_mean": float(balanced_accuracy_score(y_enc, yhat)),
                "Accuracy_mean": float(accuracy_score(y_enc, yhat)),
            }
            row["n_components"] = n_comp
            row["preprocess_steps"] = preprocess_steps
            if row.get("preprocess_steps") == "RAW":
                row["result"] = "RAW"
            else:
                row.setdefault("result", None)
            _update_row_with_best_params(row, gs.best_params_)
            proba_fn = getattr(best, "predict_proba", None)
            proba = None
            if proba_fn is not None:
                proba = proba_fn(X.values)
                brier = np.mean(
                    [
                        brier_score_loss((y_enc == k).astype(int), proba[:, k])
                        for k in range(len(le.classes_))
                    ]
                )
                row["Brier_mean"] = float(brier)
                row["ECE_mean"] = float(
                    expected_calibration_error(y_enc, proba, n_bins=15)
                )
            pls_artifacts = _save_pls_diagnostics(
                template_pipe=pipe,
                best_params=gs.best_params_,
                X=X,
                y=y_enc,
                classes=le.classes_,
                task_type="classification",
                file_path=ds["file"],
                target=ds["target"],
                model=model_label,
                preprocess_steps=row.get("preprocess_steps"),
                out_dir=out_dir,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
            )
            if pls_artifacts:
                row.update(pls_artifacts)
            if log_memory:
                mem_end = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory end %.3f MB", mem_end)
                if mem_start is not None:
                    row["memory_mb_start"] = float(mem_start)
                row["memory_mb_end"] = float(mem_end)
            _save_metrics([row], out_dir)
            if cfg.get("astm"):
                flags = check_astm_classification(
                    summary, {"external_validation": False}
                )
                with open(out_dir / "astm_classification.json", "w") as f:
                    json.dump(flags, f, indent=2)
            _save_plots(
                ds["file"],
                ds["target"],
                model_label,
                le.classes_,
                out_dir,
                y_enc,
                yhat,
                proba,
                preprocess_label=row.get("preprocess_steps"),
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
                task_type="classification",
            )
            del best, proba, yhat
            gc.collect()
            with open(out_dir / "metrics.json") as f:
                metrics_data = json.load(f)
            generate_scientific_report(metrics_data, cfg, out_dir)
            if mlflow_uri:
                try:
                    from dvc.api import DVCFileSystem

                    fs = DVCFileSystem()
                    info = fs.info(ds["file"])
                    mlflow.log_param("dvc_data_md5", info.get("md5"))
                except Exception:  # pragma: no cover - best effort
                    logger.debug("Could not log DVC metadata", exc_info=True)
                params = {
                    "model": model_label,
                    "preprocess_steps": _format_preprocess_steps(list(steps_cfg)),
                    "cv_method": cv_method,
                }
                params.update(
                    {
                        f"best_{k}": _coerce_param_value(v)
                        for k, v in gs.best_params_.items()
                    }
                )
                mlflow.log_params(params)
                mlflow.log_metrics(row)
                mlflow.log_artifact(out_dir / "model.joblib")
                mlflow.log_artifact(out_dir / "metrics.json")
                mlflow.log_artifacts(out_dir / "figures", artifact_path="figures")
            progress.update(save_task, completed=1)
    except Exception:
        logger.error("Training failed", exc_info=True)
        raise
    if email:
        enviar_email(
            subject="Treinamento concluído",
            body="Treinamento concluído. Iniciando validação...",
            to_addr=email,
        )

    return out_dir / "metrics.json", out_dir / "figures"


def _train_and_save_regression(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    random_state: int = 42,
    interactive: bool = False,
    n_jobs: int = 1,
    mlflow_uri: Optional[str] = None,
    gs_verbose: int = 0,
    show_progress: bool = True,
    log_memory: bool = False,
) -> Tuple[Path, Path]:
    """Train final model and persist artefacts for regression."""

    n_jobs = max(1, n_jobs)
    logger.info("Starting training pipeline (regression)")
    email = cfg.get("notification_email")
    try:
        mlflow_cm = nullcontext()
        if mlflow_uri:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow_cm = mlflow.start_run(run_name="train")
        with mlflow_cm:
            with Progress(
                console=Console(stderr=True), disable=not show_progress
            ) as progress:
                load_task = progress.add_task("Carregando dados", total=None)
                _validate_cfg(cfg, need_data=True, need_models=True)
                ds_list = prepare_inputs(
                    cfg["data"], use_cache=cfg.get("use_cache", True)
                )
                _maybe_run_qc(cfg, ds_list, out_dir)
                ds = ds_list[0]
            X, y_series = ds["X"], ds["Y"]
            y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
            if np.isnan(y).any():
                raise DataError("Valores alvo inválidos ou não numéricos detectados")
            groups = ds.get("groups")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_run_metadata(cfg, out_dir)

            input_hashes = {}
            if Path(ds["file"]).exists():
                input_hashes[ds["file"]] = file_hash(ds["file"])
            metadata = {
                "sys_version": sys.version,
                "versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "sklearn": sklearn.__version__,
                },
                "seed": random_state,
                "input_hashes": input_hashes,
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            name, pipe, grid = build_pipeline_from_config(cfg)[0]
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
            mem_start = None
            if log_memory:
                mem_start = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory start %.3f MB", mem_start)
            cv_cfg = cfg.get("cv", {})
            cv_method_raw = cv_cfg.get("cv_method", "kfold").lower()
            cv_method = "kfold" if cv_method_raw == "stratified" else cv_method_raw
            max_splits = int(cv_cfg.get("train_n_splits", 5) or 5)
            max_splits = max(2, max_splits)
            train_task = progress.add_task("Treinando", total=None)
            fit_kwargs: Dict[str, Any] = {}
            if cv_method == "spxy":
                allowed = len(X)
                _validate_split_capacity(allowed, kind="samples")
                n_splits = min(max_splits, allowed)
                cv = spxy_split(X.values, y, n_splits)
            elif cv_method == "group":
                if groups is None:
                    raise DataError(
                        "group_col deve ser informado para cv_method 'group'"
                    )
                group_series = pd.Series(groups)
                allowed = len(group_series.dropna().unique())
                _validate_split_capacity(allowed, kind="groups")
                n_splits = min(max_splits, allowed)
                cv = GroupKFold(n_splits=n_splits)
                fit_kwargs["groups"] = groups
            else:
                allowed = len(X)
                _validate_split_capacity(allowed, kind="samples")
                n_splits = min(max_splits, allowed)
                cv = KFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state,
                )
            gs = LoggingGridSearchCV(
                pipe,
                grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=n_jobs,
                verbose=gs_verbose,
            )
            start_ts = datetime.now().isoformat()
            logger.info("Grid search start: %s", start_ts)
            gs.fit(X.values, y, **fit_kwargs)
            logger.info("Grid search end: %s", datetime.now().isoformat())
            progress.update(train_task, completed=1)

            n_comp = _get_n_components(gs.best_params_)
            best = gs.best_estimator_
            preprocess_steps = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(best)
            )
            if preprocess_steps == "None":
                preprocess_steps = "RAW"
            model_label, _ = _format_model_and_method_labels(
                name, pipe, gs.best_params_
            )

            save_task = progress.add_task("Salvando", total=None)
            dump(best, out_dir / "model.joblib")
            export_xy_xlsx(out_dir / "prepared_XY.xlsx", X, y_series)
            (out_dir / "model_card.md").write_text(
                "\n".join(
                    [
                        "# Model Card",
                        f"Best model: {model_label}",
                        f"Best params: {gs.best_params_}",
                    ]
                ),
                encoding="utf-8",
            )

            yhat = best.predict(X.values)
            mae_pct = compute_mae_percent(y, yhat)
            rmsep = compute_rmsep(y, yhat)
            rpd = compute_rpd(y, yhat)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": False,
                "RMSE_mean": _rmse(y, yhat),
                "RMSECV_mean": None,
                "MAE_mean": float(mean_absolute_error(y, yhat)),
                "MAE_percent_mean": float(mae_pct) if np.isfinite(mae_pct) else None,
                "R2_mean": float(r2_score(y, yhat)),
                "RMSEP": float(rmsep) if np.isfinite(rmsep) else None,
                "RPD": float(rpd) if np.isfinite(rpd) else None,
                "preprocess_steps": preprocess_steps,
                "train_RMSE_mean": _rmse(y, yhat),
                "train_MAE_mean": float(mean_absolute_error(y, yhat)),
                "train_MAE_percent_mean": float(mae_pct)
                if np.isfinite(mae_pct)
                else None,
                "train_R2_mean": float(r2_score(y, yhat)),
                "train_RMSEP": float(rmsep) if np.isfinite(rmsep) else None,
                "train_RPD": float(rpd) if np.isfinite(rpd) else None,
            }
            if n_comp is not None:
                row["n_components"] = int(n_comp)
            _update_row_with_best_params(row, gs.best_params_)
            pls_artifacts = _save_pls_diagnostics(
                template_pipe=pipe,
                best_params=gs.best_params_,
                X=X,
                y=y,
                classes=None,
                task_type="regression",
                file_path=ds["file"],
                target=ds["target"],
                model=model_label,
                preprocess_steps=preprocess_steps,
                out_dir=out_dir,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
            )
            if pls_artifacts:
                row.update(pls_artifacts)
            if log_memory:
                mem_end = psutil.Process().memory_info().rss / 1e6
                logger.debug("memory end %.3f MB", mem_end)
                if mem_start is not None:
                    row["memory_mb_start"] = float(mem_start)
                row["memory_mb_end"] = float(mem_end)
            _save_metrics([row], out_dir)
            _save_plots(
                ds["file"],
                ds["target"],
                model_label,
                None,
                out_dir,
                y,
                yhat,
                None,
                preprocess_label=preprocess_steps,
                interactive=interactive,
                palette=Palette(cfg.get("palette", Palette.BLUES)),
                task_type="regression",
            )
            if mlflow_uri:
                mlflow.log_params(
                    {
                        "model": model_label,
                        "preprocess_steps": _format_preprocess_steps(list(steps_cfg)),
                        "cv_method": cv_method,
                    }
                )
                mlflow.log_params(
                    {
                        f"best_{k}": _coerce_param_value(v)
                        for k, v in gs.best_params_.items()
                    }
                )
                mlflow.log_metrics(row)
                mlflow.log_artifact(out_dir / "model.joblib")
                mlflow.log_artifact(out_dir / "metrics.json")
                mlflow.log_artifacts(out_dir / "figures", artifact_path="figures")
            progress.update(save_task, completed=1)
    except Exception:
        logger.error("Training failed", exc_info=True)
        raise
    if email:
        enviar_email(
            subject="Treinamento concluído",
            body="Treinamento concluído. Iniciando validação...",
            to_addr=email,
        )

    return out_dir / "metrics.json", out_dir / "figures"


def train_and_save(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    random_state: int = 42,
    interactive: bool = False,
    n_jobs: int = 1,
    mlflow_uri: Optional[str] = None,
    gs_verbose: int = 0,
    show_progress: bool = True,
    log_memory: bool = False,
) -> Tuple[Path, Path]:
    """Dispatch training to classification or regression pipeline."""

    task_type = _infer_task_type(cfg)
    if task_type == "regression":
        return _train_and_save_regression(
            cfg,
            out_dir,
            random_state=random_state,
            interactive=interactive,
            n_jobs=n_jobs,
            mlflow_uri=mlflow_uri,
            gs_verbose=gs_verbose,
            show_progress=show_progress,
            log_memory=log_memory,
        )
    return _train_and_save_classification(
        cfg,
        out_dir,
        random_state=random_state,
        interactive=interactive,
        n_jobs=n_jobs,
        mlflow_uri=mlflow_uri,
        gs_verbose=gs_verbose,
        show_progress=show_progress,
        log_memory=log_memory,
    )


def run_experiment(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    interactive: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
    log_memory: bool = False,
) -> Tuple[Tuple[Path, Path], List[Tuple[Path, Path]]]:
    """Execute um experimento completo usando a API de alto nível.

    A função prepara os dados de entrada, treina o melhor modelo em todo o
    conjunto e executa as estratégias de validação cruzada definidas.

    Parameters
    ----------
    cfg:
        Configuração completa do experimento.
    out_dir:
        Diretório base para os resultados.
    interactive:
        Se ``True``, gera também gráficos interativos.
    random_state:
        Semente aleatória utilizada nas rotinas de treino e validação.
    n_jobs:
        Número de trabalhos paralelos para ``GridSearchCV``.

    Returns
    -------
    tuple
        ``(train_outputs, cv_outputs)`` conforme retornado por
        :func:`train_and_save` e :func:`run_multiple_eval`.
    """

    email = cfg.get("notification_email")
    try:
        _validate_cfg(cfg, need_data=True, need_models=True, need_cv=True)
        prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))

        train_dir = out_dir / "train"
        train_outputs = train_and_save(
            cfg,
            train_dir,
            random_state=random_state,
            interactive=interactive,
            n_jobs=n_jobs,
            log_memory=log_memory,
        )

        cv_dir = out_dir / "cv"
        cv_outputs = run_multiple_eval(
            cfg,
            cv_dir,
            random_state=random_state,
            interactive=interactive,
            n_jobs=n_jobs,
            log_memory=log_memory,
        )

        return train_outputs, cv_outputs
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if email:
            enviar_email("Erro no processamento", f"{exc}\n\n{tb}", to_addr=email)
        raise


def evaluate_saved_model(model_path: Path, data_path: Path) -> None:
    try:
        est = load(model_path)
    except FileNotFoundError as e:
        raise ModelError("Modelo não encontrado. Verifique o caminho informado.") from e
    try:
        X, y_list, _ = read_xlsx_xy(data_path)
    except DataError as e:
        raise DataError(f"Erro ao ler os dados: {e}") from e
    y = y_list[0][1]
    yhat = est.predict(X.values)
    print(pd.crosstab(pd.Series(y, name="true"), pd.Series(yhat, name="pred")))


def predict_with_model(model_path: Path, data_path: Path, out_dir: Path) -> None:
    try:
        est = load(model_path)
    except FileNotFoundError as e:
        raise ModelError("Modelo não encontrado. Verifique o caminho informado.") from e
    with Progress(console=Console(stderr=True)) as progress:
        load_data_task = progress.add_task("Carregando dados", total=None)
        y = None
        try:
            if data_path.suffix.lower() in [".xlsx", ".xls"]:
                try:
                    X, y_list, _ = read_xlsx_xy(data_path)
                except DataError:
                    X, y_list, _ = read_tabular(data_path)
            else:
                X, y_list, _ = read_tabular(data_path)
        except DataError as e:
            raise DataError(f"Erro ao processar dados: {e}") from e
        if y_list:
            y = y_list[0][1]
        progress.update(load_data_task, completed=1)

        pred_task = progress.add_task("Prevendo", total=None)
        classes = None
        classes_path = model_path.with_name("classes.json")
        if classes_path.exists():
            with open(classes_path) as f:
                classes = np.array(json.load(f))
        yhat = est.predict(X.values)
        if classes is not None:
            yhat = classes[yhat]
        progress.update(pred_task, completed=1)

        save_task = progress.add_task("Salvando", total=None)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"y_pred": yhat})
        if y is not None:
            df.insert(0, "y_true", y)
        df.to_csv(out_dir / "predictions.csv", index=False)
        progress.update(save_task, completed=1)


def explain_model(model_path: Path, data_path: Path, out_dir: Path) -> None:
    """Generate SHAP explanations for a saved model and dataset.

    Parameters
    ----------
    model_path:
        Path to a previously saved model in ``joblib`` format.
    data_path:
        Dataset in tabular or Excel format. Only the feature matrix ``X`` is
        required; target columns will be ignored if present.
    out_dir:
        Directory where the generated figures will be persisted.
    """

    try:
        import shap  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "shap is required for model explanation. Install with 'pip install nircls[shap]'"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and feature matrix
    est = load(model_path)
    try:
        if data_path.suffix.lower() in [".xlsx", ".xls"]:
            try:
                X, _, _ = read_xlsx_xy(data_path)
            except DataError:
                X, _, _ = read_tabular(data_path)
        else:
            X, _, _ = read_tabular(data_path)
    except DataError as e:
        raise DataError(f"Erro ao processar dados: {e}") from e

    # Build explainer according to model and compute SHAP values
    explainer = shap.Explainer(est, X)
    shap_values = explainer(X)

    import matplotlib.pyplot as plt

    # Summary plot (PNG + interactive HTML)
    shap.plots.summary(shap_values, show=False)
    shap.save_html(out_dir / "shap_summary.html", shap_values)
    plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight")
    plt.close()

    values = shap_values.values
    if values.ndim == 3:  # multi-class -> use first class
        values = values[:, :, 0]
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = int(mean_abs.argmax())
    feature_name = X.columns[top_idx]

    # Dependence plot for the most important feature
    shap.plots.scatter(shap_values[:, top_idx], color=shap_values, show=False)
    shap.save_html(
        out_dir / f"shap_dependence_{feature_name}.html", shap_values[:, top_idx]
    )
    plt.savefig(out_dir / f"shap_dependence_{feature_name}.png", bbox_inches="tight")
    plt.close()

    # Waterfall plot for first sample
    shap.plots.waterfall(shap_values[0], show=False)
    shap.save_html(out_dir / "shap_waterfall.html", shap_values[0])
    plt.savefig(out_dir / "shap_waterfall.png", bbox_inches="tight")
    plt.close()
