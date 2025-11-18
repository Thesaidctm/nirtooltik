"""Wrappers for cross-validation and holdout execution routines."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
from sqlalchemy.orm import Session
from rich.progress import Progress
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

from .data_loading import prepare_inputs
from .evaluation import _maybe_run_qc, _save_metrics, _save_plots, _save_pls_diagnostics
from .hyperparameter_search import _get_n_components, _sanitize_n_jobs
from .metrics_utils import _rmse, _summarize_scores
from .model_factory import (
    _call_build_pipeline,
    _format_model_and_method_labels,
    _format_preprocess_steps,
    _get_preprocess_steps_from_pipeline,
    _infer_task_type,
    _update_row_with_best_params,
)
from .splitting import _validate_split_capacity
from .io import summarize_dataset
from ...core.services import runs as run_services
from ...db import crud
from ...db.models import Dataset, Run
from ...utils.logging import RunLoggerAdapter
from ..utils.compliance import check_astm_classification
from ..results.metrics.classification import expected_calibration_error
from ..results.metrics.regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmsep,
    rmse_with_ci,
)
from ..utils.errors import DataError, ModelError
from ..utils.palettes import Palette
from ..utils.split import spxy_split
from ..utils.y_scramble import y_scramble_scores

logger = logging.getLogger(__name__)


class _RunContextFilter(logging.Filter):
    """Attach run context information to all log records for observability."""

    def __init__(self, run_id: int | None = None, run_uid: str | None = None) -> None:
        super().__init__()
        self.run_id = run_id
        self.run_uid = run_uid

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging glue
        if self.run_id is not None:
            record.run_id = self.run_id
        if self.run_uid is not None:
            record.run_uid = self.run_uid
        return True


def _get_run_logger(run_id: int | None, run_uid: str | None) -> logging.LoggerAdapter:
    extra = {k: v for k, v in {"run_id": run_id, "run_uid": run_uid}.items() if v is not None}
    return RunLoggerAdapter(logger, extra)

__all__ = [
    "_run_cv",
    "_run_cv_classification",
    "_run_cv_regression",
    "_run_holdout",
    "_run_holdout_classification",
    "_run_holdout_regression",
]


def _relative_to_out_dir(path: Path, out_dir: Path) -> str:
    """Return a POSIX path relative to ``out_dir`` when possible."""

    try:
        return path.relative_to(out_dir).as_posix()
    except ValueError:
        return path.as_posix()


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

    n_jobs = _sanitize_n_jobs(n_jobs)
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
        models = _call_build_pipeline(
            cfg, random_state=random_state, n_jobs=n_jobs
        )
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
            fold_predictions: List[pd.DataFrame] = []
            proba_available = False
            n_comp_list: List[int] = []
            best_steps_repr: Optional[str] = None
            best_params_example: Optional[Dict[str, Any]] = None
            results_dir = out_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            def _export_predictions_csv() -> Optional[Path]:
                if not fold_predictions:
                    return None
                df_pred = pd.concat(fold_predictions, ignore_index=True)
                base_cols = ["sample", "fold", "y_true", "y_pred"]
                for col in base_cols:
                    if col not in df_pred.columns:
                        df_pred[col] = np.nan
                proba_cols: List[str] = []
                if proba_available:
                    proba_cols = [
                        f"y_proba_{str(cls)}" for cls in classes
                    ]
                    for col in proba_cols:
                        if col not in df_pred.columns:
                            df_pred[col] = np.nan
                ordered_cols = base_cols + proba_cols
                df_pred = df_pred[ordered_cols]
                csv_path = results_dir / "cv_predictions.csv"
                df_pred.to_csv(csv_path, index=False)
                return csv_path
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
                    calib_params = {
                        "estimator": best,
                        "method": calib_cfg.get("method", "sigmoid"),
                        "cv": calib_cfg.get("cv", 5),
                    }
                    if n_jobs is not None:
                        calib_params["n_jobs"] = n_jobs
                    best = CalibratedClassifierCV(**calib_params)
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
                proba_available = True
                fold_df = pd.DataFrame({"sample": X.index})
                fold_df["fold"] = np.arange(len(X.index))
                fold_df["y_true"] = le.inverse_transform(y_enc)
                fold_df["y_pred"] = le.inverse_transform(yhat)
                for class_idx, class_label in enumerate(classes):
                    col_name = f"y_proba_{str(class_label)}"
                    fold_df[col_name] = proba[:, class_idx]
                fold_predictions.append(fold_df)
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
                predictions_path = _export_predictions_csv()
                if predictions_path is not None:
                    row["cv_predictions_csv"] = _relative_to_out_dir(
                        predictions_path, out_dir
                    )
                row.update(_summarize_scores(scores_val))
                row.update(_summarize_scores(scores_train, prefix="train_"))
                for m in ["Brier", "ECE"]:
                    row.setdefault(f"{m}_mean", None)
                    row.setdefault(f"{m}_std", None)
                    row.setdefault(f"{m}_ci95_low", None)
                    row.setdefault(f"{m}_ci95_high", None)
                    row.setdefault(f"train_{m}_mean", None)
                    row.setdefault(f"train_{m}_std", None)
                    row.setdefault(f"train_{m}_ci95_low", None)
                    row.setdefault(f"train_{m}_ci95_high", None)
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
                    fold_predictions,
                )
                gc.collect()
                continue
            if cv_method == "spxy":
                outer_splits = spxy_split(
                    X.values,
                    y_enc,
                    outer.get_n_splits(X, y_enc),
                    random_state=random_state,
                )
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
            for fold_idx, (tr, te) in enumerate(outer_splits):
                if log_memory:
                    mem_fold_start = psutil.Process().memory_info().rss / 1e6
                    logger.debug("fold start %.3f MB", mem_fold_start)
                Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
                ytr, yte = y_enc[tr], y_enc[te]
                fit_kwargs = {}
                if cv_method == "spxy":
                    inner = spxy_split(Xtr, ytr, inner_folds, random_state=random_state)
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
                    calib_params = {
                        "estimator": best,
                        "method": calib_cfg.get("method", "sigmoid"),
                        "cv": calib_cfg.get("cv", 5),
                    }
                    if n_jobs is not None:
                        calib_params["n_jobs"] = n_jobs
                    best = CalibratedClassifierCV(**calib_params)
                    best.fit(Xtr, ytr)

                # validation metrics
                yhat = best.predict(Xte)
                scores_val["F1-macro"].append(f1_score(yte, yhat, average="macro"))
                scores_val["BalancedAcc"].append(balanced_accuracy_score(yte, yhat))
                scores_val["Accuracy"].append(accuracy_score(yte, yhat))
                proba_fn = getattr(best, "predict_proba", None)
                proba = None
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
                fold_df = pd.DataFrame({"sample": X.index[te]})
                fold_df["fold"] = fold_idx
                fold_df["y_true"] = le.inverse_transform(yte)
                fold_df["y_pred"] = le.inverse_transform(yhat)
                if proba is not None:
                    for class_idx, class_label in enumerate(classes):
                        col_name = f"y_proba_{str(class_label)}"
                        fold_df[col_name] = proba[:, class_idx]
                    proba_available = True
                fold_predictions.append(fold_df)
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
            predictions_path = _export_predictions_csv()
            if predictions_path is not None:
                row["cv_predictions_csv"] = _relative_to_out_dir(
                    predictions_path, out_dir
                )
            row.update(_summarize_scores(scores_val))
            row.update(_summarize_scores(scores_train, prefix="train_"))
            for m in ["Brier", "ECE"]:
                row.setdefault(f"{m}_mean", None)
                row.setdefault(f"{m}_std", None)
                row.setdefault(f"{m}_ci95_low", None)
                row.setdefault(f"{m}_ci95_high", None)
                row.setdefault(f"train_{m}_mean", None)
                row.setdefault(f"train_{m}_std", None)
                row.setdefault(f"train_{m}_ci95_low", None)
                row.setdefault(f"train_{m}_ci95_high", None)
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
                    observed_score = row.get("BalancedAcc_mean")
                    if observed_score is not None:
                        observed = float(observed_score)
                        extreme = np.count_nonzero(scramble_scores >= observed)
                        row["Yscramble_pvalue"] = float(
                            (extreme + 1) / (scramble_scores.size + 1)
                        )
                    else:
                        row["Yscramble_pvalue"] = None
                else:
                    row["Yscramble_mean"] = None
                    row["Yscramble_p95"] = None
                    row["Yscramble_pvalue"] = None
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
                fold_predictions,
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

    n_jobs = _sanitize_n_jobs(n_jobs)
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
        y_series = ds["Y"]
        if isinstance(y_series, pd.Series):
            y_series = y_series.reindex(X.index)
        else:
            y_series = pd.Series(y_series, index=X.index)
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
        pipelines = _call_build_pipeline(
            cfg, random_state=random_state, n_jobs=n_jobs
        )
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
            fold_predictions: List[pd.DataFrame] = []
            best_steps_repr: Optional[str] = None
            best_params_example: Optional[Dict[str, Any]] = None
            n_comp_list: List[int] = []
            results_dir = out_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            def _export_residuals_csv() -> Optional[Path]:
                if not fold_predictions:
                    return None
                df_pred = pd.concat(fold_predictions, ignore_index=True)
                ordered_cols = ["sample", "fold", "y_true", "y_pred", "residual"]
                for col in ordered_cols:
                    if col not in df_pred.columns:
                        df_pred[col] = np.nan
                df_pred = df_pred[ordered_cols]
                csv_path = results_dir / "cv_residuals.csv"
                df_pred.to_csv(csv_path, index=False)
                return csv_path

            if groups is not None:
                outer_splits = list(outer.split(X, y, groups))
            else:
                outer_splits = list(outer.split(X, y))
            fold_task = progress.add_task("Fold", total=len(outer_splits))
            for fold_idx, (tr, te) in enumerate(outer_splits):
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
                    inner = spxy_split(
                        Xtr,
                        ytr,
                        n_splits,
                        random_state=random_state,
                    )
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
                y_true_fold = np.asarray(yte, dtype=float)
                y_pred_fold = np.asarray(yhat, dtype=float)
                fold_df = pd.DataFrame({"sample": X.index[te]})
                fold_df["fold"] = fold_idx
                fold_df["y_true"] = y_true_fold
                fold_df["y_pred"] = y_pred_fold
                fold_df["residual"] = y_true_fold - y_pred_fold
                fold_predictions.append(fold_df)
                progress.advance(fold_task)
            progress.remove_task(fold_task)

            model_label, _ = _format_model_and_method_labels(name, pipe, best_params_example)
            row: Dict[str, Any] = {
                "file": ds["file"],
                "target": ds["target"],
                "model": model_label,
                "calibrated": False,
            }
            residuals_path = _export_residuals_csv()
            if residuals_path is not None:
                row["cv_residuals_csv"] = _relative_to_out_dir(
                    residuals_path, out_dir
                )
            row.update(_summarize_scores(scores_val))
            row.update(_summarize_scores(scores_train, prefix="train_"))
            if "RMSE_std" in row:
                row["RMSECV_std"] = row["RMSE_std"]
            if "train_RMSE_std" in row:
                row["train_RMSECV_std"] = row["train_RMSE_std"]

            if y_true_val_all and y_pred_val_all:
                y_true_val = np.concatenate(y_true_val_all)
                y_pred_val = np.concatenate(y_pred_val_all)
                rmsecv = compute_rmsecv(y_true_val, y_pred_val)
                row["RMSECV_mean"] = float(rmsecv) if np.isfinite(rmsecv) else None
                _, rmsecv_low, rmsecv_high = rmse_with_ci(
                    y_true_val, y_pred_val, random_state=random_state
                )
                if np.isfinite(rmsecv_low):
                    row["RMSECV_ci95_low"] = float(rmsecv_low)
                if np.isfinite(rmsecv_high):
                    row["RMSECV_ci95_high"] = float(rmsecv_high)
                rmsep = compute_rmsep(y_true_val, y_pred_val)
                if np.isfinite(rmsep):
                    row["RMSEP"] = float(rmsep)
                _, rmsep_low, rmsep_high = rmse_with_ci(
                    y_true_val, y_pred_val, random_state=random_state
                )
                if np.isfinite(rmsep_low):
                    row["RMSEP_ci95_low"] = float(rmsep_low)
                if np.isfinite(rmsep_high):
                    row["RMSEP_ci95_high"] = float(rmsep_high)
                rpd = compute_rpd(y_true_val, y_pred_val)
                if np.isfinite(rpd):
                    row["RPD"] = float(rpd)
                existing_r2 = row.get("R2_mean")
                needs_update = existing_r2 is None
                if not needs_update:
                    try:
                        needs_update = not np.isfinite(float(existing_r2))
                    except (TypeError, ValueError):
                        needs_update = True
                if needs_update:
                    agg_r2 = r2_score(y_true_val, y_pred_val)
                    if np.isfinite(agg_r2):
                        row["R2_mean"] = float(agg_r2)
            if y_true_train_all and y_pred_train_all:
                y_true_train = np.concatenate(y_true_train_all)
                y_pred_train = np.concatenate(y_pred_train_all)
                train_rmsecv = compute_rmsecv(y_true_train, y_pred_train)
                row["train_RMSECV_mean"] = (
                    float(train_rmsecv) if np.isfinite(train_rmsecv) else None
                )
                _, train_low, train_high = rmse_with_ci(
                    y_true_train, y_pred_train, random_state=random_state
                )
                if np.isfinite(train_low):
                    row["train_RMSECV_ci95_low"] = float(train_low)
                if np.isfinite(train_high):
                    row["train_RMSECV_ci95_high"] = float(train_high)
                train_rmsep = compute_rmsep(y_true_train, y_pred_train)
                if np.isfinite(train_rmsep):
                    row["train_RMSEP"] = float(train_rmsep)
                _, train_rmsep_low, train_rmsep_high = rmse_with_ci(
                    y_true_train, y_pred_train, random_state=random_state
                )
                if np.isfinite(train_rmsep_low):
                    row["train_RMSEP_ci95_low"] = float(train_rmsep_low)
                if np.isfinite(train_rmsep_high):
                    row["train_RMSEP_ci95_high"] = float(train_rmsep_high)
                train_rpd = compute_rpd(y_true_train, y_pred_train)
                if np.isfinite(train_rpd):
                    row["train_RPD"] = float(train_rpd)
                existing_train_r2 = row.get("train_R2_mean")
                needs_train_update = existing_train_r2 is None
                if not needs_train_update:
                    try:
                        needs_train_update = not np.isfinite(float(existing_train_r2))
                    except (TypeError, ValueError):
                        needs_train_update = True
                if needs_train_update:
                    agg_train_r2 = r2_score(y_true_train, y_pred_train)
                    if np.isfinite(agg_train_r2):
                        row["train_R2_mean"] = float(agg_train_r2)
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

    n_jobs = _sanitize_n_jobs(n_jobs)
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
        pipelines = _call_build_pipeline(
            cfg, random_state=random_state, n_jobs=n_jobs
        )
        Xtr, Xte, ytr, yte = train_test_split(
            X.values,
            y_enc,
            test_size=test_size,
            random_state=random_state,
            stratify=y_enc,
        )
        class_counts = np.bincount(ytr)
        nonzero_counts = class_counts[class_counts > 0]
        if nonzero_counts.size == 0:
            raise DataError(
                "Número insuficiente de amostras para validação cruzada"
            )
        n_splits = int(min(outer_folds, nonzero_counts.min()))
        n_splits = max(n_splits, 2)
        if n_splits != outer_folds:
            logger.warning(
                "Adjusted number of CV folds from %d to %d", outer_folds, n_splits
            )
        for idx, (name, pipe, grid) in enumerate(pipelines):
            if not model_cfgs:
                continue
            m_cfg = model_cfgs[idx % len(model_cfgs)]
            steps_desc = _format_preprocess_steps(
                _get_preprocess_steps_from_pipeline(pipe)
            )
            logger.info("Running %s with steps: %s", name, steps_desc)
            yscramble_iter = int(m_cfg.get("y_scramble_n_iter", 20))
            if cv_method == "spxy":
                inner = spxy_split(
                    Xtr,
                    ytr,
                    n_splits,
                    random_state=random_state,
                )
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
                calib_params = {
                    "estimator": best,
                    "method": calib_cfg.get("method", "sigmoid"),
                    "cv": calib_cfg.get("cv", 5),
                }
                if n_jobs is not None:
                    calib_params["n_jobs"] = n_jobs
                best = CalibratedClassifierCV(**calib_params)
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
                    observed_score = row.get("BalancedAcc_mean")
                    if observed_score is not None:
                        observed = float(observed_score)
                        extreme = np.count_nonzero(scramble_scores >= observed)
                        row["Yscramble_pvalue"] = float(
                            (extreme + 1) / (scramble_scores.size + 1)
                        )
                    else:
                        row["Yscramble_pvalue"] = None
                else:
                    row["Yscramble_mean"] = None
                    row["Yscramble_p95"] = None
                    row["Yscramble_pvalue"] = None
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

    n_jobs = _sanitize_n_jobs(n_jobs)
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
        pipelines = _call_build_pipeline(
            cfg, random_state=random_state, n_jobs=n_jobs
        )
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
                inner = spxy_split(
                    Xtr,
                    ytr,
                    n_splits,
                    random_state=random_state,
                )
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
            _, rmse_low, rmse_high = rmse_with_ci(
                yte, yhat, random_state=random_state
            )
            rmsep = compute_rmsep(yte, yhat)
            _, rmsep_low, rmsep_high = rmse_with_ci(
                yte, yhat, random_state=random_state
            )
            rpd = compute_rpd(yte, yhat)
            rmse_tr = _rmse(ytr, yhat_tr)
            mae_tr = mean_absolute_error(ytr, yhat_tr)
            r2_tr = r2_score(ytr, yhat_tr)
            mae_tr_pct = compute_mae_percent(ytr, yhat_tr)
            rmsep_tr = compute_rmsep(ytr, yhat_tr)
            _, rmsep_tr_low, rmsep_tr_high = rmse_with_ci(
                ytr, yhat_tr, random_state=random_state
            )
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
            if np.isfinite(rmse_low):
                row["RMSE_ci95_low"] = float(rmse_low)
            if np.isfinite(rmse_high):
                row["RMSE_ci95_high"] = float(rmse_high)
            if np.isfinite(rmsep_low):
                row["RMSEP_ci95_low"] = float(rmsep_low)
            if np.isfinite(rmsep_high):
                row["RMSEP_ci95_high"] = float(rmsep_high)
            if np.isfinite(rmsep_tr_low):
                row["train_RMSEP_ci95_low"] = float(rmsep_tr_low)
            if np.isfinite(rmsep_tr_high):
                row["train_RMSEP_ci95_high"] = float(rmsep_tr_high)
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
    resume: bool = False,
) -> Tuple[List[Tuple[Path, Path]], Path]:
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
    resume:
        When ``True``, reuse existing ``metrics.json`` files and skip already
        completed evaluations.
    Returns
    -------
    tuple
        A pair where the first element contains tuples with the paths to
        ``metrics.json`` and the ``figures`` directory for each evaluated
        strategy, and the second element is the directory containing the
        aggregated summary files.
    """

    n_jobs = _sanitize_n_jobs(n_jobs)
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
    if resume:
        logger.info("Resuming evaluation outputs from %s", out_dir)

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
            "git_commit": get_git_commit(),
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
                            metrics = sub / "metrics.json"
                            figs = sub / "figures"
                            if (
                                resume
                                and metrics.exists()
                                and figs.exists()
                                and figs.is_dir()
                            ):
                                logger.info(
                                    "Skipping %s (results already present)", sub
                                )
                                pair = (metrics, figs)
                                if pair not in seen_outputs:
                                    outputs.append(pair)
                                    seen_outputs.add(pair)
                                progress.advance(strat_task)
                                continue
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
                        metrics = sub / "metrics.json"
                        figs = sub / "figures"
                        if (
                            resume
                            and metrics.exists()
                            and figs.exists()
                            and figs.is_dir()
                        ):
                            logger.info(
                                "Skipping %s (results already present)", sub
                            )
                            pair = (metrics, figs)
                            if pair not in seen_outputs:
                                outputs.append(pair)
                                seen_outputs.add(pair)
                            progress.advance(strat_task)
                            continue
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
                        metrics = sub / "metrics.json"
                        figs = sub / "figures"
                        if (
                            resume
                            and metrics.exists()
                            and figs.exists()
                            and figs.is_dir()
                        ):
                            logger.info(
                                "Skipping %s (results already present)", sub
                            )
                            pair = (metrics, figs)
                            if pair not in seen_outputs:
                                outputs.append(pair)
                                seen_outputs.add(pair)
                            progress.advance(strat_task)
                            continue
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

        summary_dir = out_dir.parent / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_df = summarize_runs(out_dir)
        summary_csv = summary_dir / "metrics_summary.csv"
        summary_xlsx = summary_dir / "metrics_summary.xlsx"
        summary_df.to_csv(summary_csv, index=False)
        with pd.ExcelWriter(summary_xlsx) as writer:
            summary_df.to_excel(writer, index=False)

    email = cfg.get("notification_email")
    if email:
        enviar_email(
            subject="Validação concluída",
            body="A validação foi concluída.",
            to_addr=email,
        )
    return outputs, summary_dir


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
    resume: bool = False,
) -> Tuple[Path, Path]:
    """Train final model and persist artefacts for classification."""

    metrics_path = out_dir / "metrics.json"
    figs_dir = out_dir / "figures"
    if (
        resume
        and metrics_path.exists()
        and figs_dir.exists()
        and figs_dir.is_dir()
    ):
        logger.info("Skipping training (classification) because artefacts already exist")
        return metrics_path, figs_dir

    n_jobs = _sanitize_n_jobs(n_jobs)
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
                "git_commit": get_git_commit(),
                "input_hashes": input_hashes,
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            name, pipe, grid = _call_build_pipeline(
                cfg, random_state=random_state, n_jobs=n_jobs
            )[0]
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
                cv = spxy_split(
                    X.values,
                    y_enc,
                    5,
                    random_state=random_state,
                )
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
                calib_params = {
                    "estimator": best,
                    "method": calib_cfg.get("method", "sigmoid"),
                    "cv": calib_cfg.get("cv", 5),
                }
                if n_jobs is not None:
                    calib_params["n_jobs"] = n_jobs
                best = CalibratedClassifierCV(**calib_params)
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
    resume: bool = False,
) -> Tuple[Path, Path]:
    """Train final model and persist artefacts for regression."""

    metrics_path = out_dir / "metrics.json"
    figs_dir = out_dir / "figures"
    if (
        resume
        and metrics_path.exists()
        and figs_dir.exists()
        and figs_dir.is_dir()
    ):
        logger.info("Skipping training (regression) because artefacts already exist")
        return metrics_path, figs_dir

    n_jobs = _sanitize_n_jobs(n_jobs)
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
                "git_commit": get_git_commit(),
                "input_hashes": input_hashes,
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            name, pipe, grid = _call_build_pipeline(
                cfg, random_state=random_state, n_jobs=n_jobs
            )[0]
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
                cv = spxy_split(
                    X.values,
                    y,
                    n_splits,
                    random_state=random_state,
                )
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
    resume: bool = False,
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
            resume=resume,
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
        resume=resume,
    )


def run_pipeline(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    db: Session,
    run_id: int | None = None,
    run_uid: str | None = None,
    user_id: int | None = None,
    dataset_id: int | None = None,
    project_id: int | None = None,
    interactive: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
    mlflow_uri: Optional[str] = None,
    gs_verbose: int = 0,
    show_progress: bool = True,
    log_memory: bool = False,
    resume: bool = False,
) -> Run:
    """Execute a full pipeline run and register progress in the database."""

    context_filter = _RunContextFilter(run_id=run_id, run_uid=run_uid)
    logger.addFilter(context_filter)

    try:
        resolved_project_id = project_id
        if resolved_project_id is None and dataset_id is not None:
            dataset = db.get(Dataset, dataset_id)
            if dataset:
                resolved_project_id = dataset.project_id

        run: Run | None = None
        if run_id is not None:
            run = db.get(Run, run_id)
            if run is None:
                raise ValueError(f"Run with id {run_id} not found")
            run_uid = run.run_uid or run_uid
        elif run_uid is not None:
            run = crud.get_run_by_uid(db, run_uid)
            if run is None:
                raise ValueError(f"Run with uid {run_uid} not found")
        if run is None:
            run = run_services.create_run(
                db,
                project_id=resolved_project_id,
                dataset_id=dataset_id,
                user_id=user_id,
                config=cfg,
                status="running",
                artifacts={"base_dir": str(out_dir)},
                run_uid=run_uid,
            )

        context_filter.run_id = run.run_id
        context_filter.run_uid = run.run_uid
        run_logger = _get_run_logger(run.run_id, run.run_uid)
        run_logger.info("Starting pipeline execution")

        train_dir = out_dir / "train"
        cv_dir = out_dir / "cv"
        train_dir.mkdir(parents=True, exist_ok=True)
        cv_dir.mkdir(parents=True, exist_ok=True)

        try:
            _validate_cfg(cfg, need_data=True, need_models=True, need_cv=True)
            prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))

            train_and_save(
                cfg,
                train_dir,
                random_state=random_state,
                interactive=interactive,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                gs_verbose=gs_verbose,
                show_progress=show_progress,
                log_memory=log_memory,
                resume=resume,
            )
            _outputs, summary_dir = run_multiple_eval(
                cfg,
                cv_dir,
                random_state=random_state,
                interactive=interactive,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                gs_verbose=gs_verbose,
                show_progress=show_progress,
                log_memory=log_memory,
                resume=resume,
            )
            model_path = train_dir / "model.joblib"
            reports = [train_dir / "model_card.md"]
            plots = [cv_dir / "figures"]
            metrics_manifest = {
                "summary_csv": str(summary_dir / "metrics_summary.csv"),
                "summary_xlsx": str(summary_dir / "metrics_summary.xlsx"),
            }
            extras = {
                "prepared_data": str(train_dir / "prepared_XY.xlsx"),
            }
            run_logger.info("Pipeline completed successfully")
            return run_services.mark_run_completed(
                db,
                run.run_id,
                base_dir=out_dir,
                train_dir=train_dir,
                cv_dir=cv_dir,
                summary_dir=summary_dir,
                models=[model_path],
                plots=plots,
                reports=reports,
                metrics=metrics_manifest,
                extras=extras,
            )
        except Exception as exc:
            artifacts = {
                "base_dir": str(out_dir),
                "directories": {
                    "train": str(train_dir),
                    "cross_validation": str(cv_dir),
                },
            }
            run_logger.exception("Pipeline failed: %s", exc)
            run_services.mark_run_failed(db, run.run_id, str(exc), artifacts=artifacts)
            raise
    finally:
        logger.removeFilter(context_filter)


def run_experiment(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    run_id: int | None = None,
    run_uid: str | None = None,
    interactive: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
    log_memory: bool = False,
) -> Tuple[Tuple[Path, Path], Tuple[List[Tuple[Path, Path]], Path]]:
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

    context_filter = _RunContextFilter(run_id=run_id, run_uid=run_uid)
    logger.addFilter(context_filter)
    run_logger = _get_run_logger(run_id, run_uid)
    email = cfg.get("notification_email")
    try:
        _validate_cfg(cfg, need_data=True, need_models=True, need_cv=True)
        prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))

        train_dir = out_dir / "train"
        run_logger.info("Training models for experiment")
        train_outputs = train_and_save(
            cfg,
            train_dir,
            random_state=random_state,
            interactive=interactive,
            n_jobs=n_jobs,
            log_memory=log_memory,
        )

        cv_dir = out_dir / "cv"
        run_logger.info("Running evaluation for experiment")
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
        run_logger.exception("Experiment failed: %s", exc)
        raise
    finally:
        logger.removeFilter(context_filter)


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
    logger.info(
        "\n%s",
        pd.crosstab(pd.Series(y, name="true"), pd.Series(yhat, name="pred")),
    )


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