from __future__ import annotations

import json
import subprocess
import sys
import warnings
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
import yaml  # type: ignore[import-untyped]
from joblib import dump, load
from rich.progress import Progress
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config_schema import ConfigModel, ValidationError
from .io import export_xy_xlsx
from .io import prepare_inputs as _prepare_inputs
from .io import read_tabular, read_xlsx_xy, summarize_dataset
from .metrics.calibration import expected_calibration_error
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
from .utils.errors import DataError, ModelError
from .utils.logging import log_versions, logger
from .utils.misc import RESULTS_DIR, file_hash
from .utils.plots import plot_confusion, plot_reliability, plot_roc_ovr
from .utils.qc import detect_identical_spectra, find_duplicates, pca_outliers
from .utils.split import spxy_split
from .utils.y_scramble import y_scramble_scores

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)


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
            )
        else:
            _, y_list, groups = read_tabular(f, tgt, grp)
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
        n = name.upper()
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


def build_pipeline_from_config(
    cfg: Dict[str, Any]
) -> List[Tuple[str, Pipeline, Dict[str, Any]]]:
    _validate_cfg(cfg, need_models=True)
    steps_cfg = cfg.get("preprocess", {}).get("steps", [])
    pre_steps, pre_grid = _make_preprocess(steps_cfg, cfg.get("preprocess", {}))
    models = make_models_from_config(cfg["models"])
    out: List[Tuple[str, Pipeline, Dict[str, Any]]] = []
    for m in models:
        est = m["estimator"]
        grid = pre_grid.copy()
        model_grid = m.get("grid", {}) or {}
        if not hasattr(est, "predict_proba"):
            calib_cfg = m.get("calibration") or {}
            method = calib_cfg.get("method", "sigmoid")
            cv = calib_cfg.get("cv", 5)
            est = CalibratedClassifierCV(estimator=est, method=method, cv=cv)
            grid.update({f"clf__estimator__{k}": v for k, v in model_grid.items()})
        else:
            grid.update({f"clf__{k}": v for k, v in model_grid.items()})
        pipe = Pipeline(pre_steps + [("clf", est)])
        out.append((m["name"], pipe, grid))
    return out


def _save_metrics(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """Persist metrics to Excel and JSON files."""

    df = pd.DataFrame(rows)

    # Validation metrics: drop training columns
    val_df = df.loc[:, ~df.columns.str.startswith("train_")]

    # Calibration metrics: keep identifier columns and rename training columns
    id_cols = ["file", "target", "model", "calibrated"]
    train_cols = [c for c in df.columns if c.startswith("train_")]
    calib_df = df[id_cols + train_cols].copy()
    calib_df.rename(
        columns={c: c.removeprefix("train_") for c in train_cols}, inplace=True
    )

    with pd.ExcelWriter(out_dir / "metrics_summary.xlsx") as writer:
        val_df.to_excel(writer, sheet_name="validation", index=False)
        calib_df.to_excel(writer, sheet_name="calibration", index=False)

    # Convert NaN values back to ``None`` for JSON export
    json_rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(json_rows, f, indent=2)


def _maybe_run_qc(
    cfg: Dict[str, Any], data: List[Dict[str, Any]], out_dir: Path
) -> None:
    """Run optional quality control checks and persist a report."""

    if not cfg.get("qc"):
        return

    report: List[Dict[str, Any]] = []
    for ds in data:
        X, y = ds["X"], ds["Y"]
        df = X.copy()
        df[ds["target"]] = y
        report.append(
            {
                "file": ds["file"],
                "target": ds["target"],
                "duplicates": find_duplicates(df).index.tolist(),
                "identical_spectra": detect_identical_spectra(X),
                "pca_outliers": pca_outliers(X).to_dict(orient="index"),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qc_report.json", "w") as f:
        json.dump(report, f, indent=2)


def _save_plots(
    file_path: str,
    target: str,
    model: str,
    classes: np.ndarray,
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    *,
    interactive: bool = False,
) -> None:
    """Save confusion matrix and optional ROC/reliability plots.

    When ``interactive`` is ``True`` additional HTML versions of the confusion
    matrix and ROC curve are generated using Plotly.
    """

    out_fig = out_dir / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).stem
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(
        cm,
        classes,
        out_fig / f"{stem}__{target}__{model}__confusion",
        interactive=interactive,
    )
    if y_proba is not None:
        plot_roc_ovr(
            y_true,
            y_proba,
            classes,
            out_fig / f"{stem}__{target}__{model}__roc",
            interactive=interactive,
        )
        plot_reliability(
            y_true,
            y_proba,
            out_fig / f"{stem}__{target}__{model}__reliability",
            interactive=interactive,
        )


def _get_n_components(best_params: Dict[str, Any]) -> Optional[int]:
    """Return the best ``n_components`` value from GridSearchCV parameters."""

    for k, v in best_params.items():
        if k.endswith("n_components"):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def _run_cv(
    cfg: Dict[str, Any],
    out_dir: Path,
    outer: Any,
    progress: Progress,
    *,
    random_state: int,
    interactive: bool = False,
) -> None:
    """Run cross-validation for configured datasets and models.

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
    """

    try:
        data = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))
        _maybe_run_qc(cfg, data, out_dir)
    except DataError as e:
        raise DataError(
            "Arquivo de dados não encontrado. Verifique o caminho informado."
        ) from e
    summary = summarize_dataset(data)
    rows: List[Dict[str, Any]] = []
    tgt_task = progress.add_task("Target", total=len(data))
    for ds in data:
        progress.update(tgt_task, description=f"Target: {ds['target']}")
        X, y = ds["X"], ds["Y"]
        groups = ds.get("groups")
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        cv_method = cfg.get("cv", {}).get("cv_method", "stratified").lower()
        inner_folds = cfg.get("cv", {}).get("inner_folds", 3)
        models = build_pipeline_from_config(cfg)
        model_cfgs = cfg.get("models", [])
        model_task = progress.add_task("Model", total=len(models))
        for m_cfg, (name, pipe, grid) in zip(model_cfgs, models):
            progress.update(model_task, description=f"Model: {name}")
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
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
            for tr, te in outer_splits:
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
                gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1)
                gs.fit(Xtr, ytr, **fit_kwargs)

                n_comp = _get_n_components(gs.best_params_)
                if n_comp is not None:
                    n_comp_list.append(n_comp)
                best = gs.best_estimator_
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
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
                "calibrated": bool(m_cfg.get("calibration")),
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
            if m_cfg.get("overfit_check"):
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
                    n_iter=20,
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
            rows.append(row)
            y_true = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)
            y_proba = np.vstack(y_proba_all) if y_proba_all else None
            _save_plots(
                ds["file"],
                ds["target"],
                name,
                classes,
                out_dir,
                y_true,
                y_pred,
                y_proba,
                interactive=interactive,
            )
            progress.advance(model_task)
        progress.advance(tgt_task)
    _save_metrics(rows, out_dir)
    if cfg.get("astm"):
        flags = check_astm_classification(
            summary, {"external_validation": isinstance(outer, LeaveOneOut)}
        )
        with open(out_dir / "astm_classification.json", "w") as f:
            json.dump(flags, f, indent=2)


def _run_holdout(
    cfg: Dict[str, Any],
    out_dir: Path,
    test_size: float,
    random_state: int,
    *,
    interactive: bool = False,
) -> None:
    """Evaluate models using a single train/test split.

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
    """

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
        cv_method = cfg.get("cv", {}).get("cv_method", "stratified").lower()
        inner_folds = cfg.get("cv", {}).get("inner_folds", 3)
        model_cfgs = cfg.get("models", [])
        for m_cfg, (name, pipe, grid) in zip(
            model_cfgs, build_pipeline_from_config(cfg)
        ):
            Xtr, Xte, ytr, yte = train_test_split(
                X.values,
                y_enc,
                test_size=test_size,
                random_state=random_state,
                stratify=y_enc,
            )
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
            if cv_method == "spxy":
                inner = spxy_split(Xtr, ytr, inner_folds)
            else:
                inner = StratifiedKFold(
                    n_splits=inner_folds,
                    shuffle=True,
                    random_state=random_state,
                )
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1)
            gs.fit(Xtr, ytr)
            n_comp = _get_n_components(gs.best_params_)
            best = gs.best_estimator_
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
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
                "calibrated": bool(calib_cfg),
                "F1-macro_mean": float(f1_score(yte, yhat, average="macro")),
                "BalancedAcc_mean": float(balanced_accuracy_score(yte, yhat)),
                "Accuracy_mean": float(accuracy_score(yte, yhat)),
                "train_F1-macro_mean": float(f1_score(ytr, yhat_tr, average="macro")),
                "train_BalancedAcc_mean": float(balanced_accuracy_score(ytr, yhat_tr)),
                "train_Accuracy_mean": float(accuracy_score(ytr, yhat_tr)),
            }
            row["n_components"] = n_comp
            proba_fn = getattr(best, "predict_proba", None)
            proba = None
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
            if m_cfg.get("overfit_check"):
                cv_scramble = StratifiedKFold(
                    n_splits=inner_folds,
                    shuffle=True,
                    random_state=random_state,
                )
                scramble_scores = y_scramble_scores(
                    best,
                    X.values,
                    y_enc,
                    cv=cv_scramble,
                    scoring="balanced_accuracy",
                    n_iter=20,
                    random_state=random_state,
                )
                if scramble_scores.size:
                    row["Yscramble_mean"] = float(np.mean(scramble_scores))
                    row["Yscramble_p95"] = float(np.percentile(scramble_scores, 95))
                else:
                    row["Yscramble_mean"] = None
                    row["Yscramble_p95"] = None
            rows.append(row)
            _save_plots(
                ds["file"],
                ds["target"],
                name,
                classes,
                out_dir,
                yte,
                yhat,
                proba,
                interactive=interactive,
            )
    _save_metrics(rows, out_dir)
    if cfg.get("astm"):
        flags = check_astm_classification(summary, {"external_validation": True})
        with open(out_dir / "astm_classification.json", "w") as f:
            json.dump(flags, f, indent=2)


def run_multiple_eval(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    random_state: int = 42,
    interactive: bool = False,
    mlflow_uri: Optional[str] = None,
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
    Returns
    -------
    list of tuple(Path, Path)
        Pairs with the paths to ``metrics.json`` and the ``figures`` directory
        for each evaluated strategy.
    """

    _validate_cfg(cfg, need_data=True, need_cv=True, need_models=True)
    strategies = cfg.get("cv", {}).get("strategies", [])
    random_state = cfg.get("cv", {}).get("random_state", random_state)
    cv_method = cfg.get("cv", {}).get("cv_method", "stratified").lower()
    mlflow_enabled = bool(mlflow_uri)
    mlflow_cm = nullcontext()
    if mlflow_enabled:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow_cm = mlflow.start_run(run_name="crossval")
    model_names = [m.get("name", "") for m in cfg.get("models", [])]
    steps_cfg = cfg.get("preprocess", {}).get("steps", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_cm:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "models": ",".join(model_names),
                    "preprocess_steps": ",".join(steps_cfg),
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
        with Progress() as progress:
            strat_task = progress.add_task("CV strategy", total=len(strategies))
            for strat in strategies:
                stype = strat.get("type", "").lower()
                progress.update(strat_task, description=f"Strategy: {stype}")
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
                        sub = out_dir / f"kfold_{k}"
                        sub.mkdir(parents=True, exist_ok=True)
                        _run_cv(
                            cfg,
                            sub,
                            outer,
                            progress,
                            random_state=random_state,
                            interactive=interactive,
                        )
                        metrics = sub / "metrics.json"
                        figs = sub / "figures"
                        with open(metrics) as f:
                            metrics_data = json.load(f)
                        generate_scientific_report(metrics_data, cfg, sub)
                        outputs.append((metrics, figs))
                        if mlflow_enabled:
                            params = {
                                "strategy": "kfold",
                                "n_splits": int(k),
                                "n_repeats": int(n_repeats),
                            }
                            with mlflow.start_run(run_name=sub.name, nested=True):
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
                                mlflow.log_artifacts(figs, artifact_path="figures")
                elif stype == "loo":
                    if cv_method == "group":
                        raise ModelError(
                            "cv_method 'group' não suportado com estratégia LOO"
                        )
                    sub = out_dir / "loo"
                    sub.mkdir(parents=True, exist_ok=True)
                    _run_cv(
                        cfg,
                        sub,
                        LeaveOneOut(),
                        progress,
                        random_state=random_state,
                        interactive=interactive,
                    )
                    metrics = sub / "metrics.json"
                    figs = sub / "figures"
                    with open(metrics) as f:
                        metrics_data = json.load(f)
                    generate_scientific_report(metrics_data, cfg, sub)
                    outputs.append((metrics, figs))
                    if mlflow_enabled:
                        params = {"strategy": "loo"}
                        with mlflow.start_run(run_name=sub.name, nested=True):
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
                            mlflow.log_artifacts(figs, artifact_path="figures")
                elif stype == "holdout":
                    if cv_method == "group":
                        raise ModelError(
                            "cv_method 'group' não suportado com estratégia holdout"
                        )
                    sub = out_dir / "holdout"
                    sub.mkdir(parents=True, exist_ok=True)
                    test_size = float(strat.get("test_size", 0.3))
                    _run_holdout(
                        cfg, sub, test_size, random_state, interactive=interactive
                    )
                    metrics = sub / "metrics.json"
                    figs = sub / "figures"
                    with open(metrics) as f:
                        metrics_data = json.load(f)
                    generate_scientific_report(metrics_data, cfg, sub)
                    outputs.append((metrics, figs))
                    if mlflow_enabled:
                        params = {
                            "strategy": "holdout",
                            "test_size": float(test_size),
                        }
                        with mlflow.start_run(run_name=sub.name, nested=True):
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
                            mlflow.log_artifacts(figs, artifact_path="figures")
                else:
                    raise ModelError(
                        f"Estratégia de validação desconhecida: {stype}. Use 'kfold', 'loo' ou 'holdout'."
                    )
                progress.advance(strat_task)
    return outputs


def train_and_save(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    random_state: int = 42,
    interactive: bool = False,
    mlflow_uri: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Train final model and persist artefacts."""

    mlflow_cm = nullcontext()
    if mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow_cm = mlflow.start_run(run_name="train")
    with mlflow_cm:
        with Progress() as progress:
            load_task = progress.add_task("Carregando dados", total=None)
            _validate_cfg(cfg, need_data=True, need_models=True)
            ds_list = prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))
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
            gs = GridSearchCV(
                pipe,
                grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
            )
            gs.fit(X.values, y_enc, **fit_kwargs)
            progress.update(train_task, completed=1)

            n_comp = _get_n_components(gs.best_params_)
            best = gs.best_estimator_
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
                f"# Model Card\nBest model: {name}\nBest params: {gs.best_params_}\n",
            )

            yhat = best.predict(X.values)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
                "calibrated": bool(calib_cfg),
                "F1-macro_mean": float(f1_score(y_enc, yhat, average="macro")),
                "BalancedAcc_mean": float(balanced_accuracy_score(y_enc, yhat)),
                "Accuracy_mean": float(accuracy_score(y_enc, yhat)),
            }
            row["n_components"] = n_comp
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
                name,
                le.classes_,
                out_dir,
                y_enc,
                yhat,
                proba,
                interactive=interactive,
            )
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
                    "model": name,
                    "preprocess_steps": ",".join(steps_cfg),
                    "cv_method": cv_method,
                }
                params.update({f"best_{k}": v for k, v in gs.best_params_.items()})
                mlflow.log_params(params)
                mlflow.log_metrics(row)
                mlflow.log_artifact(out_dir / "model.joblib")
                mlflow.log_artifact(out_dir / "metrics.json")
                mlflow.log_artifacts(out_dir / "figures", artifact_path="figures")
            progress.update(save_task, completed=1)
    return out_dir / "metrics.json", out_dir / "figures"


def run_experiment(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    interactive: bool = False,
    random_state: int = 42,
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

    Returns
    -------
    tuple
        ``(train_outputs, cv_outputs)`` conforme retornado por
        :func:`train_and_save` e :func:`run_multiple_eval`.
    """

    _validate_cfg(cfg, need_data=True, need_models=True, need_cv=True)
    prepare_inputs(cfg["data"], use_cache=cfg.get("use_cache", True))

    train_dir = out_dir / "train"
    train_outputs = train_and_save(
        cfg, train_dir, random_state=random_state, interactive=interactive
    )

    cv_dir = out_dir / "cv"
    cv_outputs = run_multiple_eval(
        cfg, cv_dir, random_state=random_state, interactive=interactive
    )

    return train_outputs, cv_outputs


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
    with Progress() as progress:
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
