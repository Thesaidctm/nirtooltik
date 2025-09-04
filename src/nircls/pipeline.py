
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
import json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from rich.progress import Progress
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    LeaveOneOut,
    train_test_split,
)
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, brier_score_loss
from joblib import dump, load
from .io import prepare_inputs, export_xy_xlsx, read_xlsx_xy
from .preprocessors.spectro import SNV, MSC, Detrend, SavitzkyGolayTransformer, EMSC, EPO
from .models.factory import make_models_from_config
from .metrics.calibration import expected_calibration_error
from .utils.plots import plot_confusion, plot_roc_ovr, plot_reliability
from .report import generate_report
from .utils.logging import logger

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet",
    category=FutureWarning,
)

def _make_preprocess(steps_cfg, cfg):
    """Create preprocessing steps and optional parameter grid.

    For the Savitzky-Golay (SG) step it is possible to provide a list of
    parameter values (e.g. window length) which will be added to the grid for
    hyper-parameter search.  This allows testing multiple window sizes during
    cross-validation while keeping the feature optional – supplying a single
    value keeps the behaviour unchanged.
    """

    steps=[]; grid={}
    for name in steps_cfg or []:
        n=name.upper()
        if n=="SNV":
            steps.append(("snv", SNV()))
        elif n=="MSC":
            steps.append(("msc", MSC()))
        elif n=="DETREND":
            steps.append(("detrend", Detrend()))
        elif n=="SG":
            sg=cfg.get("sg", {"window":15,"polyorder":2,"deriv":1})
            wl=sg.get("window",15); po=sg.get("polyorder",2); dv=sg.get("deriv",1)
            # allow lists/tuples to trigger grid search over SG parameters
            if isinstance(wl, (list, tuple)):
                grid["sg__window_length"]=[int(w) for w in wl]
                wl=wl[0]
            if isinstance(po, (list, tuple)):
                grid["sg__polyorder"]=[int(p) for p in po]
                po=po[0]
            if isinstance(dv, (list, tuple)):
                grid["sg__deriv"]=[int(d) for d in dv]
                dv=dv[0]
            steps.append(("sg", SavitzkyGolayTransformer(wl, po, dv)))
        elif n=="EMSC":
            steps.append(("emsc", EMSC()))
        elif n=="EPO":
            steps.append(("epo", EPO()))
        elif n=="SCALE":
            steps.append(("scaler", StandardScaler()))
        else:
            raise ValueError(f"Pré-processo desconhecido: {name}")
    return steps, grid

def build_pipeline_from_config(cfg: Dict[str, Any]):
    steps_cfg = cfg.get("preprocess",{}).get("steps", [])
    pre_steps, pre_grid = _make_preprocess(steps_cfg, cfg.get("preprocess",{}))
    models = make_models_from_config(cfg["models"])
    out=[]
    for m in models:
        pipe = Pipeline(pre_steps + [("clf", m["estimator"])])
        grid = pre_grid.copy()
        grid.update({f"clf__{k}": v for k,v in (m.get("grid",{}) or {}).items()})
        out.append((m["name"], pipe, grid))
    return out


def _run_cv(cfg: Dict[str, Any], out_dir: Path, outer, progress: Progress) -> None:
    data = prepare_inputs(cfg["data"])
    rows = []
    tgt_task = progress.add_task("Target", total=len(data))
    for ds in data:
        progress.update(tgt_task, description=f"Target: {ds['target']}")
        X, y = ds["X"], ds["Y"]
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        inner = StratifiedKFold(n_splits=cfg.get("cv", {}).get("inner_folds", 3), shuffle=True, random_state=123)
        models = build_pipeline_from_config(cfg)
        model_task = progress.add_task("Model", total=len(models))
        for name, pipe, grid in models:
            progress.update(model_task, description=f"Model: {name}")
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info(f"Running {name} with steps: {steps_cfg}")
            scores = {"F1-macro": [], "BalancedAcc": [], "Brier": [], "ECE": []}
            y_true_all: List[np.ndarray] = []
            y_pred_all: List[np.ndarray] = []
            y_proba_all: List[np.ndarray] = []
            fold_task = progress.add_task("Fold", total=outer.get_n_splits())
            for tr, te in outer.split(X, y_enc):
                Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
                ytr, yte = y_enc[tr], y_enc[te]
                gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1)
                gs.fit(Xtr, ytr)
                yhat = gs.predict(Xte)
                scores["F1-macro"].append(f1_score(yte, yhat, average="macro"))
                scores["BalancedAcc"].append(balanced_accuracy_score(yte, yhat))
                proba_fn = getattr(gs.best_estimator_, "predict_proba", None)
                if proba_fn is not None:
                    proba = proba_fn(Xte)
                    y_proba_all.append(proba)
                    brier = np.mean([brier_score_loss((yte == k).astype(int), proba[:, k]) for k in range(len(classes))])
                    scores["Brier"].append(brier)
                    scores["ECE"].append(expected_calibration_error(yte, proba, n_bins=15))
                y_true_all.append(yte)
                y_pred_all.append(yhat)
                progress.advance(fold_task)
            row = {"file": ds["file"], "target": ds["target"], "model": name}
            row.update({f"{k}_mean": float(np.mean(v)) for k, v in scores.items() if v})
            row.update({f"{k}_std": float(np.std(v, ddof=1)) for k, v in scores.items() if len(v) > 1})
            rows.append(row)
            out_fig = out_dir / "figures"
            out_fig.mkdir(parents=True, exist_ok=True)
            y_true = np.concatenate(y_true_all)
            y_pred = np.concatenate(y_pred_all)
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion(cm, classes, out_fig / f"{Path(ds['file']).stem}__{ds['target']}__{name}__cm")
            if y_proba_all:
                y_proba = np.vstack(y_proba_all)
                plot_roc_ovr(
                    y_true,
                    y_proba,
                    classes,
                    out_fig / f"{Path(ds['file']).stem}__{ds['target']}__{name}__roc",
                )
                plot_reliability(
                    y_true,
                    y_proba,
                    out_fig / f"{Path(ds['file']).stem}__{ds['target']}__{name}__reliability",
                )
            progress.advance(model_task)
        progress.advance(tgt_task)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)


def _run_holdout(cfg: Dict[str, Any], out_dir: Path, test_size: float, random_state: int) -> None:
    data = prepare_inputs(cfg["data"])
    rows = []
    for ds in data:
        X, y = ds["X"], ds["Y"]
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        inner = StratifiedKFold(n_splits=cfg.get("cv", {}).get("inner_folds", 3), shuffle=True, random_state=123)
        for name, pipe, grid in build_pipeline_from_config(cfg):
            Xtr, Xte, ytr, yte = train_test_split(
                X.values,
                y_enc,
                test_size=test_size,
                random_state=random_state,
                stratify=y_enc,
            )
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info(f"Running {name} with steps: {steps_cfg}")
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1)
            gs.fit(Xtr, ytr)
            yhat = gs.predict(Xte)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
                "F1-macro_mean": float(f1_score(yte, yhat, average="macro")),
                "BalancedAcc_mean": float(balanced_accuracy_score(yte, yhat)),
            }
            proba_fn = getattr(gs.best_estimator_, "predict_proba", None)
            if proba_fn is not None:
                proba = proba_fn(Xte)
                brier = np.mean([brier_score_loss((yte == k).astype(int), proba[:, k]) for k in range(len(classes))])
                row["Brier_mean"] = float(brier)
                row["ECE_mean"] = float(expected_calibration_error(yte, proba, n_bins=15))
            rows.append(row)
            out_fig = out_dir / "figures"
            out_fig.mkdir(parents=True, exist_ok=True)
            cm = confusion_matrix(yte, yhat)
            plot_confusion(cm, classes, out_fig / f"{Path(ds['file']).stem}__{ds['target']}__{name}__cm")
            if proba_fn is not None:
                plot_roc_ovr(
                    yte,
                    proba,
                    classes,
                    out_fig / f"{Path(ds['file']).stem}__{ds['target']}__{name}__roc",
                )
                plot_reliability(
                    yte,
                    proba,
                    out_fig / f"{Path(ds['file']).stem}__{ds['target']}__{name}__reliability",
                )
    pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)


def run_multiple_eval(cfg: Dict[str, Any], out_dir: Path) -> None:
    strategies = cfg.get("cv", {}).get("strategies", [])
    random_state = cfg.get("cv", {}).get("random_state", 42)
    with Progress() as progress:
        strat_task = progress.add_task("CV strategy", total=len(strategies))
        for strat in strategies:
            stype = strat.get("type", "").lower()
            progress.update(strat_task, description=f"Strategy: {stype}")
            if stype == "kfold":
                n_repeats = strat.get("n_repeats", 1)
                for k in strat.get("n_splits", [5]):
                    if n_repeats > 1:
                        outer = RepeatedStratifiedKFold(
                            n_splits=int(k), n_repeats=int(n_repeats), random_state=random_state
                        )
                    else:
                        outer = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=random_state)
                    sub = out_dir / f"kfold_{k}"
                    sub.mkdir(parents=True, exist_ok=True)
                    _run_cv(cfg, sub, outer, progress)
                    generate_report(sub / "metrics.json", sub / "figures", sub / "report.md")
            elif stype == "loo":
                sub = out_dir / "loo"
                sub.mkdir(parents=True, exist_ok=True)
                _run_cv(cfg, sub, LeaveOneOut(), progress)
                generate_report(sub / "metrics.json", sub / "figures", sub / "report.md")
            elif stype == "holdout":
                sub = out_dir / "holdout"
                sub.mkdir(parents=True, exist_ok=True)
                test_size = float(strat.get("test_size", 0.3))
                _run_holdout(cfg, sub, test_size, random_state)
                generate_report(sub / "metrics.json", sub / "figures", sub / "report.md")
            else:
                raise ValueError(f"Unknown CV strategy: {stype}")
            progress.advance(strat_task)


def train_and_save(cfg: Dict[str, Any], out_dir: Path):
    ds = prepare_inputs(cfg["data"])[0]
    X, y = ds["X"], ds["Y"]
    le = LabelEncoder().fit(y); y_enc = le.transform(y)
    name, pipe, grid = build_pipeline_from_config(cfg)[0]
    steps_cfg = cfg.get("preprocess", {}).get("steps", [])
    logger.info(f"Running {name} with steps: {steps_cfg}")
    gs = GridSearchCV(
        pipe,
        grid,
        scoring="f1_macro",
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42),
        n_jobs=-1,
    )
    gs.fit(X.values, y_enc)
    dump(gs.best_estimator_, out_dir / "model.joblib")
    export_xy_xlsx(out_dir / "prepared_XY.xlsx", X, y)
    (out_dir/"model_card.md").write_text(f"# Model Card\nBest model: {name}\nBest params: {gs.best_params_}\n")

def evaluate_saved_model(model_path: Path, data_path: Path):
    est = load(model_path)
    X, y_list = read_xlsx_xy(data_path)
    y = y_list[0][1]
    yhat = est.predict(X.values)
    print(pd.crosstab(pd.Series(y, name="true"), pd.Series(yhat, name="pred")))

def predict_with_model(model_path: Path, data_path: Path, out_dir: Path):
    est = load(model_path)
    X, y_list = read_xlsx_xy(data_path)
    y = y_list[0][1]
    yhat = est.predict(X.values)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y, "y_pred": yhat}).to_csv(out_dir / "predictions.csv", index=False)

def explain_model(model_path: Path, data_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir/"explain.txt").write_text("Explainability hook.")
