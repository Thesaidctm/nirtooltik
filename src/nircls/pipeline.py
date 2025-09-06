
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import warnings
import json
import numpy as np
import pandas as pd
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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump, load
from .io import prepare_inputs, export_xy_xlsx, read_xlsx_xy, read_tabular
from .preprocessors.spectro import (
    SNV,
    MSC,
    Detrend,
    SavitzkyGolayTransformer,
    EMSC,
    EPO,
)
from .models.factory import make_models_from_config
from .config_schema import ConfigModel, ValidationError
from .metrics.calibration import expected_calibration_error
from .utils.plots import (
    plot_confusion,
    plot_confusion_interactive,
    plot_roc_interactive,
    plot_roc_ovr,
    plot_reliability,
)
from .report import generate_report
from .utils.logging import logger

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
        raise ValueError(
            f"Configuração inválida: {exc}. Revise os campos informados."
        ) from exc
    if need_data and model.data is None:
        raise ValueError("Seção 'data' é obrigatória. Informe o caminho dos dados.")
    if need_models and (not model.models):
        raise ValueError("Seção 'models' é obrigatória. Adicione pelo menos um modelo.")
    if need_cv and (model.cv is None or not model.cv.strategies):
        raise ValueError(
            "Seção 'cv' é obrigatória. Defina ao menos uma estratégia de validação."
        )
    return model
def _make_preprocess(steps_cfg, cfg):
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
            raise ValueError(
                f"Pré-processo desconhecido: {name}. Verifique a lista de etapas suportadas."
            )
    return steps, grid

def build_pipeline_from_config(cfg: Dict[str, Any]):
    _validate_cfg(cfg, need_models=True)
    steps_cfg = cfg.get("preprocess", {}).get("steps", [])
    pre_steps, pre_grid = _make_preprocess(
        steps_cfg, cfg.get("preprocess", {})
    )
    models = make_models_from_config(cfg["models"])
    out = []
    for m in models:
        est = m["estimator"]
        grid = pre_grid.copy()
        model_grid = m.get("grid", {}) or {}
        if not hasattr(est, "predict_proba"):
            calib_cfg = (m.get("calibration") or {})
            method = calib_cfg.get("method", "sigmoid")
            est = CalibratedClassifierCV(estimator=est, method=method)
            grid.update(
                {f'clf__estimator__{k}': v for k, v in model_grid.items()}
            )
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
    id_cols = ["file", "target", "model"]
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
        out_fig / f"{stem}__{target}__{model}__cm",
        normalize=False,
    )
    if interactive:
        plot_confusion_interactive(
            cm,
            classes,
            out_fig / f"{stem}__{target}__{model}__cm",
            normalize=False,
        )
    if y_proba is not None:
        plot_roc_ovr(
            y_true,
            y_proba,
            classes,
            out_fig / f"{stem}__{target}__{model}__roc",
        )
        plot_reliability(
            y_true,
            y_proba,
            out_fig / f"{stem}__{target}__{model}__reliability",
        )
        if interactive:
            plot_roc_interactive(
                y_true,
                y_proba,
                classes,
                out_fig / f"{stem}__{target}__{model}__roc",
            )


def _run_cv(
    cfg: Dict[str, Any],
    out_dir: Path,
    outer,
    progress: Progress,
    *,
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
    """

    data = prepare_inputs(cfg["data"])
    rows: List[Dict[str, Any]] = []
    tgt_task = progress.add_task("Target", total=len(data))
    for ds in data:
        progress.update(tgt_task, description=f"Target: {ds['target']}")
        X, y = ds["X"], ds["Y"]
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        inner = StratifiedKFold(
            n_splits=cfg.get("cv", {}).get("inner_folds", 3),
            shuffle=True,
            random_state=123,
        )
        models = build_pipeline_from_config(cfg)
        model_task = progress.add_task("Model", total=len(models))
        for name, pipe, grid in models:
            progress.update(model_task, description=f"Model: {name}")
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
            scores_val = {
                "F1-macro": [],
                "BalancedAcc": [],
                "Accuracy": [],
                "Brier": None,
                "ECE": None,
            }
            scores_train = {
                "F1-macro": [],
                "BalancedAcc": [],
                "Accuracy": [],
                "Brier": None,
                "ECE": None,
            }
            y_true_all: List[np.ndarray] = []
            y_pred_all: List[np.ndarray] = []
            y_proba_all: List[np.ndarray] = []
            fold_task = progress.add_task(
                "Fold", total=outer.get_n_splits(X, y_enc)
            )
            for tr, te in outer.split(X, y_enc):
                Xtr, Xte = X.iloc[tr].values, X.iloc[te].values
                ytr, yte = y_enc[tr], y_enc[te]
                gs = GridSearchCV(
                    pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1
                )
                gs.fit(Xtr, ytr)

                # validation metrics
                yhat = gs.predict(Xte)
                scores_val["F1-macro"].append(
                    f1_score(yte, yhat, average="macro")
                )
                scores_val["BalancedAcc"].append(
                    balanced_accuracy_score(yte, yhat)
                )
                scores_val["Accuracy"].append(
                    accuracy_score(yte, yhat)
                )
                proba_fn = getattr(gs, "predict_proba", None)
                if proba_fn is not None:
                    if scores_val["Brier"] is None:
                        scores_val["Brier"] = []
                        scores_val["ECE"] = []
                        scores_train["Brier"] = []
                        scores_train["ECE"] = []
                    proba = proba_fn(Xte)
                    y_proba_all.append(proba)
                    brier = np.mean(
                        [
                            brier_score_loss(
                                (yte == k).astype(int), proba[:, k]
                            )
                            for k in range(len(classes))
                        ]
                    )
                    scores_val["Brier"].append(brier)
                    scores_val["ECE"].append(
                        expected_calibration_error(yte, proba, n_bins=15)
                    )

                # training metrics
                yhat_tr = gs.predict(Xtr)
                scores_train["F1-macro"].append(
                    f1_score(ytr, yhat_tr, average="macro")
                )
                scores_train["BalancedAcc"].append(
                    balanced_accuracy_score(ytr, yhat_tr)
                )
                scores_train["Accuracy"].append(
                    accuracy_score(ytr, yhat_tr)
                )
                if proba_fn is not None:
                    proba_tr = proba_fn(Xtr)
                    brier_tr = np.mean(
                        [
                            brier_score_loss(
                                (ytr == k).astype(int), proba_tr[:, k]
                            )
                            for k in range(len(classes))
                        ]
                    )
                    scores_train["Brier"].append(brier_tr)
                    scores_train["ECE"].append(
                        expected_calibration_error(
                            ytr, proba_tr, n_bins=15
                        )
                    )
                y_true_all.append(yte)
                y_pred_all.append(yhat)
                progress.advance(fold_task)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
            }
            row.update(
                {
                    f"{k}_mean": float(np.mean(v))
                    for k, v in scores_val.items()
                    if v
                }
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
        Seed applied to the train/test split.
    interactive:
        If ``True``, also generate interactive Plotly HTML figures.
    """

    data = prepare_inputs(cfg["data"])
    rows: List[Dict[str, Any]] = []
    for ds in data:
        X, y = ds["X"], ds["Y"]
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        classes = le.classes_
        inner = StratifiedKFold(
            n_splits=cfg.get("cv", {}).get("inner_folds", 3),
            shuffle=True,
            random_state=123,
        )
        for name, pipe, grid in build_pipeline_from_config(cfg):
            Xtr, Xte, ytr, yte = train_test_split(
                X.values,
                y_enc,
                test_size=test_size,
                random_state=random_state,
                stratify=y_enc,
            )
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
            gs = GridSearchCV(
                pipe, grid, scoring="f1_macro", cv=inner, n_jobs=-1
            )
            gs.fit(Xtr, ytr)
            yhat = gs.predict(Xte)
            yhat_tr = gs.predict(Xtr)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
                "F1-macro_mean": float(
                    f1_score(yte, yhat, average="macro")
                ),
                "BalancedAcc_mean": float(
                    balanced_accuracy_score(yte, yhat)
                ),
                "Accuracy_mean": float(
                    accuracy_score(yte, yhat)
                ),
                "train_F1-macro_mean": float(
                    f1_score(ytr, yhat_tr, average="macro")
                ),
                "train_BalancedAcc_mean": float(
                    balanced_accuracy_score(ytr, yhat_tr)
                ),
                "train_Accuracy_mean": float(
                    accuracy_score(ytr, yhat_tr)
                ),
            }
            proba_fn = getattr(gs, "predict_proba", None)
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
                        brier_score_loss(
                            (ytr == k).astype(int), proba_tr[:, k]
                        )
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


def run_multiple_eval(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    interactive: bool = False,
) -> None:
    """Run evaluation for multiple cross-validation strategies.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    out_dir:
        Directory that receives a subfolder for each strategy.
    interactive:
        If ``True``, also generate interactive Plotly HTML figures.
    """

    _validate_cfg(cfg, need_data=True, need_cv=True, need_models=True)
    strategies = cfg.get("cv", {}).get("strategies", [])
    random_state = cfg.get("cv", {}).get("random_state", 42)
    with Progress() as progress:
        strat_task = progress.add_task(
            "CV strategy", total=len(strategies)
        )
        for strat in strategies:
            stype = strat.get("type", "").lower()
            progress.update(
                strat_task, description=f"Strategy: {stype}"
            )
            if stype == "kfold":
                n_repeats = strat.get("n_repeats", 1)
                for k in strat.get("n_splits", [5]):
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
                    _run_cv(cfg, sub, outer, progress, interactive=interactive)
                    generate_report(
                        sub / "metrics.json",
                        sub / "figures",
                        sub / "report.md",
                    )
            elif stype == "loo":
                sub = out_dir / "loo"
                sub.mkdir(parents=True, exist_ok=True)
                _run_cv(cfg, sub, LeaveOneOut(), progress, interactive=interactive)
                generate_report(
                    sub / "metrics.json",
                    sub / "figures",
                    sub / "report.md",
                )
            elif stype == "holdout":
                sub = out_dir / "holdout"
                sub.mkdir(parents=True, exist_ok=True)
                test_size = float(strat.get("test_size", 0.3))
                _run_holdout(cfg, sub, test_size, random_state, interactive=interactive)
                generate_report(
                    sub / "metrics.json",
                    sub / "figures",
                    sub / "report.md",
                )
            else:
                raise ValueError(
                    f"Estratégia de validação desconhecida: {stype}. Use 'kfold', 'loo' ou 'holdout'."
                )
            progress.advance(strat_task)


def train_and_save(
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    interactive: bool = False,
):
    """Train final model and persist artefacts.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    out_dir:
        Destination for the trained model and related files.
    interactive:
        If ``True``, also generate interactive Plotly HTML figures.
    """

    try:
        with Progress() as progress:
            load_task = progress.add_task("Carregando dados", total=None)
            _validate_cfg(cfg, need_data=True, need_models=True)
            ds = prepare_inputs(cfg["data"])[0]
            X, y = ds["X"], ds["Y"]
            le = LabelEncoder().fit(y)
            y_enc = le.transform(y)
            progress.update(load_task, completed=1)

            name, pipe, grid = build_pipeline_from_config(cfg)[0]
            steps_cfg = cfg.get("preprocess", {}).get("steps", [])
            logger.info("Running %s with steps: %s", name, steps_cfg)
            train_task = progress.add_task("Treinando", total=None)
            gs = GridSearchCV(
                pipe,
                grid,
                scoring="f1_macro",
                cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42),
                n_jobs=-1,
            )
            gs.fit(X.values, y_enc)
            progress.update(train_task, completed=1)

            save_task = progress.add_task("Salvando", total=None)
            dump(gs.best_estimator_, out_dir / "model.joblib")
            export_xy_xlsx(out_dir / "prepared_XY.xlsx", X, y)
            (out_dir / "model_card.md").write_text(
                f"# Model Card\nBest model: {name}\nBest params: {gs.best_params_}\n",
            )

            yhat = gs.predict(X.values)
            row = {
                "file": ds["file"],
                "target": ds["target"],
                "model": name,
                "F1-macro_mean": float(f1_score(y_enc, yhat, average="macro")),
                "BalancedAcc_mean": float(balanced_accuracy_score(y_enc, yhat)),
                "Accuracy_mean": float(accuracy_score(y_enc, yhat)),
            }
            proba_fn = getattr(gs, "predict_proba", None)
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
            generate_report(
                out_dir / "metrics.json",
                out_dir / "figures",
                out_dir / "report.md",
            )
            progress.update(save_task, completed=1)
    except FileNotFoundError:
        return "Arquivo de dados não encontrado. Verifique o caminho informado."
    except ValueError as e:
        return (
            f"Erro ao preparar dados ou modelo: {e}. "
            "Verifique se a coluna alvo existe na planilha e se os parâmetros do modelo estão corretos."
        )
def evaluate_saved_model(model_path: Path, data_path: Path):
    try:
        est = load(model_path)
        X, y_list = read_xlsx_xy(data_path)
        y = y_list[0][1]
        yhat = est.predict(X.values)
        print(pd.crosstab(pd.Series(y, name="true"), pd.Series(yhat, name="pred")))
    except FileNotFoundError:
        return "Modelo ou arquivo de dados não encontrado. Verifique os caminhos."
    except ValueError as e:
        return (
            f"Erro ao ler os dados: {e}. Verifique se a coluna alvo existe na planilha."
        )

def predict_with_model(model_path: Path, data_path: Path, out_dir: Path):
    try:
        with Progress() as progress:
            load_model_task = progress.add_task("Carregando modelo", total=None)
            est = load(model_path)
            progress.update(load_model_task, completed=1)

            load_data_task = progress.add_task("Carregando dados", total=None)
            y = None
            if data_path.suffix.lower() in [".xlsx", ".xls"]:
                try:
                    X, y_list = read_xlsx_xy(data_path)
                except ValueError:
                    X, y_list = read_tabular(data_path)
            else:
                X, y_list = read_tabular(data_path)
            if y_list:
                y = y_list[0][1]
            progress.update(load_data_task, completed=1)

            pred_task = progress.add_task("Prevendo", total=None)
            yhat = est.predict(X.values)
            progress.update(pred_task, completed=1)

            save_task = progress.add_task("Salvando", total=None)
            out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"y_pred": yhat})
            if y is not None:
                df.insert(0, "y_true", y)
            df.to_csv(out_dir / "predictions.csv", index=False)
            progress.update(save_task, completed=1)
    except FileNotFoundError:
        return "Arquivo de dados não encontrado. Verifique o caminho informado."
    except ValueError as e:
        return (
            f"Erro ao processar dados ou modelo: {e}. "
            "Verifique se a coluna alvo existe na planilha e se o modelo é compatível."
        )
def explain_model(model_path: Path, data_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "explain.txt").write_text("Explainability hook.")
