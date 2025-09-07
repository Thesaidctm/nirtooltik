"""Command-line interface for the NIR species classification toolkit.

This module exposes the :func:`main` function which backs the ``nircls``
command-line utility for preparing data, training models, evaluating saved
models, making predictions and generating explanations.
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .io import export_xy_xlsx, summarize_dataset
from .pipeline import (
    run_multiple_eval,
    train_and_save,
    evaluate_saved_model,
    predict_with_model,
    explain_model,
    prepare_inputs,
)
from .report import generate_report
from .dashboard import run_dashboard
from .utils.logging import setup_logging, log_versions
from .utils.misc import ensure_dirs, make_run_id, set_seed, RAW_DIR, RESULTS_DIR
from .config_schema import ConfigModel, ValidationError
from .utils.errors import DataError, ModelError

# Human-friendly names for interactive wizard choices
MODEL_DESCRIPTIONS = {
    "PLSDA": "Partial Least Squares Discriminant Analysis",
    "SVM_RBF": "Support Vector Machine (RBF)",
    "SVM_LINEAR": "Support Vector Machine (Linear)",
    "RF": "Random Forest",
    "GB": "Gradient Boosting",
    "ET": "Extra Trees",
    "ADABOOST": "AdaBoost",
    "LOGREG": "Logistic Regression",
    "QDA": "Quadratic Discriminant Analysis",
    "XGBOOST": "XGBoost",
    "LGBM": "LightGBM",
    "CATBOOST": "CatBoost",
    "MLP": "Multilayer Perceptron",
    "DT": "Decision Tree",
    "KNN": "k-Nearest Neighbors",
    "CNN1D": "1D Convolutional Neural Network",
    "LSTM1D": "1D LSTM Network",
    "GAUSSIANNB": "Gaussian Naive Bayes",
    "NB": "Gaussian Naive Bayes",
    "MULTINB": "Multinomial Naive Bayes",
    "BERNOULLINB": "Bernoulli Naive Bayes",
}

PREPROCESS_DESCRIPTIONS = {
    "SNV": "Standard Normal Variate",
    "MSC": "Multiplicative Scatter Correction",
    "DETREND": "Polynomial Detrend",
    "SG": "Savitzky-Golay Filter",
    "EMSC": "Extended MSC",
    "EPO": "External Parameter Orthogonalization",
    "SCALE": "Standard Scaling",
}


def _print_metrics_table(metrics_json: Path) -> None:
    """Print a compact table with mean ± std metrics."""

    if not metrics_json or not metrics_json.exists():
        return
    try:
        df = pd.read_json(metrics_json)
    except ValueError:
        return
    if df.empty:
        return

    id_cols = [c for c in ["target", "model"] if c in df.columns]
    metrics_cols = [c for c in df.columns if c.endswith("_mean")]
    for col in metrics_cols:
        base = col[:-5]
        std_col = f"{base}_std"
        if std_col in df.columns:
            df[base] = df.apply(
                lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}", axis=1
            )
        else:
            df[base] = df[col].map(lambda x: f"{x:.3f}")
    display_cols = id_cols + [c[:-5] for c in metrics_cols]
    print(df[display_cols].to_string(index=False))


def _print_confusion_paths(fig_dir: Path) -> None:
    """Print paths of confusion matrix images if they exist."""

    if not fig_dir or not fig_dir.exists():
        return
    for p in sorted(fig_dir.glob("*confusion_raw.png")):
        print(f"Matriz de confusão (bruta): {p}")
    for p in sorted(fig_dir.glob("*confusion_norm.png")):
        print(f"Matriz de confusão (normalizada): {p}")


def _parse_params(s: str) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a dictionary."""
    params: dict[str, Any] = {}
    for part in s.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k.strip()] = yaml.safe_load(v.strip())
    return params


def _deep_update(base: dict, updates: dict) -> dict:
    """Recursively update ``base`` with ``updates``."""

    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def main(argv=None):
    """Run the ``nircls`` command-line interface.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of arguments. When ``None`` (default), ``sys.argv`` is
        used.
    """

    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="nircls", description="NIR Species Classification Toolkit"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--run",
        choices=["one", "all"],
        default="all",
        help="Rodar somente uma ou todas as análises",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade",
    )
    parser.add_argument(
        "--mlflow",
        help="URI do servidor MLflow para rastreamento",
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    p_prep = sp.add_parser("prepare-data")
    p_prep.add_argument("--config", required=True)
    p_tr = sp.add_parser("train")
    p_tr.add_argument("--config", required=True)
    p_tr.add_argument("--report-pdf")
    p_tr.add_argument(
        "--interactive",
        action="store_true",
        help="Gerar arquivos HTML com gráficos Plotly interativos",
    )
    p_cv = sp.add_parser("crossval")
    p_cv.add_argument("--config", required=True)
    p_cv.add_argument("--report-pdf")
    p_cv.add_argument(
        "--interactive",
        action="store_true",
        help="Gerar arquivos HTML com gráficos Plotly interativos",
    )
    p_demo = sp.add_parser("demo")
    p_all = sp.add_parser(
        "run-all",
        help="Processa dados brutos e executa treino+avaliação com padrões",
    )
    p_all.add_argument(
        "--config",
        help="Arquivo de configuração para sobrescrever parâmetros padrão",
    )
    p_wz = sp.add_parser("wizard")
    p_ev = sp.add_parser("evaluate")
    p_ev.add_argument("--model", required=True)
    p_ev.add_argument("--data", required=True)
    p_pd = sp.add_parser("predict")
    p_pd.add_argument("--model", required=True)
    p_pd.add_argument("--data", required=True)
    p_pd.add_argument("--output", default=str(RESULTS_DIR))
    p_ex = sp.add_parser("explain")
    p_ex.add_argument("--model", required=True)
    p_ex.add_argument("--data", required=True)
    p_ex.add_argument("--output", default=str(RESULTS_DIR))
    p_db = sp.add_parser("dashboard")
    p_db.add_argument("--results", required=True, help="Pasta com resultados")
    args = parser.parse_args(argv)
    set_seed(args.seed)
    setup_logging()
    log_versions()
    mlflow_uri = args.mlflow
    if mlflow_uri:
        try:
            import mlflow
        except Exception:
            print(
                "'mlflow' não está instalado. Instale com `pip install mlflow`.",
                file=sys.stderr,
            )
            return 1
        mlflow.set_tracking_uri(mlflow_uri)
    if args.cmd in {"prepare-data", "train", "crossval"}:
        try:
            with open(args.config) as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError:
            print(
                "Arquivo de configuração não encontrado. Verifique o caminho informado.",
                file=sys.stderr,
            )
            return 1
        try:
            cfg = ConfigModel(**raw)
        except ValidationError as e:  # pragma: no cover - user feedback
            print(
                f"Configuração inválida: {e}. Revise os campos obrigatórios.",
                file=sys.stderr,
            )
            return 1

    if args.cmd == "prepare-data":
        if cfg.data is None:
            print("Configuração inválida: seção 'data' é obrigatória", file=sys.stderr)
            return 1
        run_id = make_run_id("prepare")
        out_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(out_dir)
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        if not summary.empty:
            print(summary.to_string(index=False))
        if args.run == "one":
            ds_list = ds_list[:1]
        for ds in ds_list:
            out_file = out_dir / f"prepared_{ds['target']}.xlsx"
            export_xy_xlsx(
                out_file,
                ds["X"],
                ds["Y"],
                cfg.data.x_sheet or "X",
                cfg.data.y_sheet or "Y",
            )
    elif args.cmd == "train":
        if cfg.data is None or not cfg.models:
            print(
                "Configuração inválida: seções 'data' e 'models' são obrigatórias",
                file=sys.stderr,
            )
            return 1
        run_id = make_run_id("train")
        out_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(out_dir)
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        if not summary.empty:
            print(summary.to_string(index=False))
        try:
            metrics_path, figs_dir = train_and_save(
                cfg.dict(),
                out_dir,
                interactive=args.interactive,
                random_state=args.seed,
                mlflow_uri=mlflow_uri,
            )
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
            return 1
        except ModelError as e:
            print(f"Modelo inválido: {e}", file=sys.stderr)
            return 1
        if metrics_path:
            _print_metrics_table(metrics_path)
        if figs_dir:
            _print_confusion_paths(figs_dir)
        if args.report_pdf and metrics_path and figs_dir:
            generate_report(metrics_path, figs_dir, Path(args.report_pdf))
        print(f"Resultados em: {out_dir}")
    elif args.cmd == "crossval":
        if cfg.data is None or not cfg.models or cfg.cv is None:
            print(
                "Configuração inválida: seções 'data', 'models' e 'cv' são obrigatórias",
                file=sys.stderr,
            )
            return 1
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        if not summary.empty:
            print(summary.to_string(index=False))
        if args.run == "one" and ds_list:
            cfg.data.target_cols = ds_list[0]["target"]
        run_id = make_run_id("cv")
        out_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(out_dir)
        try:
            outputs = run_multiple_eval(
                cfg.dict(),
                out_dir,
                interactive=args.interactive,
                random_state=args.seed,
                mlflow_uri=mlflow_uri,
            )
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
            return 1
        except ModelError as e:
            print(f"Modelo inválido: {e}", file=sys.stderr)
            return 1
        if args.report_pdf:
            pdf_path = Path(args.report_pdf)
            multi = len(outputs) > 1
        for metrics_path, figs_dir in outputs:
            out_pdf = pdf_path
            if pdf_path.suffix.lower() != ".pdf" or multi:
                base_dir = (
                    pdf_path if pdf_path.suffix.lower() != ".pdf" else pdf_path.parent
                )
                out_pdf = base_dir / f"{metrics_path.parent.name}.pdf"
            generate_report(metrics_path, figs_dir, out_pdf)
        for metrics_path, figs_dir in outputs:
            print(f"== {metrics_path.parent.name} ==")
            _print_metrics_table(metrics_path)
            _print_confusion_paths(figs_dir)
        print(f"Resultados em: {out_dir}")
    elif args.cmd == "run-all":
        default_cfg = {
            "data": {"input_path": str(RAW_DIR), "out_dir": str(RESULTS_DIR)},
            "models": [{"name": "PLSDA"}],
            "cv": {"strategies": [{"type": "kfold", "n_splits": 5}]},
        }
        if args.config:
            try:
                with open(args.config) as f:
                    user_cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                print(
                    "Arquivo de configuração não encontrado. Verifique o caminho informado.",
                    file=sys.stderr,
                )
                return 1
            _deep_update(default_cfg, user_cfg)
        run_id = make_run_id("run")
        base_dir = Path(default_cfg["data"].get("out_dir", RESULTS_DIR)) / run_id
        ensure_dirs(base_dir)
        default_cfg["data"]["out_dir"] = str(base_dir)
        try:
            ds_list = prepare_inputs(default_cfg["data"], use_cache=default_cfg.get("use_cache", True))
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        if not summary.empty:
            print(summary.to_string(index=False))
        if args.run == "one" and ds_list:
            default_cfg["data"]["target_cols"] = ds_list[0]["target"]
        train_dir = base_dir / "train"
        cv_dir = base_dir / "cv"
        ensure_dirs(train_dir, cv_dir)
        try:
            train_metrics, train_figs = train_and_save(
                default_cfg,
                train_dir,
                random_state=args.seed,
                mlflow_uri=mlflow_uri,
            )
        except (DataError, ModelError) as e:
            print(str(e), file=sys.stderr)
            return 1
        model_path = train_dir / "model.joblib"
        report_path = train_dir / "report.md"
        try:
            outputs = run_multiple_eval(
                default_cfg,
                cv_dir,
                random_state=args.seed,
                mlflow_uri=mlflow_uri,
            )
        except (DataError, ModelError) as e:
            print(str(e), file=sys.stderr)
            return 1
        for cv_metrics, cv_figs in outputs:
            print(f"== {cv_metrics.parent.name} ==")
            _print_metrics_table(cv_metrics)
            _print_confusion_paths(cv_figs)
            rpt = cv_metrics.parent / "report.md"
            if rpt.exists():
                print(f"Relatório completo: {rpt}")
        if train_metrics:
            print("== Modelo final ==")
            _print_metrics_table(train_metrics)
            _print_confusion_paths(train_figs)
            if report_path.exists():
                print(f"Relatório completo: {report_path}")
            if model_path.exists():
                print(f"Modelo salvo: {model_path}")
        print(f"Resultados em: {base_dir}")
    elif args.cmd == "demo":
        demo_cfg = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"
        with open(demo_cfg) as f:
            raw = yaml.safe_load(f)
        raw["data"]["input_path"] = str(
            (demo_cfg.parent.parent / raw["data"]["input_path"]).resolve()
        )
        try:
            cfg = ConfigModel(**raw)
        except ValidationError as e:  # pragma: no cover - user feedback
            print(f"Configuração inválida: {e}", file=sys.stderr)
            return 1
        run_id = make_run_id("demo")
        base_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(base_dir)
        ds_list = []
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
        if ds_list:
            first = ds_list[0]
            prep_file = base_dir / f"prepared_{first['target']}.xlsx"
            export_xy_xlsx(
                prep_file,
                first["X"],
                first["Y"],
                cfg.data.x_sheet or "X",
                cfg.data.y_sheet or "Y",
            )
            print(f"Dados preparados em: {prep_file}")
        train_dir = base_dir / "train"
        ensure_dirs(train_dir)
        try:
            metrics_path, figs_dir = train_and_save(
                cfg.dict(), train_dir, random_state=args.seed, mlflow_uri=mlflow_uri
            )
            if metrics_path:
                _print_metrics_table(metrics_path)
            if figs_dir:
                _print_confusion_paths(figs_dir)
            print(f"Modelo treinado em: {train_dir}")
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
        except ModelError as e:
            print(f"Modelo inválido: {e}", file=sys.stderr)
        cv_dir = base_dir / "cv"
        ensure_dirs(cv_dir)
        try:
            outputs = run_multiple_eval(
                cfg.dict(), cv_dir, random_state=args.seed, mlflow_uri=mlflow_uri
            )
            for metrics_path, figs_dir in outputs:
                _print_metrics_table(metrics_path)
                _print_confusion_paths(figs_dir)
            print(f"Resultados de cross-val em: {cv_dir}")
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
        except ModelError as e:
            print(f"Modelo inválido: {e}", file=sys.stderr)
    elif args.cmd == "wizard":
        try:  # pragma: no cover - interactive session
            import questionary
            from questionary import Choice
        except Exception:  # pragma: no cover - missing optional dependency
            print(
                "'questionary' não está instalado. Instale com `pip install questionary`.",
                file=sys.stderr,
            )
            return 1

        from typing import Union

        def _validate_path(p: str) -> Union[bool, str]:
            return Path(p).exists() or "Arquivo não encontrado"

        input_path = questionary.text(
            "Caminho dos dados:", validate=_validate_path
        ).ask()
        target = questionary.text(
            "Coluna alvo:", validate=lambda t: bool(t) or "Informe a coluna alvo"
        ).ask()

        from .models.factory import MODEL_BUILDERS

        model_choices = [
            Choice(title=f"{k} - {MODEL_DESCRIPTIONS.get(k, k)}", value=k)
            for k in MODEL_BUILDERS.keys()
        ]
        selected_models = questionary.checkbox(
            "Modelos:",
            choices=model_choices,
            validate=lambda m: bool(m) or "Selecione pelo menos um modelo",
        ).ask()

        models_cfg = []
        for name in selected_models:
            params_raw = questionary.text(
                f"Parâmetros para {name} (k=v, separados por vírgula, opcional):"
            ).ask()
            model_entry: dict[str, Any] = {"name": name}
            if params_raw:
                params = _parse_params(params_raw)
                if params:
                    model_entry["params"] = params
            models_cfg.append(model_entry)

        pre_choices = [
            Choice(title=f"{k} - {v}", value=k) for k, v in PREPROCESS_DESCRIPTIONS.items()
        ]
        preprocess_steps = questionary.checkbox(
            "Pré-processadores:", choices=pre_choices
        ).ask()

        out_dir_str = (
            questionary.text(
                "Diretório para salvar config.yaml (padrão: .):", default="."
            ).ask()
            or "."
        )
        out_path = Path(out_dir_str)
        out_path.mkdir(parents=True, exist_ok=True)
        cfg_dict = {
            "data": {"input_path": input_path, "target_cols": target},
            "models": models_cfg,
            "cv": {"strategies": [{"type": "holdout", "test_size": 0.2}]},
        }
        if preprocess_steps:
            cfg_dict["preprocess"] = {"steps": preprocess_steps}
        cfg_file = out_path / "config.yaml"
        with open(cfg_file, "w") as f:
            yaml.safe_dump(cfg_dict, f)
        print(f"Configuração salva em: {cfg_file}")
        run_now = questionary.confirm(
            "Iniciar treino e avaliação agora?"
        ).ask()
        if run_now:
            cfg_obj = ConfigModel(**cfg_dict)
            run_id = make_run_id("wizard")
            base_dir = Path(cfg_obj.data.out_dir or RESULTS_DIR) / run_id
            ensure_dirs(base_dir)
            train_dir = base_dir / "train"
            ensure_dirs(train_dir)
            try:
                metrics_path, figs_dir = train_and_save(
                    cfg_obj.dict(), train_dir, random_state=args.seed, mlflow_uri=mlflow_uri
                )
                if metrics_path:
                    _print_metrics_table(metrics_path)
                if figs_dir:
                    _print_confusion_paths(figs_dir)
                print(f"Treino em: {train_dir}")
                cv_dir = base_dir / "cv"
                ensure_dirs(cv_dir)
                outputs = run_multiple_eval(
                    cfg_obj.dict(), cv_dir, random_state=args.seed, mlflow_uri=mlflow_uri
                )
                for metrics_path, figs_dir in outputs:
                    _print_metrics_table(metrics_path)
                    _print_confusion_paths(figs_dir)
                print(f"Avaliação em: {cv_dir}")
            except DataError as e:
                print(f"Dados inválidos: {e}", file=sys.stderr)
            except ModelError as e:
                print(f"Modelo inválido: {e}", file=sys.stderr)
    elif args.cmd == "evaluate":
        try:
            evaluate_saved_model(Path(args.model), Path(args.data))
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
        except ModelError as e:
            print(f"Modelo inválido: {e}", file=sys.stderr)
    elif args.cmd == "predict":
        try:
            predict_with_model(Path(args.model), Path(args.data), Path(args.output))
        except DataError as e:
            print(f"Dados inválidos: {e}", file=sys.stderr)
        except ModelError as e:
            print(f"Modelo inválido: {e}", file=sys.stderr)
    elif args.cmd == "explain":
        explain_model(Path(args.model), Path(args.data), Path(args.output))
    elif args.cmd == "dashboard":
        run_dashboard(Path(args.results))

if __name__ == "__main__":
    main()
