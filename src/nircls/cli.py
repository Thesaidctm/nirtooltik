"""Command-line interface for the NIR species classification toolkit.

This module exposes the :func:`main` function which backs the ``nircls``
command-line utility for preparing data, training models, evaluating saved
models, making predictions and generating explanations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd
import yaml  # type: ignore[import-untyped]

from . import __version__
from .config_schema import ConfigModel, ValidationError
from .dashboard import run_dashboard
from .io import export_xy_xlsx, summarize_dataset
from .models.factory import get_model_builders, get_model_descriptions
from .pipeline import (
    evaluate_saved_model,
    explain_model,
    predict_with_model,
    prepare_inputs,
    run_multiple_eval,
    train_and_save,
)
from .utils.errors import DataError, ModelError
from .utils.logging import log_versions, setup_logging
from .utils.misc import RAW_DIR, RESULTS_DIR, ensure_dirs, make_run_id, now_ts, set_seed
from .utils.summarize import summarize_runs

# Human-friendly names for interactive wizard choices
MODEL_DESCRIPTIONS = get_model_descriptions()

PREPROCESS_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "SNV": {
        "en": "Standard Normal Variate",
        "pt": "Variação Normal Padrão",
    },
    "MSC": {
        "en": "Multiplicative Scatter Correction",
        "pt": "Correção Multiplicativa de Espalhamento",
    },
    "DETREND": {
        "en": "Polynomial Detrend",
        "pt": "Remoção de Tendência Polinomial",
    },
    "SG": {
        "en": "Savitzky-Golay Filter",
        "pt": "Filtro de Savitzky-Golay",
    },
    "EMSC": {
        "en": "Extended MSC",
        "pt": "MSC Estendido",
    },
    "EPO": {
        "en": "External Parameter Orthogonalization",
        "pt": "Ortogonalização de Parâmetro Externo",
    },
    "SCALE": {
        "en": "Standard Scaling",
        "pt": "Escalonamento Padrão",
    },
}

STRINGS = {
    "en": {
        "prog_desc": "NIR Species Classification Toolkit",
        "lang_help": "Interface language (pt or en)",
        "run_help": "Run only one or all analyses",
        "seed_help": "Seed for reproducibility",
        "mlflow_help": "MLflow server URI for tracking",
        "n_jobs_help": "Number of parallel jobs",
        "interactive_help": "Generate HTML files with interactive Plotly graphs",
        "astm_help": "Check ASTM classification criteria",
        "run_all_help": "Process raw data and run training + evaluation with defaults",
        "run_all_config_help": "Configuration file to override defaults",
        "run_all_models_help": "Use all available models (requires optional dependencies)",
        "run_all_preprocess_help": "Apply all preprocessing steps (SNV, MSC, DETREND, SG, EMSC, EPO, SCALE)",
        "dashboard_results_help": "Folder with results",
        "summary_help": "Summarize saved metrics including preprocess/result columns",
        "models_help": "List available model builders",
        "target_type_help": "Target task type (classification or regression)",
        "mlflow_not_installed": "'mlflow' is not installed. Install with `pip install mlflow`.",
        "model_dep_missing": "Skipping {name}: {err}",
        "config_not_found": "Configuration file not found. Check the provided path.",
        "invalid_config": "Invalid configuration: {err}. Review required fields.",
        "config_invalid_data_section": "Invalid configuration: 'data' section is required",
        "config_invalid_data_models_sections": "Invalid configuration: 'data' and 'models' sections are required",
        "config_invalid_data_models_cv_sections": "Invalid configuration: 'data', 'models' and 'cv' sections are required",  # noqa: E501
        "invalid_data": "Invalid data: {err}",
        "invalid_model": "Invalid model: {err}",
        "results_in": "Results in: {path}",
        "confusion": "Confusion matrix: {path}",
        "astm_min_samples": "≥ 30 samples",
        "astm_min_per_class": "≥ 10 per class",
        "astm_external_validation": "External validation",
        "full_report": "Full report: {path}",
        "model_saved": "Model saved: {path}",
        "final_model": "== Final model{steps} ==",
        "data_prepared_in": "Prepared data in: {file}",
        "model_trained_in": "Model trained in: {path}",
        "cv_results_in": "Cross-val results in: {path}",
        "wizard_data_path": "Data path:",
        "wizard_target_col": "Target column:",
        "wizard_target_required": "Provide target column",
        "wizard_file_not_found": "File not found",
        "wizard_models": "Models:",
        "wizard_models_select": "Select at least one model",
        "wizard_models_params": "Parameters for {name} (k=v, comma separated, optional):",
        "wizard_preprocessors": "Preprocessors:",
        "wizard_out_dir_prompt": "Directory to save config.yaml (default: .):",
        "wizard_config_saved": "Configuration saved at: {file}",
        "wizard_run_now": "Start training and evaluation now?",
        "wizard_train_in": "Training in: {path}",
        "wizard_eval_in": "Evaluation in: {path}",
        "questionary_not_installed": "'questionary' is not installed. Install with `pip install questionary`.",
    },
    "pt": {
        "prog_desc": "NIR Species Classification Toolkit",
        "lang_help": "Idioma da interface (pt ou en)",
        "run_help": "Rodar somente uma ou todas as análises",
        "seed_help": "Semente para reprodutibilidade",
        "mlflow_help": "URI do servidor MLflow para rastreamento",
        "n_jobs_help": "Número de jobs paralelos",
        "interactive_help": "Gerar arquivos HTML com gráficos Plotly interativos",
        "astm_help": "Verificar critérios ASTM para classificação",
        "run_all_help": "Processa dados brutos e executa treino+avaliação com padrões",
        "run_all_config_help": "Arquivo de configuração para sobrescrever parâmetros padrão",
        "run_all_models_help": "Usar todos os modelos disponíveis (requer dependências opcionais)",
        "run_all_preprocess_help": (
            "Aplicar todos os pré-processamentos disponíveis (SNV, MSC, DETREND, SG, "
            "EMSC, EPO, SCALE)"
        ),
        "dashboard_results_help": "Pasta com resultados",
        "summary_help": (
            "Resumir métricas salvas incluindo colunas de pré-processamento/resultado"
        ),
        "models_help": "Listar modelos disponíveis",
        "target_type_help": "Tipo de tarefa alvo (classificação ou regressão)",
        "mlflow_not_installed": "'mlflow' não está instalado. Instale com `pip install mlflow`.",
        "model_dep_missing": "Ignorando {name}: {err}",
        "config_not_found": "Arquivo de configuração não encontrado. Verifique o caminho informado.",
        "invalid_config": "Configuração inválida: {err}. Revise os campos obrigatórios.",
        "config_invalid_data_section": "Configuração inválida: seção 'data' é obrigatória",
        "config_invalid_data_models_sections": "Configuração inválida: seções 'data' e 'models' são obrigatórias",
        "config_invalid_data_models_cv_sections": "Configuração inválida: seções 'data', 'models' e 'cv' são obrigatórias",  # noqa: E501
        "invalid_data": "Dados inválidos: {err}",
        "invalid_model": "Modelo inválido: {err}",
        "results_in": "Resultados em: {path}",
        "confusion": "Matriz de confusão: {path}",
        "astm_min_samples": "≥ 30 amostras",
        "astm_min_per_class": "≥ 10 por classe",
        "astm_external_validation": "Validação externa",
        "full_report": "Relatório completo: {path}",
        "model_saved": "Modelo salvo: {path}",
        "final_model": "== Modelo final{steps} ==",
        "data_prepared_in": "Dados preparados em: {file}",
        "model_trained_in": "Modelo treinado em: {path}",
        "cv_results_in": "Resultados de cross-val em: {path}",
        "wizard_data_path": "Caminho dos dados:",
        "wizard_target_col": "Coluna alvo:",
        "wizard_target_required": "Informe a coluna alvo",
        "wizard_file_not_found": "Arquivo não encontrado",
        "wizard_models": "Modelos:",
        "wizard_models_select": "Selecione pelo menos um modelo",
        "wizard_models_params": "Parâmetros para {name} (k=v, separados por vírgula, opcional):",
        "wizard_preprocessors": "Pré-processadores:",
        "wizard_out_dir_prompt": "Diretório para salvar config.yaml (padrão: .):",
        "wizard_config_saved": "Configuração salva em: {file}",
        "wizard_run_now": "Iniciar treino e avaliação agora?",
        "wizard_train_in": "Treino em: {path}",
        "wizard_eval_in": "Avaliação em: {path}",
        "questionary_not_installed": "'questionary' não está instalado. Instale com `pip install questionary`.",
    },
}

LANG = "en"


def t(key: str, **kwargs: Any) -> str:
    return STRINGS.get(LANG, STRINGS["en"]).get(key, key).format(**kwargs)


def _get_preprocess_description(code: str, lang: str | None = None) -> str:
    """Return a human-friendly description for a preprocessing ``code``."""

    data = PREPROCESS_DESCRIPTIONS.get(code)
    if not data:
        return code
    language = lang or LANG
    if isinstance(data, dict):
        return data.get(language) or data.get("en") or next(iter(data.values()))
    return str(data)


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
            df[base] = df.apply(lambda r: f"{r[col]:.3f} ± {r[std_col]:.3f}", axis=1)
        else:
            df[base] = df[col].map(lambda x: f"{x:.3f}")
    display_cols = id_cols + [c[:-5] for c in metrics_cols]
    print(df[display_cols].to_string(index=False))


def _format_summary_steps(value: Any) -> str:
    """Return a user-friendly representation of ``value``."""

    items: list[str] = []

    def _collect_steps(step: Any) -> None:
        if isinstance(step, (list, tuple, set)):
            for inner in step:
                _collect_steps(inner)
            return
        if step is None:
            return
        try:
            if pd.isna(step):
                return
        except (TypeError, ValueError):
            pass
        step_str = str(step).strip()
        if step_str:
            items.append(step_str)

    _collect_steps(value)
    if not items:
        return ""
    descriptions = _describe_preprocess_steps(items)
    if descriptions:
        return ", ".join(descriptions)
    return ", ".join(items)


def _format_summary_result(value: Any) -> str:
    """Render the ``result`` value as a readable string."""

    items: list[str] = []

    def _collect(entry: Any) -> None:
        if isinstance(entry, (list, tuple, set)):
            for inner in entry:
                _collect(inner)
            return
        if entry is None:
            return
        try:
            if pd.isna(entry):
                return
        except (TypeError, ValueError):
            pass
        entry_str = str(entry).strip()
        if entry_str:
            items.append(entry_str)

    _collect(value)
    if not items:
        return ""
    return ", ".join(items)


def _prepare_summary_display(
    summary: pd.DataFrame, ordered_cols: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return a formatted copy of ``summary`` for console display."""

    if summary.empty:
        return summary
    df = summary.copy()
    if "preprocess_steps" in df.columns:
        df["preprocess_steps"] = df["preprocess_steps"].map(_format_summary_steps)
    if "result" in df.columns:
        df["result"] = df["result"].map(_format_summary_result)
    if ordered_cols:
        preferred = [col for col in ordered_cols if col in df.columns]
        remaining = [col for col in df.columns if col not in preferred]
        if remaining:
            return df[preferred + remaining]
        if preferred:
            return df[preferred]
    return df


def _print_dataset_summary(summary: pd.DataFrame) -> None:
    """Print ``summary`` with friendly preprocessing/result columns."""

    if summary.empty:
        return
    ordered_cols = [
        "target",
        "class",
        "count",
        "spectral_min",
        "spectral_max",
        "n_nans",
        "preprocess_steps",
        "result",
    ]
    df = _prepare_summary_display(summary, ordered_cols=ordered_cols)
    with pd.option_context("display.max_colwidth", None):
        print(df.to_string(index=False))


def _load_preprocess_steps(metrics_json: Path) -> list[str]:
    """Return ordered preprocessing steps found in ``metrics_json``."""

    if not metrics_json or not metrics_json.exists():
        return []
    try:
        df = pd.read_json(metrics_json)
    except ValueError:
        return []
    if df.empty or "preprocess_steps" not in df.columns:
        return []

    seen = set()
    ordered_steps: list[str] = []

    def _add_step(step: Any) -> None:
        if step is None:
            return
        try:
            if pd.isna(step):
                return
        except TypeError:
            pass
        step_str = str(step)
        if step_str not in seen:
            seen.add(step_str)
            ordered_steps.append(step_str)

    for val in df["preprocess_steps"]:
        if isinstance(val, (list, tuple)):
            for item in val:
                _add_step(item)
        else:
            _add_step(val)

    return ordered_steps


def _describe_preprocess_steps(steps: Sequence[str]) -> list[str]:
    """Convert raw ``steps`` identifiers into human-friendly descriptions."""

    pretty_steps: list[str] = []
    for raw_step in steps:
        step_str = str(raw_step).strip()
        if not step_str:
            continue
        base = step_str
        params = ""
        if "(" in step_str and step_str.endswith(")"):
            base, params = step_str.split("(", 1)
            params = "(" + params
        base_code = base.strip().upper()
        desc = _get_preprocess_description(base_code)
        if desc == base_code:
            desc = base.strip()
        if params:
            desc = f"{desc} {params}".strip()
        pretty_steps.append(desc)
    return pretty_steps


def _print_confusion_paths(fig_dir: Path) -> None:
    """Print paths of confusion matrix images if they exist."""

    if not fig_dir or not fig_dir.exists():
        return
    for p in sorted(fig_dir.glob("*confusion.png")):
        print(t("confusion", path=p))


def _print_astm_report(astm_json: Path) -> None:
    """Print ASTM compliance flags if report exists."""

    if not astm_json or not astm_json.exists():
        return
    try:
        flags = json.loads(astm_json.read_text())
    except Exception:
        return
    labels = {
        "min_samples": t("astm_min_samples"),
        "min_per_class": t("astm_min_per_class"),
        "external_validation": t("astm_external_validation"),
    }
    for key, label in labels.items():
        val = flags.get(key)
        mark = "✔" if val else "✖"
        print(f"{mark} {label}")


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``nircls`` command-line interface.

    Parameters
    ----------
    argv : list[str] | None
        Optional list of arguments. When ``None`` (default), ``sys.argv`` is
        used.
    """

    argv = list(argv) if argv is not None else sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang", choices=["pt", "en"], default="en")
    ns, argv = pre.parse_known_args(argv)
    global LANG
    LANG = ns.lang
    parser = argparse.ArgumentParser(prog="nircls", description=t("prog_desc"))
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--lang", choices=["pt", "en"], default=LANG, help=t("lang_help")
    )
    parser.add_argument(
        "--run", choices=["one", "all"], default="all", help=t("run_help")
    )
    parser.add_argument("--seed", type=int, default=42, help=t("seed_help"))
    parser.add_argument("--mlflow", help=t("mlflow_help"))
    parser.add_argument("--n-jobs", type=int, default=1, help=t("n_jobs_help"))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file (overrides default)",
    )
    parser.add_argument(
        "--cv-verbose",
        type=int,
        default=0,
        help="GridSearchCV verbosity",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log memory usage during processing",
    )
    parser.add_argument(
        "--notification-email",
        help="Email address to receive notifications",
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    p_prep = sp.add_parser("prepare-data")
    p_prep.add_argument("--config", required=True)
    p_prep.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    p_tr = sp.add_parser("train")
    p_tr.add_argument("--config", required=True)
    p_tr.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    p_tr.add_argument(
        "--interactive",
        action="store_true",
        help=t("interactive_help"),
    )
    p_tr.add_argument(
        "--astm",
        action="store_true",
        help=t("astm_help"),
    )
    p_cv = sp.add_parser("crossval")
    p_cv.add_argument("--config", required=True)
    p_cv.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    p_cv.add_argument(
        "--interactive",
        action="store_true",
        help=t("interactive_help"),
    )
    p_cv.add_argument(
        "--astm",
        action="store_true",
        help=t("astm_help"),
    )
    sp.add_parser("demo")
    p_list = sp.add_parser("list-models", help=t("models_help"))
    p_list.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        default="classification",
        help=t("target_type_help"),
    )
    p_all = sp.add_parser("run-all", help=t("run_all_help"))
    p_all.add_argument("--config", help=t("run_all_config_help"))
    p_all.add_argument(
        "--target-type",
        choices=["classification", "regression"],
        help=t("target_type_help"),
    )
    p_all.add_argument(
        "--all-preprocess",
        action="store_true",
        help=t("run_all_preprocess_help"),
    )
    p_all.add_argument(
        "--all-models",
        action="store_true",
        help=t("run_all_models_help"),
    )
    sp.add_parser("wizard")
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
    p_db.add_argument("--results", required=True, help=t("dashboard_results_help"))
    p_sum = sp.add_parser("summary", help=t("summary_help"))
    p_sum.add_argument("--results", required=True, help=t("dashboard_results_help"))
    args = parser.parse_args(argv)
    set_seed(args.seed)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    if args.log_file:
        ensure_dirs(args.log_file.parent)
    setup_logging(level=log_level, log_file=args.log_file)
    log_versions()

    def _setup_run_logging(run_id: str) -> Path:
        log_path = args.log_file or RESULTS_DIR / run_id / "nircls.log"
        ensure_dirs(log_path.parent)
        setup_logging(level=log_level, log_file=log_path)
        return log_path

    n_jobs = max(1, args.n_jobs)
    cv_verbose = args.cv_verbose
    mlflow_uri = args.mlflow
    if mlflow_uri:
        try:
            import mlflow
        except Exception:
            print(t("mlflow_not_installed"), file=sys.stderr)
            return 1
        mlflow.set_tracking_uri(mlflow_uri)
    if args.cmd == "list-models":
        target_type = getattr(args, "target_type", "classification")
        builders = get_model_builders(target_type)
        descs = get_model_descriptions(target_type)
        for name in sorted(builders.keys()):
            print(f"{name}: {descs.get(name, name)}")
        return 0
    if args.cmd in {"prepare-data", "train", "crossval"}:
        try:
            with open(args.config) as f:
                loaded_cfg = yaml.safe_load(f)
        except FileNotFoundError:
            print(t("config_not_found"), file=sys.stderr)
            return 1
        raw = cast(Dict[str, Any], loaded_cfg or {})
        if not isinstance(raw, dict):
            print(
                t("invalid_config", err="estrutura inválida"),
                file=sys.stderr,
            )
            return 1
        if args.notification_email:
            raw["notification_email"] = args.notification_email
        if getattr(args, "target_type", None):
            raw.setdefault("data", {})["target_type"] = args.target_type
        try:
            cfg = ConfigModel(**raw)
        except ValidationError as e:  # pragma: no cover - user feedback
            print(t("invalid_config", err=e), file=sys.stderr)
            return 1

    if args.cmd == "prepare-data":
        if cfg.data is None:
            print(t("config_invalid_data_section"), file=sys.stderr)
            return 1
        run_id = make_run_id("prepare")
        _setup_run_logging(run_id)
        out_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(out_dir)
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        _print_dataset_summary(summary)
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
            print(t("config_invalid_data_models_sections"), file=sys.stderr)
            return 1
        cfg.astm = args.astm
        run_id = make_run_id("train")
        _setup_run_logging(run_id)
        out_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(out_dir)
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        _print_dataset_summary(summary)
        try:
            metrics_path, figs_dir = train_and_save(
                cfg.dict(exclude_defaults=True, exclude_none=True),
                out_dir,
                interactive=args.interactive,
                random_state=args.seed,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                gs_verbose=cv_verbose,
                show_progress=not args.no_progress,
                log_memory=args.log_memory,
            )
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
            return 1
        except ModelError as e:
            print(t("invalid_model", err=e), file=sys.stderr)
            return 1
        if metrics_path:
            _print_metrics_table(metrics_path)
        if figs_dir:
            _print_confusion_paths(figs_dir)
        report_path = out_dir / "report.md"
        if report_path.exists():
            print(t("full_report", path=report_path))
        if args.astm:
            _print_astm_report(out_dir / "astm_classification.json")
        print(t("results_in", path=out_dir))
    elif args.cmd == "crossval":
        if cfg.data is None or not cfg.models or cfg.cv is None:
            print(t("config_invalid_data_models_cv_sections"), file=sys.stderr)
            return 1
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        _print_dataset_summary(summary)
        if args.run == "one" and ds_list:
            cfg.data.target_cols = ds_list[0]["target"]
        run_id = make_run_id("cv")
        _setup_run_logging(run_id)
        out_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(out_dir)
        cfg.astm = args.astm
        try:
            outputs = run_multiple_eval(
                cfg.dict(exclude_defaults=True, exclude_none=True),
                out_dir,
                interactive=args.interactive,
                random_state=args.seed,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                log_memory=args.log_memory,
            )
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
            return 1
        except ModelError as e:
            print(t("invalid_model", err=e), file=sys.stderr)
            return 1
        for metrics_path, figs_dir in outputs:
            step_descs = _describe_preprocess_steps(
                _load_preprocess_steps(metrics_path)
            )
            if step_descs:
                header = f"== {metrics_path.parent.name} ({', '.join(step_descs)}) =="
            else:
                header = f"== {metrics_path.parent.name} =="
            print(header)
            _print_metrics_table(metrics_path)
            _print_confusion_paths(figs_dir)
            rpt = metrics_path.parent / "report.md"
            if rpt.exists():
                print(t("full_report", path=rpt))
            if args.astm:
                _print_astm_report(metrics_path.parent / "astm_classification.json")
        print(t("results_in", path=out_dir))
    elif args.cmd == "run-all":
        builders = get_model_builders()
        default_cfg: Dict[str, Any] = {
            "data": {"input_path": str(RAW_DIR), "out_dir": str(RESULTS_DIR)},
            "models": [{"name": "PLSDA"}],
            "cv": {"strategies": [{"type": "kfold", "n_splits": 5}]},
        }
        if args.config:
            try:
                with open(args.config) as f:
                    user_cfg = cast(Dict[str, Any], yaml.safe_load(f) or {})
            except FileNotFoundError:
                print(t("config_not_found"), file=sys.stderr)
                return 1
            _deep_update(default_cfg, user_cfg)
        if args.target_type:
            default_cfg.setdefault("data", {})["target_type"] = args.target_type
        if args.all_preprocess:
            default_cfg["preprocess"] = {"steps": list(PREPROCESS_DESCRIPTIONS)}
        if args.all_models:
            models_cfg: list[dict[str, Any]] = []
            for name, builder in builders.items():
                try:
                    builder({})
                except ImportError as e:
                    print(t("model_dep_missing", name=name, err=e), file=sys.stderr)
                    continue
                models_cfg.append({"name": name})
            default_cfg["models"] = models_cfg
        if args.notification_email:
            default_cfg["notification_email"] = args.notification_email
        run_id = now_ts()
        _setup_run_logging(run_id)
        data_cfg = cast(Dict[str, Any], default_cfg["data"])
        base_dir = Path(data_cfg.get("out_dir", RESULTS_DIR)) / run_id
        ensure_dirs(base_dir)
        data_cfg["out_dir"] = str(base_dir)
        try:
            ds_list = prepare_inputs(
                data_cfg, use_cache=bool(default_cfg.get("use_cache", True))
            )
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
            return 1
        summary = summarize_dataset(ds_list)
        _print_dataset_summary(summary)
        if args.run == "one" and ds_list:
            data_cfg["target_cols"] = ds_list[0]["target"]
        train_dir = base_dir / "train"
        cv_dir = base_dir / "cv"
        ensure_dirs(train_dir, cv_dir)
        try:
            train_metrics, train_figs = train_and_save(
                default_cfg,
                train_dir,
                random_state=args.seed,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                gs_verbose=cv_verbose,
                show_progress=not args.no_progress,
                log_memory=args.log_memory,
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
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                log_memory=args.log_memory,
            )
        except (DataError, ModelError) as e:
            print(str(e), file=sys.stderr)
            return 1
        for cv_metrics, cv_figs in outputs:
            step_descs = _describe_preprocess_steps(
                _load_preprocess_steps(cv_metrics)
            )
            if step_descs:
                header = f"== {cv_metrics.parent.name} ({', '.join(step_descs)}) =="
            else:
                header = f"== {cv_metrics.parent.name} =="
            print(header)
            _print_metrics_table(cv_metrics)
            _print_confusion_paths(cv_figs)
            rpt = cv_metrics.parent / "report.md"
            if rpt.exists():
                print(t("full_report", path=rpt))
        if train_metrics:
            steps_list = _load_preprocess_steps(train_metrics)
            suffix = ""
            step_descs = _describe_preprocess_steps(steps_list)
            if step_descs:
                suffix = f" ({', '.join(step_descs)})"
            print(t("final_model", steps=suffix))
            _print_metrics_table(train_metrics)
            _print_confusion_paths(train_figs)
            if report_path.exists():
                print(t("full_report", path=report_path))
            if model_path.exists():
                print(t("model_saved", path=model_path))
        print(t("results_in", path=base_dir))
        return 0
    elif args.cmd == "demo":
        demo_cfg = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"
        with open(demo_cfg) as f:
            raw = yaml.safe_load(f)
        raw["data"]["input_path"] = str(
            (demo_cfg.parent.parent / raw["data"]["input_path"]).resolve()
        )
        if args.notification_email:
            raw["notification_email"] = args.notification_email
        try:
            cfg = ConfigModel(**raw)
        except ValidationError as e:  # pragma: no cover - user feedback
            print(t("demo_config_invalid", err=e), file=sys.stderr)
            return 1
        run_id = make_run_id("demo")
        _setup_run_logging(run_id)
        base_dir = Path(cfg.data.out_dir or RESULTS_DIR) / run_id
        ensure_dirs(base_dir)
        ds_list = []
        try:
            ds_list = prepare_inputs(cfg.data.dict(), use_cache=cfg.use_cache)
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
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
            print(t("data_prepared_in", file=prep_file))
        train_dir = base_dir / "train"
        ensure_dirs(train_dir)
        try:
            metrics_path, figs_dir = train_and_save(
                cfg.dict(exclude_defaults=True, exclude_none=True),
                train_dir,
                random_state=args.seed,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                gs_verbose=cv_verbose,
                show_progress=not args.no_progress,
                log_memory=args.log_memory,
            )
            if metrics_path:
                _print_metrics_table(metrics_path)
            if figs_dir:
                _print_confusion_paths(figs_dir)
            print(t("model_trained_in", path=train_dir))
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
        except ModelError as e:
            print(t("invalid_model", err=e), file=sys.stderr)
        cv_dir = base_dir / "cv"
        ensure_dirs(cv_dir)
        try:
            outputs = run_multiple_eval(
                cfg.dict(exclude_defaults=True, exclude_none=True),
                cv_dir,
                random_state=args.seed,
                n_jobs=n_jobs,
                mlflow_uri=mlflow_uri,
                log_memory=args.log_memory,
            )
            for metrics_path, figs_dir in outputs:
                _print_metrics_table(metrics_path)
                _print_confusion_paths(figs_dir)
            print(t("cv_results_in", path=cv_dir))
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
        except ModelError as e:
            print(t("invalid_model", err=e), file=sys.stderr)
    elif args.cmd == "wizard":
        try:  # pragma: no cover - interactive session
            import questionary
            from questionary import Choice
        except Exception:  # pragma: no cover - missing optional dependency
            print(t("questionary_not_installed"), file=sys.stderr)
            return 1

        from typing import Union

        def _validate_path(p: str) -> Union[bool, str]:
            return Path(p).exists() or t("wizard_file_not_found")

        input_path = questionary.text(
            t("wizard_data_path"), validate=_validate_path
        ).ask()
        target = questionary.text(
            t("wizard_target_col"),
            validate=lambda t: bool(t) or t("wizard_target_required"),
        ).ask()

        builders = get_model_builders()
        model_choices = [
            Choice(title=f"{k} - {MODEL_DESCRIPTIONS.get(k, k)}", value=k)
            for k in builders.keys()
        ]
        selected_models = questionary.checkbox(
            t("wizard_models"),
            choices=model_choices,
            validate=lambda m: bool(m) or t("wizard_models_select"),
        ).ask()

        models_cfg = []
        for name in selected_models:
            params_raw = questionary.text(t("wizard_models_params", name=name)).ask()
            model_entry: dict[str, Any] = {"name": name}
            if params_raw:
                params = _parse_params(params_raw)
                if params:
                    model_entry["params"] = params
            models_cfg.append(model_entry)

        pre_choices = [
            Choice(title=f"{k} - {_get_preprocess_description(k)}", value=k)
            for k in PREPROCESS_DESCRIPTIONS
        ]
        preprocess_steps = questionary.checkbox(
            t("wizard_preprocessors"), choices=pre_choices
        ).ask()

        out_dir_str = (
            questionary.text(t("wizard_out_dir_prompt"), default=".").ask() or "."
        )
        notification_email = questionary.text("Notification email (optional):").ask()
        out_path = Path(out_dir_str)
        out_path.mkdir(parents=True, exist_ok=True)
        cfg_dict = {
            "data": {"input_path": input_path, "target_cols": target},
            "models": models_cfg,
            "cv": {"strategies": [{"type": "holdout", "test_size": 0.2}]},
        }
        if preprocess_steps:
            cfg_dict["preprocess"] = {"steps": preprocess_steps}
        if notification_email:
            cfg_dict["notification_email"] = notification_email
        cfg_file = out_path / "config.yaml"
        with open(cfg_file, "w") as f:
            yaml.safe_dump(cfg_dict, f)
        print(t("wizard_config_saved", file=cfg_file))
        run_now = questionary.confirm(t("wizard_run_now")).ask()
        if run_now:
            cfg_obj = ConfigModel(**cfg_dict)
            run_id = make_run_id("wizard")
            _setup_run_logging(run_id)
            base_dir = Path(cfg_obj.data.out_dir or RESULTS_DIR) / run_id
            ensure_dirs(base_dir)
            train_dir = base_dir / "train"
            ensure_dirs(train_dir)
            try:
                metrics_path, figs_dir = train_and_save(
                    cfg_obj.dict(),
                    train_dir,
                    random_state=args.seed,
                    n_jobs=n_jobs,
                    mlflow_uri=mlflow_uri,
                    gs_verbose=cv_verbose,
                    show_progress=not args.no_progress,
                    log_memory=args.log_memory,
                )
                if metrics_path:
                    _print_metrics_table(metrics_path)
                if figs_dir:
                    _print_confusion_paths(figs_dir)
                print(t("wizard_train_in", path=train_dir))
                cv_dir = base_dir / "cv"
                ensure_dirs(cv_dir)
                outputs = run_multiple_eval(
                    cfg_obj.dict(),
                    cv_dir,
                    random_state=args.seed,
                    n_jobs=n_jobs,
                    mlflow_uri=mlflow_uri,
                    log_memory=args.log_memory,
                )
                for metrics_path, figs_dir in outputs:
                    _print_metrics_table(metrics_path)
                    _print_confusion_paths(figs_dir)
                print(t("wizard_eval_in", path=cv_dir))
            except DataError as e:
                print(t("invalid_data", err=e), file=sys.stderr)
            except ModelError as e:
                print(t("invalid_model", err=e), file=sys.stderr)
    elif args.cmd == "evaluate":
        try:
            evaluate_saved_model(Path(args.model), Path(args.data))
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
        except ModelError as e:
            print(t("invalid_model", err=e), file=sys.stderr)
    elif args.cmd == "predict":
        try:
            predict_with_model(Path(args.model), Path(args.data), Path(args.output))
        except DataError as e:
            print(t("invalid_data", err=e), file=sys.stderr)
        except ModelError as e:
            print(t("invalid_model", err=e), file=sys.stderr)
    elif args.cmd == "explain":
        explain_model(Path(args.model), Path(args.data), Path(args.output))
    elif args.cmd == "summary":
        df = summarize_runs(Path(args.results))
        if not df.empty:
            ordered_cols = [
                "run",
                "target",
                "model",
                "preprocess_steps",
                "result",
            ]
            df = _prepare_summary_display(df, ordered_cols=ordered_cols)
            with pd.option_context(
                "display.max_columns", None, "display.max_colwidth", None
            ):
                print(df.to_string(index=False))
    elif args.cmd == "dashboard":
        run_dashboard(Path(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
