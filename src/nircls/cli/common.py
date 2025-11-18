from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd
import yaml  # type: ignore[import-untyped]

from .. import __version__
from ..config_schema import ConfigModel, ValidationError
from ..utils.misc import RESULTS_DIR, ensure_dirs
from ..utils.logging import setup_logging

LANG = "en"


# Human-friendly names for interactive wizard choices
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
    "AQS": {
        "en": "Adaptive Quantile Segment Normalization",
        "pt": "Normalização Adaptativa por Segmentos Quantílicos",
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
        "run_all_preprocess_help": "Apply all preprocessing steps (SNV, MSC, DETREND, SG, EMSC, EPO, AQS, SCALE)",
        "run_all_output_name_help": "Rename final results directory after completion",
        "resume_help": "Resume previously started runs when outputs exist",
        "run_id_help": "Existing run identifier to reuse (requires --resume)",
        "resume_requires_run_id": "--run-id must be provided when using --resume.",
        "resume_from": "Resuming from existing directory: {path}",
        "resume_dir_missing": "Cannot resume: directory not found: {path}",
        "run_all_invalid_output_name": (
            "Output name must contain letters, numbers, '.', '_' or '-' and cannot be empty."
        ),
        "run_all_output_exists": "Cannot rename results: destination already exists: {path}",
        "run_all_train_failed": "Training failed: {err}",
        "run_all_cv_failed": "Evaluation failed: {err}",
        "run_all_resume_hint": (
            "Fix the issue and resume with: nircls run-all --resume --run-id {run_id}"
        ),
        "run_all_results_renamed": "Results directory renamed to: {path}",
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
        "config_invalid_data_models_cv_sections": "Invalid configuration: 'data', 'models' and 'cv' sections are required",
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
        "dashboard_dep_missing": (
            "Missing optional dependency '{package}'. Install with `pip install "
            "nircls[dashboard]` or `pip install \"nircls[full]\"`."
        ),
        "demo_config_invalid": "Invalid demo configuration: {err}",
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
        "run_all_preprocess_help": "Aplicar todos os pré-processamentos (SNV, MSC, DETREND, SG, EMSC, EPO, AQS, SCALE)",
        "run_all_output_name_help": "Renomear diretório final de resultados ao concluir",
        "resume_help": "Retomar execuções anteriores quando já existir saída",
        "run_id_help": "Identificador de execução existente para reutilizar (requer --resume)",
        "resume_requires_run_id": "--run-id deve ser fornecido ao usar --resume.",
        "resume_from": "Retomando do diretório existente: {path}",
        "resume_dir_missing": "Não é possível retomar: diretório não encontrado: {path}",
        "run_all_invalid_output_name": (
            "Nome de saída deve conter letras, números, '.', '_' ou '-' e não pode ser vazio."
        ),
        "run_all_output_exists": "Não é possível renomear resultados: destino já existe: {path}",
        "run_all_train_failed": "Treino falhou: {err}",
        "run_all_cv_failed": "Avaliação falhou: {err}",
        "run_all_resume_hint": (
            "Corrija o problema e retome com: nircls run-all --resume --run-id {run_id}"
        ),
        "run_all_results_renamed": "Diretório de resultados renomeado para: {path}",
        "dashboard_results_help": "Pasta com resultados",
        "summary_help": "Sumariza métricas salvas incluindo colunas de preprocess/result",
        "models_help": "Lista modelos disponíveis",
        "target_type_help": "Tipo de tarefa alvo (classification ou regression)",
        "mlflow_not_installed": "'mlflow' não está instalado. Instale com `pip install mlflow`.",
        "model_dep_missing": "Ignorando {name}: {err}",
        "config_not_found": "Arquivo de configuração não encontrado. Verifique o caminho.",
        "invalid_config": "Configuração inválida: {err}. Revise os campos obrigatórios.",
        "config_invalid_data_section": "Configuração inválida: seção 'data' é obrigatória",
        "config_invalid_data_models_sections": "Configuração inválida: seções 'data' e 'models' são obrigatórias",
        "config_invalid_data_models_cv_sections": "Configuração inválida: seções 'data', 'models' e 'cv' são obrigatórias",
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
        "dashboard_dep_missing": (
            "Dependência opcional '{package}' ausente. Instale com `pip install "
            "nircls[dashboard]` ou `pip install \"nircls[full]\"`."
        ),
        "demo_config_invalid": "Configuração de demonstração inválida: {err}",
    },
}


def set_language(lang: str) -> None:
    global LANG
    LANG = lang


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


def _sanitize_output_name(name: str) -> str:
    """Return a filesystem-friendly name based on ``name``."""

    cleaned = re.sub(r"[^\w.-]+", "_", name.strip())
    return cleaned.strip("._")


def _write_run_state(base_dir: Path, state: Dict[str, Any]) -> None:
    """Persist run state metadata under ``base_dir``."""

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        with open(base_dir / "run_state.json", "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        logging.getLogger(__name__).warning("Could not persist run_state.json", exc_info=True)


def _load_run_state(base_dir: Path) -> Optional[Dict[str, Any]]:
    """Load previously persisted run state if available."""

    state_file = base_dir / "run_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            return cast(Dict[str, Any], loaded)
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not read run_state.json from %s", state_file, exc_info=True
        )
    return None


def load_config(
    config_path: str,
    target_type: str | None = None,
    notification_email: str | None = None,
) -> ConfigModel | None:
    """Load and validate a configuration file returning a ``ConfigModel``."""

    try:
        with open(config_path) as f:
            loaded_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(t("config_not_found"), file=sys.stderr)
        return None
    raw = cast(Dict[str, Any], loaded_cfg or {})
    if not isinstance(raw, dict):
        print(t("invalid_config", err="estrutura inválida"), file=sys.stderr)
        return None
    if notification_email:
        raw["notification_email"] = notification_email
    if target_type:
        raw.setdefault("data", {})["target_type"] = target_type
    try:
        return ConfigModel(**raw)
    except ValidationError as e:  # pragma: no cover - user feedback
        print(t("invalid_config", err=e), file=sys.stderr)
        return None


def setup_run_logging(run_id: str, log_level: int, log_file: Path | None) -> Path:
    """Configure run-specific logging and return the log path."""

    log_path = log_file or RESULTS_DIR / run_id / "nircls.log"
    ensure_dirs(log_path.parent)
    setup_logging(level=log_level, log_file=log_path)
    return log_path


__all__ = [
    "LANG",
    "PREPROCESS_DESCRIPTIONS",
    "STRINGS",
    "__version__",
    "set_language",
    "t",
    "_get_preprocess_description",
    "_print_metrics_table",
    "_format_summary_steps",
    "_format_summary_result",
    "_prepare_summary_display",
    "_print_dataset_summary",
    "_load_preprocess_steps",
    "_describe_preprocess_steps",
    "_print_confusion_paths",
    "_print_astm_report",
    "_parse_params",
    "_deep_update",
    "_sanitize_output_name",
    "_write_run_state",
    "_load_run_state",
    "load_config",
    "setup_run_logging",
    "RESULTS_DIR",
]
