from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cv_runner import evaluate_saved_model as _evaluate_saved_model
from .cv_runner import run_multiple_eval as _run_multiple_eval

__all__ = ["evaluate_saved_model", "run_multiple_eval"]


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
    """Executa as estratégias de validação cruzada configuradas."""

    return _run_multiple_eval(
        cfg,
        out_dir,
        random_state=random_state,
        interactive=interactive,
        n_jobs=n_jobs,
        mlflow_uri=mlflow_uri,
        log_memory=log_memory,
        resume=resume,
    )


def evaluate_saved_model(model_path: Path, data_path: Path) -> None:
    """Avalia um modelo salvo em disco com dados de referência."""

    return _evaluate_saved_model(model_path, data_path)
