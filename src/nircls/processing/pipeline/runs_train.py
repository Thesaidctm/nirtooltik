from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ...db.models import Run
from .cv_runner import (
    _train_and_save_classification,
    _train_and_save_regression,
    run_experiment as _run_experiment,
    run_pipeline as _run_pipeline,
    train_and_save as _train_and_save,
)

__all__ = [
    "_train_and_save_classification",
    "_train_and_save_regression",
    "run_experiment",
    "run_pipeline",
    "train_and_save",
]


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
    """Treina e salva modelos finais reutilizando a implementação do runner."""

    return _train_and_save(
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
    """Executa o pipeline completo de treino e validação registrando o progresso."""

    return _run_pipeline(
        cfg,
        out_dir,
        db=db,
        run_id=run_id,
        run_uid=run_uid,
        user_id=user_id,
        dataset_id=dataset_id,
        project_id=project_id,
        interactive=interactive,
        random_state=random_state,
        n_jobs=n_jobs,
        mlflow_uri=mlflow_uri,
        gs_verbose=gs_verbose,
        show_progress=show_progress,
        log_memory=log_memory,
        resume=resume,
    )


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
    """Executa um experimento completo, incluindo treino final e validação."""

    return _run_experiment(
        cfg,
        out_dir,
        run_id=run_id,
        run_uid=run_uid,
        interactive=interactive,
        random_state=random_state,
        n_jobs=n_jobs,
        log_memory=log_memory,
    )
