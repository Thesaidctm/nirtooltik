from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from nircls.db import crud
from nircls.db.models import Run


def _require_run(run: Optional[Run], run_id: int) -> Run:
    if run is None:
        raise ValueError(f"Run with id {run_id} not found")
    return run


def create_run(
    db: Session,
    *,
    project_id: Optional[int],
    dataset_id: Optional[int],
    user_id: Optional[int],
    config: Dict[str, Any],
    status: str = "running",
    artifacts: Optional[Dict[str, Any]] = None,
) -> Run:
    """Create a Run row in the database for a new pipeline execution."""

    return crud.create_run(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=user_id,
        config=config,
        status=status,
        artifacts=artifacts,
    )


def mark_run_completed(
    db: Session,
    run_id: int,
    artifacts: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> Run:
    """Mark a run as completed and store artifacts/metrics metadata."""

    combined_artifacts: Dict[str, Any] = dict(artifacts)
    if metrics is not None:
        combined_artifacts["metrics"] = metrics

    run = crud.update_run_status(
        db, run_id, status="completed", artifacts=combined_artifacts, error_message=None
    )
    return _require_run(run, run_id)


def mark_run_failed(
    db: Session,
    run_id: int,
    error_message: str,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Run:
    """Mark a run as failed and store the error message."""

    run = crud.update_run_status(
        db,
        run_id,
        status="failed",
        artifacts=artifacts,
        error_message=error_message,
    )
    return _require_run(run, run_id)
