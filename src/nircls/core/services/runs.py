from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
    run_uid: str | None = None,
) -> Run:
    """Create a Run row in the database for a new pipeline execution."""

    resolved_run_uid = run_uid or str(uuid.uuid4())
    return crud.create_run(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=user_id,
        run_uid=resolved_run_uid,
        config=config,
        status=status,
        artifacts=artifacts,
    )


def mark_run_completed(
    db: Session,
    run_id: int,
    *,
    base_dir: Path | str,
    train_dir: Path | str,
    cv_dir: Path | str,
    summary_dir: Path | str | None = None,
    models: Iterable[Path | str] | None = None,
    plots: Iterable[Path | str] | None = None,
    reports: Iterable[Path | str] | None = None,
    metrics: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Run:
    """Mark a run as completed and store artifacts/metrics metadata.

    The ``artifacts`` column follows a predictable manifest structure to make
    it easy to locate generated outputs and debug runs:

    ``{
        "base_dir": "...",  # root directory used by the pipeline
        "models": ["models/best_model.joblib"],
        "plots": ["plots/roc_curve.png", "plots/vip.png"],
        "reports": ["reports/summary.html"],
        "metrics": {"summary_csv": "metrics/metrics.csv"},
        "extras": {...}
    }``
    """

    manifest: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "models": [str(Path(p)) for p in (models or [])],
        "plots": [str(Path(p)) for p in (plots or [])],
        "reports": [str(Path(p)) for p in (reports or [])],
        "extras": extras or {},
    }
    manifest["metrics"] = metrics or {}
    manifest.setdefault("directories", {})["train"] = str(Path(train_dir))
    manifest.setdefault("directories", {})["cross_validation"] = str(Path(cv_dir))
    if summary_dir is not None:
        manifest["directories"]["summary"] = str(Path(summary_dir))

    run = crud.update_run_status(
        db, run_id, status="completed", artifacts=manifest, error_message=None
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
