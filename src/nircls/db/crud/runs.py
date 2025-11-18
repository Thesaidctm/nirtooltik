from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models import Run


def list_runs(session: Session) -> List[Run]:
    return session.query(Run).all()


def get_run(session: Session, run_id: int) -> Optional[Run]:
    return session.get(Run, run_id)


def create_run(
    session: Session,
    *,
    project_id: int | None,
    dataset_id: int | None,
    user_id: int | None,
    config: Dict[str, Any],
    status: str = "running",
    artifacts: Dict[str, Any] | None = None,
    error_message: str | None = None,
) -> Run:
    run = Run(
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=user_id,
        config=config,
        status=status,
        artifacts=artifacts,
        error_message=error_message,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def update_run_status(
    session: Session,
    run_id: int,
    *,
    status: str,
    artifacts: Dict[str, Any] | None = None,
    error_message: str | None = None,
) -> Optional[Run]:
    run = session.get(Run, run_id)
    if not run:
        return None
    run.status = status
    if artifacts is not None:
        run.artifacts = artifacts
    run.error_message = error_message
    session.commit()
    session.refresh(run)
    return run
