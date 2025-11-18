from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models import Run


def list_runs(session: Session, *, organization_id: Optional[int] = None) -> List[Run]:
    query = session.query(Run)
    if organization_id is not None:
        query = query.filter(Run.organization_id == organization_id)
    return query.all()


def get_run(session: Session, run_id: int, *, organization_id: Optional[int] = None) -> Optional[Run]:
    run = session.get(Run, run_id)
    if organization_id is not None and run and run.organization_id != organization_id:
        return None
    return run


def get_run_by_uid(session: Session, run_uid: str) -> Optional[Run]:
    return session.query(Run).filter(Run.run_uid == run_uid).one_or_none()


def create_run(
    session: Session,
    *,
    project_id: int | None,
    dataset_id: int | None,
    user_id: int | None,
    run_uid: str | None,
    config: Dict[str, Any],
    status: str = "running",
    artifacts: Dict[str, Any] | None = None,
    error_message: str | None = None,
    organization_id: int | None = None,
) -> Run:
    run = Run(
        project_id=project_id,
        dataset_id=dataset_id,
        user_id=user_id,
        run_uid=run_uid,
        config=config,
        status=status,
        artifacts=artifacts,
        error_message=error_message,
        organization_id=organization_id,
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
