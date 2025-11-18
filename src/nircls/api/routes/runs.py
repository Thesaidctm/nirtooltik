from __future__ import annotations

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ... import i18n
from ..deps import get_current_user, get_db, is_global_admin
from ..schemas import RunCreate, RunRead
from ...core.services import runs as run_service
from ...db import crud
from ...db.models import Run, User

router = APIRouter(prefix="/runs", tags=["runs"])

@router.get("", response_model=List[RunRead])
def list_all_runs(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> List[Run]:
    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    return crud.list_runs(db, organization_id=organization_id)


@router.get("/{run_id}", response_model=RunRead)
def get_run_detail(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Run:
    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    run = crud.get_run(db, run_id, organization_id=organization_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=i18n.t("runs.not_found_by_id").format(run_id=run_id),
        )
    return run


@router.post("", response_model=RunRead, status_code=status.HTTP_201_CREATED)
def create_new_run(
    payload: RunCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Run:
    run = run_service.create_run(
        db,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        user_id=current_user.user_id,
        organization_id=current_user.organization_id,
        config=payload.config,
        status=payload.status,
        artifacts=payload.artifacts,
        run_uid=payload.run_uid,
    )

    if payload.auto_execute:
        run = run_service.execute_run(db, run)

    return run


@router.get("/{run_id}/artifacts", response_model=Dict[str, Any])
def get_run_artifacts(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    run = crud.get_run(db, run_id, organization_id=organization_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=i18n.t("runs.not_found_by_id").format(run_id=run_id),
        )
    return run.artifacts or {}


@router.post("/{run_id}/execute", response_model=RunRead)
def execute_run_endpoint(
    run_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Executa a pipeline associada a este Run de forma síncrona.
    """

    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    run = crud.get_run(db, run_id, organization_id=organization_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=i18n.t("runs.not_found_by_id").format(run_id=run_id),
        )

    return run_service.execute_run(db, run)
