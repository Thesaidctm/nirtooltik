from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...auth import get_current_user, get_db
from ...db.crud import create_run, get_run, list_runs
from ...db.models import Run, User

router = APIRouter(prefix="/runs", tags=["runs"])


class RunCreate(BaseModel):
    project_id: Optional[int] = None
    dataset_id: Optional[int] = None
    config: Dict[str, Any]
    status: Optional[str] = "running"
    artifacts: Optional[Dict[str, Any]] = None


class RunResponse(RunCreate):
    run_id: int
    user_id: Optional[int] = None
    error_message: Optional[str] = None

    class Config:
        orm_mode = True


@router.get("", response_model=List[RunResponse])
def list_all_runs(db: Session = Depends(get_db)) -> List[Run]:
    return list_runs(db)


@router.get("/{run_id}", response_model=RunResponse)
def get_run_detail(run_id: int, db: Session = Depends(get_db)) -> Run:
    run = get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


@router.post("", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_new_run(
    payload: RunCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Run:
    return create_run(
        db,
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        user_id=current_user.user_id,
        config=payload.config,
        status=payload.status or "running",
        artifacts=payload.artifacts,
    )
