from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..schemas import ModelCreate, ModelRead
from ...db import crud
from ...db.models import ModelRegistry, User

router = APIRouter()


@router.get("", response_model=List[ModelRead])
def list_all_models(db: Session = Depends(get_db)) -> List[ModelRegistry]:
    return crud.list_models(db)


@router.post("", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
def create_model_entry(
    payload: ModelCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelRegistry:  # noqa: ARG001
    return crud.create_model(
        db,
        name=payload.name,
        run_id=payload.run_id,
        dataset_id=payload.dataset_id,
        target=payload.target,
        status=payload.status,
    )


@router.post("/{model_id}/promote", response_model=ModelRead)
def promote_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelRegistry:  # noqa: ARG001
    model = crud.get_model(db, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    return crud.update_model_status(db, model, status="production")


@router.post("/{model_id}/deprecate", response_model=ModelRead)
def deprecate_model(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ModelRegistry:  # noqa: ARG001
    model = crud.get_model(db, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    return crud.update_model_status(db, model, status="deprecated")
