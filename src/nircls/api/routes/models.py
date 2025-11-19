from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..schemas import ModelCreate, ModelRead, PredictRequest, PredictResponse
from ...core import services
from ...db import crud
from ...db.models import ModelRegistry, User

router = APIRouter()


class ModelEvalRequest(BaseModel):
    dataset_id: int
    target_index: int = 0


class ModelEvalResponse(BaseModel):
    dataset_id: int
    metrics: Dict[str, float]


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
        instrument_id=payload.instrument_id,
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


@router.post("/{model_id}/predict", response_model=PredictResponse)
def predict_with_registered_model(
    model_id: int,
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    y_hat = services.models.predict_with_model(db, model_id=model_id, spectra=req.spectra)
    return PredictResponse(y_hat=y_hat)


@router.post("/{model_id}/evaluate", response_model=ModelEvalResponse)
def evaluate_model_on_dataset_endpoint(
    model_id: int,
    req: ModelEvalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    metrics = services.models.evaluate_model_on_dataset(
        db,
        model_id=model_id,
        dataset_id=req.dataset_id,
        target_index=req.target_index,
        out_dir=None,
    )
    return ModelEvalResponse(dataset_id=req.dataset_id, metrics=metrics)
