from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from ..models import ModelRegistry


def create_model(
    db: Session,
    *,
    name: str,
    run_id: int,
    dataset_id: int | None = None,
    target: str | None = None,
    status: str = "candidate",
) -> ModelRegistry:
    model = ModelRegistry(
        name=name,
        run_id=run_id,
        dataset_id=dataset_id,
        target=target,
        status=status,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def get_model(db: Session, model_id: int) -> ModelRegistry | None:
    return db.get(ModelRegistry, model_id)


def list_models(db: Session) -> List[ModelRegistry]:
    return db.query(ModelRegistry).all()


def update_model_status(db: Session, model: ModelRegistry, status: str) -> ModelRegistry:
    model.status = status
    db.commit()
    db.refresh(model)
    return model
