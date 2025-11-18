from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ... import i18n
from ..deps import get_current_user, get_db
from ..schemas import DatasetCreate, DatasetRead
from ...db import crud
from ...db.models import Dataset, User

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("", response_model=List[DatasetRead])
def list_all_datasets(db: Session = Depends(get_db)) -> List[Dataset]:
    return crud.list_datasets(db)


@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset_detail(dataset_id: int, db: Session = Depends(get_db)) -> Dataset:
    dataset = crud.get_dataset(db, dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=i18n.t("datasets.not_found_by_id").format(dataset_id=dataset_id),
        )
    return dataset


@router.post("", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def create_new_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dataset:
    return crud.create_dataset(
        db,
        name=payload.name,
        version=payload.version,
        description=payload.description,
        project_id=payload.project_id,
        created_by_id=current_user.user_id,
        storage_path=payload.storage_path,
    )
