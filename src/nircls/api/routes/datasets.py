from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...auth import get_current_user, get_db
from ...db.crud import create_dataset, get_dataset, list_datasets
from ...db.models import Dataset, User

router = APIRouter(prefix="/datasets", tags=["datasets"])


class DatasetCreate(BaseModel):
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    project_id: Optional[int] = None
    storage_path: Optional[str] = None


class DatasetResponse(DatasetCreate):
    dataset_id: int
    created_by_id: Optional[int] = None

    class Config:
        orm_mode = True


@router.get("", response_model=List[DatasetResponse])
def list_all_datasets(db: Session = Depends(get_db)) -> List[Dataset]:
    return list_datasets(db)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset_detail(dataset_id: int, db: Session = Depends(get_db)) -> Dataset:
    dataset = get_dataset(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_new_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dataset:
    return create_dataset(
        db,
        name=payload.name,
        version=payload.version,
        description=payload.description,
        project_id=payload.project_id,
        created_by_id=current_user.user_id,
        storage_path=payload.storage_path,
    )
