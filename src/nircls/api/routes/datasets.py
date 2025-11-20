from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ... import i18n
from ..deps import get_current_user, get_db, is_global_admin
from ..schemas import DatasetCreate, DatasetRead
from ...core.services.datasets import save_dataset_file
from ...db import crud
from ...db.models import Dataset, User

router = APIRouter(prefix="/datasets", tags=["datasets"])


async def _save_uploaded_dataset(
    *,
    dataset: Dataset,
    upload_file: UploadFile,
    db: Session,
    target_cols: Optional[List[str]] = None,
) -> Dataset:
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    _, ext = os.path.splitext(upload_file.filename or "")
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: .csv, .xlsx, .xls",
        )

    content = await upload_file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    return save_dataset_file(
        db,
        dataset,
        upload_file.filename or "dataset",
        content,
        target_cols=target_cols,
    )

@router.get("", response_model=List[DatasetRead])
def list_all_datasets(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> List[Dataset]:
    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    return crud.list_datasets(db, organization_id=organization_id)


@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset_detail(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dataset:
    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    dataset = crud.get_dataset(db, dataset_id, organization_id=organization_id)
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
        organization_id=current_user.organization_id,
        instrument_id=payload.instrument_id,
    )


@router.post("/upload", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_new_dataset_with_file(
    name: str = Form(...),
    instrument_id: int = Form(...),
    file: UploadFile = File(...),
    version: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    project_id: Optional[int] = Form(None),
    target_cols: Optional[List[str]] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dataset:
    dataset = crud.create_dataset(
        db,
        name=name,
        version=version,
        description=description,
        project_id=project_id,
        created_by_id=current_user.user_id,
        organization_id=current_user.organization_id,
        instrument_id=instrument_id,
    )

    return await _save_uploaded_dataset(
        dataset=dataset, upload_file=file, db=db, target_cols=target_cols
    )


@router.post("/{dataset_id}/upload", response_model=DatasetRead)
async def upload_dataset_file(
    dataset_id: int,
    file: UploadFile = File(...),
    target_cols: List[str] | None = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    organization_id = None if is_global_admin(current_user) else current_user.organization_id
    dataset = crud.get_dataset(db, dataset_id, organization_id=organization_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=i18n.t("datasets.not_found_by_id").format(dataset_id=dataset_id),
        )

    return await _save_uploaded_dataset(
        dataset=dataset, upload_file=file, db=db, target_cols=target_cols
    )
