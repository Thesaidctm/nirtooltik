from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from ..models import Dataset


def list_datasets(session: Session, *, organization_id: Optional[int] = None) -> List[Dataset]:
    query = session.query(Dataset)
    if organization_id is not None:
        query = query.filter(Dataset.organization_id == organization_id)
    return query.all()


def get_dataset(
    session: Session, dataset_id: int, *, organization_id: Optional[int] = None
) -> Optional[Dataset]:
    dataset = session.get(Dataset, dataset_id)
    if organization_id is not None and dataset and dataset.organization_id != organization_id:
        return None
    return dataset


def create_dataset(
    session: Session,
    *,
    name: str,
    version: str | None = None,
    description: str | None = None,
    project_id: int | None = None,
    created_by_id: int | None = None,
    storage_path: str | None = None,
    organization_id: int | None = None,
    instrument_id: int | None = None,
) -> Dataset:
    dataset = Dataset(
        name=name,
        version=version,
        description=description,
        project_id=project_id,
        created_by_id=created_by_id,
        storage_path=storage_path,
        organization_id=organization_id,
        instrument_id=instrument_id,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset
