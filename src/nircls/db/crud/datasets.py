from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from ..models import Dataset


def list_datasets(session: Session) -> List[Dataset]:
    return session.query(Dataset).all()


def get_dataset(session: Session, dataset_id: int) -> Optional[Dataset]:
    return session.get(Dataset, dataset_id)


def create_dataset(
    session: Session,
    *,
    name: str,
    version: str | None = None,
    description: str | None = None,
    project_id: int | None = None,
    created_by_id: int | None = None,
    storage_path: str | None = None,
) -> Dataset:
    dataset = Dataset(
        name=name,
        version=version,
        description=description,
        project_id=project_id,
        created_by_id=created_by_id,
        storage_path=storage_path,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)
    return dataset
