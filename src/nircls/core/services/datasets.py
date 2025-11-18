from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sqlalchemy.orm import Session

from nircls.core.settings import settings
from nircls.db.models import Dataset


def get_dataset_storage_dir(dataset: Dataset) -> Path:
    base = Path(settings.DATA_STORAGE_DIR)
    project_part = f"project_{dataset.project_id}" if dataset.project_id is not None else "project_none"
    version_part = dataset.version or "unversioned"
    return base / project_part / f"dataset_{dataset.dataset_id}_v{version_part}"


def save_dataset_file(
    db: Session,
    dataset: Dataset,
    filename: str,
    content: bytes,
) -> Dataset:
    storage_dir = get_dataset_storage_dir(dataset)
    storage_dir.mkdir(parents=True, exist_ok=True)
    data_file = storage_dir / filename
    data_file.write_bytes(content)

    manifest: Dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "version": dataset.version,
        "file": filename,
        "storage_dir": str(storage_dir),
    }
    manifest_path = storage_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dataset.storage_path = str(storage_dir)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


__all__ = [
    "get_dataset_storage_dir",
    "save_dataset_file",
]
