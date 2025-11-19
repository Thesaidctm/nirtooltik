from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy.orm import Session

from nircls.core.settings import settings
from nircls.db.models import Dataset, SampleFeatures, SampleMetadata, SampleTarget
from nircls.processing.io import load_dataset_xy


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
    *,
    target_cols: List[str] | None = None,
) -> Dataset:
    storage_dir = get_dataset_storage_dir(dataset)
    storage_dir.mkdir(parents=True, exist_ok=True)
    data_file = storage_dir / filename
    data_file.write_bytes(content)

    summary = ingest_dataset_to_xy_l(
        db,
        dataset,
        data_file=data_file,
        target_cols=target_cols,
    )

    manifest: Dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "version": dataset.version,
        "file": filename,
        "storage_dir": str(storage_dir),
    }
    manifest.update(
        {
            "original_filename": filename,
            "n_samples": summary["n_samples"],
            "n_features": summary["n_features"],
            "target_cols": summary["target_cols"],
        }
    )
    manifest_path = storage_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dataset.storage_path = str(storage_dir)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def ingest_dataset_to_xy_l(
    db: Session,
    dataset: Dataset,
    *,
    data_file: Path,
    target_cols: List[str] | None = None,
) -> dict:
    """
    Lê o arquivo de dados e popula as tabelas SampleMetadata (L),
    SampleFeatures (X) e SampleTarget (Y) para esse Dataset.

    Retorna um dict com resumo:
    - n_samples
    - n_features
    - target_cols
    """
    X, y_list = load_dataset_xy(data_file, target_cols=target_cols)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    target_names = [name for (name, _) in y_list]

    sample_metadatas: list[SampleMetadata] = []
    for i in range(n_samples):
        meta = SampleMetadata(
            external_id=str(i),
            metadata_json={
                "dataset_id": dataset.dataset_id,
                "sample_index": i,
            },
        )
        db.add(meta)
        sample_metadatas.append(meta)

    db.flush()

    feature_names = list(X.columns)
    for idx, meta in enumerate(sample_metadatas):
        row = X.iloc[idx]
        feat = SampleFeatures(
            sample_id=meta.sample_id,
            values={str(name): (None if pd.isna(value) else float(value)) for name, value in row.items()},
        )
        db.add(feat)

    for (tname, series) in y_list:
        for idx, meta in enumerate(sample_metadatas):
            target_value = series.iloc[idx]
            targ = SampleTarget(
                sample_id=meta.sample_id,
                target_name=tname,
                target_value=str(target_value),
            )
            db.add(targ)

    db.commit()

    return {
        "n_samples": n_samples,
        "n_features": len(feature_names),
        "target_cols": target_names,
    }


__all__ = [
    "get_dataset_storage_dir",
    "save_dataset_file",
    "ingest_dataset_to_xy_l",
]
