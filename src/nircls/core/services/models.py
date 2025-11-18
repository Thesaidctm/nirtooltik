from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from joblib import load
from sqlalchemy.orm import Session

from nircls.db import crud
from nircls.db.models import Dataset, ModelRegistry
from nircls.processing.io import read_tabular, read_xlsx_xy
from nircls.results.metrics.regression import (
    compute_mae_percent,
    compute_rpd,
    compute_rmse,
    compute_rmsep,
)
from nircls.utils.errors import DataError


def predict_with_model(
    db: Session,
    model_id: int,
    spectra: list[list[float]],
) -> list[float]:
    model = crud.get_model(db, model_id=model_id)
    if model is None:
        raise DataError("Model not found")

    run = model.run
    artifacts = run.artifacts or {}
    base_dir = artifacts.get("base_dir")
    model_paths = artifacts.get("models", [])
    if not base_dir or not model_paths:
        raise DataError("No model artifact registered for this run")

    model_path = Path(base_dir) / model_paths[0]
    estimator = load(model_path)

    X = np.array(spectra, dtype=float)
    y_hat = estimator.predict(X)
    return y_hat.tolist()


def evaluate_model_on_dataset(
    db: Session,
    model_id: int,
    dataset_id: int,
    *,
    target_index: int = 0,
    out_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Avalia um modelo registrado (ModelRegistry) em um Dataset específico.
    Lê os dados, aplica o modelo e retorna/salva métricas de regressão.
    """

    model: ModelRegistry | None = crud.get_model(db, model_id=model_id)
    if model is None:
        raise DataError(f"Model {model_id} not found")

    dataset: Dataset | None = crud.get_dataset(
        db, dataset_id, organization_id=model.run.organization_id
    )
    if dataset is None:
        raise DataError(f"Dataset {dataset_id} not found")

    if not dataset.storage_path:
        raise DataError("Dataset has no storage_path configured")

    storage_dir = Path(dataset.storage_path)

    data_path: Path | None = None
    for candidate in ("data.csv", "data.xlsx"):
        candidate_path = storage_dir / candidate
        if candidate_path.exists():
            data_path = candidate_path
            break
    if data_path is None:
        raise DataError("No data file (data.csv/data.xlsx) found in dataset storage")

    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            X, y_list, _ = read_xlsx_xy(data_path, target_type="regression")
        except DataError:
            X, y_list, _ = read_tabular(data_path, target_type="regression")
    else:
        X, y_list, _ = read_tabular(data_path, target_type="regression")

    if not y_list:
        raise DataError("No target columns found in dataset")
    if target_index < 0 or target_index >= len(y_list):
        raise DataError("Invalid target_index for dataset targets")

    y = y_list[target_index][1]

    y_hat = np.array(
        predict_with_model(db, model_id=model_id, spectra=X.values.tolist()),
        dtype=float,
    )
    y_true = np.array(y, dtype=float)

    metrics: Dict[str, Any] = {
        "rmse": compute_rmse(y_true, y_hat),
        "rmsep": compute_rmsep(y_true, y_hat),
        "rpd": compute_rpd(y_true, y_hat),
        "mae_percent": compute_mae_percent(y_true, y_hat),
    }

    output_dir = out_dir or Path("results") / f"eval_{model_id}_{dataset_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_hat})
    df.to_csv(output_dir / "external_validation.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    history = model.eval_history or []
    history_entry = {
        "dataset_id": dataset_id,
        "metrics": metrics,
    }
    history.append(history_entry)
    model.eval_history = history
    db.add(model)
    db.commit()
    db.refresh(model)

    return metrics
