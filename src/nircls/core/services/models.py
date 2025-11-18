from __future__ import annotations

from pathlib import Path

import numpy as np
from joblib import load
from sqlalchemy.orm import Session

from nircls.db import crud
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
