from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sqlalchemy.orm import Session

from nircls.db import crud
from nircls.db.models import CalibrationTransfer, Dataset
from nircls.processing.io import load_dataset_xy
from nircls.processing.transfer.calibration_transfer import fit_direct_standardization
from nircls.core.settings import settings
from nircls.core.services.models import predict_with_model
from nircls.utils.errors import DataError


def _get_dataset_X(
    dataset: Dataset,
    *,
    target_cols=None,
) -> np.ndarray:
    """
    Lê o arquivo principal do dataset e retorna apenas X (features).
    Assume que o arquivo principal é 'data.csv' ou 'data.xlsx' dentro de storage_path.
    """
    if not dataset.storage_path:
        raise DataError("Dataset has no storage_path")

    storage_dir = Path(dataset.storage_path)
    data_path = None
    for candidate in ("data.csv", "data.xlsx"):
        p = storage_dir / candidate
        if p.exists():
            data_path = p
            break
    if data_path is None:
        raise DataError(f"No data.csv or data.xlsx found in {storage_dir}")

    X, _ = load_dataset_xy(data_path, target_cols=target_cols)
    return X.values


def fit_ds_for_transfer(
    db: Session,
    transfer_id: int,
) -> CalibrationTransfer:
    """
    Ajusta Direct Standardization (DS) para um CalibrationTransfer existente,
    usando os datasets padrão master/escravo configurados na entidade.
    """
    transfer = crud.get_calibration_transfer(db, transfer_id=transfer_id)
    if transfer is None:
        raise DataError(f"CalibrationTransfer {transfer_id} not found")

    if not transfer.standard_dataset_master_id or not transfer.standard_dataset_slave_id:
        raise DataError(
            "CalibrationTransfer must have standard_dataset_master_id and standard_dataset_slave_id set"
        )

    ds_master: Dataset | None = crud.get_dataset(
        db, transfer.standard_dataset_master_id
    )
    ds_slave: Dataset | None = crud.get_dataset(
        db, transfer.standard_dataset_slave_id
    )
    if ds_master is None or ds_slave is None:
        raise DataError("Standard datasets not found")

    X_master = _get_dataset_X(ds_master)
    X_slave = _get_dataset_X(ds_slave)

    if X_master.shape != X_slave.shape:
        raise DataError(
            f"Standard datasets must have same shape, got {X_master.shape} vs {X_slave.shape}"
        )

    ds_model = fit_direct_standardization(X_primary=X_master, X_secondary=X_slave)

    # Salvar matriz F em disco, em diretório específico para transfers
    base_dir = Path(settings.RESULTS_DIR) / "calibration_transfers" / f"transfer_{transfer.id}"
    base_dir.mkdir(parents=True, exist_ok=True)
    F_path = base_dir / "F_ds.npy"
    np.save(F_path, ds_model.F)

    artifacts: Dict[str, Any] = transfer.artifacts or {}
    artifacts.update(
        {
            "base_dir": str(base_dir),
            "F_ds": str(F_path),
            "method": "ds",
        }
    )
    transfer.artifacts = artifacts

    db.add(transfer)
    db.commit()
    db.refresh(transfer)
    return transfer


def predict_with_transfer(
    db: Session,
    transfer_id: int,
    spectra: List[List[float]],
) -> List[float]:
    """
    Aplica a transferência de calibração (por enquanto DS) a espectros do instrumento escravo
    e, em seguida, usa o modelo mestre associado para prever os valores-alvo.
    """
    transfer: CalibrationTransfer | None = crud.get_calibration_transfer(
        db, transfer_id=transfer_id
    )
    if transfer is None:
        raise DataError(f"CalibrationTransfer {transfer_id} not found")

    artifacts = transfer.artifacts or {}
    method = artifacts.get("method") or transfer.method
    if method != "ds":
        raise DataError(
            f"Only DS method is supported for predict_with_transfer, got {method!r}"
        )

    base_dir = artifacts.get("base_dir")
    F_rel = artifacts.get("F_ds")
    if not base_dir or not F_rel:
        raise DataError("CalibrationTransfer artifacts must contain 'base_dir' and 'F_ds'")

    base_dir_path = Path(base_dir)
    F_path = Path(F_rel)
    if not F_path.is_absolute():
        F_path = base_dir_path / F_path

    if not F_path.exists():
        raise DataError(f"DS matrix file not found at {F_path}")

    F = np.load(F_path)

    X_slave = np.asarray(spectra, dtype=float)
    if X_slave.ndim != 2:
        raise DataError("Input 'spectra' must be a 2D array (n_samples x n_features)")

    if X_slave.shape[1] != F.shape[0]:
        raise DataError(
            f"Spectra have {X_slave.shape[1]} features but DS matrix has shape {F.shape}"
        )

    # Aplica transferência: X_mestre ≈ X_escravo @ F
    X_master = X_slave @ F

    # Usa o modelo mestre associado
    master_model_id = transfer.master_model_id
    if master_model_id is None:
        raise DataError("CalibrationTransfer has no master_model_id configured")

    # predict_with_model já sabe carregar o modelo treinado do mestre
    y_hat = predict_with_model(
        db,
        model_id=master_model_id,
        spectra=X_master.tolist(),
    )
    return y_hat
