from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sqlalchemy.orm import Session

from nircls.db import crud
from nircls.db.models import CalibrationTransfer, Dataset, ModelRegistry
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
    Usa, preferencialmente, o nome de arquivo registrado no manifest (original_filename).
    """
    if not dataset.storage_path:
        raise DataError("Dataset has no storage_path")

    storage_dir = Path(dataset.storage_path)

    manifest_path = storage_dir / "manifest.json"
    data_path: Path | None = None

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original_filename = manifest.get("original_filename")
            if original_filename:
                candidate = storage_dir / original_filename
                if candidate.exists():
                    data_path = candidate
        except Exception:
            # Se der erro ao ler o manifest, apenas segue para o fallback
            pass

    if data_path is None:
        for candidate_name in ("data.csv", "data.xlsx"):
            candidate = storage_dir / candidate_name
            if candidate.exists():
                data_path = candidate
                break

    if data_path is None:
        raise DataError(
            f"No data file found in {storage_dir} (manifest or data.csv/xlsx)"
        )

    X, _ = load_dataset_xy(data_path, target_cols=target_cols)
    return X.values


def create_calibration_transfer(
    db: Session,
    *,
    name: str,
    master_model_id: int,
    master_instrument_id: int,
    slave_instrument_id: int,
    standard_dataset_master_id: int | None = None,
    standard_dataset_slave_id: int | None = None,
    method: str = "ds",
) -> CalibrationTransfer:
    """
    Cria um CalibrationTransfer garantindo coerência entre:
    - modelo mestre e instrumento mestre;
    - datasets padrão e seus respectivos instrumentos.
    """
    model: ModelRegistry | None = crud.get_model(db, model_id=master_model_id)
    if model is None:
        raise DataError(f"Master model {master_model_id} not found")

    if model.instrument_id and model.instrument_id != master_instrument_id:
        raise DataError(
            f"Master model instrument_id={model.instrument_id} "
            f"does not match master_instrument_id={master_instrument_id}"
        )

    ds_master: Dataset | None = None
    ds_slave: Dataset | None = None

    if standard_dataset_master_id is not None:
        ds_master = crud.get_dataset(db, standard_dataset_master_id)
        if ds_master is None:
            raise DataError(f"Standard master dataset {standard_dataset_master_id} not found")
        if ds_master.instrument_id and ds_master.instrument_id != master_instrument_id:
            raise DataError(
                f"Standard master dataset instrument_id={ds_master.instrument_id} "
                f"does not match master_instrument_id={master_instrument_id}"
            )

    if standard_dataset_slave_id is not None:
        ds_slave = crud.get_dataset(db, standard_dataset_slave_id)
        if ds_slave is None:
            raise DataError(f"Standard slave dataset {standard_dataset_slave_id} not found")
        if ds_slave.instrument_id and ds_slave.instrument_id != slave_instrument_id:
            raise DataError(
                f"Standard slave dataset instrument_id={ds_slave.instrument_id} "
                f"does not match slave_instrument_id={slave_instrument_id}"
            )

    transfer = CalibrationTransfer(
        name=name,
        master_model_id=master_model_id,
        master_instrument_id=master_instrument_id,
        slave_instrument_id=slave_instrument_id,
        standard_dataset_master_id=standard_dataset_master_id,
        standard_dataset_slave_id=standard_dataset_slave_id,
        method=method,
    )
    db.add(transfer)
    db.commit()
    db.refresh(transfer)
    return transfer


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
