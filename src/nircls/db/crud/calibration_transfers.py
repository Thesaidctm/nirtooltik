from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from ..models import CalibrationTransfer


def create_transfer(
    db: Session,
    *,
    name: str,
    master_model_id: int,
    master_instrument_id: int,
    slave_instrument_id: int,
    method: str = "ds",
    artifacts: dict | None = None,
    metadata: dict | None = None,
    standard_dataset_master_id: int | None = None,
    standard_dataset_slave_id: int | None = None,
) -> CalibrationTransfer:
    transfer = CalibrationTransfer(
        name=name,
        master_model_id=master_model_id,
        master_instrument_id=master_instrument_id,
        slave_instrument_id=slave_instrument_id,
        method=method,
        artifacts=artifacts,
        metadata=metadata,
        standard_dataset_master_id=standard_dataset_master_id,
        standard_dataset_slave_id=standard_dataset_slave_id,
    )
    db.add(transfer)
    db.commit()
    db.refresh(transfer)
    return transfer


def get_transfer(db: Session, transfer_id: int) -> CalibrationTransfer | None:
    return db.get(CalibrationTransfer, transfer_id)


def list_transfers(db: Session) -> List[CalibrationTransfer]:
    return db.query(CalibrationTransfer).all()
