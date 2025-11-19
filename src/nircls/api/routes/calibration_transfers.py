from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..schemas import CalibrationTransferCreate, CalibrationTransferRead
from ...db import crud
from ...db.models import CalibrationTransfer, User

router = APIRouter()


@router.post("", response_model=CalibrationTransferRead, status_code=status.HTTP_201_CREATED)
def create_transfer(
    payload: CalibrationTransferCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CalibrationTransfer:  # noqa: ARG001
    return crud.create_transfer(
        db,
        name=payload.name,
        master_model_id=payload.master_model_id,
        master_instrument_id=payload.master_instrument_id,
        slave_instrument_id=payload.slave_instrument_id,
        method=payload.method,
        standard_dataset_master_id=payload.standard_dataset_master_id,
        standard_dataset_slave_id=payload.standard_dataset_slave_id,
    )


@router.get("", response_model=List[CalibrationTransferRead])
def list_transfers(db: Session = Depends(get_db)) -> List[CalibrationTransfer]:
    return crud.list_transfers(db)


@router.get("/{transfer_id}", response_model=CalibrationTransferRead)
def get_transfer(transfer_id: int, db: Session = Depends(get_db)) -> CalibrationTransfer:
    transfer = crud.get_transfer(db, transfer_id)
    if transfer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transfer not found")
    return transfer
