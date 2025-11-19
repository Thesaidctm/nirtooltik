from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..schemas import (
    CalibrationTransferCreate,
    CalibrationTransferRead,
    PredictRequest,
    PredictResponse,
)
from ...core import services
from ...db import crud
from ...db.models import CalibrationTransfer, User
from ...utils.errors import DataError

router = APIRouter()


class CalibrationTransferFitResponse(BaseModel):
    id: int
    artifacts: Dict[str, Any]


@router.post("", response_model=CalibrationTransferRead, status_code=status.HTTP_201_CREATED)
def create_transfer(
    payload: CalibrationTransferCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CalibrationTransfer:  # noqa: ARG001
    try:
        return services.calibration_transfers.create_calibration_transfer(
            db,
            name=payload.name,
            master_model_id=payload.master_model_id,
            master_instrument_id=payload.master_instrument_id,
            slave_instrument_id=payload.slave_instrument_id,
            standard_dataset_master_id=payload.standard_dataset_master_id,
            standard_dataset_slave_id=payload.standard_dataset_slave_id,
            method=payload.method,
        )
    except DataError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("", response_model=List[CalibrationTransferRead])
def list_transfers(db: Session = Depends(get_db)) -> List[CalibrationTransfer]:
    return crud.list_transfers(db)


@router.get("/{transfer_id}", response_model=CalibrationTransferRead)
def get_transfer(transfer_id: int, db: Session = Depends(get_db)) -> CalibrationTransfer:
    transfer = crud.get_transfer(db, transfer_id)
    if transfer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transfer not found")
    return transfer


@router.post("/{transfer_id}/fit-ds", response_model=CalibrationTransferFitResponse)
def fit_calibration_transfer_ds(
    transfer_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Opcional: checar se current_user pertence à mesma organização dos datasets/modelo
    transfer = services.calibration_transfers.fit_ds_for_transfer(db, transfer_id)
    return CalibrationTransferFitResponse(id=transfer.id, artifacts=transfer.artifacts or {})


@router.post("/{transfer_id}/predict", response_model=PredictResponse)
def predict_with_calibration_transfer(
    transfer_id: int,
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Recebe espectros medidos no instrumento escravo, aplica a transferência de calibração
    (por enquanto DS) para o espaço do instrumento mestre e, em seguida,
    usa o modelo mestre associado para prever os valores-alvo.
    """
    try:
        y_hat = services.calibration_transfers.predict_with_transfer(
            db=db,
            transfer_id=transfer_id,
            spectra=req.spectra,
        )
    except DataError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return PredictResponse(y_hat=y_hat)
