from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ..deps import get_current_user, get_db
from ..schemas import InstrumentCreate, InstrumentRead
from ...db import crud
from ...db.models import Instrument, User

router = APIRouter()


@router.post("", response_model=InstrumentRead, status_code=status.HTTP_201_CREATED)
def create_instrument(
    payload: InstrumentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Instrument:  # noqa: ARG001
    return crud.create_instrument(
        db,
        name=payload.name,
        code=payload.code,
        vendor=payload.vendor,
        model=payload.model,
        serial_number=payload.serial_number,
    )


@router.get("", response_model=List[InstrumentRead])
def list_instruments(db: Session = Depends(get_db)) -> List[Instrument]:
    return crud.list_instruments(db)
