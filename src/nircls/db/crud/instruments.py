from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from ..models import Instrument


def create_instrument(
    db: Session,
    *,
    name: str,
    code: str,
    vendor: str | None = None,
    model: str | None = None,
    serial_number: str | None = None,
) -> Instrument:
    instrument = Instrument(
        name=name,
        code=code,
        vendor=vendor,
        model=model,
        serial_number=serial_number,
    )
    db.add(instrument)
    db.commit()
    db.refresh(instrument)
    return instrument


def get_instrument(db: Session, instrument_id: int) -> Instrument | None:
    return db.get(Instrument, instrument_id)


def list_instruments(db: Session) -> List[Instrument]:
    return db.query(Instrument).all()
