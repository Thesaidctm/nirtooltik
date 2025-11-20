from __future__ import annotations

from typing import List, Optional

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
    organization_id: int | None = None,
) -> Instrument:
    instrument = Instrument(
        name=name,
        code=code,
        vendor=vendor,
        model=model,
        serial_number=serial_number,
        organization_id=organization_id,
    )
    db.add(instrument)
    db.commit()
    db.refresh(instrument)
    return instrument


def get_instrument(db: Session, instrument_id: int) -> Instrument | None:
    return db.get(Instrument, instrument_id)


def list_instruments(
    db: Session, *, organization_id: Optional[int] = None
) -> List[Instrument]:
    query = db.query(Instrument)
    if organization_id is not None:
        query = query.filter(Instrument.organization_id == organization_id)
    return query.all()


def get_instrument_by_code(
    db: Session, *, code: str, organization_id: int | None = None
) -> Instrument | None:
    query = db.query(Instrument).filter(Instrument.code == code)
    if organization_id is not None:
        query = query.filter(Instrument.organization_id == organization_id)
    return query.first()
