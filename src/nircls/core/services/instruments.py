from __future__ import annotations

from sqlalchemy.orm import Session

from nircls.db import crud

DEFAULT_INSTRUMENTS = [
    {"name": "BRUKER MPA", "code": "bruker_mpa"},
    {"name": "VIAVI MicroNIR", "code": "viavi_micronir"},
    {"name": "TRINAMIX NIR", "code": "trinamix_nir"},
]


def seed_default_instruments(db: Session, organization_id: int | None = None) -> None:
    for item in DEFAULT_INSTRUMENTS:
        existing = crud.get_instrument_by_code(
            db, code=item["code"], organization_id=organization_id
        )
        if existing:
            continue
        crud.create_instrument(
            db,
            name=item["name"],
            code=item["code"],
            organization_id=organization_id,
        )


__all__ = [
    "DEFAULT_INSTRUMENTS",
    "seed_default_instruments",
]
