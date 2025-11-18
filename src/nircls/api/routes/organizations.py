from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from ..deps import get_db, require_role
from ..schemas import OrganizationCreate, OrganizationRead
from ...db import crud
from ...db.models import Organization

router = APIRouter(prefix="/organizations", tags=["organizations"])


@router.post("", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED)
def create_organization(
    payload: OrganizationCreate,
    db: Session = Depends(get_db),
    _: None = Depends(require_role("admin")),
) -> Organization:
    return crud.create_organization(db, name=payload.name)


@router.get("", response_model=List[OrganizationRead])
def list_organizations(
    db: Session = Depends(get_db), _=Depends(require_role("admin"))
) -> List[Organization]:
    return crud.list_organizations(db)
