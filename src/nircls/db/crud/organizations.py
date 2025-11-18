from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from ..models import Organization


def list_organizations(session: Session) -> List[Organization]:
    return session.query(Organization).all()


def get_organization(session: Session, organization_id: int) -> Optional[Organization]:
    return session.get(Organization, organization_id)


def create_organization(session: Session, *, name: str) -> Organization:
    organization = Organization(name=name)
    session.add(organization)
    session.commit()
    session.refresh(organization)
    return organization
