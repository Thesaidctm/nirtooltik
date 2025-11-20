from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from ..models import Role


def create_role(session: Session, name: str) -> Role:
    role = Role(name=name)
    session.add(role)
    session.commit()
    session.refresh(role)
    return role


def get_role(session: Session, role_id: int) -> Optional[Role]:
    return session.get(Role, role_id)


def get_role_by_name(session: Session, *, name: str) -> Optional[Role]:
    return session.query(Role).filter_by(name=name).first()


def update_role(
    session: Session, role_id: int, *, name: Optional[str] = None
) -> Optional[Role]:
    role = session.get(Role, role_id)
    if not role:
        return None
    if name is not None:
        role.name = name
    session.commit()
    session.refresh(role)
    return role


def delete_role(session: Session, role_id: int) -> None:
    role = session.get(Role, role_id)
    if role:
        session.delete(role)
        session.commit()
