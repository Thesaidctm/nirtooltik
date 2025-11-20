from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from ..models import User


def create_user(
    session: Session,
    *,
    username: str,
    email: str,
    hashed_password: str,
    organization_id: Optional[int] = None,
) -> User:
    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        organization_id=organization_id,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def get_user(session: Session, user_id: int) -> Optional[User]:
    return session.get(User, user_id)


def get_user_by_username(session: Session, *, username: str) -> Optional[User]:
    return session.query(User).filter_by(username=username).first()


def get_user_by_email(session: Session, *, email: str) -> Optional[User]:
    return session.query(User).filter_by(email=email).first()


def update_user(
    session: Session,
    user_id: int,
    *,
    username: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[User]:
    user = session.get(User, user_id)
    if not user:
        return None
    if username is not None:
        user.username = username
    if email is not None:
        user.email = email
    session.commit()
    session.refresh(user)
    return user


def delete_user(session: Session, user_id: int) -> None:
    user = session.get(User, user_id)
    if user:
        session.delete(user)
        session.commit()
