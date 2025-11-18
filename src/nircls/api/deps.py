from __future__ import annotations

from typing import Generator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from .security import SESSION_COOKIE_NAME, decode_token
from ..db.database import SessionLocal
from ..db.models import Organization, User
from ..utils.errors import AuthError


def get_db() -> Generator[Session, None, None]:
    """Yield a database session."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Retrieve the authenticated user based on the session cookie."""

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    try:
        payload = decode_token(token)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    subject = payload.get("sub")
    if subject is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session payload")

    try:
        user_id = int(subject)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session payload") from exc

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


def is_global_admin(user: User) -> bool:
    roles = getattr(user, "roles", []) or []
    return any(getattr(role, "name", None) == "admin" for role in roles)


def get_current_organization(
    user: User = Depends(get_current_user),
) -> Organization | None:
    return user.organization


def require_role(*roles: str):
    def _dep(user: User = Depends(get_current_user)) -> User:
        user_roles = getattr(user, "roles", []) or []
        if not any(getattr(role, "name", None) in roles for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return _dep

__all__ = [
    "get_db",
    "get_current_user",
    "require_role",
    "get_current_organization",
    "is_global_admin",
]
