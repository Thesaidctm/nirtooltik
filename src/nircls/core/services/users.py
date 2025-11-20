from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from ...api.security import hash_password
from ...db import crud
from ...db.models import Role, User

logger = logging.getLogger(__name__)


def _ensure_admin_role(db: Session) -> Role:
    role = crud.get_role_by_name(db, name="admin")
    if role:
        return role

    logger.info("Creating default admin role for development use")
    return crud.create_role(db, name="admin")


def ensure_dev_admin_user(db: Session) -> User:
    """
    Ensure an ``admin`` user for development:
    - username: admin
    - password: admin
    - role: admin
    """

    username = "admin"
    admin_role = _ensure_admin_role(db)

    user = crud.get_user_by_username(db, username=username)
    if user:
        if admin_role not in user.roles:
            user.roles.append(admin_role)
        user.hashed_password = hash_password("admin")
        db.commit()
        db.refresh(user)
        return user

    logger.info("Creating default development admin user")
    user = crud.create_user(
        db,
        username=username,
        email="admin@example.com",
        hashed_password=hash_password("admin"),
    )
    user.roles.append(admin_role)
    db.commit()
    db.refresh(user)
    return user


__all__ = ["ensure_dev_admin_user"]
