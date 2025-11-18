from __future__ import annotations

"""Compatibility layer exposing the FastAPI app and auth utilities."""

from .api.deps import get_current_user, get_db
from .api.main import app
from .api.security import (
    ACCESS_TOKEN_EXPIRE_SECONDS,
    JWT_ALGORITHM,
    SECRET_KEY,
    SESSION_COOKIE_NAME,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

__all__ = [
    "app",
    "hash_password",
    "verify_password",
    "SESSION_COOKIE_NAME",
    "get_db",
    "get_current_user",
    "create_access_token",
    "decode_token",
    "SECRET_KEY",
    "JWT_ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_SECONDS",
]
