from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Generator, Set

import bcrypt
import jwt
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .db.database import SessionLocal
from .db.models import User

app = FastAPI()

_SECRET_KEY = os.environ.get("SESSION_SECRET", "nircls-secret")
_JWT_ALGORITHM = os.environ.get("SESSION_ALGORITHM", "HS256")
_ACCESS_TOKEN_EXPIRE_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))
SESSION_COOKIE_NAME = "nircls_session"
# Backwards compatible no-op set retained for legacy imports
SESSIONS: Set[str] = set()


def get_db() -> Generator[Session, None, None]:
    """Yield a database session."""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class LoginData(BaseModel):
    username: str
    password: str


def hash_password(password: str) -> str:
    """Hash *password* using bcrypt."""

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify that *password* matches *hashed*."""

    return bcrypt.checkpw(password.encode(), hashed.encode())


def _create_access_token(*, subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(tz=timezone.utc) + (expires_delta or timedelta(seconds=_ACCESS_TOKEN_EXPIRE_SECONDS))
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> dict[str, object]:
    try:
        return jwt.decode(token, _SECRET_KEY, algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session") from exc


@app.post("/login")
def login(data: LoginData, response: Response, db: Session = Depends(get_db)):
    """Authenticate user and issue a JWT stored in an HttpOnly cookie."""

    user = db.query(User).filter_by(username=data.username).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = _create_access_token(subject=str(user.user_id))
    expires = timedelta(seconds=_ACCESS_TOKEN_EXPIRE_SECONDS)
    SESSIONS.add(token)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=int(expires.total_seconds()),
        expires=int((datetime.now(tz=timezone.utc) + expires).timestamp()),
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"message": "login successful"}


@app.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    """Clear the authentication cookie on logout."""

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        SESSIONS.discard(token)
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"message": "logout successful"}


def get_current_user(
    request: Request, db: Session = Depends(get_db)
) -> User:
    """Retrieve the authenticated user based on the session cookie."""

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = _decode_token(token)
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


__all__ = [
    "app",
    "hash_password",
    "verify_password",
    "SESSION_COOKIE_NAME",
    "SESSIONS",
    "get_db",
    "get_current_user",
]
