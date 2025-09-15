from __future__ import annotations

import os
from typing import Generator, Set

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from itsdangerous import URLSafeSerializer
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .db.database import SessionLocal
from .db.models import User

# FastAPI application exposing authentication endpoints
app = FastAPI()

# Secret key used to sign session cookies
_SECRET_KEY = os.environ.get("SESSION_SECRET", "nircls-secret")
_serializer = URLSafeSerializer(_SECRET_KEY, salt="session")

# Simple in-memory session store mapping signed tokens to user IDs
SESSIONS: Set[str] = set()
SESSION_COOKIE_NAME = "session"


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


@app.post("/login")
def login(data: LoginData, response: Response, db: Session = Depends(get_db)):
    """Authenticate user and return a signed session cookie."""
    user = db.query(User).filter_by(username=data.username).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = _serializer.dumps({"user_id": user.user_id})
    SESSIONS.add(token)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"message": "login successful"}


@app.post("/logout")
def logout(request: Request, response: Response):
    """Invalidate the user's session by clearing the cookie."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token and token in SESSIONS:
        SESSIONS.remove(token)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"message": "logout successful"}


__all__ = [
    "app",
    "hash_password",
    "verify_password",
    "SESSION_COOKIE_NAME",
    "SESSIONS",
    "get_db",
]
