from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ... import i18n
from ...db.models import User
from ..deps import get_db
from ..security import ACCESS_TOKEN_EXPIRE_SECONDS, SESSION_COOKIE_NAME, create_access_token, verify_password

router = APIRouter(tags=["auth"])


class LoginData(BaseModel):
    username: str
    password: str


def _authenticate(username: str, password: str, db: Session) -> User:
    user = db.query(User).filter_by(username=username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=i18n.t("auth.invalid_credentials"),
        )
    return user


def _issue_token_response(user: User, response: Response) -> dict[str, str]:
    token = create_access_token(subject=str(user.user_id))
    expires = timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=int(expires.total_seconds()),
        expires=int((datetime.now(tz=timezone.utc) + expires).timestamp()),
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"access_token": token, "token_type": "bearer"}


@router.post("/login")
def login(data: LoginData, response: Response, db: Session = Depends(get_db)):
    """Authenticate user with JSON payload and return a JWT."""

    user = _authenticate(data.username, data.password, db)
    return _issue_token_response(user, response)


@router.post("/auth/token")
def token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """OAuth2-compatible login endpoint returning a bearer token."""

    user = _authenticate(form_data.username, form_data.password, db)
    return _issue_token_response(user, response)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    """Clear the authentication cookie on logout."""

    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"message": "logout successful"}
