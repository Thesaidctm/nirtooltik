from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
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


@router.post("/login")
def login(data: LoginData, response: Response, db: Session = Depends(get_db)):
    """Authenticate user and issue a JWT stored in an HttpOnly cookie."""

    user = db.query(User).filter_by(username=data.username).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=i18n.t("auth.invalid_credentials"),
        )

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
    return {"message": "login successful"}


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
