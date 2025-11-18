from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Generator

import bcrypt
try:  # pragma: no cover - optional dependency
    import jwt
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback
    class _MiniJWT:
        class ExpiredSignatureError(Exception):
            """Raised when the token has expired."""

        class InvalidTokenError(Exception):
            """Raised when the token cannot be decoded."""

        @staticmethod
        def _b64encode(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        @staticmethod
        def _b64decode(data: str) -> bytes:
            padding = "=" * (-len(data) % 4)
            return base64.urlsafe_b64decode(data + padding)

        def encode(self, payload: dict, key: str, algorithm: str = "HS256") -> str:
            if algorithm != "HS256":
                raise ValueError("Only HS256 is supported without PyJWT")
            header = {"alg": algorithm, "typ": "JWT"}
            header_b64 = self._b64encode(json.dumps(header, separators=(",", ":")).encode())
            payload_b64 = self._b64encode(
                json.dumps(payload, separators=(",", ":")).encode()
            )
            signing_input = f"{header_b64}.{payload_b64}".encode()
            signature = hmac.new(key.encode(), signing_input, hashlib.sha256).digest()
            signature_b64 = self._b64encode(signature)
            return f"{header_b64}.{payload_b64}.{signature_b64}"

        def decode(self, token: str, key: str, algorithms: list[str]) -> dict:
            if "HS256" not in algorithms:
                raise self.InvalidTokenError("Unsupported algorithm")
            try:
                header_b64, payload_b64, signature_b64 = token.split(".")
            except ValueError as exc:
                raise self.InvalidTokenError("Invalid token structure") from exc
            signing_input = f"{header_b64}.{payload_b64}".encode()
            expected = self._b64encode(
                hmac.new(key.encode(), signing_input, hashlib.sha256).digest()
            )
            if not hmac.compare_digest(expected, signature_b64):
                raise self.InvalidTokenError("Invalid signature")
            payload = json.loads(self._b64decode(payload_b64))
            exp = payload.get("exp")
            if exp is not None:
                now = datetime.now(tz=timezone.utc).timestamp()
                if now > float(exp):
                    raise self.ExpiredSignatureError("Session expired")
            return payload

    jwt = _MiniJWT()
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .core.settings import settings
from .db.database import SessionLocal
from .db.models import User
from .utils.errors import AuthError, ConfigError, DataError, DatabaseError, ModelError, PipelineError
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
async def _configure_logging() -> None:
    setup_logging()
    logger.info("FastAPI application initialized")

_SECRET_KEY = settings.JWT_SECRET_KEY
_JWT_ALGORITHM = settings.JWT_ALGORITHM
_ACCESS_TOKEN_EXPIRE_SECONDS = settings.ACCESS_TOKEN_EXPIRE_SECONDS
SESSION_COOKIE_NAME = "nircls_session"


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
    payload = {"sub": subject, "exp": int(expire.timestamp())}
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


def _format_error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(DataError)
async def data_error_handler(request: Request, exc: DataError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_400_BAD_REQUEST, str(exc))


@app.exception_handler(ConfigError)
async def config_error_handler(request: Request, exc: ConfigError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_400_BAD_REQUEST, str(exc))


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_401_UNAUTHORIZED, str(exc))


@app.exception_handler(ModelError)
@app.exception_handler(PipelineError)
async def pipeline_error_handler(
    request: Request, exc: ModelError | PipelineError
) -> JSONResponse:  # noqa: ARG001
    return _format_error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:  # noqa: ARG001
    logger.error("Database error", exc_info=exc)
    return _format_error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal database error")


__all__ = [
    "app",
    "hash_password",
    "verify_password",
    "SESSION_COOKIE_NAME",
    "get_db",
    "get_current_user",
]
