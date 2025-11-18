from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

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
            payload_b64 = self._b64encode(json.dumps(payload, separators=(",", ":")).encode())
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
            expected = self._b64encode(hmac.new(key.encode(), signing_input, hashlib.sha256).digest())
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

from ..core.settings import settings
from ..utils.errors import AuthError

SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_SECONDS = settings.ACCESS_TOKEN_EXPIRE_SECONDS
SESSION_COOKIE_NAME = "nircls_session"


def hash_password(password: str) -> str:
    """Hash *password* using bcrypt."""

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify that *password* matches *hashed*."""

    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_access_token(*, subject: str, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(tz=timezone.utc) + (expires_delta or timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS))
    payload = {"sub": subject, "exp": int(expire.timestamp())}
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, object]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:  # type: ignore[attr-defined]
        raise AuthError("Session expired") from exc
    except jwt.InvalidTokenError as exc:  # type: ignore[attr-defined]
        raise AuthError("Invalid session") from exc


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_token",
    "SESSION_COOKIE_NAME",
    "SECRET_KEY",
    "JWT_ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_SECONDS",
]
