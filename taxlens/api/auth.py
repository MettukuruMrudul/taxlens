"""JWT authentication.

Tokens are HS256-signed with JWT_SECRET and carry the user id in ``sub``.
For local development and the user study, ``POST /v1/auth/dev-token``
issues a token for any username when ALLOW_DEV_TOKENS=true. In production,
set ALLOW_DEV_TOKENS=false and issue tokens from a real identity provider
signed with the same secret.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from taxlens import config

ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)


@dataclass
class User:
    id: str


def create_token(user_id: str, minutes: int | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "iat": now,
               "exp": now + timedelta(minutes=minutes if minutes is not None else config.JWT_TTL_MINUTES)}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> User:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return User(id=str(sub))


async def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    return decode_token(creds.credentials)
