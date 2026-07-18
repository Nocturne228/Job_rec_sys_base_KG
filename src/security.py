"""Small HMAC-signed bearer-token layer for the self-contained API demo."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Iterable

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_token(subject: str, role: str = "user", expires_seconds: int = 3600) -> str:
    header = _b64url(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    )
    payload = _b64url(
        json.dumps(
            {
                "sub": subject,
                "role": role,
                "exp": int(time.time()) + expires_seconds,
            },
            separators=(",", ":"),
        ).encode()
    )
    secret = os.environ.get("JOBREC_TOKEN_SECRET", "development-only-secret").encode()
    signature = _b64url(
        hmac.new(secret, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> dict:
    try:
        header, payload, signature = token.split(".")
        secret = os.environ.get(
            "JOBREC_TOKEN_SECRET", "development-only-secret"
        ).encode()
        expected = _b64url(
            hmac.new(secret, f"{header}.{payload}".encode(), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            raise ValueError("bad signature")
        claims = json.loads(_b64decode(payload))
        if int(claims["exp"]) < int(time.time()):
            raise ValueError("expired")
        return claims
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=401, detail="Invalid or expired bearer token"
        ) from exc


def require_roles(*roles: str):
    allowed = set(roles)

    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    ) -> dict:
        if credentials is None:
            raise HTTPException(status_code=401, detail="Bearer token required")
        claims = verify_token(credentials.credentials)
        if allowed and claims.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return claims

    return dependency
