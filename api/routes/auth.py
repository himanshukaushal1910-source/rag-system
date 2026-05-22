"""
api/routes/auth.py

Session-based authentication via HttpOnly cookies (H-3 fix).

Flow:
  POST /api/login   — validate API key, set HttpOnly rag_session cookie
  POST /api/logout  — clear the cookie
  GET  /api/me      — returns 200 if session cookie is valid, else 401

Why HttpOnly cookies:
  - The API key itself never enters JavaScript — it's only sent once to /login
  - The session token stored in the cookie cannot be read by JS (HttpOnly flag)
  - If an XSS attack ever executes on the page, it cannot steal the token
  - SameSite=Strict prevents CSRF without needing a separate CSRF token
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from config import get_settings

router = APIRouter(tags=["auth"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)

_SESSION_TTL = 8 * 3600  # 8 hours

# In-memory token store: token → expiry timestamp
# For multi-process deployments, replace with Redis.
_token_store: dict[str, float] = {}


def _evict_expired() -> None:
    """Remove expired tokens from the store."""
    now = time.time()
    stale = [t for t, exp in _token_store.items() if exp < now]
    for t in stale:
        del _token_store[t]


def validate_session_token(token: str) -> bool:
    """Return True if token exists and has not expired.

    Called by APIKeyMiddleware on every request that has a rag_session cookie
    but no X-API-Key header.
    """
    expiry = _token_store.get(token)
    if expiry is None:
        return False
    if time.time() > expiry:
        del _token_store[token]
        return False
    return True


@router.post("/api/login")
async def login(request: Request) -> JSONResponse:
    """Validate API key and issue an HttpOnly session cookie.

    Request body: {"api_key": "<key>"}
    On success: sets rag_session cookie + returns {"status": "ok"}
    On failure: 401
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content={"error": "Invalid JSON body."})

    provided_key = body.get("api_key", "")
    settings = get_settings()

    if not hmac.compare_digest(
        hashlib.sha256(provided_key.encode()).digest(),
        hashlib.sha256(settings.api_key.encode()).digest(),
    ):
        logger.warning("auth.login_failed", remote=request.client.host if request.client else "unknown")
        return JSONResponse(status_code=401, content={"error": "Invalid API key."})

    _evict_expired()
    token = secrets.token_urlsafe(32)
    _token_store[token] = time.time() + _SESSION_TTL

    logger.info("auth.login_ok", token_prefix=token[:8])

    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        key="rag_session",
        value=token,
        httponly=True,          # not readable by JavaScript
        samesite="strict",      # CSRF protection
        max_age=_SESSION_TTL,
        path="/",
        secure=False,           # set True in production when serving over HTTPS
    )
    return response


@router.post("/api/logout")
async def logout() -> Response:
    """Clear the session cookie."""
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(key="rag_session", path="/")
    return response


@router.get("/api/me")
async def me() -> JSONResponse:
    """Probe endpoint to check if the session cookie is still valid.

    This route goes through APIKeyMiddleware normally — returns 200 if the
    middleware accepts the request (valid cookie or valid X-API-Key header),
    401 if not. The frontend calls this on page load to decide whether to
    show the login overlay.
    """
    return JSONResponse({"authenticated": True})
