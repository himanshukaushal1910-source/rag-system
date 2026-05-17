from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from collections import defaultdict, deque

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_PUBLIC_PATHS = {"/", "/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/static"}

# In-memory rate limit store: api_key → deque of request timestamps
_rate_limit_store: dict[str, deque] = defaultdict(deque)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request-id into every request and response."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.clear_contextvars()
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on all non-public routes."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        settings = get_settings()
        provided_key = request.headers.get("X-API-Key", "")

        if not hmac.compare_digest(
            hashlib.sha256(provided_key.encode()).digest(),
            hashlib.sha256(settings.api_key.encode()).digest(),
        ):
            request_id = getattr(request.state, "request_id", None)
            logger.warning("Unauthorized request", path=path, request_id=request_id)
            return JSONResponse(
                status_code=401,
                content={
                    "error_code": "authentication_error",
                    "message": "Invalid or missing X-API-Key header",
                    "detail": "Provide a valid API key in the X-API-Key header",
                    "request_id": request_id,
                },
            )

        # Store the validated key on request state for rate limiter
        request.state.api_key = provided_key
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter per API key.

    Tracks request timestamps in a deque per key. On each request:
    1. Evict timestamps older than the window.
    2. If count >= limit → 429.
    3. Otherwise append current timestamp and allow.

    Uses in-memory storage — for multi-process deployments replace with Redis.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        settings = get_settings()

        if not settings.rate_limit_enabled:
            return await call_next(request)

        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        # Use API key as the rate limit identity
        api_key = getattr(request.state, "api_key", None) or request.headers.get("X-API-Key", "anonymous")
        now = time.monotonic()
        window = settings.rate_limit_window_seconds
        limit = settings.rate_limit_requests

        timestamps = _rate_limit_store[api_key]

        # Evict old timestamps outside the window
        while timestamps and now - timestamps[0] > window:
            timestamps.popleft()

        if len(timestamps) >= limit:
            retry_after = int(window - (now - timestamps[0])) + 1
            request_id = getattr(request.state, "request_id", None)
            logger.warning(
                "Rate limit exceeded",
                api_key=api_key[:8] + "...",
                count=len(timestamps),
                limit=limit,
                request_id=request_id,
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error_code": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: {limit} requests per {window}s",
                    "detail": f"Try again in {retry_after} seconds",
                    "request_id": request_id,
                },
            )

        timestamps.append(now)
        response: Response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - len(timestamps))
        response.headers["X-RateLimit-Window"] = str(window)
        return response
