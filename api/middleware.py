from __future__ import annotations

import hashlib
import hmac
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Paths that don't require API key authentication.
_PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/static"}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request-id into every request and response.

    The request-id is bound to structlog so all log lines within a request
    carry the same ID for easy tracing.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind to structlog context for this request
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        structlog.contextvars.clear_contextvars()
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate X-API-Key header on all non-public routes.

    Uses ``hmac.compare_digest`` for constant-time comparison to prevent
    timing attacks.
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        # Allow public paths and all static file paths
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
            logger.warning(
                "Unauthorized request",
                path=path,
                request_id=request_id,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error_code": "authentication_error",
                    "message": "Invalid or missing X-API-Key header",
                    "detail": "Provide a valid API key in the X-API-Key header",
                    "request_id": request_id,
                },
            )

        return await call_next(request)
