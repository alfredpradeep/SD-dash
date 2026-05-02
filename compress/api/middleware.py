"""
Request timing middleware.

Adds X-Processing-Time-Ms header to every response.
Logs request path, method, status code, and latency.
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class TimingMiddleware(BaseHTTPMiddleware):
    """Request timing middleware with latency header and logging."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "{} {} -> {} ({:.1f}ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
