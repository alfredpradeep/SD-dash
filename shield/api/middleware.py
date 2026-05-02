"""
SHIELD API Middleware.

Implements request timing and logging middleware.
"""

import time
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from loguru import logger


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request processing time.

    Adds X-Processing-Time-Ms header to responses.
    Logs all requests with method, path, status, and duration.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and measure timing."""
        start_time = time.time()
        request_id = str(int(time.time() * 1000000))

        # Log request start
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} started"
        )

        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"failed with {type(e).__name__}: {e} ({duration_ms}ms)"
            )
            raise

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Add header
        response.headers["X-Processing-Time-Ms"] = str(processing_time_ms)
        response.headers["X-Request-ID"] = request_id

        # Log request completion
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"completed with {response.status_code} ({processing_time_ms}ms)"
        )

        return response
