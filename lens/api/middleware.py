"""
ASGI middleware for transparent request interception.

Intercepts AI API calls passing through the LENS HTTP proxy mode.
Attaches entropy profiling as a non-blocking side-channel:
  - Does NOT modify the request body
  - Does NOT add latency to the critical path (async fire-and-forget)
  - Adds X-LENS-* response headers with key metrics

Headers added:
  X-LENS-IDS: Information Density Score
  X-LENS-ETR-Inequity: Entropy-Token Ratio inequity
  X-LENS-Waste-Type: 2D waste classification
  X-LENS-Low-IDS-Alert: "true" if below threshold
  X-LENS-Request-ID: LENS request tracking ID
"""

import time
import json
import asyncio
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class LENSMiddleware(BaseHTTPMiddleware):
    """
    Non-destructive ASGI middleware for entropy profiling.

    Intercepts /v1/chat/completions and similar AI API endpoints.
    Reads the request body, fires async profiling, passes request unchanged.
    """

    # Paths to intercept (AI API proxy mode)
    INTERCEPT_PATHS = {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",         # Anthropic Claude
    }

    def __init__(self, app, engine):
        super().__init__(app)
        self.engine = engine

    async def dispatch(self, request: Request, call_next) -> Response:
        start_ms = time.monotonic() * 1000

        # Only instrument AI API paths
        if request.url.path not in self.INTERCEPT_PATHS:
            return await call_next(request)

        # Read body (non-destructively)
        body_bytes = await request.body()
        body_text = ""
        language = None
        model_name = "gpt-4o"

        try:
            body_json = json.loads(body_bytes)
            model_name = body_json.get("model", "gpt-4o")
            messages = body_json.get("messages", [])
            # Concatenate user messages for profiling
            body_text = " ".join(
                m.get("content", "") for m in messages
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            )
        except Exception:
            pass

        # Patch request to allow re-reading body downstream
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

        # Call downstream handler
        response = await call_next(request)

        # Fire-and-forget entropy profiling (non-blocking)
        if body_text:
            customer_id = request.headers.get("X-Customer-ID", "default")
            profile_task = asyncio.create_task(
                self._profile_and_attach(
                    text=body_text,
                    model_name=model_name,
                    customer_id=customer_id,
                    language=language,
                    response=response,
                )
            )

        elapsed_ms = (time.monotonic() * 1000) - start_ms
        logger.debug("Middleware intercept complete in {:.1f}ms", elapsed_ms)
        return response

    async def _profile_and_attach(
        self,
        text: str,
        model_name: str,
        customer_id: str,
        language: str | None,
        response: Response,
    ) -> None:
        """Run profiling and attach results to response headers."""
        try:
            profile = await self.engine.profile_request(
                text=text,
                model_name=model_name,
                customer_id=customer_id,
                pre_detected_language=language,
            )
            response.headers["X-LENS-IDS"] = str(round(profile.ids_score, 3))
            response.headers["X-LENS-ETR-Inequity"] = str(round(profile.etr_inequity_ratio, 2))
            response.headers["X-LENS-Waste-Type"] = profile.waste_type
            response.headers["X-LENS-Low-IDS-Alert"] = "true" if profile.low_ids_alert else "false"
            response.headers["X-LENS-Request-ID"] = profile.request_id
        except Exception as e:
            logger.debug("Middleware profiling failed: {}", e)
