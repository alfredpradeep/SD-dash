"""
SHIELD API Module.

Implements FastAPI routes and request/response schemas.
"""

# Lazy imports to avoid circular dependency (engine <-> api)
# Import schemas directly (no circular risk)
# Import router and middleware only when needed

from shield.api.schemas import (
    ScanRequest,
    ScanResponse,
    ProbeResultResponse,
    HealthResponse,
)

__all__ = [
    "ScanRequest",
    "ScanResponse",
    "ProbeResultResponse",
    "HealthResponse",
]
