"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditRequest(BaseModel):
    languages: List[str] = Field(default_factory=lambda: ["en", "hi", "ta", "ar", "zh"])
    categories: List[str] = Field(default_factory=lambda: ["financial", "hiring", "medical"])
    n_probes_per_cell: int = 2
    target_endpoint: Optional[str] = None
    target_api_key: Optional[str] = None
    target_model: Optional[str] = None


class GuardRequest(BaseModel):
    text: str
    language: str = "en"
    session_id: str = "default"
    direction: str = "INBOUND"  # INBOUND | OUTBOUND
    prompt: Optional[str] = None   # required for OUTBOUND: the original user prompt


class GuardStreamTurn(BaseModel):
    session_id: str
    user_message: str
    ai_response: str
    language: str = "en"


class HealthResponse(BaseModel):
    status: str
    version: str
    pillars: Dict[str, str]
    algorithms: Dict[str, bool]
