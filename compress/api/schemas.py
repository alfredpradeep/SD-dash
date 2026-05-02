"""
Pydantic request/response models for the COMPRESS API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CompressRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    language: str = Field(
        ..., description="ISO 639-1 language code e.g. 'ta', 'hi', 'ar'"
    )
    target_tokenizer: str = Field(default="gpt-4o")
    customer_id: str = Field(default="default")
    min_reduction_threshold: float = Field(default=0.15, ge=0.05, le=0.80)
    semantic_threshold: float = Field(default=0.91, ge=0.80, le=0.99)
    graph_threshold: float = Field(default=0.88, ge=0.75, le=0.99)

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be whitespace only")
        return v


class CompressResponse(BaseModel):
    original_text: str
    compressed_text: str
    language: str
    target_tokenizer: str
    original_tokens: int
    compressed_tokens: int
    reduction_ratio: float
    semantic_similarity: float
    graph_jaccard: float
    stes_score: float
    compression_applied: bool
    rejection_reason: Optional[str]
    processing_ms: float
    candidates_evaluated: int


class BatchItem(BaseModel):
    text: str
    language: str
    tokenizer: str = "gpt-4o"


class BatchCompressRequest(BaseModel):
    items: list[BatchItem] = Field(..., max_length=100)
    customer_id: str = "default"


class BatchCompressResponse(BaseModel):
    results: list[CompressResponse]


class HealthResponse(BaseModel):
    status: str
    extractor: str
    searcher: str
    gate: str
    reconstructor: str = "unknown"
    decompressor: str = "unknown"
    distiller: str = "unknown"
    cache: str


# ── Context Distillation Schemas ──

class SessionTurnRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    user_text: str = Field(..., min_length=1, max_length=8000)
    assistant_text: str = Field(default="", max_length=8000)
    language: str = Field(default="en")


class SessionTurnResponse(BaseModel):
    session_id: str
    turn_count: int
    compressed_context: str
    fact_count: int
    active_facts: int
    entity_count: int
    open_threads: int
    contradictions: int
    context_tokens_estimate: int


class SessionStateResponse(BaseModel):
    session_id: str
    turn_count: int
    fact_count: int
    active_facts: int
    entity_count: int
    open_threads: int
    contradictions: int
    sentiment: Optional[tuple] = None


# ── Decompression Schemas ──

class DecompressRequest(BaseModel):
    compressed_text: str = Field(
        ..., min_length=1, max_length=8000,
        description="Compressed English text to translate back",
    )
    target_language: str = Field(
        ..., description="ISO 639-1 code for target language (e.g. 'ta')",
    )
    original_text: str = Field(
        default="",
        description="Optional original text for quality comparison",
    )
    n_candidates: int = Field(
        default=3, ge=1, le=5,
        description="Number of translation candidates to generate",
    )
