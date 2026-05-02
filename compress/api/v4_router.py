"""
COMPRESS v4.0 — API Router
============================
FastAPI router for the v4.0 engine pipeline.
Prefix: /v4
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from compress.v4.engines import (
    V4Pipeline,
    LANGUAGE_CONFIG,
    CompressionMode,
    RequiredCapabilityError,
)

# ── Singleton pipeline instance ──────────────────────────────
_pipeline: Optional[V4Pipeline] = None


def get_pipeline() -> V4Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = V4Pipeline()
    return _pipeline


# ── Request / Response models ────────────────────────────────

class V4CompressRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to compress")
    language: str = Field("auto", description="ISO 639-1 code or 'auto'")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    conversation_history: Optional[List[str]] = Field(None, description="Prior exchanges")
    exchange_number: int = Field(1, ge=1, description="Exchange number in conversation")


class V4RouteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("auto")


class V4VerifyRequest(BaseModel):
    original_text: str = Field(..., min_length=1)
    compressed_text: str = Field(..., min_length=1)
    language: str = Field("en")


class V4CacheRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("en")


class V4DecompressRequest(BaseModel):
    compressed_text: str = Field(..., min_length=1)
    target_language: str = Field(..., min_length=2)
    original_text: str = Field("", description="Optional source text for scoring")
    n_candidates: int = Field(3, ge=1, le=5)


class V4RoundTripRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("auto")
    session_id: Optional[str] = None
    exchange_number: int = Field(1, ge=1)
    n_candidates: int = Field(3, ge=1, le=5)


class V4InterviewRunRequest(BaseModel):
    role: str = Field("Backend Engineer", min_length=2)
    seniority: str = Field("Senior", min_length=2)
    topic: str = Field("distributed systems, APIs, and production debugging", min_length=2)
    language: str = Field("en", min_length=2)
    duration_minutes: int = Field(15, ge=1, le=60)
    turn_count: int = Field(12, ge=1, le=30)


class V4InterviewTurnRequest(BaseModel):
    interview_id: str = Field(..., min_length=1)


# ── Router ───────────────────────────────────────────────────

v4_router = APIRouter(prefix="/v4", tags=["COMPRESS v4.0"])


@v4_router.post("/compress")
async def v4_compress(req: V4CompressRequest) -> Dict[str, Any]:
    """Run the full v4.0 compression pipeline."""
    try:
        pipeline = get_pipeline()
        result = await pipeline.process(
            text=req.text,
            language=req.language,
            session_id=req.session_id,
            conversation_history=req.conversation_history,
            exchange_number=req.exchange_number,
        )
        return result
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={
            "capability": e.capability,
            "reason": e.reason,
            "message": "COMPRESS v4 strict mode will not simulate this capability.",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/route")
async def v4_route(req: V4RouteRequest) -> Dict[str, Any]:
    """Run only the PolyCompress Router (language detection + mode selection)."""
    try:
        pipeline = get_pipeline()
        return await pipeline.router.process(req.text, req.language)
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={"capability": e.capability, "reason": e.reason})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/verify")
async def v4_verify(req: V4VerifyRequest) -> Dict[str, Any]:
    """Run only the Five-Stage Verification."""
    try:
        pipeline = get_pipeline()
        return await pipeline.verification.process(
            req.original_text, req.compressed_text, req.language
        )
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={"capability": e.capability, "reason": e.reason})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/cache/check")
async def v4_cache_check(req: V4CacheRequest) -> Dict[str, Any]:
    """Check the cross-lingual semantic cache."""
    try:
        pipeline = get_pipeline()
        return await pipeline.cache.check(req.text, req.language)
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={"capability": e.capability, "reason": e.reason})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/decompress")
async def v4_decompress(req: V4DecompressRequest) -> Dict[str, Any]:
    """Translate compressed v4 output back to the original/source language."""
    try:
        pipeline = get_pipeline()
        return await pipeline.decompress(
            compressed_text=req.compressed_text,
            target_language=req.target_language,
            original_text=req.original_text,
            n_candidates=req.n_candidates,
        )
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={"capability": e.capability, "reason": e.reason})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/round-trip")
async def v4_round_trip(req: V4RoundTripRequest) -> Dict[str, Any]:
    """Run compress -> decompress and return round-trip token/quality summary."""
    try:
        pipeline = get_pipeline()
        return await pipeline.round_trip(
            text=req.text,
            language=req.language,
            session_id=req.session_id,
            exchange_number=req.exchange_number,
            n_candidates=req.n_candidates,
        )
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={"capability": e.capability, "reason": e.reason})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/interview/run")
async def v4_interview_run(req: V4InterviewRunRequest) -> Dict[str, Any]:
    """Run an accelerated real-LLM 15-minute interview scenario."""
    try:
        pipeline = get_pipeline()
        return await pipeline.run_interview(
            role=req.role,
            seniority=req.seniority,
            topic=req.topic,
            language=req.language,
            duration_minutes=req.duration_minutes,
            turn_count=req.turn_count,
        )
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={
            "capability": e.capability,
            "reason": e.reason,
            "message": "AI Interview requires real LLM/ML capabilities in strict mode.",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.post("/interview/turn")
async def v4_interview_turn(req: V4InterviewTurnRequest) -> Dict[str, Any]:
    """Generate the next real-LLM interview turn for an existing interview."""
    try:
        pipeline = get_pipeline()
        return await pipeline.interview_turn(req.interview_id)
    except RequiredCapabilityError as e:
        raise HTTPException(status_code=503, detail={"capability": e.capability, "reason": e.reason})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@v4_router.get("/health")
async def v4_health() -> Dict[str, Any]:
    """Health check with per-engine status."""
    pipeline = get_pipeline()
    return pipeline.get_health()


@v4_router.get("/metrics")
async def v4_metrics() -> Dict[str, Any]:
    """Live pipeline metrics."""
    pipeline = get_pipeline()
    return pipeline.get_metrics()


@v4_router.get("/capabilities")
async def v4_capabilities() -> Dict[str, Any]:
    """Strict-mode capability registry for all v4 ML/provider components."""
    pipeline = get_pipeline()
    return {
        "strict_mode": True,
        "status": pipeline.capabilities.overall_status(),
        "capabilities": pipeline.capabilities.statuses(),
    }


@v4_router.get("/engines")
async def v4_engines() -> Dict[str, Any]:
    """Engine configuration and status."""
    pipeline = get_pipeline()
    capabilities = pipeline.capabilities.statuses()
    return {
        "version": "4.0.0",
        "strict_mode": True,
        "capabilities": capabilities,
        "engines": [
            {
                "id": 1,
                "name": "PolyCompress Router",
                "model": "Script/language metadata + span routing",
                "status": "ready",
                "latency_p95_ms": 2,
                "description": "Language detection + compression mode routing at span level",
                "what": "Chooses Mode A, B, C, or passthrough from source language token tax and span structure.",
                "input": "source prompt + selected/auto language",
                "output": "source language, spans, token tax, compression mode",
            },
            {
                "id": 2,
                "name": "KV-Distill Memory",
                "model": "Fact/entity/open-thread memory distillation",
                "status": "ready",
                "latency_p95_ms": 3,
                "description": "Compresses conversation history into structured memory",
                "what": "Keeps a compact semantic state for long conversations so context cost stays nearly flat.",
                "input": "session id + new utterance",
                "output": "facts, entities, open questions, distilled token count",
            },
            {
                "id": 3,
                "name": "Five-Stage Verification",
                "model": "LaBSE + hard facts + semantic graph + token gate + optional LLM judge",
                "status": capabilities["labse_embeddings"]["status"],
                "latency_p95_ms": 8,
                "description": "5 independent gates — all must pass or original is returned unchanged",
                "what": "Checks cross-lingual meaning, entities/numbers/negations, graph overlap, real token reduction, and optional LLM judge escalation.",
                "input": "original text + candidate compressed text",
                "output": "PASS/KILL verdict and gate reasons",
            },
            {
                "id": 4,
                "name": "Cross-Lingual Semantic Cache",
                "model": "LaBSE (109 languages)",
                "status": capabilities["labse_embeddings"]["status"],
                "latency_p95_ms": 2,
                "description": "Same meaning in any language = cache hit, zero API call",
                "what": "Uses cross-lingual semantic identity so repeated questions across languages avoid model calls.",
                "input": "source text",
                "output": "embedding match, similarity score, hit/miss",
            },
            {
                "id": 5,
                "name": "xRAG Graph-to-Token Bridge",
                "model": "Learned xRAG bridge when asset exists; structured-fact accounting otherwise",
                "status": capabilities["xrag_bridge_model"]["status"],
                "latency_p95_ms": 3,
                "description": "Encodes structured facts into token-efficient memory",
                "what": "Only claims learned xRAG when COMPRESS_XRAG_BRIDGE_PATH exists; otherwise shows honest structured-fact compression accounting.",
                "input": "extracted facts/entities",
                "output": "fact count, text tokens, xRAG/accounting tokens",
            },
            {
                "id": 6,
                "name": "Adaptive Model Router",
                "model": "Deterministic complexity/risk router",
                "status": "ready",
                "latency_p95_ms": 1,
                "description": "Routes 40% of exchanges to mini model at 1/17th cost",
                "what": "Sends simple/factual turns to cheaper models and reserves full models for hard turns.",
                "input": "prompt + verification risk",
                "output": "model tier and cost rate",
            },
            {
                "id": 7,
                "name": "AI Interview Runner",
                "model": "Configured LLM provider + v4 compression pipeline",
                "status": capabilities["interview_provider"]["status"],
                "latency_p95_ms": 0,
                "description": "Generates interviewer and candidate turns with real LLM calls",
                "what": "Runs an accelerated 15-minute interview and compresses context every turn to demonstrate token savings.",
                "input": "role, seniority, topic, language",
                "output": "transcript, compressed memory, per-turn and cumulative token savings",
            },
        ],
    }


@v4_router.get("/languages")
async def v4_languages() -> Dict[str, Any]:
    """Supported languages with token tax and mode info."""
    langs = []
    for code, cfg in LANGUAGE_CONFIG.items():
        langs.append({
            "code": code,
            "name": cfg["name"],
            "script": cfg["script"],
            "token_tax": cfg["tax"],
            "mode": cfg["mode"].value,
            "mode_label": {
                CompressionMode.MODE_A: "Full Semantic Code-Switch",
                CompressionMode.MODE_B: "Selective Native Pruning",
                CompressionMode.MODE_C: "Hybrid Span-Level Routing",
                CompressionMode.PASSTHROUGH: "Passthrough",
            }[cfg["mode"]],
            "zone": cfg["zone"],
            "sample_prompt": cfg["sample"],
            "sample_compressed": cfg["sample_compressed"],
            "expected_savings_min": cfg["savings_range"][0],
            "expected_savings_max": cfg["savings_range"][1],
        })
    langs.sort(key=lambda x: -x["token_tax"])
    return {"languages": langs, "total": len(langs)}
