"""
FastAPI router for the COMPRESS API.

Endpoints:
  POST /compress/          - Compress single text
  POST /compress/batch     - Compress multiple texts concurrently
  GET  /compress/health    - Health check
  GET  /compress/supported - List supported languages and tokenizers
"""

from fastapi import APIRouter, HTTPException, Depends
from compress.api.schemas import (
    CompressRequest,
    CompressResponse,
    BatchCompressRequest,
    BatchCompressResponse,
    HealthResponse,
    SessionTurnRequest,
    SessionTurnResponse,
    SessionStateResponse,
    DecompressRequest,
)
from compress.exceptions import (
    UnsupportedLanguageError,
    ExtractionError,
    SearchError,
)
from loguru import logger

router = APIRouter(prefix="/compress", tags=["COMPRESS"])


def get_engine():
    from compress.main import engine_instance
    return engine_instance


@router.post("/", response_model=CompressResponse, status_code=200)
async def compress_text(
    request: CompressRequest,
    engine=Depends(get_engine),
) -> CompressResponse:
    """
    Compress input text using Semantic Compression Lattice.

    Returns compressed text with semantic preservation guarantee.
    If no valid compression found, returns original with
    compression_applied=False.
    """
    try:
        result = await engine.compress(
            text=request.text,
            source_language=request.language,
            target_tokenizer=request.target_tokenizer,
            customer_id=request.customer_id,
            min_reduction_threshold=request.min_reduction_threshold,
            semantic_threshold=request.semantic_threshold,
            graph_threshold=request.graph_threshold,
        )
        return CompressResponse(
            original_text=result.original_text,
            compressed_text=result.compressed_text,
            language=result.source_language,
            target_tokenizer=result.target_tokenizer,
            original_tokens=result.original_token_count,
            compressed_tokens=result.compressed_token_count,
            reduction_ratio=result.reduction_ratio,
            semantic_similarity=result.semantic_similarity,
            graph_jaccard=result.graph_jaccard,
            stes_score=result.stes_score,
            compression_applied=result.compression_applied,
            rejection_reason=result.rejection_reason,
            processing_ms=result.processing_ms,
            candidates_evaluated=result.candidates_evaluated,
        )
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (ExtractionError, SearchError) as e:
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {e}"
        )
    except Exception as e:
        logger.exception("Unhandled error in /compress")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/batch", response_model=BatchCompressResponse, status_code=200
)
async def batch_compress(
    request: BatchCompressRequest,
    engine=Depends(get_engine),
) -> BatchCompressResponse:
    """Compress multiple texts concurrently."""
    if len(request.items) > 100:
        raise HTTPException(
            status_code=422, detail="Maximum 100 items per batch"
        )
    results = await engine.batch_compress(
        texts=[item.model_dump() for item in request.items],
        customer_id=request.customer_id,
    )
    return BatchCompressResponse(
        results=[
            CompressResponse(
                original_text=r.original_text,
                compressed_text=r.compressed_text,
                language=r.source_language,
                target_tokenizer=r.target_tokenizer,
                original_tokens=r.original_token_count,
                compressed_tokens=r.compressed_token_count,
                reduction_ratio=r.reduction_ratio,
                semantic_similarity=r.semantic_similarity,
                graph_jaccard=r.graph_jaccard,
                stes_score=r.stes_score,
                compression_applied=r.compression_applied,
                rejection_reason=r.rejection_reason,
                processing_ms=r.processing_ms,
                candidates_evaluated=r.candidates_evaluated,
            )
            for r in results
        ]
    )


@router.post("/trace", status_code=200)
async def compress_with_trace(
    request: CompressRequest,
    engine=Depends(get_engine),
) -> dict:
    """
    Compress with full pipeline trace for visualization.
    Returns compression result + semantic graph + embedding viz + candidates.
    """
    try:
        if hasattr(engine, "compress_with_trace"):
            return await engine.compress_with_trace(
                text=request.text,
                source_language=request.language,
                target_tokenizer=request.target_tokenizer,
                customer_id=request.customer_id,
                min_reduction_threshold=request.min_reduction_threshold,
                semantic_threshold=request.semantic_threshold,
                graph_threshold=request.graph_threshold,
            )
        else:
            # Fallback for production engine without trace
            result = await engine.compress(
                text=request.text,
                source_language=request.language,
                target_tokenizer=request.target_tokenizer,
                customer_id=request.customer_id,
                min_reduction_threshold=request.min_reduction_threshold,
                semantic_threshold=request.semantic_threshold,
                graph_threshold=request.graph_threshold,
            )
            return {
                "result": {
                    "original_text": result.original_text,
                    "compressed_text": result.compressed_text,
                    "language": result.source_language,
                    "target_tokenizer": result.target_tokenizer,
                    "original_tokens": result.original_token_count,
                    "compressed_tokens": result.compressed_token_count,
                    "reduction_ratio": result.reduction_ratio,
                    "semantic_similarity": result.semantic_similarity,
                    "graph_jaccard": result.graph_jaccard,
                    "stes_score": result.stes_score,
                    "compression_applied": result.compression_applied,
                    "rejection_reason": result.rejection_reason,
                    "processing_ms": result.processing_ms,
                    "candidates_evaluated": result.candidates_evaluated,
                },
                "pipeline": {"stages": [], "total_ms": result.processing_ms},
                "semantic_graph": {"nodes": [], "edges": []},
                "embedding_visualization": None,
                "beam_candidates": [],
            }
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error in /compress/trace")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/decompress", status_code=200)
async def decompress_text(
    request: DecompressRequest,
    engine=Depends(get_engine),
) -> dict:
    """
    Decompress: translate compressed English back to the original language.

    The full COMPRESS cycle:
      1. COMPRESS:   Tamil (566 tok) → English (67 tok)   [88% savings]
      2. STORE/SEND: English (67 tok)                     [cheap API calls]
      3. DECOMPRESS: English (67 tok) → Tamil (~200 tok)  [natural output]

    Net round-trip savings: 50-70% token reduction.
    """
    try:
        result = await engine.decompressor.decompress(
            compressed_text=request.compressed_text,
            target_language=request.target_language,
            original_text=request.original_text,
            n_candidates=request.n_candidates,
        )
        return result
    except Exception as e:
        logger.exception("Unhandled error in /compress/decompress")
        raise HTTPException(
            status_code=500, detail=f"Decompression failed: {e}"
        )


@router.get("/health", response_model=HealthResponse)
async def health(
    engine=Depends(get_engine),
) -> HealthResponse:
    status = await engine.health_check()
    return HealthResponse(**status)


@router.get("/supported")
async def supported(engine=Depends(get_engine)) -> dict:
    return {
        "languages": sorted(engine.SUPPORTED_LANGUAGES),
        "tokenizers": sorted(engine.SUPPORTED_TOKENIZERS.keys()),
    }


# ── Context Distillation Endpoints ──────────────────────────────

@router.post(
    "/session/turn",
    response_model=SessionTurnResponse,
    status_code=200,
)
async def session_turn(
    request: SessionTurnRequest,
    engine=Depends(get_engine),
) -> SessionTurnResponse:
    """
    Process a conversation turn and update the distilled context.

    Send each user+assistant exchange to maintain a living semantic state.
    Returns compressed context prompt (~2000 tokens constant size)
    regardless of conversation length.
    """
    try:
        extractor = engine.extractor if hasattr(engine, "extractor") else None
        state = await engine.distiller.add_turn(
            session_id=request.session_id,
            user_text=request.user_text,
            assistant_text=request.assistant_text,
            language=request.language,
            extractor=extractor,
        )
        compressed_context = state.to_prompt()

        # Estimate tokens in compressed context
        token_estimate = len(compressed_context.split()) * 1.3

        stats = state.to_dict()
        return SessionTurnResponse(
            session_id=request.session_id,
            turn_count=stats["turn_count"],
            compressed_context=compressed_context,
            fact_count=stats["fact_count"],
            active_facts=stats["active_facts"],
            entity_count=stats["entity_count"],
            open_threads=stats["open_threads"],
            contradictions=stats["contradictions"],
            context_tokens_estimate=int(token_estimate),
        )
    except Exception as e:
        logger.exception("Error in /session/turn")
        raise HTTPException(
            status_code=500, detail=f"Session turn failed: {e}"
        )


@router.get(
    "/session/{session_id}",
    response_model=SessionStateResponse,
    status_code=200,
)
async def session_state(
    session_id: str,
    engine=Depends(get_engine),
) -> SessionStateResponse:
    """Get the current state of a distillation session."""
    try:
        state = engine.distiller.get_or_create_session(session_id)
        stats = state.to_dict()
        return SessionStateResponse(**stats)
    except Exception as e:
        logger.exception("Error in /session/{session_id}")
        raise HTTPException(
            status_code=500, detail=f"Session state failed: {e}"
        )


@router.get("/session/{session_id}/context", status_code=200)
async def session_context(
    session_id: str,
    engine=Depends(get_engine),
) -> dict:
    """
    Get the compressed context prompt for a session.

    This is the prompt that replaces the entire conversation history.
    Target: ~2000 tokens regardless of conversation length.
    """
    try:
        context = engine.distiller.get_compressed_context(session_id)
        return {
            "session_id": session_id,
            "compressed_context": context,
            "token_estimate": int(len(context.split()) * 1.3),
        }
    except Exception as e:
        logger.exception("Error in /session/{session_id}/context")
        raise HTTPException(
            status_code=500, detail=f"Context retrieval failed: {e}"
        )


@router.get("/sessions", status_code=200)
async def list_sessions(
    engine=Depends(get_engine),
) -> dict:
    """List all active distillation sessions."""
    try:
        sessions = {}
        for sid, state in engine.distiller._sessions.items():
            sessions[sid] = state.to_dict()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.exception("Error in /sessions")
        raise HTTPException(
            status_code=500, detail=f"Sessions list failed: {e}"
        )
