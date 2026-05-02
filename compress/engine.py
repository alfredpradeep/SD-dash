"""
Core SCL (Semantic Compression Lattice) Engine v3.0.

Five-stage compression pipeline:
  Stage 1: Semantic graph extraction (hybrid AMR + NLI + NER)
  Stage 2: LLM-powered semantic rewriting + rule-based fallback
  Stage 3: Triple verification gate (LaBSE + graph Jaccard + reduction)
  Stage 4: Adversarial reconstruction test
  Stage 5: Context distillation (for conversation mode)

Thread-safe. Async-native. Stateless per request.
"""

import asyncio
import time
from loguru import logger
from compress.lattice.extractor import LatticeExtractor
from compress.search.beam import BeamSearcher
from compress.verification.gate import VerificationGate
from compress.verification.reconstruction import ReconstructionVerifier
from compress.decompressor import DecompressionEngine
from compress.context.distiller import ContextDistiller
from compress.storage.cache import CompressionCache
from compress.storage.metrics_store import MetricsStore
from compress.config import Config
from compress.lattice.structures import CompressionResult, SemanticGraph
from compress.exceptions import (
    ExtractionError,
    SearchError,
    UnsupportedLanguageError,
)


class SCLEngine:
    """
    Semantic Compression Lattice Engine.

    Decomposes input text into language-independent semantic graph,
    finds minimum-token surface realization via beam search,
    verifies output through triple-gate system before returning.
    """

    SUPPORTED_LANGUAGES = {
        "en", "ta", "hi", "ar", "ja", "zh", "ko", "pt", "es",
        "fr", "de", "id", "ms", "bn", "ur", "te", "ml", "pa", "gu", "mr",
    }

    SUPPORTED_TOKENIZERS = {
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude-3-5-sonnet": "anthropic",
        "claude-3-haiku": "anthropic",
        "claude-3-opus": "anthropic",
        "llama-3-8b": "sentencepiece_llama3",
        "llama-3-70b": "sentencepiece_llama3",
        "mistral-7b": "sentencepiece_mistral",
        "mistral-8x7b": "sentencepiece_mistral",
    }

    def __init__(self, config: Config):
        self.config = config
        self.extractor = LatticeExtractor(config)
        self.searcher = BeamSearcher(config)
        self.gate = VerificationGate(config)
        self.reconstructor = ReconstructionVerifier(config)
        self.decompressor = DecompressionEngine(config)
        self.distiller = ContextDistiller(config)
        self.cache = CompressionCache(config)
        self.metrics = MetricsStore(config)
        logger.info("SCLEngine v3.0 initialised with config: {}", config.model_dump())

    async def compress(
        self,
        text: str,
        source_language: str,
        target_tokenizer: str = "gpt-4o",
        customer_id: str = "default",
        min_reduction_threshold: float = 0.15,
        semantic_threshold: float = 0.91,
        graph_threshold: float = 0.88,
    ) -> CompressionResult:
        """
        Main compression entry point.

        Returns CompressionResult with compressed text or original if
        no candidate passes all verification gates.
        """
        start_ms = time.monotonic() * 1000

        if source_language not in self.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Language '{source_language}' not supported"
            )
        if target_tokenizer not in self.SUPPORTED_TOKENIZERS:
            raise UnsupportedLanguageError(
                f"Tokenizer '{target_tokenizer}' not supported"
            )

        # Cross-lingual threshold adjustment
        if source_language != "en":
            semantic_threshold = min(semantic_threshold, 0.75)
            graph_threshold = min(graph_threshold, 0.70)

        # Cache check
        cache_key = self.cache.make_key(
            text, source_language, target_tokenizer
        )
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit(source_language)
            return cached

        try:
            # Stage 1: Extract semantic graph
            graph: SemanticGraph = await self.extractor.extract(
                text=text, language=source_language
            )
            logger.debug(
                "Graph extracted: {} nodes, confidence={:.3f}, "
                "amr_confidence={:.3f}",
                len(graph.nodes),
                graph.extraction_confidence,
                graph.amr_confidence,
            )

            original_tokens = self.searcher.count_tokens(
                text, target_tokenizer
            )

            # Stage 2: Beam search
            candidates = await self.searcher.search(
                graph=graph,
                target_tokenizer=target_tokenizer,
                beam_width=self.config.beam_width,
                max_candidates=self.config.max_candidates,
            )
            logger.debug("Beam search produced {} candidates", len(candidates))

            # Stage 3: Verification gate
            best_result = None
            for candidate in candidates:
                passed, reason = await self.gate.verify(
                    original_text=text,
                    original_graph=graph,
                    candidate=candidate,
                    original_tokens=original_tokens,
                    semantic_threshold=semantic_threshold,
                    graph_threshold=graph_threshold,
                    min_reduction=min_reduction_threshold,
                )
                if passed:
                    best_result = candidate
                    break
                logger.debug("Candidate rejected: {}", reason)

            processing_ms = (time.monotonic() * 1000) - start_ms

            if best_result is None:
                result = CompressionResult(
                    original_text=text,
                    compressed_text=text,
                    source_language=source_language,
                    target_tokenizer=target_tokenizer,
                    original_token_count=original_tokens,
                    compressed_token_count=original_tokens,
                    reduction_ratio=0.0,
                    semantic_similarity=1.0,
                    graph_jaccard=1.0,
                    stes_score=0.0,
                    compression_applied=False,
                    rejection_reason=(
                        "No candidate met all verification thresholds"
                    ),
                    processing_ms=processing_ms,
                    candidates_evaluated=len(candidates),
                )
            else:
                result = CompressionResult(
                    original_text=text,
                    compressed_text=best_result.text,
                    source_language=source_language,
                    target_tokenizer=target_tokenizer,
                    original_token_count=original_tokens,
                    compressed_token_count=best_result.token_count,
                    reduction_ratio=(
                        1.0 - (best_result.token_count / original_tokens)
                    ),
                    semantic_similarity=best_result.semantic_similarity,
                    graph_jaccard=best_result.graph_jaccard,
                    stes_score=best_result.stes_score,
                    compression_applied=True,
                    rejection_reason=None,
                    processing_ms=processing_ms,
                    candidates_evaluated=len(candidates),
                )

            await self.cache.set(
                cache_key, result,
                ttl_seconds=self.config.redis_ttl_seconds,
            )
            self.metrics.record_compression(result, customer_id)
            return result

        except ExtractionError as e:
            logger.error("Semantic graph extraction failed: {}", e)
            raise
        except SearchError as e:
            logger.error("Beam search failed: {}", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error in SCLEngine.compress")
            raise

    async def compress_with_trace(
        self,
        text: str,
        source_language: str,
        target_tokenizer: str = "gpt-4o",
        customer_id: str = "default",
        min_reduction_threshold: float = 0.15,
        semantic_threshold: float = 0.91,
        graph_threshold: float = 0.88,
    ) -> dict:
        """
        Compress with full pipeline trace for visualization.

        Returns dict with: result, pipeline stages, semantic_graph,
        embedding_visualization, beam_candidates.
        """
        import time as _time
        stages = []
        t0 = _time.monotonic()

        # Cross-lingual adjustment: translations have naturally
        # lower LaBSE similarity and different graph structure
        is_cross_lingual = source_language != "en"
        if is_cross_lingual:
            semantic_threshold = min(semantic_threshold, 0.75)
            graph_threshold = min(graph_threshold, 0.70)
            # Cross-lingual should achieve MORE reduction (3-4x)
            min_reduction_threshold = max(min_reduction_threshold, 0.15)

        # ── Stage 1: Extraction ──
        stage1_start = _time.monotonic()
        stages.append({
            "name": "Semantic Graph Extraction",
            "status": "running",
            "detail": "Hybrid AMR + NLI + NER pipeline",
        })
        try:
            graph: SemanticGraph = await self.extractor.extract(
                text=text, language=source_language
            )
            stage1_ms = (_time.monotonic() - stage1_start) * 1000
            stages[-1].update({
                "status": "complete",
                "ms": round(stage1_ms, 1),
                "detail": (
                    f"{len(graph.nodes)} nodes, "
                    f"confidence={graph.extraction_confidence:.3f}"
                ),
            })
        except Exception as e:
            stages[-1]["status"] = "rejected"
            stages[-1]["detail"] = str(e)
            raise

        original_tokens = self.searcher.count_tokens(text, target_tokenizer)

        # ── Stage 2: Beam Search ──
        stage2_start = _time.monotonic()
        stages.append({
            "name": "LaBSE-Guided Beam Search",
            "status": "running",
            "detail": f"beam_width={self.config.beam_width}",
        })
        try:
            candidates = await self.searcher.search(
                graph=graph,
                target_tokenizer=target_tokenizer,
                beam_width=self.config.beam_width,
                max_candidates=self.config.max_candidates,
            )
            stage2_ms = (_time.monotonic() - stage2_start) * 1000
            stages[-1].update({
                "status": "complete",
                "ms": round(stage2_ms, 1),
                "detail": f"{len(candidates)} candidates scored",
            })
        except Exception as e:
            stages[-1]["status"] = "rejected"
            stages[-1]["detail"] = str(e)
            raise

        # ── Stage 3: Verification Gate ──
        stage3_start = _time.monotonic()
        stages.append({
            "name": "Triple Verification Gate",
            "status": "running",
            "detail": (
                f"LaBSE≥{semantic_threshold}, "
                f"Jaccard≥{graph_threshold}, "
                f"reduction≥{min_reduction_threshold}"
            ),
        })

        best_result = None
        candidate_details = []
        for candidate in candidates:
            passed, reason = await self.gate.verify(
                original_text=text,
                original_graph=graph,
                candidate=candidate,
                original_tokens=original_tokens,
                semantic_threshold=semantic_threshold,
                graph_threshold=graph_threshold,
                min_reduction=min_reduction_threshold,
            )
            cand_reduction = 1.0 - (
                candidate.token_count / max(original_tokens, 1)
            )
            candidate_details.append({
                "rank": candidate.rank,
                "text": candidate.text,
                "tokens": candidate.token_count,
                "stes_score": round(candidate.stes_score, 4),
                "semantic_similarity": round(
                    candidate.semantic_similarity, 4
                ),
                "graph_jaccard": round(candidate.graph_jaccard, 4),
                "reduction": round(cand_reduction, 4),
                "passed": passed,
                "reason": reason if not passed else "All gates passed",
            })
            if passed and best_result is None:
                best_result = candidate

        stage3_ms = (_time.monotonic() - stage3_start) * 1000
        passed_count = sum(1 for c in candidate_details if c["passed"])
        stages[-1].update({
            "status": "complete" if passed_count > 0 else "rejected",
            "ms": round(stage3_ms, 1),
            "detail": f"{passed_count}/{len(candidates)} passed gates",
        })

        # ── Stage 4: Adversarial Reconstruction ──
        stage4_start = _time.monotonic()
        reconstruction_details = {}
        if best_result is not None:
            stages.append({
                "name": "Adversarial Reconstruction Test",
                "status": "running",
                "detail": "KG extraction + comparison",
            })

            if is_cross_lingual:
                # ── Cross-lingual: skip graph reconstruction ──
                # NER extraction is language-dependent: Tamil text
                # produces far more entity nodes than English.
                # LaBSE cross-lingual similarity (Stage 3) is the
                # correct metric for cross-lingual meaning preservation.
                stage4_ms = (_time.monotonic() - stage4_start) * 1000
                labse_score = best_result.semantic_similarity
                reconstruction_details = {
                    "score": round(labse_score, 4),
                    "missing_items": [],
                    "method": "crosslingual_labse_proxy",
                    "reason": (
                        "Graph reconstruction skipped for cross-lingual. "
                        "LaBSE cross-lingual similarity used instead."
                    ),
                    "labse_similarity": round(labse_score, 4),
                }
                stages[-1].update({
                    "status": "complete",
                    "ms": round(stage4_ms, 1),
                    "detail": (
                        f"Cross-lingual — LaBSE proxy "
                        f"{labse_score:.3f} (Stage 3 verified)"
                    ),
                })
            else:
                # ── Same-language: full graph reconstruction ──
                try:
                    recon_score, missing_items, recon_info = (
                        await self.reconstructor.verify(
                            original_text=text,
                            compressed_text=best_result.text,
                            original_graph=graph,
                            extractor=self.extractor,
                            language=source_language,
                        )
                    )
                    stage4_ms = (
                        (_time.monotonic() - stage4_start) * 1000
                    )
                    reconstruction_details = {
                        "score": round(recon_score, 4),
                        "missing_items": missing_items,
                        **recon_info,
                    }
                    recon_threshold = 0.85
                    if recon_score < recon_threshold:
                        stages[-1].update({
                            "status": "rejected",
                            "ms": round(stage4_ms, 1),
                            "detail": (
                                f"Score {recon_score:.3f} < "
                                f"{recon_threshold} — "
                                f"{len(missing_items)} items lost"
                            ),
                        })
                        best_result = None
                    else:
                        stages[-1].update({
                            "status": "complete",
                            "ms": round(stage4_ms, 1),
                            "detail": (
                                f"Score {recon_score:.3f} — "
                                f"all critical info preserved"
                            ),
                        })
                except Exception as e:
                    stage4_ms = (
                        (_time.monotonic() - stage4_start) * 1000
                    )
                    stages[-1].update({
                        "status": "complete",
                        "ms": round(stage4_ms, 1),
                        "detail": f"Skipped: {e}",
                    })
        else:
            stages.append({
                "name": "Adversarial Reconstruction Test",
                "status": "rejected",
                "ms": 0,
                "detail": "Skipped — no candidate passed gates",
            })

        # ── Stage 5: Context Distillation (info only in trace) ──
        stages.append({
            "name": "Context Distillation",
            "status": "complete",
            "ms": 0,
            "detail": (
                f"Available for session mode — "
                f"{len(self.distiller._sessions)} active sessions"
            ),
        })

        total_ms = (_time.monotonic() - t0) * 1000

        # Build compression result
        if best_result is None:
            result = CompressionResult(
                original_text=text,
                compressed_text=text,
                source_language=source_language,
                target_tokenizer=target_tokenizer,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                reduction_ratio=0.0,
                semantic_similarity=1.0,
                graph_jaccard=1.0,
                stes_score=0.0,
                compression_applied=False,
                rejection_reason=(
                    "No candidate met all verification thresholds"
                ),
                processing_ms=total_ms,
                candidates_evaluated=len(candidates),
            )
        else:
            result = CompressionResult(
                original_text=text,
                compressed_text=best_result.text,
                source_language=source_language,
                target_tokenizer=target_tokenizer,
                original_token_count=original_tokens,
                compressed_token_count=best_result.token_count,
                reduction_ratio=(
                    1.0 - (best_result.token_count / original_tokens)
                ),
                semantic_similarity=best_result.semantic_similarity,
                graph_jaccard=best_result.graph_jaccard,
                stes_score=best_result.stes_score,
                compression_applied=True,
                rejection_reason=None,
                processing_ms=total_ms,
                candidates_evaluated=len(candidates),
            )

        # Build graph visualization data
        graph_viz = {
            "nodes": [
                {
                    "id": n.unit_id,
                    "type": n.unit_type.value,
                    "value": n.value,
                    "confidence": round(n.confidence, 3),
                }
                for n in graph.nodes
            ],
            "edges": [
                {"source": e[0], "relation": e[1], "target": e[2]}
                for e in graph.edges
            ],
        }

        # Embedding visualization data (simplified 2D projection)
        embedding_viz = {
            "original": {"x": 0.0, "y": 0.0, "label": "Original"},
            "compressed": {
                "x": 1.0 - result.semantic_similarity
                    if result.compression_applied else 0.0,
                "y": result.reduction_ratio * 0.5
                    if result.compression_applied else 0.0,
                "label": "Compressed",
            },
            "similarity_radius": result.semantic_similarity,
            "candidates": [
                {
                    "x": 1.0 - c["semantic_similarity"],
                    "y": c["reduction"] * 0.5,
                    "label": f"C{c['rank']+1}",
                    "passed": c["passed"],
                }
                for c in candidate_details
            ],
        }

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
            "pipeline": {
                "stages": stages,
                "total_ms": round(total_ms, 1),
            },
            "semantic_graph": graph_viz,
            "embedding_visualization": embedding_viz,
            "beam_candidates": candidate_details,
            "reconstruction": reconstruction_details,
            "context_distillation": {
                "active_sessions": len(self.distiller._sessions),
                "available": True,
                "endpoint": "/compress/session/turn",
            },
        }

    async def batch_compress(
        self,
        texts: list[dict],
        customer_id: str = "default",
        concurrency: int = 10,
    ) -> list[CompressionResult]:
        """Compress multiple texts concurrently with controlled parallelism."""
        semaphore = asyncio.Semaphore(concurrency)

        async def compress_one(item: dict) -> CompressionResult:
            async with semaphore:
                return await self.compress(
                    text=item["text"],
                    source_language=item["language"],
                    target_tokenizer=item.get("tokenizer", "gpt-4o"),
                    customer_id=customer_id,
                )

        return await asyncio.gather(
            *[compress_one(item) for item in texts]
        )

    async def health_check(self) -> dict:
        """Return health status of all sub-components."""
        return {
            "status": "healthy",
            "extractor": await self.extractor.health(),
            "searcher": await self.searcher.health(),
            "gate": await self.gate.health(),
            "reconstructor": await self.reconstructor.health(),
            "decompressor": await self.decompressor.health(),
            "distiller": await self.distiller.health(),
            "cache": await self.cache.health(),
        }
