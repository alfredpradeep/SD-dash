"""
Prometheus metrics collector for the SCL engine.

Exposed metrics:
- scl_compressions_total: Counter by language, tokenizer, applied/rejected
- scl_compression_ratio: Histogram of token reduction ratios
- scl_processing_seconds: Histogram of request latency
- scl_semantic_similarity: Histogram of LaBSE scores
- scl_graph_jaccard: Histogram of graph Jaccard scores
- scl_cache_hits_total: Counter of cache hits by language
- scl_candidates_evaluated: Histogram of candidates per request
"""

from prometheus_client import Counter, Histogram, start_http_server
from compress.lattice.structures import CompressionResult
from compress.config import Config
from loguru import logger


class MetricsStore:
    """Prometheus metrics collector for the SCL engine."""

    def __init__(self, config: Config):
        self.config = config

        self.compressions_total = Counter(
            "scl_compressions_total",
            "Total compression requests",
            ["language", "tokenizer", "status"],
        )
        self.compression_ratio = Histogram(
            "scl_compression_ratio",
            "Token reduction ratio (0.0-1.0)",
            buckets=[
                0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
                0.80, 1.0,
            ],
        )
        self.processing_seconds = Histogram(
            "scl_processing_seconds",
            "Request processing time in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )
        self.semantic_similarity = Histogram(
            "scl_semantic_similarity",
            "LaBSE cosine similarity scores",
            buckets=[
                0.70, 0.80, 0.85, 0.88, 0.91, 0.93, 0.95, 0.97, 0.99,
                1.0,
            ],
        )
        self.graph_jaccard = Histogram(
            "scl_graph_jaccard",
            "Graph Jaccard similarity scores",
            buckets=[0.60, 0.70, 0.80, 0.85, 0.88, 0.91, 0.95, 1.0],
        )
        self.cache_hits = Counter(
            "scl_cache_hits_total", "Cache hits", ["language"]
        )
        self.candidates_evaluated = Histogram(
            "scl_candidates_evaluated",
            "Number of candidates evaluated per request",
            buckets=[0, 1, 2, 4, 6, 8, 12, 16],
        )

        try:
            start_http_server(config.metrics_port)
            logger.info(
                "Prometheus metrics server started on port {}",
                config.metrics_port,
            )
        except Exception as e:
            logger.warning("Failed to start metrics server: {}", e)

    def record_compression(
        self, result: CompressionResult, customer_id: str
    ) -> None:
        """Record metrics for a completed compression."""
        status = "applied" if result.compression_applied else "rejected"
        self.compressions_total.labels(
            language=result.source_language,
            tokenizer=result.target_tokenizer,
            status=status,
        ).inc()

        self.compression_ratio.observe(result.reduction_ratio)
        self.processing_seconds.observe(result.processing_ms / 1000.0)
        self.semantic_similarity.observe(result.semantic_similarity)
        self.graph_jaccard.observe(result.graph_jaccard)
        self.candidates_evaluated.observe(result.candidates_evaluated)

    def record_cache_hit(self, language: str) -> None:
        """Record a cache hit."""
        self.cache_hits.labels(language=language).inc()
