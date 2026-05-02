"""
MetricsStore: Prometheus metrics collection.

Tracks scans, probes, judgments, and performance metrics.
"""

from typing import Dict, Any
from datetime import datetime

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from loguru import logger


class MetricsStore:
    """Prometheus-based metrics collection."""

    def __init__(self, namespace: str = "shield"):
        """Initialize metrics store."""
        self.namespace = namespace
        self.available = PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            return

        # Counters
        self.scans_total = Counter(
            f"{namespace}_scans_total",
            "Total scans completed",
            ["language", "category", "architecture", "status"],
        )
        self.probes_total = Counter(
            f"{namespace}_probes_total",
            "Total probes executed",
            ["language", "category"],
        )
        self.judgments_total = Counter(
            f"{namespace}_judgments_total",
            "Total judgments made",
            ["level", "result"],
        )
        self.cache_hits = Counter(
            f"{namespace}_cache_hits_total",
            "Total cache hits",
            ["result"],
        )

        # Histograms
        self.bypass_rate = Histogram(
            f"{namespace}_bypass_rate_bucket",
            "Rate of safety bypasses",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.hasd_score = Histogram(
            f"{namespace}_hasd_score",
            "HASD (Harm-Aware Semantic Distance) scores",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.processing_seconds = Histogram(
            f"{namespace}_processing_seconds",
            "Processing time in seconds",
            ["stage"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )
        self.entailment_score = Histogram(
            f"{namespace}_entailment_score",
            "Semantic entailment scores",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.smatch_score = Histogram(
            f"{namespace}_smatch_score",
            "Semantic matching (SMaT) scores",
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.certified_radius = Histogram(
            f"{namespace}_certified_radius",
            "Certified adversarial robustness radius",
            buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5),
        )

        # Gauge
        self.active_scans = Gauge(
            f"{namespace}_active_scans",
            "Number of active scans",
        )

        logger.info("Metrics store initialized")

    # Counter recording methods
    def record_scan_start(self):
        """Record scan start."""
        if self.available:
            self.active_scans.inc()

    def record_scan_completion(
        self,
        language: str,
        category: str,
        architecture: str,
        status: str,
    ):
        """Record scan completion."""
        if self.available:
            self.scans_total.labels(
                language=language,
                category=category,
                architecture=architecture,
                status=status,
            ).inc()
            self.active_scans.dec()

    def record_probe_execution(self, language: str, category: str):
        """Record probe execution."""
        if self.available:
            self.probes_total.labels(language=language, category=category).inc()

    def record_judgment(self, level: str, result: str):
        """Record judgment decision."""
        if self.available:
            self.judgments_total.labels(level=level, result=result).inc()

    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss."""
        if self.available:
            self.cache_hits.labels(result="hit" if hit else "miss").inc()

    # Histogram recording methods
    def record_bypass_rate(self, rate: float):
        """Record bypass rate metric."""
        if self.available:
            self.bypass_rate.observe(min(1.0, max(0.0, rate)))

    def record_hasd_score(self, score: float):
        """Record HASD score."""
        if self.available:
            self.hasd_score.observe(min(1.0, max(0.0, score)))

    def record_processing_time(self, stage: str, seconds: float):
        """Record processing time for a stage."""
        if self.available:
            self.processing_seconds.labels(stage=stage).observe(seconds)

    def record_entailment_score(self, score: float):
        """Record entailment score."""
        if self.available:
            self.entailment_score.observe(min(1.0, max(0.0, score)))

    def record_smatch_score(self, score: float):
        """Record semantic matching score."""
        if self.available:
            self.smatch_score.observe(min(1.0, max(0.0, score)))

    def record_certified_radius(self, radius: float):
        """Record certified robustness radius."""
        if self.available:
            self.certified_radius.observe(radius)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        if not self.available:
            return {"status": "disabled"}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_scans": int(self.active_scans._value.get()),
            "metrics_available": True,
        }
