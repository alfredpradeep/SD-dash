"""
Provider Health & SLA Monitor

Monitors the health and performance of AI providers by analyzing real production
request data. Tracks latency trends, error rates, and degradation over time.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
import statistics


@dataclass
class HealthProbe:
    """A single health observation from a production request."""
    provider: str
    model: str
    timestamp: datetime
    latency_ms: float
    status: str              # "healthy", "degraded", "down", "error"
    error_message: str = ""
    tokens_per_second: float = 0.0


@dataclass
class ProviderSLA:
    """SLA compliance report for a provider over a time period."""
    provider: str
    period_hours: int
    total_probes: int
    successful_probes: int
    uptime_pct: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate_pct: float
    latency_trend: str       # "improving", "stable", "degrading"
    incidents: list[dict] = field(default_factory=list)  # [{start, end, duration_min, type}]


class HealthMonitor:
    """
    Tracks provider health from actual request data.

    NOT by making synthetic probe calls — by analyzing real production
    request latencies and error rates. Every request through LENS
    automatically contributes to the health profile.

    For single-provider enterprises: shows latency trends, detects
    degradation, and tracks SLA compliance over time.
    """

    # Thresholds for status classification
    LATENCY_HEALTHY_MS = 1000.0
    LATENCY_DEGRADED_MS = 3000.0
    ERROR_RATE_HEALTHY_PCT = 1.0
    ERROR_RATE_DEGRADED_PCT = 5.0

    # Incident detection
    DEGRADATION_THRESHOLD_PCT = 50.0  # 50% increase triggers incident
    INCIDENT_DURATION_MIN_PROBES = 3   # At least 3 probes to call it an incident

    def __init__(self):
        self._probes: dict[str, list[HealthProbe]] = defaultdict(list)
        self._incidents: dict[str, list[dict]] = defaultdict(list)
        self._current_status: dict[str, str] = {}
        self._last_incident_check: dict[str, datetime] = {}

    def record_request(self, provider: str, model: str, latency_ms: float,
                      success: bool, error_msg: str = "", tokens_generated: int = 0,
                      generation_time_ms: float = 0):
        """Record a real production request as a health probe."""

        # Calculate tokens_per_second if possible
        tokens_per_second = 0.0
        if generation_time_ms > 0 and tokens_generated > 0:
            tokens_per_second = (tokens_generated / generation_time_ms) * 1000

        # Determine status
        if not success:
            status = "error"
        elif latency_ms > self.LATENCY_DEGRADED_MS:
            status = "degraded"
        elif latency_ms > self.LATENCY_HEALTHY_MS:
            status = "degraded"
        else:
            status = "healthy"

        # Create probe record
        probe = HealthProbe(
            provider=provider,
            model=model,
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            status=status,
            error_message=error_msg,
            tokens_per_second=tokens_per_second
        )

        # Store probe
        key = f"{provider}:{model}"
        self._probes[key].append(probe)

        # Keep only last 10,000 probes per provider:model to avoid memory bloat
        if len(self._probes[key]) > 10000:
            self._probes[key] = self._probes[key][-10000:]

        # Check for incidents
        self._detect_incident(provider, probe)

    def get_status(self, provider: str = None) -> dict:
        """Current health status of provider(s)."""
        result = {}

        # Get all relevant keys
        if provider:
            keys = [k for k in self._probes.keys() if k.startswith(f"{provider}:")]
        else:
            keys = list(self._probes.keys())

        for key in keys:
            probes = self._probes[key]
            if not probes:
                continue

            # Analyze last 100 probes (or all if fewer)
            recent = probes[-100:] if len(probes) > 100 else probes

            # Last 5 minutes
            five_min_ago = datetime.utcnow() - timedelta(minutes=5)
            recent_5min = [p for p in recent if p.timestamp > five_min_ago]

            # Calculate metrics
            latencies = [p.latency_ms for p in recent]
            recent_latencies = [p.latency_ms for p in recent_5min] if recent_5min else latencies
            errors = [p for p in recent if p.status == "error"]

            avg_latency = statistics.mean(latencies) if latencies else 0
            recent_avg_latency = statistics.mean(recent_latencies) if recent_latencies else avg_latency
            error_rate = (len(errors) / len(recent)) * 100 if recent else 0

            # Determine trend
            trend = self._detect_degradation(key)

            # Overall status
            if error_rate > self.ERROR_RATE_DEGRADED_PCT:
                overall_status = "degraded"
            elif avg_latency > self.LATENCY_DEGRADED_MS:
                overall_status = "degraded"
            else:
                overall_status = "healthy"

            provider_name, model_name = key.split(":")
            result[key] = {
                "provider": provider_name,
                "model": model_name,
                "status": overall_status,
                "avg_latency_last_5min": recent_avg_latency if recent_5min else avg_latency,
                "avg_latency_overall": avg_latency,
                "error_rate_last_5min": (len([p for p in recent_5min if p.status == "error"]) / len(recent_5min)) * 100 if recent_5min else error_rate,
                "error_rate_overall": error_rate,
                "tokens_per_second": statistics.mean([p.tokens_per_second for p in recent if p.tokens_per_second > 0]) if [p for p in recent if p.tokens_per_second > 0] else 0,
                "trend": trend,
                "total_requests_analyzed": len(recent)
            }

        return result

    def get_sla_report(self, provider: str, hours: int = 24) -> ProviderSLA:
        """Generate SLA compliance report."""

        # Find all probes for this provider
        keys = [k for k in self._probes.keys() if k.startswith(f"{provider}:")]
        all_probes = []
        for key in keys:
            all_probes.extend(self._probes[key])

        # Filter by time window
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        probes = [p for p in all_probes if p.timestamp > cutoff]

        if not probes:
            return ProviderSLA(
                provider=provider,
                period_hours=hours,
                total_probes=0,
                successful_probes=0,
                uptime_pct=100.0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                error_rate_pct=0,
                latency_trend="stable",
                incidents=[]
            )

        # Calculate metrics
        successful = [p for p in probes if p.status != "error"]
        errors = [p for p in probes if p.status == "error"]
        latencies = [p.latency_ms for p in successful]

        latencies_sorted = sorted(latencies) if latencies else [0]

        uptime_pct = (len(successful) / len(probes)) * 100
        error_rate_pct = (len(errors) / len(probes)) * 100
        avg_latency = statistics.mean(latencies) if latencies else 0

        # Percentiles
        p50 = latencies_sorted[int(len(latencies_sorted) * 0.5)] if latencies_sorted else 0
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0

        # Detect trend
        trend = self._detect_degradation(f"{provider}:")

        return ProviderSLA(
            provider=provider,
            period_hours=hours,
            total_probes=len(probes),
            successful_probes=len(successful),
            uptime_pct=uptime_pct,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            error_rate_pct=error_rate_pct,
            latency_trend=trend,
            incidents=self._incidents.get(provider, [])
        )

    def get_latency_history(self, provider: str, hours: int = 1) -> list[dict]:
        """Latency measurements over time (for charting)."""

        keys = [k for k in self._probes.keys() if k.startswith(f"{provider}:")]
        all_probes = []
        for key in keys:
            all_probes.extend(self._probes[key])

        # Filter by time window
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        probes = [p for p in all_probes if p.timestamp > cutoff]

        # Sort by timestamp
        probes = sorted(probes, key=lambda p: p.timestamp)

        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "latency_ms": p.latency_ms,
                "status": p.status,
                "model": p.model
            }
            for p in probes
        ]

    def _detect_degradation(self, provider_key: str) -> str:
        """Check if latency is trending up (degradation)."""

        probes = self._probes.get(provider_key, [])
        if len(probes) < 10:
            return "stable"

        # Last 5 minutes vs last 30 minutes
        now = datetime.utcnow()
        five_min_ago = now - timedelta(minutes=5)
        thirty_min_ago = now - timedelta(minutes=30)

        last_5min = [p.latency_ms for p in probes if p.timestamp > five_min_ago]
        last_30min = [p.latency_ms for p in probes if five_min_ago > p.timestamp > thirty_min_ago]

        if not last_5min or not last_30min:
            return "stable"

        avg_5min = statistics.mean(last_5min)
        avg_30min = statistics.mean(last_30min)

        # Check for >50% increase
        if avg_5min > avg_30min * 1.5:
            return "degrading"
        elif avg_5min < avg_30min * 0.8:
            return "improving"
        else:
            return "stable"

    def _detect_incident(self, provider: str, probe: HealthProbe):
        """Detect and track incidents (sustained errors or high latency)."""

        probes = self._probes[f"{provider}:{probe.model}"]

        # Look at last N probes for incident pattern
        if len(probes) < self.INCIDENT_DURATION_MIN_PROBES:
            return

        recent = probes[-self.INCIDENT_DURATION_MIN_PROBES:]

        # Incident if most recent probes are errors or degraded
        error_count = len([p for p in recent if p.status == "error"])
        degraded_count = len([p for p in recent if p.status == "degraded"])

        is_incident = (error_count >= 2) or (degraded_count >= 2)

        if not is_incident:
            return

        # Check if we already have an open incident
        incidents = self._incidents.get(provider, [])
        if incidents and incidents[-1].get("end") is None:
            # Incident is ongoing
            return

        # Create new incident
        incident_start = recent[0].timestamp
        incident = {
            "start": incident_start.isoformat(),
            "end": None,
            "duration_min": None,
            "type": "error" if error_count >= 2 else "degradation",
            "probe_count": len(recent)
        }

        self._incidents[provider].append(incident)

        # Keep only last 100 incidents
        if len(self._incidents[provider]) > 100:
            self._incidents[provider] = self._incidents[provider][-100:]
