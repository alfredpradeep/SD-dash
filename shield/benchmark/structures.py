"""
Dataclass structures for benchmark module results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TrendDirection(str, Enum):
    """Safety trend direction over time."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


@dataclass
class BenchmarkPercentile:
    """Percentile ranking for a model."""
    model_name: str
    category: str
    language: str
    bypass_rate: float
    percentile: float  # 0-100 (0=worst, 100=best)
    rank: int  # 1=best performer
    total_models: int
    reference_group: str  # "published_AdvBench" or "internal_history"


@dataclass
class TrendAnalysis:
    """Temporal trend analysis for a model."""
    model_name: str
    category: str
    language: str
    direction: TrendDirection
    weekly_change: float  # Percentage point change per week
    days_of_data: int
    first_bypass_rate: float
    latest_bypass_rate: float
    confidence: float


@dataclass
class BenchmarkComparison:
    """Complete benchmark comparison result."""
    model_name: str
    benchmark_date: str
    percentiles: List[BenchmarkPercentile]
    trends: List[TrendAnalysis]
    published_comparison: Dict[str, float]  # AdvBench, HarmBench scores
    internal_comparison: Dict[str, float]  # Historical scores
    overall_percentile: float
    safety_improving: bool
    summary: str
