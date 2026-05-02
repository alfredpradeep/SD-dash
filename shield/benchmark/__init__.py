"""
SHIELD Benchmark Module: Living benchmark and adaptive scan allocation.

Engine 6: Real-time performance benchmarking and adaptive resource allocation.

Exports:
  - LivingBenchmark: Compare against published and empirical baselines
  - ThompsonScanAllocator: Adaptive budget allocation using Thompson sampling
"""

from shield.benchmark.living import LivingBenchmark
from shield.benchmark.thompson import ThompsonScanAllocator
from shield.benchmark.structures import (
    BenchmarkComparison,
    BenchmarkPercentile,
    TrendAnalysis,
)

__all__ = [
    "LivingBenchmark",
    "ThompsonScanAllocator",
    "BenchmarkComparison",
    "BenchmarkPercentile",
    "TrendAnalysis",
]
