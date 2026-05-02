"""
SHIELD Drift Detection Module: Semantic and distribution drift analysis.

Engines 4-5: Detect how harm intent changes across transformations.

Exports:
  - HASDCalculator: Harm-Aware Semantic Distance
  - JSDDriftAnalyzer: Jensen-Shannon Divergence drift
  - CertifiedDriftBounds: Interval Bound Propagation certification
"""

from shield.drift.hasd import HASDCalculator
from shield.drift.jsd import JSDDriftAnalyzer
from shield.drift.ibp import CertifiedDriftBounds
from shield.drift.structures import (
    HASDResult,
    JSDResult,
    IBPResult,
)

__all__ = [
    "HASDCalculator",
    "JSDDriftAnalyzer",
    "CertifiedDriftBounds",
    "HASDResult",
    "JSDResult",
    "IBPResult",
]
