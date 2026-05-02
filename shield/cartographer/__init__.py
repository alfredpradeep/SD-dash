"""
SHIELD Cartographer Module: Safety gap analysis and remediation.

Engine 5: Identify, rank, and remediate safety gaps.

Exports:
  - SafetyGapMatrix: Build and aggregate bypass rates
  - BayesianRiskEstimator: Hierarchical Bayesian posterior estimation
  - ParetoRemediationRanker: NSGA-II multi-objective optimization
  - ComplianceReportGenerator: Generate compliance and regulatory reports
"""

from shield.cartographer.matrix import SafetyGapMatrix
from shield.cartographer.bayesian import BayesianRiskEstimator
from shield.cartographer.pareto import ParetoRemediationRanker
from shield.cartographer.report import ComplianceReportGenerator
from shield.cartographer.structures import (
    GapMatrix,
    GapCell,
    PosteriorEstimate,
    RemediationAction,
    RemediationRoadmap,
    RegulatoryReport,
)

__all__ = [
    "SafetyGapMatrix",
    "BayesianRiskEstimator",
    "ParetoRemediationRanker",
    "ComplianceReportGenerator",
    "GapMatrix",
    "GapCell",
    "PosteriorEstimate",
    "RemediationAction",
    "RemediationRoadmap",
    "RegulatoryReport",
]
