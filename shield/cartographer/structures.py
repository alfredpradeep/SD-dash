"""
Dataclass structures for cartographer module results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RemediationTier(str, Enum):
    """NSGA-II Pareto tier assignment."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"


class RootCauseCategory(str, Enum):
    """Root cause categories for failures."""
    PROMPT_INJECTION = "prompt_injection"
    TRAINING_DATA = "training_data"
    ARCHITECTURE_FLAW = "architecture_flaw"
    FINE_TUNING = "fine_tuning"
    MULTILINGUAL_TRANSFER = "multilingual_transfer"
    JAILBREAK_TECHNIQUE = "jailbreak_technique"
    INSTRUCTION_FOLLOWING = "instruction_following"


class RegulatorStandard(str, Enum):
    """Regulatory compliance standards."""
    EU_AI_ACT = "eu_ai_act"
    NIST_AI_RMF = "nist_ai_rmf"
    ISO_42001 = "iso_42001"
    CALIFORNIA_SB_1047 = "california_sb_1047"


@dataclass
class GapCell:
    """Single cell in safety gap matrix."""
    language: str
    architecture: str  # translate_sandwich, native, hybrid
    harm_category: str
    bypass_rate: float  # 0-1
    harm_severity_avg: float  # 0-10
    confidence: float  # 0-1
    n_observations: int
    description: str


@dataclass
class GapMatrix:
    """2D safety gap matrix: (language × architecture) × harm_category."""
    cells: Dict[Tuple[str, str, str], GapCell]  # (lang, arch, category) -> cell
    languages: List[str]
    architectures: List[str]
    categories: List[str]
    timestamp: str
    summary: str


@dataclass
class PosteriorEstimate:
    """Bayesian posterior estimate for bypass rate."""
    posterior_mean: float  # E[bypass_rate | observations]
    credible_interval_95: Tuple[float, float]  # 95% credible interval
    prior_mean: float
    prior_source: str  # "published_baseline" or "language_family"
    language_family: str  # e.g., "Dravidian", "Indo-Aryan", "CJK"
    n_successes: int  # harm bypassed
    n_failures: int  # properly refused
    posterior_std: float


@dataclass
class RemediationAction:
    """Single remediation action with Pareto tier."""
    action_id: str
    title: str
    description: str
    target_cell: Tuple[str, str, str]  # (language, architecture, category)
    root_cause: RootCauseCategory
    bypass_reduction: float  # Expected reduction in bypass rate
    harm_severity_reduction: float  # Expected reduction in severity
    implementation_effort: float  # Estimated effort (1-10)
    coverage: float  # Fraction of cells affected (0-1)
    pareto_tier: RemediationTier
    crowding_distance: float  # NSGA-II diversity measure


@dataclass
class RemediationRoadmap:
    """Complete remediation roadmap prioritized by Pareto."""
    roadmap_title: str
    actions: List[RemediationAction]  # Sorted by tier then crowding distance
    total_estimated_effort: float
    total_bypass_reduction: float
    total_cells_impacted: int
    timeline_weeks: int
    timestamp: str


@dataclass
class RegulatorySection:
    """Single section of regulatory compliance report."""
    standard: RegulatorStandard
    section_title: str
    requirement: str
    compliance_status: str  # "compliant", "partial", "non_compliant"
    gaps: List[str]  # Specific gaps
    evidence: List[str]  # Evidence of compliance/gaps
    remediation_actions: List[str]  # Related action IDs


@dataclass
class RegulatoryReport:
    """Regulatory compliance report."""
    report_title: str
    generated_date: str
    standards: List[RegulatorySection]
    overall_compliance_score: float  # 0-1
    critical_gaps: List[str]
    executive_summary: str
    detailed_findings: str
