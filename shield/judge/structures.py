"""
Dataclass structures for judge module results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RefusalSignal(str, Enum):
    """Tier 1 structural refusal signal classification."""
    LIKELY_REFUSAL = "likely_refusal"
    AMBIGUOUS = "ambiguous"
    LIKELY_COMPLIANCE = "likely_compliance"


class SemanticClassificationResult(str, Enum):
    """Tier 2 semantic classification result."""
    REFUSED = "refused"
    DEFLECTED = "deflected"
    COMPLIED = "complied"
    PARTIAL = "partial"


class SeverityGrade(str, Enum):
    """Harm severity grading scale A-F."""
    GRADE_A = "A"
    GRADE_B = "B"
    GRADE_C = "C"
    GRADE_D = "D"
    GRADE_E = "E"
    GRADE_F = "F"


@dataclass
class StructuralSignal:
    """Result from Tier 1 structural signal detection."""
    signal: RefusalSignal
    confidence: float  # 0-1
    matched_patterns: List[str]
    response_length_ratio: float
    has_list_markers: bool
    has_code_blocks: bool
    has_apology_disclaimers: bool
    refusal_phrases_matched: List[str]
    language: str
    timestamp: str


@dataclass
class SemanticClassification:
    """Result from Tier 2 semantic refusal detection."""
    classification: SemanticClassificationResult
    confidence: float  # 0-1
    harm_info_score: int  # 0-10
    reasoning: str
    language: str
    timestamp: str
    refusal_patterns_detected: List[str] = field(default_factory=list)


@dataclass
class HarmElement:
    """Structured representation of harmful content element."""
    element_type: str  # method_specificity, actionability, target_identification, etc.
    content: str
    severity: float  # 0-1
    weight: float  # importance weight


@dataclass
class FulfillmentResult:
    """Result from Tier 3 harm fulfillment graph comparison."""
    fulfillment_score: float  # 0-1
    matched_elements: List[HarmElement]
    severity_grade: SeverityGrade
    requested_harm_vector: List[HarmElement]
    provided_harm_vector: List[HarmElement]
    method_specificity_match: float
    actionability_match: float
    target_identification_match: float
    general_knowledge_match: float
    category: str
    timestamp: str


@dataclass
class RepresentationResult:
    """Result from safety representation engineering probe."""
    safety_engagement: float  # 0-1
    refusal_polarity: float  # -1 to +1
    cross_lingual_consistency: float  # 0-1
    embedding_norm: float
    safety_direction_projection: float
    language: str
    timestamp: str


@dataclass
class MMDResult:
    """Result from MMD distribution-level safety assessment."""
    mmd_vs_refusals: float
    mmd_vs_compliances: float
    p_value: float
    distribution_classification: str  # refusal_like/neutral/compliance_like
    kernel_type: str
    bandwidth: float
    category: str
    language: str
    timestamp: str
    reference_distribution_size_safe: int
    reference_distribution_size_compliant: int


@dataclass
class WLResult:
    """Result from Weisfeiler-Leman graph kernel analysis."""
    kernel_value: float
    structural_match: bool
    causal_chain_preserved: bool
    wl_iterations: int
    tree1_subtrees: int
    tree2_subtrees: int
    language: str
    timestamp: str
    structural_similarity: float


@dataclass
class TierResults:
    """Aggregated results from all tiers."""
    tier1_structural: StructuralSignal
    tier2_semantic: SemanticClassification
    tier3_fulfillment: FulfillmentResult
    representation_probe: Optional[RepresentationResult] = None
    mmd_assessor: Optional[MMDResult] = None
    wl_analyzer: Optional[WLResult] = None


@dataclass
class JudgmentResult:
    """Final judgment result from three-tier pipeline."""
    final_classification: SemanticClassificationResult
    confidence: float  # 0-1
    tier_agreement: float  # agreement score 0-1
    disagreement_flags: List[str]
    all_tier_results: TierResults
    explanation: str
    timestamp: str
    language: str
    category: str
