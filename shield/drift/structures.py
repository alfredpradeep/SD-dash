"""
Dataclass structures for drift detection results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class DriftClassification(str, Enum):
    """Drift classification result."""
    PRESERVED = "preserved"
    DILUTED = "diluted"
    AMPLIFIED = "amplified"
    DRIFTED = "drifted"


class DriftSeverity(str, Enum):
    """Drift severity classification."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HarmVector:
    """6-dimensional harm vector representation."""
    method_specificity: float  # 0-1
    actionability: float  # 0-1
    target_identification: float  # 0-1
    general_knowledge: float  # 0-1
    temporal_urgency: float  # 0-1
    cultural_adaptation: float  # 0-1

    def to_list(self) -> List[float]:
        """Convert to list for distance computation."""
        return [
            self.method_specificity,
            self.actionability,
            self.target_identification,
            self.general_knowledge,
            self.temporal_urgency,
            self.cultural_adaptation,
        ]


@dataclass
class HASDResult:
    """Result from Harm-Aware Semantic Distance analysis."""
    hasd_score: float  # Wasserstein distance
    source_harm_vector: HarmVector
    target_harm_vector: HarmVector
    drift_classification: DriftClassification
    magnitude_ratio: float  # target_magnitude / source_magnitude
    dimension_changes: Dict[str, float]  # Changes per dimension
    category: str
    timestamp: str


@dataclass
class JSDResult:
    """Result from Jensen-Shannon Divergence drift analysis."""
    jsd_score: float  # Jensen-Shannon Divergence (0-1)
    terms_lost: List[str]  # Safety terms no longer in response
    terms_gained: List[str]  # New safety terms in response
    terms_preserved: List[str]  # Terms preserved from source
    drift_severity: DriftSeverity
    source_term_distribution: Dict[str, float]
    target_term_distribution: Dict[str, float]
    category: str
    timestamp: str


@dataclass
class IBPResult:
    """Result from Interval Bound Propagation certification."""
    certified: bool  # All outputs guaranteed to be safe
    uncertified_fraction: float  # Fraction of outputs outside safety bounds
    boundary_points: List[str]  # Paraphrases near safety boundary
    input_interval_radius: float  # Input perturbation radius
    output_interval_radius: float  # Propagated output radius
    n_paraphrases_tested: int
    safety_region_margin: float  # Distance to safety boundary
    timestamp: str


@dataclass
class DriftAnalysisBundle:
    """Aggregated drift analysis across multiple methods."""
    hasd_result: Optional[HASDResult] = None
    jsd_result: Optional[JSDResult] = None
    ibp_result: Optional[IBPResult] = None
    overall_drift: DriftClassification = DriftClassification.PRESERVED
    overall_severity: DriftSeverity = DriftSeverity.MINIMAL
    timestamp: str = ""
