"""
Data structures for probe execution and results.

Defines: ProbeResult, StepResult, SmatchResult, CertificationResult, EntailmentResult.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from shield.attacks.structures import AdversarialProbe


@dataclass
class StepResult:
    """Result from a single pipeline step."""
    step_name: str                           # e.g., "translation_forward", "target_llm"
    input_text: str
    output_text: str
    classification: str                      # e.g., "SAFE", "HARMFUL", "BORDERLINE"
    harm_preserved: bool                     # Did harmful intent survive this step?
    latency_ms: float                        # Execution time


@dataclass
class EntailmentResult:
    """Cross-lingual entailment analysis result."""
    forward_score: float                     # Source → Target entailment (0-1)
    backward_score: float                    # Target → Source entailment (0-1)
    drift_classification: str                # "preserved", "diluted", "amplified", "drifted"


@dataclass
class SmatchResult:
    """AMR Semantic Match comparison result."""
    score: float                             # Smatch F1 score (0-1)
    unmatched_source: List[str]              # AMR triples in source not in target
    unmatched_target: List[str]              # AMR triples in target not in source
    structural_drift: str                    # Type of drift: "agent_patient_swap", "relation_loss", etc.


@dataclass
class CertificationResult:
    """Certified robustness analysis result."""
    certified_accuracy: float                # Provable accuracy under perturbation budget
    certified_radius: float                  # Radius of robustness (max perturbation budget)
    n_samples: int                           # Number of perturbation samples tested
    confidence_level: float                  # Statistical confidence (alpha=0.001)
    votes_distribution: Dict[str, int]       # Counts of each classification across samples


@dataclass
class ProbeResult:
    """Complete result from running probe through pipeline."""
    probe: AdversarialProbe
    architecture: str                        # "translate_sandwich", "native_multilingual", "hybrid"
    steps: List[StepResult]                  # Per-step execution results
    overall_classification: str              # Final safety classification
    hasd_scores: Dict[str, float]            # HASD dimension scores
    entailment_scores: Optional[EntailmentResult] = None
    smatch_scores: Optional[SmatchResult] = None
    certification: Optional[CertificationResult] = None
    metadata: dict = field(default_factory=dict)

    def was_harmful(self) -> bool:
        """Whether probe was classified as harmful."""
        return self.overall_classification in ("HARMFUL", "BORDERLINE")

    def get_step_by_name(self, name: str) -> Optional[StepResult]:
        """Retrieve step result by name."""
        for step in self.steps:
            if step.step_name == name:
                return step
        return None

    def harm_preservation_at_step(self, step_idx: int) -> bool:
        """Check if harm was preserved at specific step."""
        if 0 <= step_idx < len(self.steps):
            return self.steps[step_idx].harm_preserved
        return False

    def total_latency_ms(self) -> float:
        """Sum of all step latencies."""
        return sum(s.latency_ms for s in self.steps)
