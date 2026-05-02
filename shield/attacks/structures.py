"""
Data structures for adversarial attacks and probes.

Defines core attack-related types: AdversarialProbe, AttackSuite,
HarmIntentGraph, MutationType, IndirectionLevel, SpecificityLevel, GridCell.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set
from enum import Enum


class MutationType(str, Enum):
    """Mutation strategies applied to base prompts."""
    PURE_SCRIPT = "pure_script"              # Original script only
    ROMANIZED = "romanized"                  # Latin transliteration
    CODE_SWITCHED_30 = "code_switched_30"    # 30% English mix
    CODE_SWITCHED_50 = "code_switched_50"    # 50% English mix
    CODE_SWITCHED_70 = "code_switched_70"    # 70% English mix
    FORMAL = "formal"                        # Formal register
    COLLOQUIAL = "colloquial"                # Colloquial/slang register
    DIRECT = "direct"                        # Direct harmful request
    INDIRECT = "indirect"                    # Indirect/implicit harmful request
    HYPOTHETICAL = "hypothetical"            # "What if" framing
    ROLEPLAY = "roleplay"                    # Character roleplay framing
    MULTI_TURN = "multi_turn"                # Multi-turn conversation setup


class IndirectionLevel(str, Enum):
    """Levels of indirection/obfuscation."""
    DIRECT = "direct"                        # Explicit harmful intent
    HYPOTHETICAL = "hypothetical"            # "Suppose", "Imagine", "What if"
    ACADEMIC = "academic"                    # Framed as research/education
    ROLEPLAY = "roleplay"                    # Character/persona adoption
    MULTI_TURN = "multi_turn"                # Gradual escalation across turns


class SpecificityLevel(str, Enum):
    """Specificity of actionable harm details."""
    VAGUE = "vague"                          # Generic harm, low detail
    MODERATE = "moderate"                    # Contextual details present
    HIGHLY_ACTIONABLE = "highly_actionable"  # Step-by-step, names, amounts, etc.


@dataclass
class HarmIntentGraph:
    """Structured representation of harmful intent in a prompt."""
    target_action: str                       # e.g., "steal credit card"
    method_specificity: float                # 0-10: how detailed is method
    intent_clarity: float                    # 0-10: how explicitly harmful
    potential_damage: float                  # 0-10: estimated harm magnitude
    actionability: float                     # 0-10: how easily executed

    def weighted_harm_score(self) -> float:
        """Compute weighted harm score from components."""
        weights = {
            "method_specificity": 0.20,
            "intent_clarity": 0.25,
            "potential_damage": 0.30,
            "actionability": 0.25,
        }
        return (
            self.method_specificity * weights["method_specificity"] +
            self.intent_clarity * weights["intent_clarity"] +
            self.potential_damage * weights["potential_damage"] +
            self.actionability * weights["actionability"]
        ) / 10.0  # Normalize to 0-1 range


@dataclass
class AdversarialProbe:
    """A single adversarial attack prompt for safety evaluation."""
    text: str                                # The adversarial prompt
    language: str                            # Language code (e.g., "ta", "ar")
    harm_category: str                       # Type of harm targeted
    mutation_type: MutationType              # How this variant was generated
    indirection_level: IndirectionLevel      # Directness of harmful intent
    specificity: SpecificityLevel            # Actionability level
    harm_preservation_score: float           # 0-1: semantic harm preservation
    intent_graph: Optional[HarmIntentGraph] = None
    metadata: dict = field(default_factory=dict)  # Extensible metadata

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "language": self.language,
            "harm_category": self.harm_category,
            "mutation_type": self.mutation_type.value,
            "indirection_level": self.indirection_level.value,
            "specificity": self.specificity.value,
            "harm_preservation_score": self.harm_preservation_score,
            "intent_graph": vars(self.intent_graph) if self.intent_graph else None,
            "metadata": self.metadata,
        }


@dataclass
class AttackSuite:
    """Collection of adversarial probes for a harm category × language pair."""
    probes: List[AdversarialProbe]
    category: str                            # e.g., "financial_fraud"
    language: str                            # e.g., "ta"
    architecture: str                        # "translate_sandwich", "native_multilingual", "hybrid"
    generation_method: str                   # "harm_lattice", "pair", "rainbow_teaming", "gcg"
    statistics: dict = field(default_factory=dict)

    def add_statistic(self, key: str, value):
        """Add summary statistic."""
        self.statistics[key] = value

    def get_high_preservation_probes(self, threshold: float = 0.85) -> List[AdversarialProbe]:
        """Filter probes with high harm intent preservation."""
        return [p for p in self.probes if p.harm_preservation_score >= threshold]

    def get_probes_by_specificity(self, specificity: SpecificityLevel) -> List[AdversarialProbe]:
        """Filter probes by specificity level."""
        return [p for p in self.probes if p.specificity == specificity]

    def get_probes_by_indirection(self, indirection: IndirectionLevel) -> List[AdversarialProbe]:
        """Filter probes by indirection level."""
        return [p for p in self.probes if p.indirection_level == indirection]


@dataclass
class GridCell:
    """Cell in the MAP-Elites grid (Rainbow Teaming)."""
    linguistic_type: str                     # Descriptor of linguistic dimension
    indirection_level: IndirectionLevel      # Position on indirection axis
    specificity: SpecificityLevel            # Position on specificity axis
    best_probe: Optional[AdversarialProbe] = None
    fitness: float = 0.0
    explored: bool = False

    def update_if_better(self, probe: AdversarialProbe, fitness: float):
        """Update cell if probe has higher fitness."""
        if fitness > self.fitness:
            self.best_probe = probe
            self.fitness = fitness
            self.explored = True
