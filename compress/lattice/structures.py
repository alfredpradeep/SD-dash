"""
Semantic graph data structures for the SCL engine.

Defines the core types: GraphNode, SemanticGraph, CompressionCandidate,
and CompressionResult. All graph-related dataclasses live here.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SemanticUnitType(str, Enum):
    ENTITY      = "entity"
    SENTIMENT   = "sentiment"
    TEMPORAL    = "temporal"
    RELATIONAL  = "relational"
    MODAL       = "modal"
    NEGATION    = "negation"
    QUANTIFIER  = "quantifier"
    ATTRIBUTE   = "attribute"
    CAUSAL      = "causal"
    CONDITIONAL = "conditional"


@dataclass
class GraphNode:
    unit_id: str
    unit_type: SemanticUnitType
    value: str
    intensity: Optional[float] = None      # for sentiment: 0.0–1.0
    specificity: Optional[float] = None    # for temporal: 0.0–1.0
    certainty: Optional[float] = None      # for modal: 0.0–1.0
    confidence: float = 1.0                # extraction confidence for this node
    dependencies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SemanticGraph:
    source_text: str
    source_language: str
    nodes: list[GraphNode]
    edges: list[tuple[str, str, str]]   # (from_id, relation, to_id)
    amr_penman: str                      # raw AMR penman notation
    extraction_confidence: float         # 0.0–1.0 (harmonic mean)
    amr_confidence: float                # 0.0–1.0 (AMR parser confidence alone)
    component_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nodes": [vars(n) for n in self.nodes],
            "edges": self.edges,
            "amr_penman": self.amr_penman,
            "extraction_confidence": self.extraction_confidence,
            "amr_confidence": self.amr_confidence,
            "component_scores": self.component_scores,
        }

    def node_by_id(self, unit_id: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.unit_id == unit_id:
                return n
        return None

    def nodes_by_type(self, unit_type: SemanticUnitType) -> list[GraphNode]:
        return [n for n in self.nodes if n.unit_type == unit_type]


@dataclass
class CompressionCandidate:
    text: str
    token_count: int
    stes_score: float
    semantic_similarity: float
    graph_jaccard: float
    rank: int


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    source_language: str
    target_tokenizer: str
    original_token_count: int
    compressed_token_count: int
    reduction_ratio: float              # 0.0–1.0, higher = more compressed
    semantic_similarity: float          # LaBSE cosine sim, must be >= 0.91
    graph_jaccard: float                # graph similarity, must be >= 0.88
    stes_score: float
    compression_applied: bool           # False if validation gate rejected all
    rejection_reason: Optional[str]     # populated if compression_applied=False
    processing_ms: float
    candidates_evaluated: int
