"""
SHIELD Judge Module: Multi-tier safety classification pipeline.

Exports:
  - StructuralSignalDetector: Tier 1 structural analysis
  - SemanticRefusalDetector: Tier 2 LLM-based classification
  - HarmFulfillmentAnalyzer: Tier 3 harm graph comparison
  - SafetyRepresentationProbe: Representation engineering probe
  - MMDSafetyAssessor: Distribution-level safety assessment
  - WLStructureAnalyzer: Graph kernel structural analysis
  - ThreeTierJudge: Unified three-tier pipeline
"""

from shield.judge.structural import StructuralSignalDetector
from shield.judge.semantic import SemanticRefusalDetector
from shield.judge.fulfillment import HarmFulfillmentAnalyzer
from shield.judge.representation import SafetyRepresentationProbe
from shield.judge.mmd import MMDSafetyAssessor
from shield.judge.wl_kernel import WLStructureAnalyzer
from shield.judge.pipeline import ThreeTierJudge
from shield.judge.structures import (
    StructuralSignal,
    SemanticClassification,
    FulfillmentResult,
    RepresentationResult,
    MMDResult,
    WLResult,
    JudgmentResult,
)

__all__ = [
    "StructuralSignalDetector",
    "SemanticRefusalDetector",
    "HarmFulfillmentAnalyzer",
    "SafetyRepresentationProbe",
    "MMDSafetyAssessor",
    "WLStructureAnalyzer",
    "ThreeTierJudge",
    "StructuralSignal",
    "SemanticClassification",
    "FulfillmentResult",
    "RepresentationResult",
    "MMDResult",
    "WLResult",
    "JudgmentResult",
]
