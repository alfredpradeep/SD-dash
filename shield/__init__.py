"""
SHIELD (Cross-Lingual Safety Gap Scanner) v1.0.

Multi-engine adversarial safety evaluation system for multilingual LLM safety assessment.
Core components:
  - Harm Lattice Generator: Generate culturally-adapted adversarial probes
  - PAIR Attacker: Iterative adversarial attack loop with judge feedback
  - Rainbow Teaming: MAP-Elites diversity optimization of attacks
  - GCG Transfer: Cross-lingual universal suffix discovery
  - Pipeline Probe Runner: Test attacks across translate-sandwich, native, hybrid architectures
  - Drift Detection: Entailment, smoothing, AMR-based semantic drift analysis

Part of the Haiku AI Governance Platform alongside LENS and COMPRESS modules.
"""

__version__ = "1.0.0"
__author__ = "Haiku AI Safety Team"
