"""
SHIELD v2 — Cross-Lingual Safety Gap Scanner & Guardrail Auditor.

Four-Pillar Architecture:
  Pillar 1: LLM API Calls (the Brain) — cultural/semantic reasoning only
  Pillar 2: Deterministic Algorithms (the Reflexes) — script mapping, BPE drift, attack-graph routing
  Pillar 3: Distilled Student Models — fast-path judge replacing 80% of LLM calls
  Pillar 4: Novel Manifold Alignment Layer — cross-lingual latent-space projection

Code name: CALYX (Cross-lingual Adversarial Latent-sYnergy eXaminer).
"""

__version__ = "2.0.0"
__codename__ = "CALYX"
