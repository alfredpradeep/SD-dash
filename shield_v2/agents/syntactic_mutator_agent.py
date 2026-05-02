"""
Agent 1: The Syntactic Mutator.

Generates cross-script, cross-tokenizer adversarial variants of a seed
prompt. Uses Pillar 2 (DeterministicSyntacticMutator) for its action space
and Pillar 4 (AntigenicDriftMutator) for embedding-space search.

Persona: purely mechanical. Does not understand meaning. Cares only
about syntactic surface and tokenizer boundaries. Think of it as a
careful typesetter who's never read the manuscript.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from shield_v2.core.syntactic_mutator import (
    DeterministicSyntacticMutator, MutationResult, BPEDriftFingerprint,
)
from shield_v2.core.manifold_alignment import (
    AntigenicDriftMutator, SafetyBoundaryProjection,
)
from shield_v2.core.attack_graph import AttackGraph, AttackRouter, LanguageNode

logger = logging.getLogger(__name__)


@dataclass
class MutationProposal:
    text: str
    transform_chain: List[str]
    target_lang: str
    bpe_risk: str
    embedding_drift_score: float
    confidence: float


class SyntacticMutatorAgent:
    """Generates adversarial mutations. Never evaluates harm."""

    def __init__(
        self,
        attack_graph: Optional[AttackGraph] = None,
        drift_mutator: Optional[AntigenicDriftMutator] = None,
    ):
        self.det = DeterministicSyntacticMutator()
        self.graph = attack_graph
        self.drift = drift_mutator or AntigenicDriftMutator()

    def propose(
        self,
        seed: str,
        target_lang: str,
        seed_embedding: Optional[np.ndarray] = None,
        projection: Optional[SafetyBoundaryProjection] = None,
        n_variants: int = 6,
    ) -> List[MutationProposal]:
        proposals: List[MutationProposal] = []

        # Path 1: deterministic script/register mutations (Pillar 2)
        surface = self.det.mutate_all(seed, target_lang)
        for m in surface[:n_variants]:
            fp = self.det.bpe_fingerprint(m.mutated)
            proposals.append(MutationProposal(
                text=m.mutated,
                transform_chain=[m.mutation_type],
                target_lang=target_lang,
                bpe_risk=fp.risk_flag,
                embedding_drift_score=0.0,
                confidence=0.9 if fp.risk_flag == "LOW" else 0.7,
            ))

        # Path 2: graph-routed multi-hop mutations (Pillar 2)
        if self.graph:
            router = AttackRouter(self.graph)
            src = LanguageNode("en", "native", "formal", 0)
            targets = router.top_k_targets(src, k=3)
            for tgt_node, cost in targets:
                if tgt_node.lang != target_lang:
                    continue
                path = router.shortest_path(src, tgt_node)
                if path is None:
                    continue
                proposals.append(MutationProposal(
                    text=seed,  # router supplies the chain; text fills in downstream
                    transform_chain=[e.transform for e in path.edges],
                    target_lang=target_lang,
                    bpe_risk="MEDIUM",
                    embedding_drift_score=cost,
                    confidence=max(0.3, 1.0 - cost / 10.0),
                ))

        # Path 3: antigenic drift in latent space (Pillar 4)
        if projection is not None and seed_embedding is not None:
            x_final, trajectory = self.drift.drift(seed_embedding, projection)
            drift_magnitude = float(np.linalg.norm(x_final - seed_embedding))
            final_score = projection.safety_score(x_final)
            proposals.append(MutationProposal(
                text=seed,
                transform_chain=[f"antigenic_drift(T={len(trajectory)})"],
                target_lang=target_lang,
                bpe_risk="LOW",
                embedding_drift_score=drift_magnitude,
                confidence=0.85 if final_score < 0 else 0.55,
            ))

        return proposals

    def name(self) -> str:
        return "SyntacticMutator"
