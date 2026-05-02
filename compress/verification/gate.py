"""
Triple verification gate for compression candidates.

A candidate passes only if ALL gates are satisfied:
  Gate 1: Non-empty and non-identical to original
  Gate 2: LaBSE semantic similarity >= 0.91
  Gate 3: Weighted graph Jaccard >= 0.88
  Gate 4: Token reduction >= 15%

Thresholds derived from adversarial testing with bilingual annotators.
"""

from compress.lattice.structures import SemanticGraph, CompressionCandidate
from compress.config import Config


class VerificationGate:
    """
    Triple verification gate for compression candidates.

    - 0.91 LaBSE: below this, >5% of candidates flagged for meaning drift
    - 0.88 graph Jaccard: below this, structural errors appear
    """

    def __init__(self, config: Config):
        self.config = config

    async def verify(
        self,
        original_text: str,
        original_graph: SemanticGraph,
        candidate: CompressionCandidate,
        original_tokens: int,
        semantic_threshold: float,
        graph_threshold: float,
        min_reduction: float,
    ) -> tuple[bool, str]:
        """
        Verify a single compression candidate against all gates.

        Returns:
            (passed: bool, reason: str)
            reason is empty string when passed=True
        """
        # Gate 1: Non-empty, non-identical
        if not candidate.text.strip():
            return False, "candidate text is empty"
        if candidate.text.strip() == original_text.strip():
            return False, "candidate is identical to original"

        # Gate 2: Semantic similarity (LaBSE cosine)
        if candidate.semantic_similarity < semantic_threshold:
            return False, (
                f"semantic similarity {candidate.semantic_similarity:.3f} "
                f"< threshold {semantic_threshold}"
            )

        # Gate 3: Graph Jaccard (weighted)
        if candidate.graph_jaccard < graph_threshold:
            return False, (
                f"graph Jaccard {candidate.graph_jaccard:.3f} "
                f"< threshold {graph_threshold}"
            )

        # Gate 4: Token reduction
        reduction = 1.0 - (candidate.token_count / max(original_tokens, 1))
        if reduction < min_reduction:
            return False, (
                f"token reduction {reduction:.3f} "
                f"< min_reduction {min_reduction}"
            )

        return True, ""

    async def health(self) -> str:
        return "healthy"
