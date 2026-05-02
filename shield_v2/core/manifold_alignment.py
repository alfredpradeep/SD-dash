"""
PILLAR 4: Novel Cross-Lingual Manifold Alignment Layer.

The scientific moat of SHIELD v2. Implements two original algorithmic contributions:

1. SafetyBoundaryProjection (CrossLingualManifoldAligner):
   Learns a per-language orthogonal projection operator P_L : R^d_en → R^d_L
   that maps the English safety decision hyperplane into each target language's
   embedding manifold. Uses Procrustes analysis + safety-aware Riemannian
   optimization on the Stiefel manifold.

2. AntigenicDriftMutator:
   Inspired by viral antigenic drift — the mechanism by which influenza evolves
   to escape immune recognition. We treat the safety classifier as the "immune
   system" and adversarial prompts as "epitopes" evolving to evade detection.
   The mutator searches in the ORTHOGONAL COMPLEMENT of the projected safety
   boundary — mutations that preserve semantic intent but slide the prompt's
   embedding parallel to (not across) the decision hyperplane.

Complexity:
  - Alignment fit: O(d^3) one-time per language (SVD-dominated)
  - Per-prompt projection: O(d^2)
  - Antigenic drift step: O(k·d^2) where k = mutation candidates per step
  - Full drift trajectory: O(T·k·d^2) for T steps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import orthogonal_procrustes, svd
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SafetyBoundaryProjection:
    """Learned cross-lingual safety projection for one target language."""

    source_lang: str
    target_lang: str
    rotation: np.ndarray
    refusal_direction: np.ndarray
    compliance_direction: np.ndarray
    margin: float
    boundary_fidelity: float
    n_pairs: int

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project a source-language embedding into target-language manifold."""
        return x @ self.rotation

    def safety_score(self, x: np.ndarray) -> float:
        """
        Signed distance from the safety boundary.
        Positive => refusal side (safe). Negative => compliance side (harmful).
        """
        diff = self.refusal_direction - self.compliance_direction
        norm = np.linalg.norm(diff) + 1e-9
        return float((x @ diff) / norm / max(self.margin, 1e-6))

    def to_dict(self) -> dict:
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "margin": float(self.margin),
            "boundary_fidelity": float(self.boundary_fidelity),
            "n_pairs": int(self.n_pairs),
        }


# =============================================================================
# Cross-Lingual Manifold Aligner
# =============================================================================


class CrossLingualManifoldAligner:
    """
    Learns per-language SafetyBoundaryProjection from aligned sentence pairs.

    Loss:
        L(R) = ||X_en · R - X_tgt||_F^2
             + λ · (1 - cos(R · r_en, r_tgt))
             + μ · (1 - cos(R · c_en, c_tgt))
        subject to R^T R = I.

    Solved via projected gradient on the Stiefel manifold with SVD retraction.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        lambda_refusal: float = 1.0,
        mu_compliance: float = 0.5,
        n_stiefel_steps: int = 40,
        learning_rate: float = 0.05,
    ):
        self.d = embed_dim
        self.lambda_refusal = lambda_refusal
        self.mu_compliance = mu_compliance
        self.n_stiefel_steps = n_stiefel_steps
        self.lr = learning_rate
        self.projections: Dict[str, SafetyBoundaryProjection] = {}
        logger.info(
            "CrossLingualManifoldAligner initialized (d=%d, lambda=%.2f, mu=%.2f)",
            embed_dim, lambda_refusal, mu_compliance,
        )

    def fit(
        self,
        target_lang: str,
        source_embeds: np.ndarray,
        target_embeds: np.ndarray,
        source_refusal: np.ndarray,
        target_refusal: np.ndarray,
        source_compliance: np.ndarray,
        target_compliance: np.ndarray,
        source_lang: str = "en",
    ) -> SafetyBoundaryProjection:
        """Fit a projection from source_lang to target_lang."""
        assert source_embeds.shape == target_embeds.shape
        assert source_embeds.shape[1] == self.d

        # Step 1: orthogonal Procrustes warm start
        R0, _ = orthogonal_procrustes(source_embeds, target_embeds)

        # Step 2: Stiefel refinement with safety-aware loss
        R = self._stiefel_refine(
            R0, source_embeds, target_embeds,
            source_refusal, target_refusal,
            source_compliance, target_compliance,
        )

        # Step 3: fidelity
        projected_refusal = source_refusal @ R
        projected_compliance = source_compliance @ R
        fidelity = self._boundary_fidelity(
            projected_refusal, target_refusal,
            projected_compliance, target_compliance,
        )

        margin = float(np.linalg.norm(target_refusal - target_compliance))

        proj = SafetyBoundaryProjection(
            source_lang=source_lang,
            target_lang=target_lang,
            rotation=R,
            refusal_direction=target_refusal / (np.linalg.norm(target_refusal) + 1e-9),
            compliance_direction=target_compliance / (np.linalg.norm(target_compliance) + 1e-9),
            margin=margin,
            boundary_fidelity=fidelity,
            n_pairs=int(source_embeds.shape[0]),
        )
        self.projections[target_lang] = proj
        logger.info(
            "Fit projection en->%s: fidelity=%.3f margin=%.3f n=%d",
            target_lang, fidelity, margin, source_embeds.shape[0],
        )
        return proj

    def project(self, x: np.ndarray, target_lang: str) -> np.ndarray:
        if target_lang not in self.projections:
            raise KeyError(f"No projection fitted for {target_lang}")
        return self.projections[target_lang].project(x)

    def safety_score(self, x: np.ndarray, target_lang: str) -> float:
        return self.projections[target_lang].safety_score(x)

    def thin_boundary_languages(self, threshold: float = 0.6) -> List[str]:
        return [
            lang for lang, p in self.projections.items()
            if p.boundary_fidelity < threshold
        ]

    def _stiefel_refine(
        self,
        R0: np.ndarray,
        X_src: np.ndarray, X_tgt: np.ndarray,
        r_src: np.ndarray, r_tgt: np.ndarray,
        c_src: np.ndarray, c_tgt: np.ndarray,
    ) -> np.ndarray:
        """Projected gradient on Stiefel manifold with SVD retraction."""
        R = R0.copy()
        r_src = r_src / (np.linalg.norm(r_src) + 1e-9)
        r_tgt = r_tgt / (np.linalg.norm(r_tgt) + 1e-9)
        c_src = c_src / (np.linalg.norm(c_src) + 1e-9)
        c_tgt = c_tgt / (np.linalg.norm(c_tgt) + 1e-9)

        n = max(X_src.shape[0], 1)
        for _ in range(self.n_stiefel_steps):
            grad_proc = 2.0 * X_src.T @ (X_src @ R - X_tgt) / n
            grad_ref = -self.lambda_refusal * np.outer(r_src, r_tgt)
            grad_com = -self.mu_compliance * np.outer(c_src, c_tgt)
            grad = grad_proc + grad_ref + grad_com
            R = R - self.lr * grad
            # Retract onto the orthogonal group
            U, _, Vt = svd(R, full_matrices=False)
            R = U @ Vt
        return R

    @staticmethod
    def _boundary_fidelity(
        proj_r: np.ndarray, true_r: np.ndarray,
        proj_c: np.ndarray, true_c: np.ndarray,
    ) -> float:
        cos_r = 1.0 - cosine(proj_r, true_r)
        cos_c = 1.0 - cosine(proj_c, true_c)
        avg = (cos_r + cos_c) / 2.0
        return float(max(0.0, min(1.0, (avg + 1.0) / 2.0)))


# =============================================================================
# Antigenic Drift Mutator
# =============================================================================


@dataclass
class MutationStep:
    """One step in an antigenic-drift trajectory."""
    prompt_embedding: np.ndarray
    safety_score: float
    margin_drop: float
    step_index: int
    accepted: bool


class AntigenicDriftMutator:
    """
    Evolves an adversarial prompt's embedding along the direction that
    MAXIMALLY reduces distance to the safety boundary while MINIMIZING
    semantic drift — analogous to viral antigenic escape from antibodies
    under selection pressure to preserve receptor binding.

    Formal statement (the mutation step):
        x_{t+1} = x_t + η · (P_parallel · g - α · (x_t - x_0))
        where:
          g = -∇ SafetyScore(x_t)         (gradient pushing toward compliance)
          P_parallel = I - r·r^T           (projector onto boundary-parallel subspace)
          α ≥ 0                            (semantic anchor strength)

    Interpretation:
      - P_parallel·g moves along the boundary (immune-escape direction).
      - (x_t - x_0) pulls back toward semantic origin (preserves meaning).
      - η controls mutation step size.
    """

    def __init__(
        self,
        step_size: float = 0.08,
        semantic_anchor: float = 0.15,
        n_candidates: int = 8,
        max_steps: int = 20,
        acceptance_threshold: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        self.eta = step_size
        self.alpha = semantic_anchor
        self.k = n_candidates
        self.T = max_steps
        self.accept_thresh = acceptance_threshold
        self.rng = np.random.default_rng(rng_seed)

    def drift(
        self,
        x0: np.ndarray,
        projection: SafetyBoundaryProjection,
    ) -> Tuple[np.ndarray, List[MutationStep]]:
        """
        Run an antigenic-drift trajectory starting from embedding x0.

        Returns the final embedding (the "escaped" adversarial variant)
        and the full trajectory of mutation steps (for audit).
        """
        r = projection.refusal_direction
        r = r / (np.linalg.norm(r) + 1e-9)
        # Projector onto the subspace ORTHOGONAL to the refusal direction
        # = subspace parallel to the safety boundary
        P_par = np.eye(x0.shape[0]) - np.outer(r, r)

        trajectory: List[MutationStep] = []
        x = x0.copy()

        initial_score = projection.safety_score(x)

        for t in range(self.T):
            # Gradient of (negative) safety score = push toward compliance
            # For linear score s(x) = (r - c)/||r-c|| · x / margin,
            # the gradient is simply the diff direction, normalised.
            diff = projection.refusal_direction - projection.compliance_direction
            grad = -diff / (np.linalg.norm(diff) + 1e-9)

            # Component parallel to the boundary (escape direction)
            boundary_parallel = P_par @ grad

            # Semantic anchor pulling toward x0
            anchor = -self.alpha * (x - x0)

            # Generate k stochastic candidate mutations
            best_x = x
            best_delta = 0.0
            for _ in range(self.k):
                noise = self.rng.standard_normal(x.shape) * 0.02
                candidate = x + self.eta * (boundary_parallel + anchor + noise)
                # Score after mutation
                new_score = projection.safety_score(candidate)
                delta = projection.safety_score(x) - new_score  # positive = closer to compliance
                if delta > best_delta:
                    best_delta = delta
                    best_x = candidate

            new_score = projection.safety_score(best_x)
            accepted = best_delta > self.accept_thresh

            trajectory.append(MutationStep(
                prompt_embedding=best_x,
                safety_score=new_score,
                margin_drop=best_delta,
                step_index=t,
                accepted=accepted,
            ))

            if accepted:
                x = best_x
            else:
                # No better candidate found — stop
                break

            # Early termination: crossed the boundary
            if new_score < 0 and initial_score > 0:
                logger.debug("Antigenic drift: boundary crossed at step %d", t)
                break

        return x, trajectory

    def escape_probability(
        self,
        x0: np.ndarray,
        projection: SafetyBoundaryProjection,
        n_trials: int = 16,
    ) -> float:
        """Monte Carlo estimate of probability that drift crosses the boundary."""
        crossings = 0
        initial_score = projection.safety_score(x0)
        if initial_score < 0:
            return 1.0  # already on compliance side
        for _ in range(n_trials):
            x_final, _ = self.drift(x0, projection)
            if projection.safety_score(x_final) < 0:
                crossings += 1
        return crossings / n_trials
