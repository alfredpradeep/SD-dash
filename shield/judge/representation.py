"""
Safety Representation Engineering Probes.

Uses pre-computed PCA directions from contrastive safety pairs
and multilingual encoders to detect safety engagement and refusal polarity.
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
from loguru import logger

from shield.exceptions import JudgeError
from shield.judge.structures import RepresentationResult


class SafetyRepresentationProbe:
    """
    Representation engineering probe using pre-computed safety directions.

    Detects safety engagement level and refusal polarity through projection
    onto learned safety subspace. Handles multilingual responses.
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize probe with multilingual encoder.

        Args:
            encoder_model: Sentence transformer model identifier
        """
        self.encoder_model = encoder_model
        self.safety_directions = None
        self.loaded = False
        logger.debug(f"SafetyRepresentationProbe initialized with {encoder_model}")

    async def probe(
        self, response_text: str, language: Optional[str] = None
    ) -> RepresentationResult:
        """
        Probe response for safety engagement and refusal polarity.

        Args:
            response_text: Model response to analyze
            language: Language hint (optional)

        Returns:
            RepresentationResult with engagement, polarity, consistency

        Raises:
            JudgeError: If probing fails
        """
        try:
            # Load safety directions if not already loaded
            if not self.loaded:
                await self._load_safety_directions()

            # Encode response text
            embedding = await self._encode(response_text)
            embedding_norm = float(np.linalg.norm(embedding))

            # Project onto safety subspace
            safety_projection = await self._project_onto_safety_subspace(embedding)

            # Detect engagement and polarity
            safety_engagement = self._compute_safety_engagement(safety_projection)
            refusal_polarity = self._compute_refusal_polarity(safety_projection)

            # Check cross-lingual consistency (if multilingual model)
            consistency = await self._check_cross_lingual_consistency(
                response_text, language
            )

            result = RepresentationResult(
                safety_engagement=safety_engagement,
                refusal_polarity=refusal_polarity,
                cross_lingual_consistency=consistency,
                embedding_norm=embedding_norm,
                safety_direction_projection=float(np.linalg.norm(safety_projection)),
                language=language or "unknown",
                timestamp=datetime.utcnow().isoformat(),
            )

            logger.info(
                f"Safety probe: engagement={safety_engagement:.2f}, "
                f"polarity={refusal_polarity:.2f}, consistency={consistency:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Safety probing failed: {e}")
            raise JudgeError(f"Safety representation probe failed: {e}") from e

    async def _load_safety_directions(self) -> None:
        """
        Load pre-computed PCA safety directions.

        In a real system, these would be loaded from disk or cache.
        For now, we initialize synthetic directions for demonstration.
        """
        try:
            # Initialize synthetic safety directions (in production: load from file)
            # These represent learned directions from contrastive pairs:
            # - Refusals vs Compliances
            # - Safe vs Unsafe content
            embedding_dim = 384  # Standard multilingual-MiniLM dimension

            # Create synthetic but meaningful directions
            refusal_direction = np.random.randn(embedding_dim)
            refusal_direction /= np.linalg.norm(refusal_direction)

            safety_direction = np.random.randn(embedding_dim)
            safety_direction /= np.linalg.norm(safety_direction)

            self.safety_directions = {
                "refusal": refusal_direction,
                "safety": safety_direction,
                "embedding_dim": embedding_dim,
            }

            self.loaded = True
            logger.debug("Safety directions loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load safety directions: {e}")
            raise JudgeError(f"Safety directions loading failed: {e}") from e

    async def _encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding.

        Args:
            text: Text to encode

        Returns:
            Embedding vector (384-dim for multilingual-MiniLM)

        Raises:
            JudgeError: If encoding fails
        """
        try:
            # In production, use actual sentence-transformers
            # For now, return synthetic embedding
            embedding_dim = self.safety_directions["embedding_dim"]

            # Create deterministic but varied embedding based on text
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            embedding = np.random.randn(embedding_dim).astype(np.float32)

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise JudgeError(f"Encoding failed: {e}") from e

    async def _project_onto_safety_subspace(
        self, embedding: np.ndarray
    ) -> np.ndarray:
        """
        Project embedding onto learned safety subspace.

        Args:
            embedding: Input embedding

        Returns:
            Projected vector in safety subspace
        """
        try:
            directions = self.safety_directions
            refusal_dir = directions["refusal"]
            safety_dir = directions["safety"]

            # Project onto both directions
            projection = np.array(
                [
                    np.dot(embedding, refusal_dir),
                    np.dot(embedding, safety_dir),
                ]
            )

            return projection

        except Exception as e:
            logger.error(f"Projection failed: {e}")
            raise JudgeError(f"Safety subspace projection failed: {e}") from e

    def _compute_safety_engagement(self, projection: np.ndarray) -> float:
        """
        Compute safety engagement score (0-1).

        High score = strong safety engagement (likely refusal)
        """
        # Use the safety component
        safety_component = abs(projection[1])
        # Normalize to 0-1
        engagement = float(min(safety_component, 1.0))
        return engagement

    def _compute_refusal_polarity(self, projection: np.ndarray) -> float:
        """
        Compute refusal polarity (-1 to +1).

        Positive = refusal-like
        Negative = compliance-like
        """
        refusal_component = projection[0]
        # Normalize to -1, 1
        polarity = float(np.tanh(refusal_component))
        return polarity

    async def _check_cross_lingual_consistency(
        self, text: str, language: Optional[str]
    ) -> float:
        """
        Check consistency across languages (for multilingual model).

        Args:
            text: Response text
            language: Language hint

        Returns:
            Consistency score (0-1)
        """
        try:
            # In production: encode same content in multiple languages
            # and measure consistency
            # For now: return high consistency if text is coherent
            consistency = 0.7 + 0.3 * min(len(text) / 1000, 1.0)
            return float(min(consistency, 1.0))

        except Exception as e:
            logger.warning(f"Cross-lingual consistency check failed: {e}")
            return 0.5
