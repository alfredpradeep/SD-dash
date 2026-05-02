"""
Cross-Lingual Entailment Analysis

Uses XLM-RoBERTa fine-tuned on XNLI to compute bidirectional entailment
between source and target texts. Detects if harmful intent drifts across
translation boundaries or architectural boundaries.
"""

import asyncio
import time
from typing import Tuple, Optional
from loguru import logger

from shield.config import Config
from shield.exceptions import ProbeError
from shield.probes.structures import EntailmentResult


class EntailmentAnalyzer:
    """Cross-lingual entailment analysis."""

    # Lazy-loaded model
    _model = None
    _tokenizer = None

    def __init__(self, config: Config):
        self.config = config
        logger.info("EntailmentAnalyzer initialised")

    @classmethod
    def _load_model(cls, config: Config):
        """Lazy-load XNLI entailment model."""
        if cls._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info("Loading XLM-RoBERTa XNLI model: {}", config.xnli_model_path)
                cls._model = CrossEncoder(config.xnli_model_path)
                logger.info("XNLI model loaded successfully")
            except Exception as e:
                logger.error("Failed to load XNLI model: {}", e)
                cls._model = None
        return cls._model

    async def compute_bidirectional_entailment(
        self, source: str, target: str
    ) -> Tuple[float, float]:
        """
        Compute bidirectional entailment scores between source and target.

        Returns:
            (forward_score, backward_score) both in [0, 1]
            forward = P(target entails source)
            backward = P(source entails target)
        """
        try:
            logger.info(
                "Computing entailment: source={:.40s}..., target={:.40s}...",
                source, target
            )

            # Load model
            model = self._load_model(self.config)
            if model is None:
                logger.warning("XNLI model unavailable, returning default scores")
                return (0.5, 0.5)

            # Forward: target → source (does target imply source?)
            try:
                forward_scores = model.predict([source, target])
                # forward_scores is [contradiction, neutral, entailment]
                # Extract entailment probability
                forward_entailment = float(forward_scores[2]) if len(forward_scores) > 2 else 0.5
            except Exception as e:
                logger.warning("Forward entailment computation failed: {}", e)
                forward_entailment = 0.5

            # Backward: source → target (does source imply target?)
            try:
                backward_scores = model.predict([target, source])
                backward_entailment = float(backward_scores[2]) if len(backward_scores) > 2 else 0.5
            except Exception as e:
                logger.warning("Backward entailment computation failed: {}", e)
                backward_entailment = 0.5

            logger.info(
                "Entailment scores: forward={:.3f}, backward={:.3f}",
                forward_entailment, backward_entailment
            )

            return (forward_entailment, backward_entailment)

        except Exception as e:
            logger.error("Entailment computation failed: {}", e)
            raise ProbeError(f"Entailment analysis failed: {e}") from e

    async def classify_drift(
        self, forward_score: float, backward_score: float
    ) -> str:
        """
        Classify type of semantic drift based on entailment scores.

        Returns:
            "preserved": High bidirectional entailment (intent preserved)
            "diluted": Forward high, backward low (target weaker than source)
            "amplified": Backward high, forward low (target stronger than source)
            "drifted": Low bidirectional entailment (intent significantly changed)
        """
        try:
            threshold = self.config.entailment_threshold

            if forward_score >= threshold and backward_score >= threshold:
                return "preserved"
            elif forward_score >= threshold and backward_score < threshold:
                return "diluted"
            elif forward_score < threshold and backward_score >= threshold:
                return "amplified"
            else:
                return "drifted"

        except Exception as e:
            logger.warning("Drift classification failed: {}", e)
            return "unknown"

    async def analyze_translation_drift(
        self, original: str, translated: str, language_pair: str
    ) -> EntailmentResult:
        """
        Analyze drift across translation boundary.

        Args:
            original: Original text
            translated: Text after translation
            language_pair: e.g., "en-ta" (English to Tamil)

        Returns:
            EntailmentResult with forward/backward scores and drift type
        """
        try:
            logger.info(
                "Analyzing translation drift for pair: {}",
                language_pair
            )

            forward, backward = await self.compute_bidirectional_entailment(
                original, translated
            )

            drift_type = await self.classify_drift(forward, backward)

            result = EntailmentResult(
                forward_score=forward,
                backward_score=backward,
                drift_classification=drift_type,
            )

            logger.info(
                "Translation drift analysis: type={}, forward={:.3f}, backward={:.3f}",
                drift_type, forward, backward
            )

            return result

        except Exception as e:
            logger.error("Translation drift analysis failed: {}", e)
            raise ProbeError(f"Translation drift analysis failed: {e}") from e
