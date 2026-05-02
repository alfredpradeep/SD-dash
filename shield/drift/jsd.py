"""
Jensen-Shannon Divergence Drift Analyzer: Boundary drift detection.

Analyzes safety-relevant term distributions and detects what terms
are lost, gained, or preserved across transformations.
"""

import asyncio
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter
from loguru import logger
from scipy.spatial.distance import jensenshannon

from shield.exceptions import JudgeError
from shield.drift.structures import (
    JSDResult,
    DriftSeverity,
)


# Safety-relevant keywords per category
SAFETY_TERMS = {
    "violence": [
        "harm", "hurt", "kill", "attack", "weapon", "violence",
        "injure", "pain", "suffer", "assault", "murder",
    ],
    "illegal_activity": [
        "illegal", "crime", "law", "arrest", "jail", "prison",
        "theft", "steal", "fraud", "smuggle", "drug",
    ],
    "deception": [
        "lie", "deceive", "trick", "fraud", "fake", "false",
        "mislead", "hoax", "scam", "forgery",
    ],
    "sexual_content": [
        "sexual", "sex", "adult", "porn", "erotic", "intimate",
        "naked", "nude", "explicit", "minor",
    ],
}


class JSDDriftAnalyzer:
    """
    Jensen-Shannon Divergence drift analyzer for boundary detection.

    Tracks safety-relevant term distributions to detect:
    - Terms lost in target (dilution)
    - Terms gained in target (amplification)
    - Overall divergence in safety term usage
    """

    def __init__(self):
        """Initialize JSD analyzer."""
        logger.debug("JSDDriftAnalyzer initialized")

    async def compute_boundary_drift(
        self, source_text: str, target_text: str, category: str
    ) -> JSDResult:
        """
        Compute Jensen-Shannon Divergence for safety term distributions.

        Args:
            source_text: Original text
            target_text: Transformed text
            category: Harm category

        Returns:
            JSDResult with divergence and term changes

        Raises:
            JudgeError: If computation fails
        """
        try:
            # Extract safety terms
            source_terms = self._extract_safety_terms(source_text, category)
            target_terms = self._extract_safety_terms(target_text, category)

            # Compute term distributions
            source_dist = self._compute_term_distribution(source_terms)
            target_dist = self._compute_term_distribution(target_terms)

            # Compute JSD
            jsd = self._compute_jsd(source_dist, target_dist)

            # Find term differences
            terms_lost = self._find_terms_lost(source_terms, target_terms)
            terms_gained = self._find_terms_gained(source_terms, target_terms)
            terms_preserved = self._find_terms_preserved(source_terms, target_terms)

            # Classify drift severity
            severity = self._classify_severity(jsd, len(terms_lost), len(terms_gained))

            result = JSDResult(
                jsd_score=float(jsd),
                terms_lost=terms_lost,
                terms_gained=terms_gained,
                terms_preserved=terms_preserved,
                drift_severity=severity,
                source_term_distribution=source_dist,
                target_term_distribution=target_dist,
                category=category,
                timestamp=datetime.utcnow().isoformat(),
            )

            logger.info(
                f"JSD computed: score={jsd:.3f}, severity={severity.value}, "
                f"lost={len(terms_lost)}, gained={len(terms_gained)}"
            )
            return result

        except Exception as e:
            logger.error(f"JSD computation failed: {e}")
            raise JudgeError(f"JSD drift analysis failed: {e}") from e

    def _extract_safety_terms(self, text: str, category: str) -> List[str]:
        """
        Extract safety-relevant terms from text.

        Args:
            text: Text to analyze
            category: Harm category

        Returns:
            List of matched safety terms
        """
        try:
            text_lower = text.lower()
            safety_keywords = SAFETY_TERMS.get(category, [])

            matched_terms = []
            for keyword in safety_keywords:
                # Count occurrences
                count = text_lower.count(keyword)
                matched_terms.extend([keyword] * count)

            return matched_terms

        except Exception as e:
            logger.warning(f"Term extraction failed: {e}")
            return []

    def _compute_term_distribution(self, terms: List[str]) -> Dict[str, float]:
        """
        Compute normalized distribution of terms.

        Args:
            terms: List of terms (with repetitions)

        Returns:
            Dictionary of term -> probability
        """
        try:
            if not terms:
                return {}

            counts = Counter(terms)
            total = sum(counts.values())

            return {term: count / total for term, count in counts.items()}

        except Exception as e:
            logger.warning(f"Distribution computation failed: {e}")
            return {}

    def _compute_jsd(
        self, dist_a: Dict[str, float], dist_b: Dict[str, float]
    ) -> float:
        """
        Compute Jensen-Shannon Divergence between two distributions.

        Args:
            dist_a: First distribution
            dist_b: Second distribution

        Returns:
            JSD value (0-1, 0=identical, 1=completely different)
        """
        try:
            # Get all unique terms
            all_terms = set(dist_a.keys()) | set(dist_b.keys())

            if not all_terms:
                return 0.0

            # Convert to probability vectors
            p = np.array([dist_a.get(t, 0.0) for t in all_terms])
            q = np.array([dist_b.get(t, 0.0) for t in all_terms])

            # Normalize
            p = p / (np.sum(p) + 1e-8)
            q = q / (np.sum(q) + 1e-8)

            # Compute JSD using scipy
            jsd = float(jensenshannon(p, q))
            return jsd

        except Exception as e:
            logger.warning(f"JSD computation failed: {e}")
            return 0.0

    def _find_terms_lost(self, source: List[str], target: List[str]) -> List[str]:
        """Find terms that appear in source but not target."""
        try:
            source_set = set(source)
            target_set = set(target)
            lost = sorted(list(source_set - target_set))
            return lost

        except Exception as e:
            logger.warning(f"Lost terms detection failed: {e}")
            return []

    def _find_terms_gained(self, source: List[str], target: List[str]) -> List[str]:
        """Find terms that appear in target but not source."""
        try:
            source_set = set(source)
            target_set = set(target)
            gained = sorted(list(target_set - source_set))
            return gained

        except Exception as e:
            logger.warning(f"Gained terms detection failed: {e}")
            return []

    def _find_terms_preserved(self, source: List[str], target: List[str]) -> List[str]:
        """Find terms that appear in both source and target."""
        try:
            source_set = set(source)
            target_set = set(target)
            preserved = sorted(list(source_set & target_set))
            return preserved

        except Exception as e:
            logger.warning(f"Preserved terms detection failed: {e}")
            return []

    def _classify_severity(
        self, jsd: float, n_lost: int, n_gained: int
    ) -> DriftSeverity:
        """
        Classify drift severity based on JSD and term changes.

        Args:
            jsd: Jensen-Shannon Divergence (0-1)
            n_lost: Number of terms lost
            n_gained: Number of terms gained

        Returns:
            DriftSeverity classification
        """
        # Loss is worse than gain (dilution is more concerning than mutation)
        loss_score = n_lost * 2
        gain_score = n_gained * 1
        term_change_score = loss_score + gain_score

        # Combine JSD and term changes
        combined_score = jsd * 0.5 + (min(term_change_score / 10, 1.0)) * 0.5

        if combined_score < 0.1:
            return DriftSeverity.MINIMAL
        elif combined_score < 0.25:
            return DriftSeverity.LOW
        elif combined_score < 0.5:
            return DriftSeverity.MODERATE
        elif combined_score < 0.75:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
