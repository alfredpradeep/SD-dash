"""
Semantic-Token Efficiency Score (STES) calculator.

STES = (graph_preservation ^ alpha) / (token_ratio ^ beta)

Higher STES = better candidate (more meaning preserved, fewer tokens used).
alpha/beta are per-language learned parameters tuned for morphological complexity.
"""

import numpy as np
from compress.search.tokenizer_profiles import TokenizerProfiles
from compress.config import Config


class STESScorer:
    """
    Semantic-Token Efficiency Score calculator.

    alpha governs semantic preservation penalty (higher = stricter fidelity).
    beta governs token savings reward (higher = more aggressive compression).

    Languages with richer morphology get higher alpha because rephrasing
    carries more risk of meaning loss in agglutinative systems.
    """

    LANG_PARAMS: dict[str, dict[str, float]] = {
        "ta": {"alpha": 1.3, "beta": 0.9},
        "te": {"alpha": 1.3, "beta": 0.9},
        "ml": {"alpha": 1.3, "beta": 0.9},
        "hi": {"alpha": 1.2, "beta": 0.85},
        "bn": {"alpha": 1.2, "beta": 0.85},
        "mr": {"alpha": 1.2, "beta": 0.85},
        "gu": {"alpha": 1.2, "beta": 0.85},
        "pa": {"alpha": 1.2, "beta": 0.85},
        "ur": {"alpha": 1.2, "beta": 0.85},
        "ar": {"alpha": 1.25, "beta": 0.9},
        "ja": {"alpha": 1.15, "beta": 0.8},
        "zh": {"alpha": 1.1, "beta": 0.75},
        "ko": {"alpha": 1.15, "beta": 0.8},
        "id": {"alpha": 1.05, "beta": 0.7},
        "ms": {"alpha": 1.05, "beta": 0.7},
        "pt": {"alpha": 1.05, "beta": 0.7},
        "es": {"alpha": 1.05, "beta": 0.7},
        "fr": {"alpha": 1.05, "beta": 0.7},
        "de": {"alpha": 1.1, "beta": 0.75},
        "en": {"alpha": 1.0, "beta": 0.65},
    }

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = TokenizerProfiles()

    def count_tokens(self, text: str, tokenizer_name: str) -> int:
        """Count tokens using the actual target tokenizer."""
        return self.tokenizer.count_tokens(text, tokenizer_name)

    def compute_stes(
        self,
        graph_preservation: float,
        token_count: int,
        source_language: str,
        baseline_token_count: int,
    ) -> float:
        """
        Compute Semantic-Token Efficiency Score.

        When token_ratio < 1.0 (compression achieved), STES > 1.
        When graph_preservation is high and token_ratio is low, STES is max.
        """
        params = self.LANG_PARAMS.get(
            source_language, {"alpha": 1.0, "beta": 0.7}
        )
        alpha = params["alpha"]
        beta = params["beta"]

        token_ratio = token_count / max(baseline_token_count, 1)
        stes = (graph_preservation ** alpha) / (token_ratio ** beta)

        return float(np.clip(stes, 0.0, 10.0))
