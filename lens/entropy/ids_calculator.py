"""
Information Density Score (IDS) calculator.

IDS = language-normalised mean token surprisal.

High IDS (> 1.0) = information-dense content — tokens are earning their cost.
Low IDS (< 0.6) = wasteful content — tokens are predictable, redundant, inflated.

The language normalisation ensures fair comparison: a Tamil request with the same
semantic content as an English request should score ~1.0 on IDS, not 0.2.
"""

import numpy as np
from lens.entropy.structures import TokenEntropyRecord
from lens.config import Config


class IDSCalculator:
    """Information Density Score calculator with COMPRESS bridge signal generation."""

    # Language-specific baseline IDS from empirical corpus analysis
    # (mean IDS of 10,000 representative texts per language)
    LANGUAGE_BASELINE_IDS: dict[str, float] = {
        "en": 4.2, "es": 4.1, "fr": 4.0, "de": 4.3, "pt": 4.1,
        "hi": 3.8, "bn": 3.7, "ur": 3.7, "ta": 3.5, "te": 3.4,
        "ml": 3.4, "kn": 3.5, "gu": 3.6, "mr": 3.7, "pa": 3.8,
        "ar": 3.9, "ja": 4.4, "zh": 4.6, "ko": 4.1,
        "id": 4.0, "ms": 4.0,
    }

    # Window size (tokens) for span-level IDS, used for COMPRESS bridge targeting
    SPAN_WINDOW = 10

    def __init__(self, config: Config):
        self.config = config

    def compute(
        self,
        token_records: list[TokenEntropyRecord],
        language: str,
    ) -> float:
        """
        Compute normalised IDS for a request.

        Returns:
            Float normalised to language baseline:
              1.0 = at language baseline
              > 1 = high information density (efficient)
              < 1 = low information density (wasteful)
        """
        if not token_records:
            return 0.0

        surprisals = np.array([r.surprisal_bits for r in token_records])
        mean_surprisal = float(np.mean(surprisals))

        baseline = self.LANGUAGE_BASELINE_IDS.get(language, 4.0)
        ids = mean_surprisal / baseline

        return float(np.clip(ids, 0.0, 5.0))

    def find_low_ids_spans(
        self,
        token_records: list[TokenEntropyRecord],
        language: str,
        threshold_multiplier: float = 0.5,
    ) -> list[tuple[int, int, float]]:
        """
        Identify token spans with below-threshold IDS for COMPRESS bridge targeting.

        Returns list of (start_token_idx, end_token_idx, span_ids) for spans
        where IDS is below threshold_multiplier * language_baseline.

        Used by: lens/engine.py to populate EntropyProfile.low_ids_token_spans
        """
        if not token_records:
            return []

        baseline = self.LANGUAGE_BASELINE_IDS.get(language, 4.0)
        alert_threshold = baseline * threshold_multiplier

        low_spans: list[tuple[int, int, float]] = []
        n = len(token_records)
        step = self.SPAN_WINDOW // 2  # 50% overlap

        for start in range(0, n, step):
            end = min(start + self.SPAN_WINDOW, n)
            window = token_records[start:end]
            if len(window) < 3:
                continue
            window_surprisals = np.array([r.surprisal_bits for r in window])
            window_ids = float(np.mean(window_surprisals))
            if window_ids < alert_threshold:
                low_spans.append((start, end, round(window_ids / baseline, 3)))

        # Merge overlapping spans
        if not low_spans:
            return []
        merged: list[tuple[int, int, float]] = []
        current_start, current_end, current_ids = low_spans[0]
        for s, e, ids in low_spans[1:]:
            if s <= current_end:
                current_end = max(current_end, e)
                current_ids = min(current_ids, ids)
            else:
                merged.append((current_start, current_end, current_ids))
                current_start, current_end, current_ids = s, e, ids
        merged.append((current_start, current_end, current_ids))
        return merged
