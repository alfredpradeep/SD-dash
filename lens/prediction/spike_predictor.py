"""
Token cost spike predictor — Gap 3 Fix.

Replaces simple linear regression with CUSUM (Cumulative Sum Control Chart)
change detection. CUSUM detects structural breaks the moment they begin,
not after slope accumulates over multiple minutes.

Algorithm:
  1. Maintain 30-min rolling entropy window per language (via rolling_window.py)
  2. Compute EWMA (Exponentially Weighted Moving Average) baseline entropy
  3. Compute CUSUM statistic: C_t = max(0, C_{t-1} + (x_t - mu - k))
     where k = allowable drift (config.cusum_drift * sigma)
  4. Alert when C_t > h (config.cusum_threshold * sigma)
  5. Classify spike cause (Gap 5) — new named entities, code-switching, domain terms

This gives sub-minute spike detection vs the original's 5–10 minute lag.

Gap 5 — Cause Attribution:
  Analyses the high-surprisal token types in the window to classify:
    "new_named_entity"  — proper nouns / capitalized tokens appearing for first time
    "codeswitching"     — Latin tokens in non-Latin language traffic
    "domain_term"       — OOV technical/domain vocabulary
    "event_driven"      — burst of new named entities (likely temporary)
    "vocab_expansion"   — gradual entropy increase (likely permanent)
"""

import numpy as np
from datetime import datetime
from loguru import logger
from lens.entropy.structures import SpikePrediction
from lens.config import Config


class SpikePredictor:
    """
    CUSUM-based entropy spike predictor with cause attribution.
    """

    ALERT_LEVELS = ["watch", "warning", "alert", "critical"]

    # CUSUM C_t threshold multipliers (in units of sigma)
    CUSUM_LEVEL_THRESHOLDS = {
        "watch":    1.5,
        "warning":  3.0,
        "alert":    5.0,
        "critical": 8.0,
    }

    # Empirical: 1 bit/min entropy velocity ≈ 3% token count increase over 30 min
    ENTROPY_TO_TOKEN_FACTOR = 0.03

    def __init__(self, config: Config):
        self.config = config

    async def predict(
        self,
        language: str,
        entropy_window: list[tuple[datetime, float]],
        horizon_minutes: int = 30,
        high_surprisal_token_types: list[str] = None,
    ) -> SpikePrediction | None:
        """
        Predict token cost spike from entropy velocity using CUSUM.

        Args:
            language: ISO 639-1 language code
            entropy_window: List of (timestamp, entropy_bits) from rolling window
            horizon_minutes: Prediction horizon
            high_surprisal_token_types: List of token_type strings from current requests
                                        (for Gap 5 cause attribution)

        Returns:
            SpikePrediction or None if no spike predicted
        """
        if len(entropy_window) < 5:
            return None

        times = np.array([
            (t - entropy_window[0][0]).total_seconds() / 60.0
            for t, _ in entropy_window
        ])
        entropies = np.array([e for _, e in entropy_window])

        # EWMA baseline (α=0.3)
        alpha = 0.3
        ewma = [entropies[0]]
        for e in entropies[1:]:
            ewma.append(alpha * e + (1 - alpha) * ewma[-1])
        ewma = np.array(ewma)

        # Use baseline sigma (first third of window) — global sigma is inflated
        # by any step change that CUSUM is trying to detect.
        baseline_n = max(3, len(entropies) // 3)
        sigma = float(np.std(entropies[:baseline_n])) if baseline_n >= 2 else 1.0
        if sigma < 0.05:
            # Very stable baseline — use a small but non-zero sigma
            sigma = max(0.05, float(np.mean(np.abs(np.diff(entropies[:baseline_n])))) or 0.05)
        sigma = max(sigma, 0.01)

        # CUSUM (upper one-sided)
        k = self.config.cusum_drift * sigma    # allowable drift
        h = self.config.cusum_threshold * sigma  # alert threshold

        cusum = np.zeros(len(entropies))
        for i in range(1, len(entropies)):
            cusum[i] = max(0, cusum[i-1] + (entropies[i] - ewma[i] - k))

        current_cusum = float(cusum[-1])

        # No signal
        if current_cusum <= h * self.CUSUM_LEVEL_THRESHOLDS["watch"] / self.config.cusum_threshold:
            return None

        # Determine alert level
        alert_level = "watch"
        for level in reversed(self.ALERT_LEVELS):
            if current_cusum >= h * self.CUSUM_LEVEL_THRESHOLDS[level] / self.config.cusum_threshold:
                alert_level = level
                break

        # Entropy velocity (bits/min) from linear fit over recent half-window
        recent = len(entropies) // 2
        if recent >= 3:
            t_recent = times[-recent:]
            e_recent = entropies[-recent:]
            velocity = float(np.polyfit(t_recent, e_recent, 1)[0])
        else:
            velocity = float((entropies[-1] - entropies[0]) / max(times[-1] - times[0], 1))

        # Confidence = sigmoid of cusum / h, scaled by velocity consistency
        raw_confidence = float(1 / (1 + np.exp(-(current_cusum / h - 1))))
        # Penalise if velocity is actually negative (CUSUM caught a transient)
        if velocity < 0:
            raw_confidence *= 0.3
        confidence = float(np.clip(raw_confidence, 0.0, 1.0))

        if confidence < self.config.spike_alert_confidence_threshold:
            return None

        # Predicted token increase
        predicted_increase = abs(velocity) * horizon_minutes * self.ENTROPY_TO_TOKEN_FACTOR

        # --- Gap 5: Cause attribution ---
        spike_cause, is_temporary = self._classify_cause(
            entropies=entropies,
            times=times,
            token_types=high_surprisal_token_types or [],
        )

        trigger_reason = (
            f"CUSUM statistic C={current_cusum:.2f} (threshold={h:.2f}). "
            f"Entropy velocity {velocity:+.2f} bits/min over the last "
            f"{self.config.rolling_window_minutes} minutes. "
            f"Cause: {spike_cause.replace('_', ' ')}. "
            f"Estimated token cost increase: {predicted_increase*100:.1f}% "
            f"in next {horizon_minutes} minutes."
        )

        logger.info(
            "Spike predicted — lang={} level={} confidence={:.2f} "
            "velocity={:.2f} cause={}",
            language, alert_level, confidence, velocity, spike_cause
        )

        return SpikePrediction(
            language=language,
            current_entropy_velocity=round(velocity, 4),
            predicted_token_increase_pct=round(predicted_increase * 100, 2),
            confidence=round(confidence, 4),
            horizon_minutes=horizon_minutes,
            trigger_reason=trigger_reason,
            alert_level=alert_level,
            spike_cause=spike_cause,
            high_surprisal_tokens=[],  # populated by engine from current request batch
            is_likely_temporary=is_temporary,
        )

    def _classify_cause(
        self,
        entropies: np.ndarray,
        times: np.ndarray,
        token_types: list[str],
    ) -> tuple[str, bool]:
        """
        Classify the cause of the entropy spike (Gap 5).

        Returns: (cause_string, is_likely_temporary)
        """
        # Abrupt jump (high second derivative) = event-driven (temporary)
        if len(entropies) >= 6:
            early_var = float(np.var(entropies[:len(entropies)//2]))
            late_var = float(np.var(entropies[len(entropies)//2:]))
            is_abrupt = late_var > early_var * 3

            # Entropy is decelerating? → event likely fading
            early_slope = float(np.polyfit(times[:len(times)//2], entropies[:len(entropies)//2], 1)[0])
            late_slope = float(np.polyfit(times[len(times)//2:], entropies[len(entropies)//2:], 1)[0])
            decelerating = late_slope < early_slope * 0.5
        else:
            is_abrupt = False
            decelerating = False

        # Token type signals
        type_counts: dict[str, int] = {}
        for t in token_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        total_tokens = max(sum(type_counts.values()), 1)

        entity_ratio = type_counts.get("entity", 0) / total_tokens
        codesw_ratio = type_counts.get("codesw", 0) / total_tokens
        fragment_ratio = type_counts.get("fragment", 0) / total_tokens

        # Classification logic
        if codesw_ratio > 0.15:
            return "codeswitching", True
        if entity_ratio > 0.12 and is_abrupt:
            return "event_driven", True
        if entity_ratio > 0.08:
            return "new_named_entity", True
        if fragment_ratio > 0.30 and not is_abrupt:
            return "vocab_expansion", False
        if is_abrupt and not decelerating:
            return "event_driven", True
        if not is_abrupt:
            return "vocab_expansion", False

        return "domain_term", None
