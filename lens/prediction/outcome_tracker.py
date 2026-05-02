"""
Prediction Outcome Tracker — Gap 6 Fix.

Tracks predicted spikes vs actual token count changes.
When a spike materialises (or doesn't), records the outcome and
recalibrates per-language confidence thresholds.

Storage: Redis hashes + sorted sets (falls back to in-memory dict).

Recalibration:
  - If predictions for language X are 80% accurate → keep threshold
  - If accuracy drops below 50% → raise threshold by 0.05 (be more conservative)
  - If accuracy > 90% → lower threshold by 0.02 (be more sensitive)
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from loguru import logger
from lens.entropy.structures import PredictionOutcome
from lens.config import Config

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class PredictionOutcomeTracker:
    """
    Tracks spike prediction outcomes and recalibrates confidence thresholds.
    """

    ACCURACY_TOLERANCE = 0.25      # Prediction is "accurate" if within 25% of actual
    RECAL_WINDOW_HOURS = 24        # Look at last 24h of predictions for recalibration
    MIN_OUTCOMES_FOR_RECAL = 10    # Need at least 10 outcomes to recalibrate

    def __init__(self, config: Config):
        self.config = config
        self._redis = None
        self._redis_ok = False
        # In-memory fallback
        self._pending: dict[str, PredictionOutcome] = {}
        self._resolved: list[PredictionOutcome] = []
        # Per-language adjusted thresholds
        self._thresholds: dict[str, float] = {}

    async def _get_redis(self):
        if not REDIS_AVAILABLE or self._redis_ok is False:
            return None
        if self._redis:
            return self._redis
        try:
            self._redis = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._redis.ping()
            self._redis_ok = True
            return self._redis
        except Exception:
            self._redis_ok = False
            return None

    async def record_prediction(
        self,
        language: str,
        predicted_increase_pct: float,
        confidence: float,
        horizon_minutes: int,
    ) -> str:
        """Record a new prediction. Returns prediction_id."""
        pred_id = str(uuid.uuid4())[:8]
        outcome = PredictionOutcome(
            prediction_id=pred_id,
            language=language,
            predicted_at=datetime.utcnow(),
            predicted_increase_pct=predicted_increase_pct,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
        )
        self._pending[pred_id] = outcome
        return pred_id

    async def record_outcome(
        self,
        prediction_id: str,
        actual_increase_pct: float,
    ) -> PredictionOutcome | None:
        """Record actual outcome for a prediction."""
        outcome = self._pending.pop(prediction_id, None)
        if outcome is None:
            return None

        outcome.actual_increase_pct = actual_increase_pct
        outcome.outcome_recorded_at = datetime.utcnow()

        # Was accurate within tolerance?
        predicted = outcome.predicted_increase_pct
        if predicted > 0:
            relative_error = abs(actual_increase_pct - predicted) / predicted
            outcome.was_accurate = relative_error <= self.ACCURACY_TOLERANCE
        else:
            outcome.was_accurate = actual_increase_pct < 5.0  # No spike predicted, was that right?

        self._resolved.append(outcome)
        # Keep last 1000 resolved outcomes
        if len(self._resolved) > 1000:
            self._resolved = self._resolved[-1000:]

        # Trigger recalibration for this language
        await self._recalibrate(outcome.language)
        return outcome

    async def _recalibrate(self, language: str) -> None:
        """Adjust confidence threshold for language based on recent accuracy."""
        cutoff = datetime.utcnow() - timedelta(hours=self.RECAL_WINDOW_HOURS)
        recent = [
            o for o in self._resolved
            if o.language == language
            and o.outcome_recorded_at is not None
            and o.outcome_recorded_at >= cutoff
            and o.was_accurate is not None
        ]
        if len(recent) < self.MIN_OUTCOMES_FOR_RECAL:
            return

        accuracy = sum(1 for o in recent if o.was_accurate) / len(recent)
        current_threshold = self._thresholds.get(
            language, self.config.spike_alert_confidence_threshold
        )

        if accuracy < 0.50:
            # Too many false positives — raise threshold
            new_threshold = min(0.95, current_threshold + 0.05)
            logger.info(
                "Recalibrating {} threshold: {:.2f} → {:.2f} (accuracy={:.0%})",
                language, current_threshold, new_threshold, accuracy
            )
            self._thresholds[language] = new_threshold
        elif accuracy > 0.90:
            # High accuracy — be more sensitive
            new_threshold = max(0.40, current_threshold - 0.02)
            self._thresholds[language] = new_threshold
        # else: accuracy 50–90% → keep threshold

    def get_threshold(self, language: str) -> float:
        """Return the current (possibly recalibrated) confidence threshold for a language."""
        return self._thresholds.get(language, self.config.spike_alert_confidence_threshold)

    def accuracy_report(self) -> dict[str, dict]:
        """Return accuracy statistics per language."""
        report = {}
        langs = {o.language for o in self._resolved}
        for lang in langs:
            lang_outcomes = [o for o in self._resolved if o.language == lang]
            resolved = [o for o in lang_outcomes if o.was_accurate is not None]
            if not resolved:
                continue
            accuracy = sum(1 for o in resolved if o.was_accurate) / len(resolved)
            report[lang] = {
                "total_predictions": len(lang_outcomes),
                "resolved": len(resolved),
                "accuracy": round(accuracy, 3),
                "current_threshold": self.get_threshold(lang),
            }
        return report
