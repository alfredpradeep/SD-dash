"""
Isolation Forest anomaly detector for entropy time series.

Detects anomalous entropy values that are statistical outliers,
independent of directional trends. Used alongside CUSUM (which detects trends)
to catch sudden point anomalies (e.g. single requests with extreme token counts
from an unusual document type).
"""

import numpy as np
from loguru import logger
from lens.config import Config

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available — anomaly detection disabled")


class AnomalyDetector:
    """
    Per-language Isolation Forest for entropy anomaly detection.

    Each language gets its own model, trained on the last N observations.
    Anomaly score < threshold → flag as anomalous.
    """

    # Contamination fraction: expected fraction of anomalies in training data
    CONTAMINATION = 0.05
    # Anomaly score threshold (-1 = anomaly in sklearn)
    ANOMALY_LABEL = -1
    # Minimum observations to train on
    MIN_TRAIN_SAMPLES = 20

    def __init__(self, config: Config):
        self.config = config
        self._models: dict[str, "IsolationForest"] = {}
        self._history: dict[str, list[float]] = {}  # language -> recent entropy values

    def observe(self, language: str, entropy: float) -> None:
        """Record a new entropy observation for the language."""
        if language not in self._history:
            self._history[language] = []
        self._history[language].append(entropy)
        # Keep last 500 observations
        if len(self._history[language]) > 500:
            self._history[language] = self._history[language][-500:]

    def is_anomalous(self, language: str, entropy: float) -> tuple[bool, float]:
        """
        Check if entropy value is anomalous for this language.

        Returns:
            (is_anomaly: bool, anomaly_score: float)
            score < 0 indicates anomaly; more negative = more anomalous
        """
        if not SKLEARN_AVAILABLE:
            return False, 0.0

        history = self._history.get(language, [])
        if len(history) < self.MIN_TRAIN_SAMPLES:
            return False, 0.0

        # Fit or refit model on recent history
        model = self._get_or_fit_model(language, history)
        if model is None:
            return False, 0.0

        X = np.array([[entropy]])
        try:
            label = model.predict(X)[0]
            score = float(model.score_samples(X)[0])
            return label == self.ANOMALY_LABEL, round(score, 4)
        except Exception as e:
            logger.debug("Anomaly detection failed: {}", e)
            return False, 0.0

    def _get_or_fit_model(
        self, language: str, history: list[float]
    ) -> "IsolationForest | None":
        """Return fitted Isolation Forest, refitting every 50 new observations."""
        if not SKLEARN_AVAILABLE:
            return None
        key = language
        # Refit if model doesn't exist or history has grown significantly
        should_fit = (
            key not in self._models or
            len(history) % 50 == 0
        )
        if should_fit:
            try:
                X_train = np.array(history).reshape(-1, 1)
                model = IsolationForest(
                    contamination=self.CONTAMINATION,
                    random_state=42,
                    n_estimators=50,
                )
                model.fit(X_train)
                self._models[key] = model
            except Exception as e:
                logger.debug("IsolationForest fit failed for {}: {}", language, e)
                return None
        return self._models.get(key)
