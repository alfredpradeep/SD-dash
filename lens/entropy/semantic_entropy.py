"""
Semantic Entropy Calculator — Gap 2 Fix.

Adds the second dimension of the waste matrix:

  Axis 1 (lexical):  Are tokens predictable?  → IDS / ETR
  Axis 2 (semantic): Are tokens meaningful?   → Semantic Entropy

Two texts can have identical IDS but very different semantic entropy:
  - High lexical entropy + low semantic entropy = verbose/inflated waste
    (tokens are individually surprising but carry no new meaning overall)
  - High lexical entropy + high semantic entropy = genuinely dense content
    (tokens are surprising AND carry new meaning — efficient and substantive)

The 2D classification enables more precise waste attribution:
  waste_type:
    "lexical"  — token inefficiency (handled by ETR/IDS)
    "semantic" — meaning dilution (handled by Semantic Entropy)
    "combined" — both issues
    "efficient"— high IDS, high semantic entropy
"""

import numpy as np
from loguru import logger
from lens.entropy.structures import SemanticEntropyWindow
from lens.config import Config

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not available — semantic entropy disabled")


class SemanticEntropyCalculator:
    """
    Computes semantic entropy from embedding variance in sliding windows.

    Method:
      1. Split text into sentences (or fixed-token windows)
      2. Encode each sentence with a multilingual embedding model (LaBSE or paraphrase-multilingual)
      3. Compute the variance of embeddings within each window
      4. High variance = each sentence is semantically distant from its neighbours = rich content
      5. Low variance = repetitive / redundant sentences = semantic waste
    """

    # Prefer LaBSE (language-agnostic BERT) for multilingual coverage
    # Fall back to paraphrase-multilingual-MiniLM if LaBSE is unavailable
    MODEL_PREFERENCE = [
        "LaBSE",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "all-MiniLM-L6-v2",
    ]

    # Semantic entropy thresholds (bits, derived from variance)
    HIGH_SEMANTIC_ENTROPY_THRESHOLD = 2.5

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> bool:
        if self._model_loaded:
            return self._model is not None
        self._model_loaded = True
        if not SBERT_AVAILABLE or not self.config.semantic_entropy_enabled:
            return False
        for model_name in self.MODEL_PREFERENCE:
            try:
                logger.info("Loading semantic embedding model: {}", model_name)
                self._model = SentenceTransformer(model_name)
                logger.info("Semantic entropy model loaded: {}", model_name)
                return True
            except Exception as e:
                logger.debug("Failed to load {}: {}", model_name, e)
        logger.warning("No semantic entropy model available")
        return False

    def _split_sentences(self, text: str, language: str) -> list[str]:
        """Simple sentence splitter — avoids spaCy dependency for now."""
        import re
        # Split on sentence-ending punctuation, including Tamil/Arabic/Japanese stops
        pattern = r'(?<=[.!?।।。！？\n])\s+'
        sentences = re.split(pattern, text.strip())
        # Also split on newlines for structured inputs
        result = []
        for s in sentences:
            for sub in s.split("\n"):
                sub = sub.strip()
                if len(sub) > 5:
                    result.append(sub)
        return result if result else [text]

    def _variance_to_entropy(self, variance: float) -> float:
        """
        Convert embedding variance to semantic entropy in bits.

        We use a calibrated sigmoid mapping:
          variance ≈ 0.0  → entropy ≈ 0 bits  (all sentences identical)
          variance ≈ 0.5  → entropy ≈ 2 bits  (moderate diversity)
          variance ≈ 1.0+ → entropy ≈ 4+ bits (high diversity)
        """
        # Logistic scaling: entropy = 4 * sigmoid(variance * 8 - 3)
        return float(4.0 / (1.0 + np.exp(-(variance * 8 - 3))))

    async def compute(
        self,
        text: str,
        language: str,
        window_size: int = None,
    ) -> tuple[float, list[SemanticEntropyWindow]]:
        """
        Compute semantic entropy for text.

        Returns:
            (mean_semantic_entropy_bits, list[SemanticEntropyWindow])
        """
        window_size = window_size or self.config.semantic_entropy_window_size

        if not self._load_model():
            return 0.0, []

        sentences = self._split_sentences(text, language)
        if len(sentences) < 2:
            return 0.0, []

        try:
            embeddings = self._model.encode(sentences, batch_size=32, show_progress_bar=False)
        except Exception as e:
            logger.debug("Embedding failed: {}", e)
            return 0.0, []

        windows: list[SemanticEntropyWindow] = []
        for i in range(0, len(sentences) - window_size + 1):
            window_embeddings = embeddings[i : i + window_size]
            # Variance across embedding dimensions (mean of per-dimension variance)
            variance = float(np.mean(np.var(window_embeddings, axis=0)))
            entropy_bits = self._variance_to_entropy(variance)
            window_text = " | ".join(sentences[i : i + window_size])
            windows.append(SemanticEntropyWindow(
                window_text=window_text[:200],
                embedding_variance=round(variance, 4),
                semantic_entropy_bits=round(entropy_bits, 4),
                is_high_semantic_entropy=entropy_bits > self.HIGH_SEMANTIC_ENTROPY_THRESHOLD,
            ))

        if not windows:
            return 0.0, []

        mean_entropy = float(np.mean([w.semantic_entropy_bits for w in windows]))
        return round(mean_entropy, 4), windows

    def classify_waste_type(
        self,
        ids_score: float,
        semantic_entropy: float,
    ) -> str:
        """
        2D waste matrix classification.

        High IDS + High Semantic = efficient
        High IDS + Low Semantic  = semantic waste (repetitive meaning in novel tokens)
        Low IDS  + High Semantic = lexical waste (rich meaning, inefficient phrasing)
        Low IDS  + Low Semantic  = combined waste (both problems)
        """
        ids_high = ids_score >= 0.8
        sem_high = semantic_entropy >= self.HIGH_SEMANTIC_ENTROPY_THRESHOLD

        if ids_high and sem_high:
            return "efficient"
        elif ids_high and not sem_high:
            return "semantic"
        elif not ids_high and sem_high:
            return "lexical"
        else:
            return "combined"
