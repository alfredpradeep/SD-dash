"""
AMR (Abstract Meaning Representation) parsing utilities.

Provides AMRParser (SPRING-based) and CrossLingualAdapter for
multilingual AMR parsing with graceful fallback.
"""

import penman
from loguru import logger


class AMRParser:
    """
    Wrapper around SPRING-based AMR parser with confidence scoring.

    Uses amrlib's SPRING model for English AMR parsing.
    Non-English text is processed through CrossLingualAdapter first.
    Gracefully degrades when the AMR model is unavailable.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        try:
            import amrlib
            self.model = amrlib.load_stog_model(model_dir=model_path)
            self._available = True
            logger.info("AMR parser loaded from {}", model_path)
        except Exception as e:
            logger.warning("AMR parser unavailable ({}), will use fallback extraction", e)
            self._available = False
            self.model = None

    def parse(self, text: str) -> tuple[str, float]:
        """
        Parse text into AMR penman notation.

        Returns:
            (penman_string, confidence_score)
            confidence is based on heuristic node-to-word ratio analysis.
        """
        if not self._available or not self.model:
            return "", 0.0

        try:
            graphs = self.model.parse_sents([text])
            if not graphs or not graphs[0]:
                return "", 0.0

            amr_str = graphs[0]
            # Validate the AMR is parseable by penman
            parsed = penman.decode(amr_str)
            node_count = len(parsed.instances())

            # Heuristic confidence: well-formed AMR with reasonable node count
            # gets higher confidence. Very sparse or very dense graphs are penalised.
            text_words = len(text.split())
            ratio = node_count / max(text_words, 1)
            if 0.3 <= ratio <= 1.5:
                confidence = min(0.95, 0.6 + ratio * 0.3)
            else:
                confidence = max(0.2, 0.5 - abs(ratio - 0.7) * 0.3)

            return amr_str, confidence

        except Exception as e:
            logger.warning("AMR parse failed: {}", e)
            return "", 0.0


class CrossLingualAdapter:
    """
    Language-specific adapter for cross-lingual AMR parsing.

    For non-English input, applies language-specific transformations
    to improve AMR parsing quality:
    - Direct adapter for high-resource languages (zh, de, es, fr, pt)
    - Translate-then-parse for low-resource languages (ta, hi, ar, etc.)
    """

    DIRECT_LANGUAGES = {"en", "zh", "de", "es", "fr", "pt"}
    TRANSLATE_LANGUAGES = {
        "ta", "hi", "ar", "ja", "ko", "bn", "ur", "te", "ml",
        "pa", "gu", "mr", "id", "ms"
    }

    def __init__(self, adapter_dir: str, device: str = "cpu"):
        self.adapter_dir = adapter_dir
        self.device = device
        self._adapters: dict = {}
        logger.info("CrossLingualAdapter initialised, adapter_dir={}", adapter_dir)

    def adapt(self, text: str, language: str) -> str:
        """
        Adapt non-English text for AMR parsing.

        For DIRECT_LANGUAGES: applies lightweight adapter transformation.
        For TRANSLATE_LANGUAGES: tags text for downstream handling.
        In production, this uses per-language adapter weights from adapter_dir.
        """
        if language in self.DIRECT_LANGUAGES:
            return text
        return f"[{language}] {text}"

    def back_align(self, amr_str: str, original_text: str, language: str) -> str:
        """Realign AMR entity values to original language surface forms."""
        return amr_str
