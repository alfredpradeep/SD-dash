"""
Language detection using fastText lid.176 model.

Detects language with confidence score. Supports 176 languages.
Falls back to a lightweight character-script heuristic when fastText is unavailable.
"""

import asyncio
import unicodedata
from loguru import logger
from lens.config import Config

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    logger.warning("fasttext not available — using script heuristic fallback")

try:
    import huggingface_hub
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LanguageDetector:
    """
    Language detector with fastText lid.176 as primary engine.
    Falls back to Unicode script heuristics when fastText is unavailable.
    """

    SUPPORTED_LANGUAGES = {
        "en", "ta", "hi", "ar", "ja", "zh", "ko", "pt", "es",
        "fr", "de", "id", "ms", "bn", "ur", "te", "ml", "pa", "gu", "mr", "kn",
    }

    # Character script heuristics for offline fallback
    SCRIPT_RANGES = {
        "ta": (0x0B80, 0x0BFF),   # Tamil
        "te": (0x0C00, 0x0C7F),   # Telugu
        "kn": (0x0C80, 0x0CFF),   # Kannada
        "ml": (0x0D00, 0x0D7F),   # Malayalam
        "hi": (0x0900, 0x097F),   # Devanagari (Hindi, Marathi, Nepali)
        "mr": (0x0900, 0x097F),   # Same as Hindi
        "bn": (0x0980, 0x09FF),   # Bengali
        "gu": (0x0A80, 0x0AFF),   # Gujarati
        "pa": (0x0A00, 0x0A7F),   # Gurmukhi (Punjabi)
        "ur": (0x0600, 0x06FF),   # Arabic script (also Urdu)
        "ar": (0x0600, 0x06FF),   # Arabic
        "ja": (0x3040, 0x30FF),   # Hiragana + Katakana
        "zh": (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        "ko": (0xAC00, 0xD7FF),   # Hangul
    }

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> bool:
        if self._model_loaded:
            return self._model is not None
        self._model_loaded = True
        if not FASTTEXT_AVAILABLE:
            return False
        try:
            # Try to load from HuggingFace cache or local path
            model_path = self._get_model_path()
            if model_path:
                self._model = fasttext.load_model(model_path)
                logger.info("fastText lid.176 model loaded from {}", model_path)
                return True
        except Exception as e:
            logger.warning("Failed to load fastText model: {}", e)
        return False

    def _get_model_path(self) -> str:
        """Try to find or download the fastText lid.176.bin model."""
        import os
        # Check common paths
        candidates = [
            "/models/lid.176.bin",
            os.path.expanduser("~/.fasttext/lid.176.bin"),
            "/tmp/lid.176.bin",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        # Try to download via huggingface_hub
        if HF_AVAILABLE:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id="facebook/fasttext-language-identification",
                    filename="model.bin",
                    cache_dir="/tmp/fasttext",
                )
                return path
            except Exception as e:
                logger.debug("HF download failed: {}", e)

        return ""

    def _script_heuristic(self, text: str) -> tuple[str, float]:
        """
        Fast Unicode script heuristic — used when fastText is unavailable.
        Returns (language_code, confidence).
        """
        if not text.strip():
            return "en", 0.5

        script_counts: dict[str, int] = {}
        total = 0
        for char in text:
            cp = ord(char)
            for lang, (lo, hi) in self.SCRIPT_RANGES.items():
                if lo <= cp <= hi:
                    script_counts[lang] = script_counts.get(lang, 0) + 1
                    total += 1
                    break

        if not script_counts:
            # Mostly ASCII — assume English
            return "en", 0.7

        best_lang = max(script_counts, key=lambda k: script_counts[k])
        confidence = script_counts[best_lang] / max(total, 1)

        # Disambiguate scripts that share range (e.g. Arabic/Urdu, Hindi/Marathi)
        if best_lang in ("ur", "ar"):
            # Urdu uses Nastaliq style; without ML we can't distinguish perfectly
            best_lang = "ar"
        if best_lang in ("hi", "mr"):
            best_lang = "hi"

        return best_lang, round(min(confidence, 0.95), 3)

    async def detect(self, text: str) -> tuple[str, float]:
        """
        Detect language of text.

        Returns:
            (language_code: str, confidence: float)
        """
        if not self._load_model():
            return self._script_heuristic(text)

        def _predict():
            # fastText wants single-line input
            clean = text.replace("\n", " ").strip()[:500]
            labels, probs = self._model.predict(clean, k=1)
            lang = labels[0].replace("__label__", "")
            conf = float(probs[0])
            # Map to supported languages; fallback to 'en' if unknown
            if lang not in self.SUPPORTED_LANGUAGES:
                lang = "en"
                conf = 0.5
            return lang, round(conf, 3)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _predict)

    async def health(self) -> str:
        if self._load_model():
            return "healthy"
        return "degraded (script heuristic mode)"
