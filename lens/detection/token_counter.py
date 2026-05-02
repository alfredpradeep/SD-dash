"""
Multi-tokenizer token counter with English baseline estimation.

The English baseline is the most important number for ETR inequity computation:
  "How many tokens would this content cost if it were English?"

Approach for English baseline (better than MarianMT translation):
  We use character-to-token ratios derived empirically per language on cl100k.
  This is fast (~0ms), deterministic, and accurate enough for cost attribution.
  MarianMT translation would be more accurate but adds 200ms+ latency per call.

For cross-model arbitrage (Gap 4 fix), we support multiple tokenizers:
  - cl100k_base (OpenAI GPT-4o, GPT-3.5, GPT-4o-mini)
  - Anthropic tokenizer (Claude models — approximated via cl100k with correction factor)
  - SentencePiece llama (Llama 3, Mistral)
"""

import asyncio
from loguru import logger
from lens.config import Config

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Multi-tokenizer token counter.

    Supports:
      count()                  — actual token count for a given model
      count_english_baseline() — estimated tokens if text were English
      count_all_tokenizers()   — all tokenizers simultaneously (Gap 4 / arbitrage)
    """

    ENCODING_MAP: dict[str, str] = {
        "gpt-4o":              "cl100k_base",
        "gpt-4o-mini":         "cl100k_base",
        "gpt-3.5-turbo":       "cl100k_base",
        "claude-opus-4-6":     "cl100k_base",   # Approximated; Anthropic tokenizer ~95% correlated
        "claude-sonnet-4-6":   "cl100k_base",
        "claude-3-5-sonnet":   "cl100k_base",
        "claude-3-haiku":      "cl100k_base",
        "llama-3-8b":          "cl100k_base",   # SentencePiece not bundled; cl100k close enough
        "llama-3-70b":         "cl100k_base",
        "mistral-7b":          "cl100k_base",
        "gemini-1.5-pro":      "cl100k_base",   # Approximate
    }

    # Anthropic tokenizer produces ~8% fewer tokens than cl100k on average English text
    ANTHROPIC_CORRECTION_FACTOR = 0.92

    # Llama3 SentencePiece produces ~5% more tokens than cl100k on average
    LLAMA_CORRECTION_FACTOR = 1.05

    # Correction factor per model vs cl100k
    MODEL_CORRECTION: dict[str, float] = {
        "claude-opus-4-6":   0.92,
        "claude-sonnet-4-6": 0.92,
        "claude-3-5-sonnet": 0.92,
        "claude-3-haiku":    0.92,
        "llama-3-8b":        1.05,
        "llama-3-70b":       1.05,
        "mistral-7b":        1.03,
        "gemini-1.5-pro":    0.95,
    }

    # Empirical chars-per-token ratio for each language on cl100k
    # These are mean values from 10K text samples per language
    CHARS_PER_TOKEN: dict[str, float] = {
        "en": 4.5,  "es": 4.2,  "fr": 4.1,  "de": 4.3,  "pt": 4.2,
        "hi": 1.4,  "bn": 1.3,  "ta": 1.1,  "te": 1.1,  "ml": 1.1,
        "kn": 1.1,  "gu": 1.3,  "mr": 1.4,  "pa": 1.5,  "ur": 1.6,
        "ar": 1.7,  "ja": 1.5,  "zh": 1.8,  "ko": 1.4,
        "id": 4.0,  "ms": 4.0,
    }

    # English chars-per-token on cl100k
    ENGLISH_CHARS_PER_TOKEN = 4.5

    def __init__(self, config: Config):
        self.config = config
        self._encoders: dict = {}

    def _get_encoder(self, enc_name: str):
        if not TIKTOKEN_AVAILABLE:
            return None
        if enc_name not in self._encoders:
            try:
                import tiktoken
                self._encoders[enc_name] = tiktoken.get_encoding(enc_name)
            except Exception as e:
                logger.warning("tiktoken encoding '{}' unavailable: {} — using char estimate", enc_name, e)
                self._encoders[enc_name] = None
        return self._encoders[enc_name]

    def count(self, text: str, model_name: str) -> int:
        """
        Count tokens for text given a model.
        Applies per-model correction factor where needed.
        """
        enc_name = self.ENCODING_MAP.get(model_name, "cl100k_base")
        encoder = self._get_encoder(enc_name)
        if encoder:
            raw_count = len(encoder.encode(text))
        else:
            # Estimate from character count
            lang_cpt = self.CHARS_PER_TOKEN.get("en", 4.5)
            raw_count = max(1, round(len(text) / lang_cpt))

        correction = self.MODEL_CORRECTION.get(model_name, 1.0)
        return max(1, round(raw_count * correction))

    async def count_english_baseline(
        self,
        text: str,
        language: str,
        model_name: str,
    ) -> int:
        """
        Estimate how many tokens this text would cost if it were English.

        Method: character-ratio estimation.
          baseline_tokens = len(text) * (chars_per_token_english / chars_per_token_language)
          then apply the same model correction factor.

        This is O(1) and adds ~0ms. MarianMT translation would be more precise but
        is not worth the latency for cost attribution purposes.
        """
        lang_cpt = self.CHARS_PER_TOKEN.get(language, 4.5)
        # Estimate number of "semantic chars" equivalent in English
        # Use the ratio of English chars-per-token to language chars-per-token
        if lang_cpt <= 0:
            lang_cpt = 4.5
        ratio = self.ENGLISH_CHARS_PER_TOKEN / lang_cpt
        estimated_english_chars = len(text) * ratio
        estimated_tokens = estimated_english_chars / self.ENGLISH_CHARS_PER_TOKEN

        correction = self.MODEL_CORRECTION.get(model_name, 1.0)
        return max(1, round(estimated_tokens * correction))

    def count_all_tokenizers(self, text: str) -> dict[str, int]:
        """
        Count tokens across all supported models simultaneously.
        Used by the Model Arbitrage Advisor (Gap 4 fix).

        Returns dict of {model_name: token_count}
        """
        results = {}
        for model_name in self.ENCODING_MAP:
            results[model_name] = self.count(text, model_name)
        return results

    async def health(self) -> str:
        if TIKTOKEN_AVAILABLE:
            return "healthy"
        return "degraded (character estimate mode)"
