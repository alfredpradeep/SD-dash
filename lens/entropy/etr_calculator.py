"""
Entropy-Token Ratio (ETR) calculator.

ETR = total_linguistic_entropy (bits) / token_count

Measures information delivered per token. A fair tokenizer produces similar ETR
across all languages. Dramatically lower ETR for Tamil vs English = the tokenizer
is delivering less information per token for Tamil = tokenization inequity.

ETR inequity ratio = ETR_english / ETR_language
  > 1.5 → significant inequity
  > 3.0 → severe inequity (common for Tamil, Arabic, CJK scripts)
"""

import numpy as np
from lens.config import Config


class ETRCalculator:
    """Entropy-Token Ratio calculator with equity reporting."""

    # Character-level entropy per character (bits), derived from large corpora
    LINGUISTIC_ENTROPY_PER_CHAR: dict[str, float] = {
        "en": 4.03, "es": 3.97, "fr": 3.94, "de": 4.15, "pt": 3.95,
        "hi": 4.40, "bn": 4.38, "ta": 4.55, "te": 4.52, "ml": 4.55,
        "kn": 4.50, "gu": 4.35, "mr": 4.38, "pa": 4.30, "ur": 4.32,
        "ar": 4.32, "ja": 5.11, "zh": 5.35, "ko": 4.43,
        "id": 3.88, "ms": 3.90,
    }

    # English baseline: bits per token for well-formed English text
    ENGLISH_BITS_PER_TOKEN = 4.03 * 4.5   # ~4.5 chars per English token on cl100k

    def __init__(self, config: Config):
        self.config = config

    def compute(
        self,
        total_entropy_bits: float,
        token_count: int,
        language: str,
    ) -> float:
        """
        Compute ETR: bits of linguistic information per token.

        Returns bits/token. English baseline ≈ 18 bits/token.
        Tamil with fragmented tokenization ≈ 3–6 bits/token → inequity ratio 3–6x.
        """
        if token_count <= 0:
            return 0.0
        etr = total_entropy_bits / token_count
        return float(np.clip(etr, 0.0, 100.0))

    def compute_from_text(self, text: str, token_count: int, language: str) -> float:
        """Estimate ETR from text character count and language entropy stats."""
        char_entropy_rate = self.LINGUISTIC_ENTROPY_PER_CHAR.get(language, 4.0)
        estimated_entropy = len(text) * char_entropy_rate
        return self.compute(estimated_entropy, token_count, language)

    def compute_inequity_ratio(self, etr_language: float, etr_english: float) -> float:
        """
        Compute inequity ratio: how many times more tokens the language needs
        per unit of information compared to English.
        """
        if etr_language <= 0:
            return 0.0
        return float(np.clip(etr_english / etr_language, 1.0, 50.0))

    def equity_statement(
        self,
        language: str,
        etr_language: float,
        etr_english: float,
        cost_usd: float,
        waste_cost_usd: float,
    ) -> str:
        """
        Generate a human-readable equity statement for enterprise reporting.

        Example: "Tamil text carries the same information content as English text,
        but the tokenizer produces 4.8x more tokens, meaning you pay 4.8x more
        for the same information. ETR inequity ratio: 4.8 (Tamil: 3.8 bits/token
        vs English baseline: 18.2 bits/token)."
        """
        ratio = self.compute_inequity_ratio(etr_language, etr_english)
        waste_pct = (waste_cost_usd / cost_usd * 100) if cost_usd > 0 else 0
        lang_name = _LANG_NAMES.get(language, language.upper())
        return (
            f"{lang_name} text carries the same information as English text, but the "
            f"tokenizer produces {ratio:.1f}x more tokens to encode it — you pay "
            f"{ratio:.1f}x more for the same information. "
            f"ETR: {lang_name} {etr_language:.1f} bits/token vs "
            f"English baseline {etr_english:.1f} bits/token "
            f"({waste_pct:.0f}% of this request's cost is tokenization waste)."
        )


_LANG_NAMES: dict[str, str] = {
    "ta": "Tamil", "ml": "Malayalam", "te": "Telugu", "kn": "Kannada",
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu",
    "ar": "Arabic", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "id": "Indonesian", "ms": "Malay",
}
