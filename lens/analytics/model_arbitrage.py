"""
Model Arbitrage Advisor — Gap 4 Fix.

Cross-model tokenizer comparison: given text in any language,
compute token count + cost + ETR inequity across ALL supported models simultaneously.

Key insight: the same Tamil text tokenizes to different counts on different models.
GPT-4o's cl100k may produce 150 tokens while Llama3's SentencePiece produces 120.
For high-volume multilingual traffic, routing Tamil requests to the right model
can reduce token costs by 20–40%.

Output: ranked list of models by cost-efficiency for the given language + text.
"""

from loguru import logger
from lens.entropy.structures import TokenizerArbitrageResult
from lens.detection.token_counter import TokenCounter
from lens.entropy.etr_calculator import ETRCalculator
from lens.config import Config


class ModelArbitrageAdvisor:
    """
    Computes token cost across all supported models and recommends the most efficient.
    """

    # Token cost per 1M input tokens (USD) — updated periodically
    # claude-opus-4-6 and claude-sonnet-4-6 added as requested
    MODEL_COSTS_PER_M: dict[str, float] = {
        "gpt-4o":              5.00,
        "gpt-4o-mini":         0.15,
        "gpt-3.5-turbo":       0.50,
        "claude-opus-4-6":     15.00,    # Opus 4 pricing tier
        "claude-sonnet-4-6":   3.00,
        "claude-3-5-sonnet":   3.00,
        "claude-3-haiku":      0.25,
        "llama-3-8b":          0.06,     # Fireworks/Together pricing
        "llama-3-70b":         0.90,
        "mistral-7b":          0.25,
        "gemini-1.5-pro":      3.50,
    }

    def __init__(self, config: Config, token_counter: TokenCounter, etr_calc: ETRCalculator):
        self.config = config
        self.token_counter = token_counter
        self.etr_calc = etr_calc

    async def compute(
        self,
        text: str,
        language: str,
        customer_id: str = "default",
    ) -> TokenizerArbitrageResult:
        """
        Compute token efficiency across all models for the given text.

        Returns TokenizerArbitrageResult with per-model breakdown and recommendation.
        """
        # Count tokens for all models simultaneously
        all_token_counts = self.token_counter.count_all_tokenizers(text)

        results: dict[str, dict] = {}
        for model_name, token_count in all_token_counts.items():
            cost_per_m = self.MODEL_COSTS_PER_M.get(model_name, 3.0)
            cost_usd = token_count * cost_per_m / 1_000_000

            # ETR for this model (same entropy, different token count)
            char_entropy = sum(
                self.etr_calc.LINGUISTIC_ENTROPY_PER_CHAR.get(language, 4.0)
                for _ in text
            )
            etr = self.etr_calc.compute(char_entropy, token_count, language)

            # English baseline tokens for this model
            en_baseline = await self.token_counter.count_english_baseline(
                text, language, model_name
            )
            etr_english = self.etr_calc.compute(char_entropy, en_baseline, "en")
            inequity = self.etr_calc.compute_inequity_ratio(etr, etr_english)

            results[model_name] = {
                "tokens": token_count,
                "cost_usd": round(cost_usd, 8),
                "etr": round(etr, 4),
                "inequity": round(inequity, 3),
                "cost_per_m": cost_per_m,
            }

        # Find best model: minimum cost_usd (practical criterion)
        best_model = min(results, key=lambda m: results[m]["cost_usd"])
        best_cost = results[best_model]["cost_usd"]

        # Compute max savings vs most common baseline (gpt-4o)
        gpt4o_cost = results.get("gpt-4o", {}).get("cost_usd", 0)
        max_savings_pct = 0.0
        if gpt4o_cost > 0:
            max_savings_pct = (gpt4o_cost - best_cost) / gpt4o_cost * 100

        lang_name = _LANG_NAMES.get(language, language.upper())
        savings_summary = (
            f"For {lang_name} text, {best_model} uses "
            f"{results[best_model]['tokens']} tokens vs "
            f"gpt-4o's {results.get('gpt-4o', {}).get('tokens', '?')} tokens "
            f"({max_savings_pct:.1f}% cost reduction). "
            f"ETR inequity: {results[best_model]['inequity']:.1f}x (lower is better)."
        )

        return TokenizerArbitrageResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            language=language,
            results=results,
            recommended_model=best_model,
            max_savings_pct=round(max_savings_pct, 2),
            savings_summary=savings_summary,
        )


_LANG_NAMES: dict[str, str] = {
    "ta": "Tamil", "ml": "Malayalam", "te": "Telugu", "kn": "Kannada",
    "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu",
    "ar": "Arabic", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "id": "Indonesian", "ms": "Malay",
}
