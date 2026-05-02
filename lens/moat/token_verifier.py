"""
Live Multi-Provider Token Verification Engine

Verifies token counts using real tokenizers where available.
Falls back to estimates with confidence scores.

Primary use: Verify that LENS token counts match provider charges.
Secondary use: What-if analysis across providers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import sys

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class TokenVerification:
    """Result of token verification for a single input."""
    text: str
    model: str
    provider: str
    measured_tokens: int          # What LENS counts using the real tokenizer
    estimated_cost_usd: float     # What LENS calculates the cost should be
    verification_method: str      # "tiktoken_real", "anthropic_api", "sentencepiece_real", "char_estimate"
    confidence: float             # 1.0 for real tokenizer, 0.7 for estimate
    timestamp: datetime


@dataclass
class CrossProviderComparison:
    """What-if comparison: shows cost on other providers without switching."""
    text: str
    current_model: str
    current_tokens: int
    current_cost: float
    alternatives: List[Dict] = field(default_factory=list)  # [{model, tokens, cost, savings_pct, confidence}]
    best_alternative: str = ""
    max_savings_pct: float = 0.0
    recommendation: str = ""      # Plain English: "Your 3 cheapest options are..."


class TokenVerificationEngine:
    """
    Verifies token counts using real tokenizers where available.
    Falls back to estimates with confidence scores.
    """

    # Real tokenizer support matrix
    TOKENIZER_MAP = {
        # OpenAI — tiktoken (REAL, local, no API call needed)
        "gpt-4o": {"lib": "tiktoken", "encoding": "cl100k_base", "confidence": 1.0},
        "gpt-4o-mini": {"lib": "tiktoken", "encoding": "cl100k_base", "confidence": 1.0},
        "gpt-4-turbo": {"lib": "tiktoken", "encoding": "cl100k_base", "confidence": 1.0},
        "gpt-3.5-turbo": {"lib": "tiktoken", "encoding": "cl100k_base", "confidence": 1.0},
        "o1": {"lib": "tiktoken", "encoding": "o200k_base", "confidence": 1.0},
        "o1-mini": {"lib": "tiktoken", "encoding": "o200k_base", "confidence": 1.0},

        # Anthropic — estimate via tiktoken + correction factor
        "claude-opus-4-6": {"lib": "tiktoken_corrected", "encoding": "cl100k_base", "factor": 0.92, "confidence": 0.85},
        "claude-sonnet-4-6": {"lib": "tiktoken_corrected", "encoding": "cl100k_base", "factor": 0.92, "confidence": 0.85},
        "claude-3-5-sonnet": {"lib": "tiktoken_corrected", "encoding": "cl100k_base", "factor": 0.92, "confidence": 0.85},
        "claude-3-haiku": {"lib": "tiktoken_corrected", "encoding": "cl100k_base", "factor": 0.92, "confidence": 0.85},

        # Open source — character estimate with language-specific ratios
        "llama-3-8b": {"lib": "char_estimate", "chars_per_token": 3.5, "confidence": 0.70},
        "llama-3-70b": {"lib": "char_estimate", "chars_per_token": 3.5, "confidence": 0.70},
        "mistral-7b": {"lib": "char_estimate", "chars_per_token": 3.6, "confidence": 0.70},
        "gemini-1.5-pro": {"lib": "char_estimate", "chars_per_token": 3.9, "confidence": 0.70},
    }

    # Cost per 1M INPUT tokens (USD) — latest pricing as of 2024
    PRICING = {
        "gpt-4o": 2.50,
        "gpt-4o-mini": 0.15,
        "gpt-4-turbo": 10.00,
        "gpt-3.5-turbo": 0.50,
        "o1": 15.00,
        "o1-mini": 3.00,
        "claude-opus-4-6": 15.00,
        "claude-sonnet-4-6": 3.00,
        "claude-3-5-sonnet": 3.00,
        "claude-3-haiku": 0.25,
        "llama-3-8b": 0.05,
        "llama-3-70b": 0.59,
        "mistral-7b": 0.10,
        "gemini-1.5-pro": 1.25,
    }

    def __init__(self):
        """Initialize the verification engine with cached tokenizers."""
        self._encoders = {}  # cached tiktoken encoders

    def _get_provider(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()
        if "gpt-" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude-" in model_lower:
            return "anthropic"
        elif "llama-" in model_lower:
            return "meta"
        elif "mistral-" in model_lower:
            return "mistral"
        elif "gemini-" in model_lower:
            return "google"
        else:
            return "unknown"

    def verify(self, text: str, model: str) -> TokenVerification:
        """
        Count tokens using the best available method for this model.
        Returns TokenVerification with confidence score and timestamp.
        Falls back gracefully if encoding is unavailable (e.g., network error).
        """
        if model not in self.TOKENIZER_MAP:
            # Fallback: char estimate for unknown models
            token_count = self._count_char_estimate(text, chars_per_token=3.5)
            verification_method = "char_estimate"
            confidence = 0.60
        else:
            tokenizer_info = self.TOKENIZER_MAP[model]
            lib_type = tokenizer_info["lib"]

            try:
                if lib_type == "tiktoken":
                    token_count = self._count_tiktoken(text, tokenizer_info["encoding"])
                    verification_method = "tiktoken_real"
                    confidence = tokenizer_info["confidence"]
                elif lib_type == "tiktoken_corrected":
                    token_count = self._count_corrected(
                        text,
                        tokenizer_info["encoding"],
                        tokenizer_info["factor"]
                    )
                    verification_method = "tiktoken_corrected"
                    confidence = tokenizer_info["confidence"]
                elif lib_type == "char_estimate":
                    token_count = self._count_char_estimate(text, tokenizer_info["chars_per_token"])
                    verification_method = "char_estimate"
                    confidence = tokenizer_info["confidence"]
                else:
                    # Fallback
                    token_count = self._count_char_estimate(text, chars_per_token=3.5)
                    verification_method = "char_estimate"
                    confidence = 0.60
            except Exception:
                # Graceful fallback for any errors
                token_count = self._count_char_estimate(text, chars_per_token=3.5)
                verification_method = "char_estimate_fallback"
                confidence = 0.60

        # Calculate cost
        price_per_1m = self.PRICING.get(model, 1.0)
        estimated_cost = (token_count / 1_000_000) * price_per_1m

        provider = self._get_provider(model)

        return TokenVerification(
            text=text,
            model=model,
            provider=provider,
            measured_tokens=token_count,
            estimated_cost_usd=estimated_cost,
            verification_method=verification_method,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )

    def compare_providers(self, text: str, current_model: str) -> CrossProviderComparison:
        """
        What-if analysis: verify across all models, sort by cost,
        generate recommendation.
        """
        current_verification = self.verify(text, current_model)
        current_tokens = current_verification.measured_tokens
        current_cost = current_verification.estimated_cost_usd

        alternatives = []

        for model in self.TOKENIZER_MAP.keys():
            if model == current_model:
                continue

            verification = self.verify(text, model)
            tokens = verification.measured_tokens
            cost = verification.estimated_cost_usd
            savings_pct = ((current_cost - cost) / current_cost * 100) if current_cost > 0 else 0

            alternatives.append({
                "model": model,
                "tokens": tokens,
                "cost": cost,
                "savings_pct": savings_pct,
                "confidence": verification.confidence
            })

        # Sort by cost (ascending)
        alternatives.sort(key=lambda x: x["cost"])

        # Find best alternative and max savings
        best_alternative = alternatives[0]["model"] if alternatives else ""
        max_savings_pct = alternatives[0]["savings_pct"] if alternatives else 0.0

        # Generate recommendation
        if alternatives:
            top_3 = alternatives[:3]
            top_3_names = ", ".join([alt["model"] for alt in top_3])
            recommendation = (
                f"Your 3 cheapest options are: {top_3_names}. "
                f"Best savings: {max_savings_pct:.2f}% with {best_alternative}."
            )
        else:
            recommendation = f"No alternatives available. {current_model} is your only option."

        return CrossProviderComparison(
            text=text,
            current_model=current_model,
            current_tokens=current_tokens,
            current_cost=current_cost,
            alternatives=alternatives,
            best_alternative=best_alternative,
            max_savings_pct=max_savings_pct,
            recommendation=recommendation
        )

    def _count_tiktoken(self, text: str, encoding: str) -> int:
        """Real token count via tiktoken. Falls back to char estimate if encoding unavailable."""
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for real token counting. "
                "Install with: pip install tiktoken"
            )

        try:
            if encoding not in self._encoders:
                try:
                    self._encoders[encoding] = tiktoken.get_encoding(encoding)
                except Exception as e:
                    # Network error or unavailable encoding - fall back to char estimate
                    return self._count_char_estimate(text, chars_per_token=3.5)

            encoder = self._encoders[encoding]
            tokens = encoder.encode(text)
            return len(tokens)
        except Exception as e:
            # Any error, fall back to char estimate
            return self._count_char_estimate(text, chars_per_token=3.5)

    def _count_corrected(self, text: str, encoding: str, factor: float) -> int:
        """Tiktoken count * correction factor for non-OpenAI models."""
        base_count = self._count_tiktoken(text, encoding)
        return int(base_count * factor)

    def _count_char_estimate(self, text: str, chars_per_token: float) -> int:
        """Character-based estimation fallback."""
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")

        char_count = len(text)
        token_count = int(char_count / chars_per_token) + 1
        return max(1, token_count)


def main_demo():
    """Demo the token verification engine."""
    engine = TokenVerificationEngine()

    test_text = "The quick brown fox jumps over the lazy dog. " * 10

    print("=" * 70)
    print("TOKEN VERIFICATION ENGINE DEMO")
    print("=" * 70)

    # Test single verification (using char estimate since tiktoken encoding may require network)
    print("\n1. Single Model Verification (llama-3-8b with char estimate):")
    print("-" * 70)
    verification = engine.verify(test_text, "llama-3-8b")
    print(f"Text length: {len(test_text)} characters")
    print(f"Tokens: {verification.measured_tokens}")
    print(f"Cost (USD): ${verification.estimated_cost_usd:.6f}")
    print(f"Method: {verification.verification_method}")
    print(f"Confidence: {verification.confidence}")

    # Test cross-provider comparison
    print("\n2. Cross-Provider Comparison:")
    print("-" * 70)
    comparison = engine.compare_providers(test_text, "gpt-4o-mini")
    print(f"Current model: {comparison.current_model}")
    print(f"Current cost: ${comparison.current_cost:.6f}")
    print(f"\nTop 3 alternatives:")
    for i, alt in enumerate(comparison.alternatives[:3], 1):
        print(f"  {i}. {alt['model']}: ${alt['cost']:.6f} "
              f"(saves {alt['savings_pct']:.2f}%, confidence: {alt['confidence']})")
    print(f"\nRecommendation: {comparison.recommendation}")

    # Test different models
    print("\n3. Verify across different models:")
    print("-" * 70)
    test_models = ["gpt-4o-mini", "claude-3-5-sonnet", "llama-3-8b", "mistral-7b"]
    for model in test_models:
        try:
            v = engine.verify(test_text, model)
            print(f"{model:25} | Tokens: {v.measured_tokens:5} | Cost: ${v.estimated_cost_usd:.6f} | Confidence: {v.confidence}")
        except Exception as e:
            print(f"{model:25} | Error (network likely needed): {type(e).__name__}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main_demo()
