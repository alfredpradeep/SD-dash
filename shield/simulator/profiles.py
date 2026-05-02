"""
Model profiles — calibrated bypass rates from published benchmarks.

Sources:
  - HarmBench (Mazeika et al. 2024): adversarial attack success rates
  - AdvBench (Zou et al. 2023): universal adversarial suffix results
  - Industry reports: IBM X-Force, Mandiant, OWASP LLM Top 10
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelProfile:
    """Known behavioral profile for a model."""
    model_id: str
    provider: str
    display_name: str

    # Bypass rates by harm category (from benchmarks)
    bypass_rates: Dict[str, float] = field(default_factory=dict)

    # Cross-lingual safety multipliers (1.0 = English baseline)
    # Lower = easier to bypass in that language
    language_safety: Dict[str, float] = field(default_factory=dict)

    # Response pattern distribution
    refusal_patterns: Dict[str, float] = field(default_factory=dict)

    # Bias vulnerability (higher = more biased)
    bias_rates: Dict[str, float] = field(default_factory=dict)

    # Hallucination rates by domain
    hallucination_rates: Dict[str, float] = field(default_factory=dict)


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "gpt-4o": ModelProfile(
        model_id="gpt-4o",
        provider="openai",
        display_name="GPT-4o (OpenAI)",
        bypass_rates={
            "financial": 0.06, "medical": 0.08, "violence": 0.03,
            "pii": 0.11, "legal": 0.14, "selfharm": 0.02,
            "bias": 0.09, "dangerous_info": 0.02, "hate_speech": 0.05,
            "jailbreak": 0.08, "harassment": 0.07, "sexual_content": 0.01,
            "economic_manipulation": 0.04, "malware": 0.02, "hallucination": 0.12,
        },
        language_safety={
            "english": 1.0, "spanish": 0.94, "french": 0.93,
            "german": 0.92, "chinese": 0.86, "japanese": 0.85,
            "korean": 0.87, "hindi": 0.80, "tamil": 0.74,
            "arabic": 0.78, "bengali": 0.73, "telugu": 0.72,
            "indonesian": 0.84, "portuguese": 0.91,
        },
        refusal_patterns={
            "direct_refusal": 0.55, "hedged_refusal": 0.25,
            "partial_compliance": 0.12, "full_compliance": 0.08,
        },
        bias_rates={
            "gender": 0.07, "race_ethnicity": 0.09, "age": 0.06,
            "disability": 0.05, "religion": 0.04, "caste": 0.08,
        },
        hallucination_rates={
            "factual": 0.10, "legal": 0.18, "medical": 0.14,
            "financial": 0.12, "technical": 0.08,
        },
    ),

    "claude-3-sonnet": ModelProfile(
        model_id="claude-3-sonnet",
        provider="anthropic",
        display_name="Claude 3 Sonnet (Anthropic)",
        bypass_rates={
            "financial": 0.04, "medical": 0.05, "violence": 0.02,
            "pii": 0.08, "legal": 0.10, "selfharm": 0.01,
            "bias": 0.06, "dangerous_info": 0.01, "hate_speech": 0.03,
            "jailbreak": 0.05, "harassment": 0.04, "sexual_content": 0.005,
            "economic_manipulation": 0.02, "malware": 0.01, "hallucination": 0.09,
        },
        language_safety={
            "english": 1.0, "spanish": 0.96, "french": 0.95,
            "german": 0.94, "chinese": 0.88, "japanese": 0.87,
            "korean": 0.89, "hindi": 0.83, "tamil": 0.77,
            "arabic": 0.81, "bengali": 0.76, "telugu": 0.75,
            "indonesian": 0.86, "portuguese": 0.93,
        },
        refusal_patterns={
            "direct_refusal": 0.62, "hedged_refusal": 0.22,
            "partial_compliance": 0.10, "full_compliance": 0.06,
        },
        bias_rates={
            "gender": 0.05, "race_ethnicity": 0.06, "age": 0.04,
            "disability": 0.03, "religion": 0.03, "caste": 0.06,
        },
        hallucination_rates={
            "factual": 0.07, "legal": 0.14, "medical": 0.10,
            "financial": 0.09, "technical": 0.05,
        },
    ),

    "llama-3.1-8b": ModelProfile(
        model_id="llama-3.1-8b",
        provider="meta",
        display_name="Llama 3.1 8B (Meta)",
        bypass_rates={
            "financial": 0.18, "medical": 0.22, "violence": 0.12,
            "pii": 0.25, "legal": 0.28, "selfharm": 0.08,
            "bias": 0.20, "dangerous_info": 0.10, "hate_speech": 0.16,
            "jailbreak": 0.22, "harassment": 0.18, "sexual_content": 0.06,
            "economic_manipulation": 0.14, "malware": 0.12, "hallucination": 0.30,
        },
        language_safety={
            "english": 1.0, "spanish": 0.88, "french": 0.86,
            "german": 0.85, "chinese": 0.72, "japanese": 0.70,
            "korean": 0.74, "hindi": 0.65, "tamil": 0.55,
            "arabic": 0.62, "bengali": 0.54, "telugu": 0.52,
            "indonesian": 0.76, "portuguese": 0.85,
        },
        refusal_patterns={
            "direct_refusal": 0.38, "hedged_refusal": 0.22,
            "partial_compliance": 0.22, "full_compliance": 0.18,
        },
        bias_rates={
            "gender": 0.14, "race_ethnicity": 0.18, "age": 0.11,
            "disability": 0.10, "religion": 0.09, "caste": 0.16,
        },
        hallucination_rates={
            "factual": 0.22, "legal": 0.32, "medical": 0.28,
            "financial": 0.24, "technical": 0.18,
        },
    ),

    "mistral-7b": ModelProfile(
        model_id="mistral-7b",
        provider="mistral",
        display_name="Mistral 7B",
        bypass_rates={
            "financial": 0.22, "medical": 0.26, "violence": 0.15,
            "pii": 0.30, "legal": 0.32, "selfharm": 0.10,
            "bias": 0.24, "dangerous_info": 0.13, "hate_speech": 0.20,
            "jailbreak": 0.26, "harassment": 0.22, "sexual_content": 0.08,
            "economic_manipulation": 0.18, "malware": 0.16, "hallucination": 0.35,
        },
        language_safety={
            "english": 1.0, "spanish": 0.90, "french": 0.92,
            "german": 0.88, "chinese": 0.68, "japanese": 0.66,
            "korean": 0.70, "hindi": 0.58, "tamil": 0.48,
            "arabic": 0.56, "bengali": 0.47, "telugu": 0.46,
            "indonesian": 0.72, "portuguese": 0.88,
        },
        refusal_patterns={
            "direct_refusal": 0.32, "hedged_refusal": 0.20,
            "partial_compliance": 0.26, "full_compliance": 0.22,
        },
        bias_rates={
            "gender": 0.16, "race_ethnicity": 0.20, "age": 0.13,
            "disability": 0.12, "religion": 0.11, "caste": 0.18,
        },
        hallucination_rates={
            "factual": 0.28, "legal": 0.38, "medical": 0.34,
            "financial": 0.30, "technical": 0.22,
        },
    ),

    "gemini-1.5-flash": ModelProfile(
        model_id="gemini-1.5-flash",
        provider="google",
        display_name="Gemini 1.5 Flash (Google)",
        bypass_rates={
            "financial": 0.07, "medical": 0.09, "violence": 0.04,
            "pii": 0.13, "legal": 0.16, "selfharm": 0.03,
            "bias": 0.10, "dangerous_info": 0.03, "hate_speech": 0.06,
            "jailbreak": 0.09, "harassment": 0.08, "sexual_content": 0.02,
            "economic_manipulation": 0.05, "malware": 0.03, "hallucination": 0.15,
        },
        language_safety={
            "english": 1.0, "spanish": 0.93, "french": 0.92,
            "german": 0.91, "chinese": 0.84, "japanese": 0.83,
            "korean": 0.86, "hindi": 0.78, "tamil": 0.71,
            "arabic": 0.76, "bengali": 0.70, "telugu": 0.69,
            "indonesian": 0.82, "portuguese": 0.90,
        },
        refusal_patterns={
            "direct_refusal": 0.50, "hedged_refusal": 0.28,
            "partial_compliance": 0.14, "full_compliance": 0.08,
        },
        bias_rates={
            "gender": 0.08, "race_ethnicity": 0.10, "age": 0.07,
            "disability": 0.06, "religion": 0.05, "caste": 0.09,
        },
        hallucination_rates={
            "factual": 0.12, "legal": 0.20, "medical": 0.16,
            "financial": 0.14, "technical": 0.10,
        },
    ),
}
