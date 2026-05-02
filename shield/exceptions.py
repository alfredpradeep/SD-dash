"""
Custom exception hierarchy for the SHIELD module.

All module-specific exceptions inherit from ShieldError.
"""


class ShieldError(Exception):
    """Base exception for SHIELD module."""


class ScanError(ShieldError):
    """Error during adversarial scanning or probe generation."""


class ProbeError(ShieldError):
    """Error during probe execution or analysis."""


class JudgeError(ShieldError):
    """Error in judge LLM feedback or classification."""


class DriftError(ShieldError):
    """Error during semantic drift detection."""


class BenchmarkError(ShieldError):
    """Error during benchmark execution or evaluation."""


class ProviderError(ShieldError):
    """Error with LLM provider, endpoint, or API call."""


class UnsupportedLanguageError(ShieldError):
    """Language not in supported set."""


class UnsupportedHarmCategoryError(ShieldError):
    """Harm category not in supported set."""


class ConfigurationError(ShieldError):
    """Configuration validation or initialization error."""


class AttackTimeoutError(ShieldError):
    """Attack iteration exceeded max iterations or timeout."""
