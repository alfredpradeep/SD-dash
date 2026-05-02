"""
Custom exception hierarchy for the COMPRESS module.

All module-specific exceptions inherit from CompressError.
"""


class CompressError(Exception):
    """Base exception for COMPRESS module."""


class UnsupportedLanguageError(CompressError):
    """Language or tokenizer not in supported set."""


class ExtractionError(CompressError):
    """Semantic graph extraction failed."""


class SearchError(CompressError):
    """Beam search failed to produce candidates."""


class VerificationError(CompressError):
    """Unexpected error during verification gate."""


class CacheError(CompressError):
    """Redis cache read/write failure."""
