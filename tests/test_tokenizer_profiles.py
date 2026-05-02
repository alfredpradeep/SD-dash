"""
Tests for TokenizerProfiles (search/tokenizer_profiles.py).

Covers:
  - Profile registration for all tokenizer models
  - Token counting with cl100k (tiktoken)
  - Graceful fallback for unavailable backends
  - Consistent encoding name mapping
"""

import pytest
from compress.search.tokenizer_profiles import TokenizerProfiles


@pytest.fixture
def profiles():
    return TokenizerProfiles()


class TestProfileRegistry:
    def test_all_tokenizers_registered(self, profiles):
        """All 10 expected tokenizer profiles are registered."""
        expected = [
            "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
            "claude-3-5-sonnet", "claude-3-haiku", "claude-3-opus",
            "llama-3-8b", "llama-3-70b",
            "mistral-7b", "mistral-8x7b",
        ]
        for tok in expected:
            assert tok in profiles.PROFILES

    def test_profile_has_required_keys(self, profiles):
        """Each profile has backend and encoding_name."""
        for name, profile in profiles.PROFILES.items():
            assert "backend" in profile, f"{name} missing 'backend'"
            assert "encoding_name" in profile, f"{name} missing 'encoding_name'"


class TestTokenCounting:
    def test_gpt4o_counts_tokens(self, profiles):
        """GPT-4o tokenizer returns positive count for text."""
        count = profiles.count_tokens("Hello, world!", "gpt-4o")
        assert isinstance(count, int)
        assert count > 0

    def test_gpt4o_empty_string(self, profiles):
        """Empty string returns 0 tokens."""
        count = profiles.count_tokens("", "gpt-4o")
        assert count == 0

    def test_unknown_tokenizer_falls_back(self, profiles):
        """Unknown tokenizer falls back to cl100k without error."""
        count = profiles.count_tokens("Fallback test", "nonexistent-model")
        assert isinstance(count, int)
        assert count > 0

    def test_multilingual_text_counts(self, profiles):
        """Multilingual text produces valid token counts."""
        texts = {
            "en": "The quick brown fox",
            "ta": "வணக்கம் உலகம்",
            "ja": "こんにちは世界",
            "ar": "مرحبا بالعالم",
        }
        for lang, text in texts.items():
            count = profiles.count_tokens(text, "gpt-4o")
            assert count > 0, f"Failed for {lang}: {text}"

    def test_different_tokenizers_different_counts(self, profiles):
        """Different tokenizers may produce different counts for the same text."""
        text = "Semantic compression reduces token costs significantly."
        gpt4o = profiles.count_tokens(text, "gpt-4o")
        # At minimum, cl100k should be consistent with itself
        gpt35 = profiles.count_tokens(text, "gpt-3.5-turbo")
        assert gpt4o == gpt35  # Both use cl100k_base
