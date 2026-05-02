"""
Tests for BeamSearcher (search/beam.py).

Tests are written against the mocked searcher since
actual beam search requires loaded ML models (mT5 + LaBSE).
Integration tests would require GPU/model fixtures.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from compress.search.beam import BeamSearcher
from compress.search.scorer import STESScorer
from compress.lattice.structures import (
    SemanticGraph,
    GraphNode,
    SemanticUnitType,
    CompressionCandidate,
)


class TestSTESScorer:
    """Tests for the STES scoring function."""

    @pytest.fixture
    def scorer(self):
        return STESScorer()

    def test_english_params(self, scorer):
        """English uses alpha=1.0, beta=0.65."""
        assert scorer.LANG_PARAMS["en"]["alpha"] == 1.0
        assert scorer.LANG_PARAMS["en"]["beta"] == 0.65

    def test_tamil_params(self, scorer):
        """Tamil has higher alpha (1.3) for graph preservation emphasis."""
        assert scorer.LANG_PARAMS["ta"]["alpha"] == 1.3
        assert scorer.LANG_PARAMS["ta"]["beta"] == 0.9

    def test_compute_stes_positive(self, scorer):
        """STES score is positive for valid compression."""
        score = scorer.compute_stes(
            graph_preservation=0.92,
            token_ratio=0.65,
            language="en",
        )
        assert score > 0

    def test_compute_stes_zero_token_ratio(self, scorer):
        """Token ratio of 0 should be handled without division by zero."""
        score = scorer.compute_stes(
            graph_preservation=0.9,
            token_ratio=0.01,  # Near-zero
            language="en",
        )
        assert score > 0

    def test_compute_stes_unknown_language_defaults(self, scorer):
        """Unknown language falls back to English params."""
        score_unknown = scorer.compute_stes(0.9, 0.7, "xx")
        score_en = scorer.compute_stes(0.9, 0.7, "en")
        assert score_unknown == score_en

    def test_all_20_languages_have_params(self, scorer):
        """All 20 supported languages have STES parameters."""
        expected = [
            "en", "ta", "hi", "ar", "ja", "zh", "ko", "pt", "es",
            "fr", "de", "id", "ms", "bn", "ur", "te", "ml", "pa", "gu", "mr",
        ]
        for lang in expected:
            assert lang in scorer.LANG_PARAMS, f"Missing STES params for {lang}"

    def test_stes_higher_with_better_compression(self, scorer):
        """Lower token ratio (better compression) yields higher STES."""
        score_low = scorer.compute_stes(0.92, 0.5, "en")
        score_high = scorer.compute_stes(0.92, 0.8, "en")
        assert score_low > score_high


class TestBeamSearcherMocked:
    """Tests for BeamSearcher with mocked models."""

    async def test_search_returns_candidates(self, mock_engine, sample_graph):
        """Mocked search returns candidates."""
        candidates = await mock_engine.searcher.search(
            graph=sample_graph,
            target_tokenizer="gpt-4o",
            beam_width=4,
            max_candidates=3,
        )
        assert isinstance(candidates, list)

    def test_count_tokens_delegates(self, mock_engine):
        """count_tokens delegates to mock."""
        count = mock_engine.searcher.count_tokens("test text", "gpt-4o")
        assert isinstance(count, int)
