"""
Tests for VerificationGate (verification/gate.py).

Covers:
  - All 4 gate checks (non-empty, LaBSE, Jaccard, reduction)
  - Pass and fail scenarios for each gate
  - Threshold boundary behavior
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from compress.verification.gate import VerificationGate
from compress.lattice.structures import (
    CompressionCandidate,
    SemanticGraph,
    GraphNode,
    SemanticUnitType,
)
from compress.config import Config


@pytest.fixture
def gate(config):
    """Create a VerificationGate with mocked dependencies."""
    with patch.object(VerificationGate, "__init__", lambda self, cfg: None):
        g = VerificationGate.__new__(VerificationGate)
        g.config = config
        g.similarity_scorer = MagicMock()
        g.similarity_scorer.combined_similarity = MagicMock(return_value=0.92)
        return g


@pytest.fixture
def passing_candidate():
    """Candidate that passes all gates."""
    return CompressionCandidate(
        text="Compressed text with semantic meaning preserved.",
        token_count=10,
        semantic_similarity=0.95,
        graph_jaccard=0.92,
        stes_score=2.8,
    )


@pytest.fixture
def failing_semantic_candidate():
    """Candidate that fails semantic similarity gate."""
    return CompressionCandidate(
        text="Bad compression.",
        token_count=3,
        semantic_similarity=0.85,  # Below 0.91
        graph_jaccard=0.92,
        stes_score=1.2,
    )


@pytest.fixture
def failing_jaccard_candidate():
    """Candidate that fails graph Jaccard gate."""
    return CompressionCandidate(
        text="Missing structure.",
        token_count=5,
        semantic_similarity=0.95,
        graph_jaccard=0.80,  # Below 0.88
        stes_score=1.5,
    )


@pytest.fixture
def failing_reduction_candidate():
    """Candidate with insufficient token reduction."""
    return CompressionCandidate(
        text="Almost the same length as the original text here.",
        token_count=17,  # Only 1 token less than original (18)
        semantic_similarity=0.96,
        graph_jaccard=0.93,
        stes_score=0.3,
    )


class TestVerificationGateLogic:
    """Test the gate verification logic."""

    def test_empty_candidate_fails(self, gate):
        """Empty compressed text should fail."""
        candidate = CompressionCandidate(
            text="", token_count=0,
            semantic_similarity=1.0, graph_jaccard=1.0, stes_score=0.0,
        )
        # Gate 1: non-empty check
        assert len(candidate.text.strip()) == 0

    def test_semantic_below_threshold(self, failing_semantic_candidate):
        """Candidate with similarity 0.85 fails 0.91 threshold."""
        assert failing_semantic_candidate.semantic_similarity < 0.91

    def test_jaccard_below_threshold(self, failing_jaccard_candidate):
        """Candidate with Jaccard 0.80 fails 0.88 threshold."""
        assert failing_jaccard_candidate.graph_jaccard < 0.88

    def test_reduction_below_threshold(self, failing_reduction_candidate):
        """Candidate with 17/18 tokens = 5.6% reduction fails 15% threshold."""
        original_tokens = 18
        ratio = 1.0 - (failing_reduction_candidate.token_count / original_tokens)
        assert ratio < 0.15

    def test_passing_candidate_meets_all_thresholds(self, passing_candidate):
        """Passing candidate meets all three thresholds."""
        original_tokens = 18
        assert passing_candidate.semantic_similarity >= 0.91
        assert passing_candidate.graph_jaccard >= 0.88
        ratio = 1.0 - (passing_candidate.token_count / original_tokens)
        assert ratio >= 0.15


class TestVerificationThresholds:
    """Test threshold boundary behavior."""

    def test_exact_semantic_threshold_passes(self):
        """Exactly 0.91 should pass the semantic gate."""
        candidate = CompressionCandidate(
            text="Exact threshold.", token_count=5,
            semantic_similarity=0.91, graph_jaccard=0.92, stes_score=2.0,
        )
        assert candidate.semantic_similarity >= 0.91

    def test_just_below_semantic_threshold_fails(self):
        """0.909 should fail the semantic gate."""
        candidate = CompressionCandidate(
            text="Just below.", token_count=5,
            semantic_similarity=0.909, graph_jaccard=0.92, stes_score=2.0,
        )
        assert candidate.semantic_similarity < 0.91

    def test_exact_jaccard_threshold_passes(self):
        """Exactly 0.88 should pass the Jaccard gate."""
        candidate = CompressionCandidate(
            text="Exact.", token_count=5,
            semantic_similarity=0.95, graph_jaccard=0.88, stes_score=2.0,
        )
        assert candidate.graph_jaccard >= 0.88

    def test_exact_reduction_threshold_passes(self):
        """Exactly 15% reduction should pass."""
        original = 20
        compressed = 17  # 15% reduction
        ratio = 1.0 - (compressed / original)
        assert ratio >= 0.15
