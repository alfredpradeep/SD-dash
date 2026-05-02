"""
Tests for lattice structures and extractor.

Covers:
  - SemanticUnitType enum values
  - GraphNode creation and validation
  - SemanticGraph helper methods
  - CompressionResult fields
"""

import pytest
from compress.lattice.structures import (
    SemanticUnitType,
    GraphNode,
    SemanticGraph,
    CompressionCandidate,
    CompressionResult,
)


class TestSemanticUnitType:
    def test_has_10_types(self):
        assert len(SemanticUnitType) == 10

    def test_expected_types_exist(self):
        expected = [
            "entity", "relation", "attribute", "event",
            "negation", "quantifier", "temporal", "modal",
            "sentiment", "condition",
        ]
        for name in expected:
            assert hasattr(SemanticUnitType, name.upper()) or \
                   name in [t.value for t in SemanticUnitType]


class TestGraphNode:
    def test_create_valid_node(self):
        node = GraphNode(
            id="n1",
            node_type=SemanticUnitType.ENTITY,
            value="test entity",
            confidence=0.95,
            intensity=0.8,
            specificity=0.9,
            certainty=0.85,
            source="amr",
        )
        assert node.id == "n1"
        assert node.node_type == SemanticUnitType.ENTITY
        assert node.value == "test entity"
        assert node.confidence == 0.95

    def test_node_equality_by_id(self):
        n1 = GraphNode(id="n1", node_type=SemanticUnitType.ENTITY, value="a",
                       confidence=0.9, intensity=0.5, specificity=0.5,
                       certainty=0.5, source="amr")
        n2 = GraphNode(id="n1", node_type=SemanticUnitType.ENTITY, value="a",
                       confidence=0.9, intensity=0.5, specificity=0.5,
                       certainty=0.5, source="amr")
        assert n1.id == n2.id


class TestSemanticGraph:
    def test_graph_node_count(self, sample_graph):
        assert len(sample_graph.nodes) == 4

    def test_graph_edge_count(self, sample_graph):
        assert len(sample_graph.edges) == 3

    def test_graph_has_language(self, sample_graph):
        assert sample_graph.source_language == "en"

    def test_graph_extraction_confidence(self, sample_graph):
        assert 0 <= sample_graph.extraction_confidence <= 1

    def test_graph_amr_confidence(self, sample_graph):
        assert 0 <= sample_graph.amr_confidence <= 1


class TestCompressionCandidate:
    def test_create_candidate(self, sample_candidate):
        assert sample_candidate.token_count == 12
        assert sample_candidate.semantic_similarity == 0.94
        assert sample_candidate.graph_jaccard == 0.91
        assert sample_candidate.stes_score == 2.45

    def test_candidate_text_not_empty(self, sample_candidate):
        assert len(sample_candidate.text) > 0


class TestCompressionResult:
    def test_result_has_all_fields(self, sample_result, expected_outputs):
        """CompressionResult has all required fields."""
        schema = expected_outputs["compression_result_schema"]
        for field in schema["required_fields"]:
            assert hasattr(sample_result, field), f"Missing field: {field}"

    def test_result_applied_compression(self, sample_result):
        assert sample_result.compression_applied is True
        assert sample_result.rejection_reason is None

    def test_result_reduction_ratio_valid(self, sample_result):
        assert 0.0 <= sample_result.reduction_ratio <= 1.0

    def test_result_semantic_similarity_valid(self, sample_result):
        assert 0.0 <= sample_result.semantic_similarity <= 1.0

    def test_result_no_compression(self):
        """No-compression result has ratio=0 and applied=False."""
        result = CompressionResult(
            original_text="Original",
            compressed_text="Original",
            source_language="en",
            target_tokenizer="gpt-4o",
            original_token_count=5,
            compressed_token_count=5,
            reduction_ratio=0.0,
            semantic_similarity=1.0,
            graph_jaccard=1.0,
            stes_score=0.0,
            compression_applied=False,
            rejection_reason="No candidate met thresholds",
            processing_ms=50.0,
            candidates_evaluated=3,
        )
        assert result.compression_applied is False
        assert result.reduction_ratio == 0.0
        assert result.rejection_reason is not None
