"""
Tests for GraphSimilarityScorer (lattice/similarity.py).

Covers:
  - Node similarity with exact matches
  - Node similarity with fuzzy entity matching
  - Edge similarity scoring
  - Combined score weighting
  - Edge cases: empty graphs, single nodes
"""

import pytest
from compress.lattice.similarity import GraphSimilarityScorer
from compress.lattice.structures import (
    SemanticGraph,
    GraphNode,
    SemanticUnitType,
)


@pytest.fixture
def scorer():
    return GraphSimilarityScorer()


@pytest.fixture
def graph_a():
    """Graph with 3 nodes and 2 edges."""
    return SemanticGraph(
        nodes=[
            GraphNode(id="n1", node_type=SemanticUnitType.ENTITY, value="machine learning",
                      confidence=0.9, intensity=0.8, specificity=0.9, certainty=0.9, source="amr"),
            GraphNode(id="n2", node_type=SemanticUnitType.EVENT, value="perform",
                      confidence=0.85, intensity=0.7, specificity=0.6, certainty=0.8, source="amr"),
            GraphNode(id="n3", node_type=SemanticUnitType.ATTRIBUTE, value="well",
                      confidence=0.8, intensity=0.6, specificity=0.5, certainty=0.75, source="nli"),
        ],
        edges=[("n1", "n2", "agent"), ("n2", "n3", "manner")],
        source_language="en",
        amr_penman="",
        extraction_confidence=0.85,
        amr_confidence=0.80,
        component_scores={},
    )


@pytest.fixture
def graph_b_similar():
    """Graph similar to graph_a with slight variation."""
    return SemanticGraph(
        nodes=[
            GraphNode(id="m1", node_type=SemanticUnitType.ENTITY, value="machine learning model",
                      confidence=0.92, intensity=0.8, specificity=0.9, certainty=0.9, source="amr"),
            GraphNode(id="m2", node_type=SemanticUnitType.EVENT, value="perform",
                      confidence=0.88, intensity=0.7, specificity=0.6, certainty=0.85, source="amr"),
            GraphNode(id="m3", node_type=SemanticUnitType.ATTRIBUTE, value="well",
                      confidence=0.83, intensity=0.65, specificity=0.5, certainty=0.78, source="nli"),
        ],
        edges=[("m1", "m2", "agent"), ("m2", "m3", "manner")],
        source_language="en",
        amr_penman="",
        extraction_confidence=0.87,
        amr_confidence=0.82,
        component_scores={},
    )


@pytest.fixture
def graph_empty():
    """Empty graph."""
    return SemanticGraph(
        nodes=[], edges=[], source_language="en", amr_penman="",
        extraction_confidence=0.0, amr_confidence=0.0, component_scores={},
    )


class TestNodeSimilarity:
    def test_identical_nodes_score_1(self, scorer, graph_a):
        score = scorer.node_similarity(graph_a, graph_a)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar_graphs_high_score(self, scorer, graph_a, graph_b_similar):
        score = scorer.node_similarity(graph_a, graph_b_similar)
        assert score > 0.8

    def test_empty_graphs_score_1(self, scorer, graph_empty):
        score = scorer.node_similarity(graph_empty, graph_empty)
        assert score == pytest.approx(1.0)


class TestEdgeSimilarity:
    def test_identical_edges_score_1(self, scorer, graph_a):
        score = scorer.edge_similarity(graph_a, graph_a)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_similar_edges_high_score(self, scorer, graph_a, graph_b_similar):
        score = scorer.edge_similarity(graph_a, graph_b_similar)
        assert score > 0.8

    def test_empty_edges_score_1(self, scorer, graph_empty):
        score = scorer.edge_similarity(graph_empty, graph_empty)
        assert score == pytest.approx(1.0)


class TestCombinedScore:
    def test_combined_weighted_70_30(self, scorer, graph_a, graph_b_similar):
        node_sim = scorer.node_similarity(graph_a, graph_b_similar)
        edge_sim = scorer.edge_similarity(graph_a, graph_b_similar)
        combined = scorer.combined_similarity(graph_a, graph_b_similar)

        expected = 0.7 * node_sim + 0.3 * edge_sim
        assert combined == pytest.approx(expected, abs=0.01)

    def test_combined_score_range(self, scorer, graph_a, graph_b_similar):
        score = scorer.combined_similarity(graph_a, graph_b_similar)
        assert 0.0 <= score <= 1.0

    def test_identical_graphs_score_1(self, scorer, graph_a):
        score = scorer.combined_similarity(graph_a, graph_a)
        assert score == pytest.approx(1.0, abs=0.01)
