"""
Weighted graph Jaccard similarity scorer.

Computes similarity between two SemanticGraphs using:
- Node matching: type + value equality with fuzzy matching for entities
- Edge matching: exact relation-type match between matched node pairs
- Weighted scoring: entity/negation nodes weighted 1.5x (loss is dangerous)
"""

import Levenshtein
from compress.lattice.structures import (
    SemanticGraph, GraphNode, SemanticUnitType,
)
from loguru import logger


class GraphSimilarityScorer:
    """
    Weighted Jaccard similarity between two semantic graphs.

    Node type weights (higher = more important to preserve):
        ENTITY:      1.5  -- entity loss is the most dangerous failure
        NEGATION:    1.5  -- negation inversion reverses meaning entirely
        QUANTIFIER:  1.3  -- numerical accuracy is critical
        ATTRIBUTE:   1.2  -- attribute loss changes factual content
        CAUSAL:      1.2  -- causal relationships affect reasoning
        CONDITIONAL: 1.2  -- conditional clauses carry meaning
        SENTIMENT:   1.0  -- sentiment shift is noticeable but less dangerous
        TEMPORAL:    1.0  -- temporal shift is noticeable but less dangerous
        MODAL:       1.0  -- modality changes are subtle
        RELATIONAL:  0.8  -- relational structure can be implicit
    """

    NODE_TYPE_WEIGHTS: dict[SemanticUnitType, float] = {
        SemanticUnitType.ENTITY:      1.5,
        SemanticUnitType.NEGATION:    1.5,
        SemanticUnitType.QUANTIFIER:  1.3,
        SemanticUnitType.ATTRIBUTE:   1.2,
        SemanticUnitType.CAUSAL:      1.2,
        SemanticUnitType.CONDITIONAL: 1.2,
        SemanticUnitType.SENTIMENT:   1.0,
        SemanticUnitType.TEMPORAL:    1.0,
        SemanticUnitType.MODAL:       1.0,
        SemanticUnitType.RELATIONAL:  0.8,
    }

    ENTITY_FUZZY_THRESHOLD = 0.85

    def __init__(self):
        pass

    def jaccard(
        self,
        original: SemanticGraph,
        candidate_text: str,
        language: str,
        candidate_graph: SemanticGraph | None = None,
        extractor=None,
    ) -> float:
        """
        Compute weighted Jaccard similarity between original and candidate.

        If candidate_graph is not provided, attempts extraction via extractor.
        Returns weighted Jaccard score between 0.0 and 1.0.
        """
        if candidate_graph is None and extractor is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        candidate_graph = pool.submit(
                            asyncio.run,
                            extractor.extract(candidate_text, language),
                        ).result(timeout=10)
                else:
                    candidate_graph = asyncio.run(
                        extractor.extract(candidate_text, language)
                    )
            except Exception as e:
                logger.warning(
                    "Failed to extract candidate graph for similarity: {}", e
                )
                return 0.0

        if candidate_graph is None:
            return 0.0

        node_score = self._node_similarity(
            original.nodes, candidate_graph.nodes
        )
        edge_score = self._edge_similarity(
            original.edges,
            candidate_graph.edges,
            original.nodes,
            candidate_graph.nodes,
        )

        # Combined: 70% node similarity + 30% edge similarity
        return 0.7 * node_score + 0.3 * edge_score

    def _node_similarity(
        self,
        orig_nodes: list[GraphNode],
        cand_nodes: list[GraphNode],
    ) -> float:
        """Weighted Jaccard over node sets."""
        if not orig_nodes and not cand_nodes:
            return 1.0
        if not orig_nodes or not cand_nodes:
            return 0.0

        matched_weight = 0.0
        total_weight = 0.0
        used_cand_ids: set[str] = set()

        for orig_node in orig_nodes:
            weight = self.NODE_TYPE_WEIGHTS.get(orig_node.unit_type, 1.0)
            total_weight += weight

            best_match = None
            best_score = 0.0
            for cand_node in cand_nodes:
                if cand_node.unit_id in used_cand_ids:
                    continue
                score = self._node_match_score(orig_node, cand_node)
                if score > best_score:
                    best_score = score
                    best_match = cand_node

            if best_match and best_score > 0.5:
                matched_weight += weight * best_score
                used_cand_ids.add(best_match.unit_id)

        # Penalise hallucinated nodes in candidate
        unmatched_cand = len(cand_nodes) - len(used_cand_ids)
        extra_penalty = unmatched_cand * 0.3

        denominator = total_weight + extra_penalty
        return matched_weight / denominator if denominator > 0 else 0.0

    def _node_match_score(self, a: GraphNode, b: GraphNode) -> float:
        """Score how well two nodes match. Returns 0.0-1.0."""
        if a.unit_type != b.unit_type:
            return 0.0

        if a.value.lower().strip() == b.value.lower().strip():
            return 1.0

        # Fuzzy match for entities
        if a.unit_type == SemanticUnitType.ENTITY:
            ratio = Levenshtein.ratio(a.value.lower(), b.value.lower())
            return ratio if ratio >= self.ENTITY_FUZZY_THRESHOLD else 0.0

        # Same type but different value = partial match
        if a.unit_type in {
            SemanticUnitType.SENTIMENT,
            SemanticUnitType.TEMPORAL,
            SemanticUnitType.MODAL,
        }:
            return 0.6

        return 0.0

    def _edge_similarity(
        self,
        orig_edges: list[tuple],
        cand_edges: list[tuple],
        orig_nodes: list[GraphNode],
        cand_nodes: list[GraphNode],
    ) -> float:
        """Compute edge overlap between graphs."""
        if not orig_edges and not cand_edges:
            return 1.0
        if not orig_edges or not cand_edges:
            return 0.0

        def normalise(
            edges: list[tuple], nodes: list[GraphNode]
        ) -> set[tuple]:
            node_map = {n.unit_id: n.unit_type.value for n in nodes}
            result = set()
            for from_id, rel, to_id in edges:
                ft = node_map.get(from_id, "?")
                tt = node_map.get(to_id, "?")
                result.add((ft, rel, tt))
            return result

        orig_set = normalise(orig_edges, orig_nodes)
        cand_set = normalise(cand_edges, cand_nodes)

        intersection = len(orig_set & cand_set)
        union = len(orig_set | cand_set)
        return intersection / union if union > 0 else 0.0
