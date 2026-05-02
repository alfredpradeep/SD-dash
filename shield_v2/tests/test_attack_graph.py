"""Tests for Pillar 2 Part 2: AttackGraph + AttackRouter (Dijkstra)."""

from __future__ import annotations

import pytest

from shield_v2.core.attack_graph import (
    AttackGraph,
    AttackRouter,
    LanguageNode,
    FAMILY_DISTANCE,
    family_distance,
)


class TestFamilyDistance:
    def test_same_family_small_distance(self):
        assert family_distance("en", "de") < family_distance("en", "zh")
        assert family_distance("hi", "bn") <= family_distance("hi", "en")

    def test_unknown_language_default(self):
        # Falls back to a defined pair or default
        d = family_distance("en", "xx_unknown")
        assert d > 0

    def test_symmetric_lookup_works(self):
        # japonic/koreanic is declared one way only; helper symmetrises.
        a = family_distance("ja", "ko")
        b = family_distance("ko", "ja")
        assert a == b


class TestAttackGraphBuild:
    def test_standard_build_creates_english_seed(self, attack_graph):
        seed = LanguageNode("en", "native", "formal", 0)
        assert seed.key() in attack_graph.nodes

    def test_standard_build_connects_seed_to_targets(self, attack_graph):
        seed = LanguageNode("en", "native", "formal", 0)
        neighbors = attack_graph.neighbors(seed)
        langs_reached = {e.dst.lang for e in neighbors}
        assert {"hi", "ta", "ar", "zh"} <= langs_reached

    def test_edge_costs_non_negative(self, attack_graph):
        for edges in attack_graph.edges.values():
            for e in edges:
                assert e.cost >= 0

    def test_update_bypass_stats_lowers_cost(self):
        g = AttackGraph()
        src = LanguageNode("en", "native", "formal", 0)
        dst = LanguageNode("hi", "native", "formal", 0)
        g.update_bypass_stat("en", "hi", rate=0.8)
        g.add_edge(src, dst, transform="test", base_cost=1.0)
        edge = g.edges[src.key()][0]
        # With high empirical bypass rate, fused cost should be noticeably less
        assert edge.cost < 1.0 + family_distance("en", "hi")


class TestAttackRouter:
    def test_shortest_path_exists(self, attack_graph):
        router = AttackRouter(attack_graph)
        src = LanguageNode("en", "native", "formal", 0)
        dst = LanguageNode("hi", "native", "formal", 0)
        path = router.shortest_path(src, dst)
        assert path is not None
        assert path.nodes[0].key() == src.key()
        assert path.nodes[-1].key() == dst.key()
        assert path.total_cost > 0

    def test_shortest_path_missing_target(self, attack_graph):
        router = AttackRouter(attack_graph)
        src = LanguageNode("en", "native", "formal", 0)
        dst = LanguageNode("xx_unreachable", "native", "formal", 0)
        assert router.shortest_path(src, dst) is None

    def test_top_k_targets(self, attack_graph):
        router = AttackRouter(attack_graph)
        src = LanguageNode("en", "native", "formal", 0)
        tops = router.top_k_targets(src, k=5)
        assert 1 <= len(tops) <= 6
        # Sorted ascending by cost
        costs = [c for _, c in tops]
        assert costs == sorted(costs)

    def test_describe_path(self, attack_graph):
        router = AttackRouter(attack_graph)
        src = LanguageNode("en", "native", "formal", 0)
        dst = LanguageNode("ta", "native", "formal", 0)
        path = router.shortest_path(src, dst)
        assert path is not None
        desc = path.describe()
        assert "en" in desc and "ta" in desc
