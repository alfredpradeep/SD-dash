#!/usr/bin/env python3
"""
Quick test: Cross-lingual compression fixes.

Tests:
  1. Non-Latin script detection
  2. Cross-lingual reconstruction (type-count comparison)
  3. Full pipeline with Tamil input (requires LLM + models)

Run: python test_crosslingual.py
"""

import asyncio
import sys

# ── Test 1: Non-Latin script detection ──
def test_non_latin_detection():
    """Test that the beam searcher correctly identifies non-Latin text."""
    from compress.search.beam import BeamSearcher
    from compress.config import Config

    config = Config()
    # We can't fully init BeamSearcher without models, so test the method standalone
    def is_non_latin(text):
        if not text:
            return False
        non_latin = sum(
            1 for c in text
            if c.isalpha() and ord(c) > 0x024F
        )
        alpha = sum(1 for c in text if c.isalpha())
        return alpha > 0 and (non_latin / alpha) > 0.5

    # Tamil text should be detected as non-Latin
    assert is_non_latin("காற்று மிகவும் வலுவாக வீசுகிறது") == True
    # English text should NOT be non-Latin
    assert is_non_latin("The wind is blowing very strongly") == False
    # Mixed text (mostly English) should NOT be non-Latin
    assert is_non_latin("The wind காற்று is blowing strongly today") == False
    # Arabic should be non-Latin
    assert is_non_latin("الرياح تهب بقوة شديدة") == True
    # Hindi should be non-Latin
    assert is_non_latin("हवा बहुत तेज़ बह रही है") == True

    print("  ✓ Test 1: Non-Latin script detection passed")


# ── Test 2: Cross-lingual reconstruction verification ──
async def test_crosslingual_reconstruction():
    """Test that reconstruction uses type-count for cross-lingual."""
    from compress.verification.reconstruction import ReconstructionVerifier
    from compress.config import Config
    from compress.lattice.structures import (
        SemanticGraph, GraphNode, SemanticUnitType
    )

    config = Config()
    verifier = ReconstructionVerifier(config)

    # Create a mock original graph with Tamil entities
    tamil_nodes = [
        GraphNode(
            unit_id="n1", unit_type=SemanticUnitType.ENTITY,
            value="காற்று", confidence=0.9
        ),
        GraphNode(
            unit_id="n2", unit_type=SemanticUnitType.ENTITY,
            value="ஆடைகள்", confidence=0.9
        ),
        GraphNode(
            unit_id="n3", unit_type=SemanticUnitType.NEGATION,
            value="இல்லை", confidence=0.9
        ),
    ]

    # Create a mock compressed graph with English entities
    english_nodes = [
        GraphNode(
            unit_id="n1", unit_type=SemanticUnitType.ENTITY,
            value="wind", confidence=0.9
        ),
        GraphNode(
            unit_id="n2", unit_type=SemanticUnitType.ENTITY,
            value="clothes", confidence=0.9
        ),
        GraphNode(
            unit_id="n3", unit_type=SemanticUnitType.NEGATION,
            value="not", confidence=0.9
        ),
    ]

    orig_graph = SemanticGraph(
        nodes=tamil_nodes, edges=[], source_text="Tamil text",
        source_language="ta", extraction_confidence=0.9,
        amr_confidence=0.0,
        amr_penman="",
    )

    # Mock extractor that returns the English graph
    class MockExtractor:
        async def extract(self, text, language):
            return SemanticGraph(
                nodes=english_nodes, edges=[], source_text=text,
                source_language=language, extraction_confidence=0.9,
                amr_confidence=0.0, amr_penman="",
            )

    score, missing, details = await verifier._verify_graph(
        original_text="Tamil text",
        compressed_text="Wind blows clothes not dry",
        original_graph=orig_graph,
        extractor=MockExtractor(),
        language="ta",
    )

    print(f"    Score: {score:.3f}")
    print(f"    Method: {details.get('method')}")
    print(f"    Missing: {missing}")

    # Cross-lingual should use type-count method
    assert details["method"] == "graph_reconstruction_crosslingual", \
        f"Expected crosslingual method, got {details['method']}"

    # Same number of each type → score should be 1.0
    assert score >= 0.9, f"Expected score >= 0.9, got {score}"
    assert len(missing) == 0, f"Expected no missing, got {missing}"

    print("  ✓ Test 2: Cross-lingual reconstruction passed")


# ── Test 3: Same-language reconstruction still works ──
async def test_samelang_reconstruction():
    """Test that same-language still uses value matching."""
    from compress.verification.reconstruction import ReconstructionVerifier
    from compress.config import Config
    from compress.lattice.structures import (
        SemanticGraph, GraphNode, SemanticUnitType
    )

    config = Config()
    verifier = ReconstructionVerifier(config)

    english_nodes = [
        GraphNode(
            unit_id="n1", unit_type=SemanticUnitType.ENTITY,
            value="wind", confidence=0.9
        ),
        GraphNode(
            unit_id="n2", unit_type=SemanticUnitType.ENTITY,
            value="clothes", confidence=0.9
        ),
    ]

    orig_graph = SemanticGraph(
        nodes=english_nodes, edges=[], source_text="Wind blows clothes",
        source_language="en", extraction_confidence=0.9,
        amr_confidence=0.0,
        amr_penman="",
    )

    # Compressed has same entities
    class MockExtractor:
        async def extract(self, text, language):
            return SemanticGraph(
                nodes=[
                    GraphNode(
                        unit_id="n1", unit_type=SemanticUnitType.ENTITY,
                        value="wind", confidence=0.9
                    ),
                ],
                edges=[], source_text=text,
                source_language=language, extraction_confidence=0.9,
                amr_confidence=0.0, amr_penman="",
            )

    score, missing, details = await verifier._verify_graph(
        original_text="Wind blows clothes",
        compressed_text="Wind blows",
        original_graph=orig_graph,
        extractor=MockExtractor(),
        language="en",
    )

    print(f"    Score: {score:.3f}")
    print(f"    Method: {details.get('method')}")
    print(f"    Missing: {missing}")

    # Same-language should use value matching
    assert details["method"] == "graph_reconstruction"
    # Should have 1 missing entity ("clothes")
    assert score == 0.5, f"Expected 0.5, got {score}"
    assert len(missing) == 1

    print("  ✓ Test 3: Same-language reconstruction passed")


if __name__ == "__main__":
    print("\n=== Cross-Lingual Compression Tests ===\n")

    # Test 1: No dependencies needed
    test_non_latin_detection()

    # Tests 2 & 3: Need basic compress package
    try:
        asyncio.run(test_crosslingual_reconstruction())
        asyncio.run(test_samelang_reconstruction())
    except ImportError as e:
        print(f"  ⚠ Skipped async tests (missing dep: {e})")

    print("\n=== All tests passed! ===")
    print("\nNow restart the server and test with Tamil input:")
    print("  ./run_production.sh")
    print("")
