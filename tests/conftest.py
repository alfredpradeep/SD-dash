"""
Shared pytest fixtures for COMPRESS test suite.

Provides:
  - config: test-mode Config instance
  - mock_engine: SCLEngine with mocked sub-components
  - sample_inputs: loaded from fixtures/sample_inputs.json
  - expected_outputs: loaded from fixtures/expected_outputs.json
  - async_client: httpx AsyncClient for API integration tests
"""

import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from compress.config import Config
from compress.engine import SCLEngine
from compress.lattice.structures import (
    SemanticGraph,
    GraphNode,
    SemanticUnitType,
    CompressionCandidate,
    CompressionResult,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def config():
    """Test configuration with sensible defaults."""
    return Config(
        compress_model_path="/tmp/test-models",
        amr_model_path="/tmp/test-models/amr",
        amr_adapter_dir="/tmp/test-models/amr-adapters",
        device="cpu",
        beam_width=4,
        max_candidates=3,
        redis_host="localhost",
        redis_port=6379,
        redis_ttl_seconds=60,
        api_host="127.0.0.1",
        api_port=8000,
        metrics_port=9101,
    )


@pytest.fixture
def sample_inputs():
    """Load sample inputs from fixtures."""
    with open(FIXTURES_DIR / "sample_inputs.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def expected_outputs():
    """Load expected outputs from fixtures."""
    with open(FIXTURES_DIR / "expected_outputs.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_graph():
    """Create a sample SemanticGraph for testing."""
    nodes = [
        GraphNode(
            id="n1",
            node_type=SemanticUnitType.ENTITY,
            value="machine learning model",
            confidence=0.95,
            intensity=0.8,
            specificity=0.9,
            certainty=0.92,
            source="amr",
        ),
        GraphNode(
            id="n2",
            node_type=SemanticUnitType.EVENT,
            value="demonstrate",
            confidence=0.88,
            intensity=0.7,
            specificity=0.6,
            certainty=0.85,
            source="amr",
        ),
        GraphNode(
            id="n3",
            node_type=SemanticUnitType.ATTRIBUTE,
            value="exceptional",
            confidence=0.82,
            intensity=0.9,
            specificity=0.5,
            certainty=0.80,
            source="nli",
        ),
        GraphNode(
            id="n4",
            node_type=SemanticUnitType.ENTITY,
            value="natural language processing",
            confidence=0.91,
            intensity=0.7,
            specificity=0.95,
            certainty=0.90,
            source="ner",
        ),
    ]
    edges = [
        ("n1", "n2", "agent"),
        ("n2", "n3", "manner"),
        ("n2", "n4", "domain"),
    ]
    return SemanticGraph(
        nodes=nodes,
        edges=edges,
        source_language="en",
        amr_penman="(d / demonstrate :ARG0 (m / model) :manner (e / exceptional))",
        extraction_confidence=0.89,
        amr_confidence=0.85,
        component_scores={"amr": 0.85, "nli": 0.82, "ner": 0.91},
    )


@pytest.fixture
def sample_candidate():
    """Create a sample CompressionCandidate."""
    return CompressionCandidate(
        text="ML model showed outstanding NLP performance across benchmarks.",
        token_count=12,
        semantic_similarity=0.94,
        graph_jaccard=0.91,
        stes_score=2.45,
    )


@pytest.fixture
def sample_result():
    """Create a sample CompressionResult."""
    return CompressionResult(
        original_text="The machine learning model demonstrated exceptional performance in natural language processing tasks.",
        compressed_text="ML model showed outstanding NLP performance across benchmarks.",
        source_language="en",
        target_tokenizer="gpt-4o",
        original_token_count=18,
        compressed_token_count=12,
        reduction_ratio=0.333,
        semantic_similarity=0.94,
        graph_jaccard=0.91,
        stes_score=2.45,
        compression_applied=True,
        rejection_reason=None,
        processing_ms=145.3,
        candidates_evaluated=5,
    )


@pytest.fixture
def mock_extractor():
    """Mock LatticeExtractor."""
    extractor = AsyncMock()
    extractor.health = AsyncMock(return_value="healthy")
    return extractor


@pytest.fixture
def mock_searcher():
    """Mock BeamSearcher."""
    searcher = AsyncMock()
    searcher.health = AsyncMock(return_value="healthy")
    searcher.count_tokens = MagicMock(return_value=18)
    return searcher


@pytest.fixture
def mock_gate():
    """Mock VerificationGate."""
    gate = AsyncMock()
    gate.health = AsyncMock(return_value="healthy")
    return gate


@pytest.fixture
def mock_cache():
    """Mock CompressionCache."""
    cache = AsyncMock()
    cache.health = AsyncMock(return_value="healthy")
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.make_key = MagicMock(return_value="test_cache_key_sha256")
    return cache


@pytest.fixture
def mock_metrics():
    """Mock MetricsStore."""
    metrics = MagicMock()
    metrics.record_compression = MagicMock()
    metrics.record_cache_hit = MagicMock()
    return metrics


@pytest.fixture
def mock_engine(
    config, mock_extractor, mock_searcher, mock_gate, mock_cache, mock_metrics,
    sample_graph, sample_candidate,
):
    """
    Fully mocked SCLEngine for unit tests.

    Wires up sub-components so compress() runs through the full pipeline
    without loading actual ML models.
    """
    with patch.object(SCLEngine, "__init__", lambda self, cfg: None):
        engine = SCLEngine.__new__(SCLEngine)
        engine.config = config
        engine.extractor = mock_extractor
        engine.searcher = mock_searcher
        engine.gate = mock_gate
        engine.cache = mock_cache
        engine.metrics = mock_metrics

        # Wire up the pipeline
        mock_extractor.extract = AsyncMock(return_value=sample_graph)
        mock_searcher.search = AsyncMock(return_value=[sample_candidate])
        mock_gate.verify = AsyncMock(return_value=(True, None))

        return engine


@pytest_asyncio.fixture
async def async_client(mock_engine):
    """httpx AsyncClient pointed at the test app."""
    import compress.main as main_module

    main_module.engine_instance = mock_engine

    from compress.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
