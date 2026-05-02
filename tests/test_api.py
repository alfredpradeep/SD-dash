"""
Tests for the COMPRESS FastAPI API layer.

Covers:
  - POST /compress/ — successful compression
  - POST /compress/ — validation errors
  - POST /compress/batch — batch processing
  - GET  /compress/health — health endpoint
  - GET  /compress/supported — supported languages/tokenizers
  - GET  / — UI serving
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient


@pytest.mark.asyncio
class TestCompressEndpoint:
    """Tests for POST /compress/."""

    async def test_compress_success(self, async_client):
        """Valid request returns 200 with compression result."""
        resp = await async_client.post("/compress/", json={
            "text": "The machine learning model demonstrated exceptional results.",
            "language": "en",
            "target_tokenizer": "gpt-4o",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "original_text" in data
        assert "compressed_text" in data
        assert "compression_applied" in data
        assert "semantic_similarity" in data
        assert "graph_jaccard" in data
        assert "reduction_ratio" in data
        assert "processing_ms" in data

    async def test_compress_with_all_params(self, async_client):
        """Request with all optional params returns 200."""
        resp = await async_client.post("/compress/", json={
            "text": "Full params test.",
            "language": "ta",
            "target_tokenizer": "claude-3-5-sonnet",
            "customer_id": "test-customer",
            "min_reduction_threshold": 0.20,
            "semantic_threshold": 0.92,
            "graph_threshold": 0.90,
        })
        assert resp.status_code == 200

    async def test_compress_empty_text_422(self, async_client):
        """Empty text returns 422 validation error."""
        resp = await async_client.post("/compress/", json={
            "text": "",
            "language": "en",
        })
        assert resp.status_code == 422

    async def test_compress_whitespace_only_422(self, async_client):
        """Whitespace-only text returns 422."""
        resp = await async_client.post("/compress/", json={
            "text": "   \n\t  ",
            "language": "en",
        })
        assert resp.status_code == 422

    async def test_compress_missing_language_422(self, async_client):
        """Missing language field returns 422."""
        resp = await async_client.post("/compress/", json={
            "text": "No language specified.",
        })
        assert resp.status_code == 422

    async def test_compress_unsupported_language_422(self, async_client):
        """Unsupported language returns 422."""
        resp = await async_client.post("/compress/", json={
            "text": "Unsupported lang test.",
            "language": "xx",
        })
        assert resp.status_code == 422


@pytest.mark.asyncio
class TestBatchEndpoint:
    """Tests for POST /compress/batch."""

    async def test_batch_success(self, async_client):
        """Valid batch returns 200 with results list."""
        resp = await async_client.post("/compress/batch", json={
            "items": [
                {"text": "First text.", "language": "en", "tokenizer": "gpt-4o"},
                {"text": "Second text.", "language": "en", "tokenizer": "gpt-4o"},
            ],
            "customer_id": "batch-test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2

    async def test_batch_too_many_items_422(self, async_client):
        """More than 100 items returns 422."""
        items = [
            {"text": f"Item {i}", "language": "en", "tokenizer": "gpt-4o"}
            for i in range(101)
        ]
        resp = await async_client.post("/compress/batch", json={
            "items": items,
        })
        assert resp.status_code == 422


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for GET /compress/health."""

    async def test_health_returns_200(self, async_client):
        """Health endpoint returns 200 with component statuses."""
        resp = await async_client.get("/compress/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "extractor" in data
        assert "searcher" in data
        assert "gate" in data
        assert "cache" in data


@pytest.mark.asyncio
class TestSupportedEndpoint:
    """Tests for GET /compress/supported."""

    async def test_supported_returns_200(self, async_client):
        """Supported endpoint returns languages and tokenizers."""
        resp = await async_client.get("/compress/supported")
        assert resp.status_code == 200
        data = resp.json()
        assert "languages" in data
        assert "tokenizers" in data
        assert len(data["languages"]) == 20
        assert len(data["tokenizers"]) == 10

    async def test_supported_languages_sorted(self, async_client):
        """Languages list is sorted alphabetically."""
        resp = await async_client.get("/compress/supported")
        data = resp.json()
        assert data["languages"] == sorted(data["languages"])

    async def test_supported_tokenizers_sorted(self, async_client):
        """Tokenizers list is sorted alphabetically."""
        resp = await async_client.get("/compress/supported")
        data = resp.json()
        assert data["tokenizers"] == sorted(data["tokenizers"])


@pytest.mark.asyncio
class TestUIServing:
    """Tests for GET / (UI)."""

    async def test_root_returns_html(self, async_client):
        """Root endpoint serves HTML dashboard."""
        resp = await async_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Haiku" in resp.text
