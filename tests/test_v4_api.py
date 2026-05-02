"""Tests for COMPRESS v4 strict API endpoints."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from compress.api import v4_router
from compress.api.v4_router import v4_router as router


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router)
    v4_router._pipeline = None
    yield test_app
    v4_router._pipeline = None


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


def _make_strict_test_pipeline():
    pipeline = v4_router.get_pipeline()
    pipeline.capabilities.require = lambda *caps: None
    pipeline.semantics.require_labse = lambda: None
    pipeline.semantics.similarity = lambda a, b: 0.96
    pipeline.cache.check = AsyncMock(return_value={
        "engine": "semantic_cache",
        "status": "ready",
        "cache_hit": False,
        "cache_key": "test",
        "similarity_score": None,
        "hit_rate": 0,
        "total_lookups": 1,
        "api_calls_saved": 0,
        "latency_ms": 0.1,
    })
    pipeline.verification._graph_overlap = AsyncMock(return_value=(0.95, "mock graph overlap"))
    return pipeline


@pytest.mark.asyncio
async def test_v4_languages_returns_canonical_20(client):
    resp = await client.get("/v4/languages")
    assert resp.status_code == 200
    data = resp.json()
    codes = {lang["code"] for lang in data["languages"]}
    assert data["total"] == 20
    assert codes == {
        "en", "ta", "hi", "ar", "ja", "zh", "ko", "bn", "ur", "te",
        "ml", "pa", "gu", "mr", "pt", "es", "fr", "de", "id", "ms",
    }


@pytest.mark.asyncio
async def test_v4_capabilities_explain_strict_status(client):
    resp = await client.get("/v4/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strict_mode"] is True
    assert "llm_provider" in data["capabilities"]
    assert "labse_embeddings" in data["capabilities"]


@pytest.mark.asyncio
async def test_v4_compress_without_provider_returns_503_not_truncation(client):
    pipeline = v4_router.get_pipeline()
    pipeline.compressor.provider._provider = None

    resp = await client.post("/v4/compress", json={
        "text": "என் கணக்கில் தவறான கட்டணம் உள்ளது. தயவு செய்து முழு விவரங்களையும் சரிபார்க்கவும்.",
        "language": "ta",
    })

    assert resp.status_code == 503
    body = resp.json()["detail"]
    assert body["capability"] in {"llm_provider", "labse_embeddings", "semantic_graph", "ner_entity_extraction"}
    assert "simulate" in body.get("message", "") or body["reason"]


@pytest.mark.asyncio
async def test_v4_compress_applies_only_verified_shorter_provider_output(client):
    pipeline = _make_strict_test_pipeline()
    pipeline.compressor.provider._provider = "test-provider"
    pipeline.compressor.provider.generate = AsyncMock(return_value="incorrect account charge; please investigate")

    resp = await client.post("/v4/compress", json={
        "text": "The customer says there is an incorrect account charge and asks the support team to investigate it.",
        "language": "en",
        "session_id": "strict-test",
        "exchange_number": 2,
    })

    assert resp.status_code == 200
    data = resp.json()
    assert data["compression"]["provider"] == "test-provider"
    assert data["compression"]["fallback_used"] is False
    assert data["compression"]["compression_applied"] is True
    assert data["compression"]["compressed_text"] == "incorrect account charge; please investigate"
    assert data["engines"]["verification"]["verdict"] == "PASS"


@pytest.mark.asyncio
async def test_v4_compress_failed_verification_returns_original(client):
    pipeline = _make_strict_test_pipeline()
    pipeline.compressor.provider._provider = "test-provider"
    pipeline.compressor.provider.generate = AsyncMock(return_value="account charge")
    pipeline.verification.process = AsyncMock(return_value={
        "engine": "five_stage_verification",
        "status": "ready",
        "verdict": "KILL",
        "action": "use_original",
        "gates": [{"gate": 1, "name": "mock", "passed": False, "skipped": False}],
        "gates_passed": 0,
        "gates_total": 5,
        "frontier_judge_invoked": False,
        "latency_ms": 0.1,
    })
    source = "The account has an incorrect charge of 25 dollars and must not be closed."

    resp = await client.post("/v4/compress", json={"text": source, "language": "en"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["compression"]["compression_applied"] is False
    assert data["compression"]["compressed_text"] == source
    assert "Verification failed" in data["compression"]["failure_reason"]


@pytest.mark.asyncio
async def test_v4_round_trip_requires_successful_compression_and_decompression(client):
    pipeline = _make_strict_test_pipeline()
    pipeline.compressor.provider._provider = "test-provider"
    pipeline.compressor.provider.generate = AsyncMock(return_value="incorrect account charge; investigate")
    pipeline.decompressor._translator._providers = [{"name": "mock"}]
    pipeline.decompressor.decompress = AsyncMock(return_value={
        "decompressed_text": "The account has an incorrect charge. Please investigate.",
        "target_language": "en",
        "language_name": "English",
        "method": "mock",
        "provider": "mock",
        "candidates": [],
        "quality": {"labse_to_original": 0.95},
        "tokens": {
            "original": 14,
            "compressed_english": 5,
            "decompressed": 9,
            "net_reduction_vs_original": 0.35,
        },
        "processing_ms": 1.2,
    })

    resp = await client.post("/v4/round-trip", json={
        "text": "The account has an incorrect charge and the support team should investigate.",
        "language": "en",
        "session_id": "rt-test",
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["round_trip"]["provider"] == "mock"
    assert body["round_trip"]["semantic_similarity"] == 0.95


@pytest.mark.asyncio
async def test_v4_interview_run_uses_llm_and_reports_savings(client):
    pipeline = _make_strict_test_pipeline()
    pipeline.compressor.provider._provider = "test-provider"
    pipeline.decompressor._translator._providers = [{"name": "mock"}]

    async def fake_generate(system_prompt, user_prompt, **kwargs):
        if "interviewer" in system_prompt.lower():
            return "Can you describe a production incident you debugged?"
        if "candidate" in system_prompt.lower():
            return "I debugged a latency spike by tracing API calls, isolating a slow database query, and adding an index."
        return "debugged latency spike; slow database query; added index"

    pipeline.compressor.provider.generate = AsyncMock(side_effect=fake_generate)

    resp = await client.post("/v4/interview/run", json={
        "role": "Backend Engineer",
        "seniority": "Senior",
        "topic": "production debugging",
        "language": "en",
        "turn_count": 1,
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "test-provider"
    assert body["summary"]["turns"] == 1
    assert body["turns"][0]["interviewer_message"]
    assert body["turns"][0]["candidate_message"]
    assert "compressed_context" in body["turns"][0]

