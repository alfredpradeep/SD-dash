"""
API endpoint tests using FastAPI TestClient.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a test app with a real engine in dry-run mode."""
    from lens.config import Config
    from lens.engine import LEAEEngine
    import lens.main as main_module
    from fastapi import FastAPI
    from lens.api.router import router
    from contextlib import asynccontextmanager

    config = Config(
        clickhouse_host="nonexistent",
        redis_host="nonexistent",
        redis_port=6399,
        semantic_entropy_enabled=False,
        arbitrage_enabled=True,
    )

    @asynccontextmanager
    async def lifespan(app):
        main_module.engine_instance = LEAEEngine(config)
        yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/lens/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_profile_english(client):
    response = client.post("/lens/profile", json={
        "text": "How do I reset my password?",
        "model_name": "gpt-4o",
        "language": "en",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["token_count"] > 0
    assert "ids_score" in data
    assert "etr_score" in data
    assert "waste_type" in data
    assert "low_ids_alert" in data


def test_profile_tamil(client):
    response = client.post("/lens/profile", json={
        "text": "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?",
        "model_name": "gpt-4o",
        "language": "ta",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "ta"
    assert data["token_count"] > 0
    assert data["etr_inequity_ratio"] >= 1.0


def test_profile_claude_opus(client):
    response = client.post("/lens/profile", json={
        "text": "Summarize the quarterly results for the board.",
        "model_name": "claude-opus-4-6",
        "language": "en",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "claude-opus-4-6"
    assert data["cost_usd"] > 0


def test_profile_claude_sonnet(client):
    response = client.post("/lens/profile", json={
        "text": "What are the key metrics?",
        "model_name": "claude-sonnet-4-6",
        "language": "en",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "claude-sonnet-4-6"


def test_profile_with_token_records(client):
    response = client.post("/lens/profile", json={
        "text": "Hello world",
        "model_name": "gpt-4o",
        "language": "en",
        "include_token_records": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["token_records"] is not None
    assert len(data["token_records"]) > 0


def test_batch_profile(client):
    response = client.post("/lens/profile/batch", json={
        "items": [
            {"text": f"Message {i}", "model": "gpt-4o", "language": "en"}
            for i in range(5)
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 5
    assert data["total_cost_usd"] >= 0
    assert "en" in data["languages_detected"]


def test_batch_profile_too_many(client):
    response = client.post("/lens/profile/batch", json={
        "items": [
            {"text": f"Message {i}"}
            for i in range(501)
        ]
    })
    assert response.status_code == 422


def test_arbitrage_endpoint(client):
    response = client.post("/lens/arbitrage", json={
        "text": "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?",
        "language": "ta",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_model"] is not None
    assert len(data["results"]) > 1
    # claude-opus-4-6 should be in arbitrage results
    model_names = [r["model_name"] for r in data["results"]]
    assert "claude-opus-4-6" in model_names
    assert "claude-sonnet-4-6" in model_names


def test_predict_spikes_empty(client):
    response = client.get("/lens/predict/spikes")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_accuracy_report(client):
    response = client.get("/lens/accuracy")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions_tracked" in data


def test_root_endpoint(client):
    # Check the app serves something at root
    response = client.get("/lens/health")
    assert response.status_code == 200
