"""
Tests for storage subsystem: RedisCache, ClickHouseStore (dry-run mode).
"""

import pytest
import asyncio
from lens.config import Config
from lens.storage.redis_cache import RedisCache
from lens.storage.clickhouse_store import ClickHouseStore


@pytest.fixture
def config():
    return Config(
        redis_host="nonexistent",  # Forces in-memory fallback
        redis_port=6399,
        clickhouse_host="nonexistent",  # Forces dry-run
    )


@pytest.fixture
def cache(config):
    return RedisCache(config)


@pytest.fixture
def ch_store(config):
    return ClickHouseStore(config)


# ---- RedisCache ----

@pytest.mark.asyncio
async def test_cache_profile_roundtrip(cache):
    data = {"ids_score": 1.2, "etr": 3.5, "cost": 0.001}
    await cache.set_profile("test text", "gpt-4o", "en", data)
    result = await cache.get_profile("test text", "gpt-4o", "en")
    assert result == data


@pytest.mark.asyncio
async def test_cache_miss_returns_none(cache):
    result = await cache.get_profile("missing text xyz", "gpt-4o", "en")
    assert result is None


@pytest.mark.asyncio
async def test_cache_language_roundtrip(cache):
    await cache.set_language("hello world", "en", 0.99)
    result = await cache.get_language("hello world")
    assert result is not None
    lang, conf = result
    assert lang == "en"
    assert abs(conf - 0.99) < 0.01


@pytest.mark.asyncio
async def test_cache_arbitrage_roundtrip(cache):
    data = {"results": {"gpt-4o": {"tokens": 10, "cost_usd": 0.0001}}}
    await cache.set_arbitrage("test", "ta", data)
    result = await cache.get_arbitrage("test", "ta")
    assert result == data


@pytest.mark.asyncio
async def test_cache_health_returns_string(cache):
    health = await cache.health()
    assert isinstance(health, str)


# ---- ClickHouseStore (dry-run) ----

@pytest.mark.asyncio
async def test_ch_store_is_dry_run(ch_store):
    assert ch_store._dry_run is True


@pytest.mark.asyncio
async def test_ch_store_insert_profile_noop(ch_store):
    """Dry-run insert should not raise."""
    from lens.entropy.structures import EntropyProfile
    from datetime import datetime
    profile = EntropyProfile(
        text="test",
        language="ta",
        model_name="gpt-4o",
        token_count=10,
        english_baseline_tokens=3,
        efficiency_ratio=3.3,
        total_entropy_bits=45.0,
        mean_entropy_per_token=4.5,
        ids_score=0.9,
        etr_score=4.5,
        etr_english_baseline=18.2,
        etr_inequity_ratio=4.0,
        cost_usd=0.0001,
        waste_cost_usd=0.00007,
        request_id="test-123",
        customer_id="test",
    )
    # Should not raise even though ClickHouse isn't available
    await ch_store.insert_profile(profile)


@pytest.mark.asyncio
async def test_ch_store_health_dry_run(ch_store):
    health = await ch_store.health()
    assert "dry-run" in health.lower() or "unhealthy" in health.lower()
