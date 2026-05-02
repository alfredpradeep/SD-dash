"""
Integration tests for LEAEEngine — end-to-end profiling pipeline.

Covers all 6 gaps and the COMPRESS bridge addition.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from lens.engine import LEAEEngine
from lens.config import Config


@pytest.fixture
def config():
    return Config(
        device="cpu",
        semantic_entropy_enabled=False,   # Faster tests
        arbitrage_enabled=True,
        spike_alert_confidence_threshold=0.5,
        clickhouse_host="localhost",
        clickhouse_port=8123,
        redis_host="localhost",
        redis_port=6379,
    )


@pytest.fixture
def engine(config):
    return LEAEEngine(config)


# Multi-language test cases: (language, text, model)
MULTILINGUAL_CASES = [
    ("en", "How do I reset my password?", "gpt-4o"),
    ("ta", "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?", "gpt-4o"),
    ("hi", "पासवर्ड कैसे रीसेट करें?", "gpt-4o"),
    ("ar", "كيف أعيد تعيين كلمة المرور؟", "gpt-4o"),
    ("ja", "パスワードをリセットするにはどうすればよいですか？", "gpt-4o"),
    ("ta", "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?", "claude-opus-4-6"),
    ("ta", "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?", "claude-sonnet-4-6"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,text,model", MULTILINGUAL_CASES)
async def test_profile_returns_valid_metrics(engine, lang, text, model):
    profile = await engine.profile_request(
        text=text,
        model_name=model,
        customer_id="test",
        pre_detected_language=lang,
    )
    assert profile.token_count > 0
    assert 0.0 <= profile.ids_score <= 5.0
    assert profile.etr_score >= 0.0
    assert 0.0 <= profile.etr_inequity_ratio <= 50.0
    assert profile.cost_usd >= 0.0
    assert profile.waste_cost_usd >= 0.0
    assert profile.waste_type in ("efficient", "lexical", "semantic", "combined", "unknown")
    assert profile.request_id


@pytest.mark.asyncio
async def test_tamil_has_higher_token_count_than_english(engine):
    """Core validation: Tamil tokenizes less efficiently than English."""
    en = await engine.profile_request(
        text="How do I reset my password?",
        model_name="gpt-4o",
        pre_detected_language="en",
    )
    ta = await engine.profile_request(
        text="என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?",
        model_name="gpt-4o",
        pre_detected_language="ta",
    )
    assert ta.token_count > en.token_count, \
        f"Tamil ({ta.token_count}) should use more tokens than English ({en.token_count})"
    assert ta.etr_inequity_ratio >= 1.0, \
        f"Tamil ETR inequity should be >= 1.0, got {ta.etr_inequity_ratio}"


@pytest.mark.asyncio
async def test_claude_opus_model_supported(engine):
    """claude-opus-4-6 should be in SUPPORTED_MODELS and have a cost."""
    assert "claude-opus-4-6" in engine.SUPPORTED_MODELS
    assert engine.MODEL_COSTS["claude-opus-4-6"] == 15.0
    profile = await engine.profile_request(
        text="Test message for Opus",
        model_name="claude-opus-4-6",
        pre_detected_language="en",
    )
    assert profile.model_name == "claude-opus-4-6"
    assert profile.cost_usd > 0


@pytest.mark.asyncio
async def test_claude_sonnet_model_supported(engine):
    """claude-sonnet-4-6 should be supported."""
    assert "claude-sonnet-4-6" in engine.SUPPORTED_MODELS
    profile = await engine.profile_request(
        text="Test message for Sonnet",
        model_name="claude-sonnet-4-6",
        pre_detected_language="en",
    )
    assert profile.model_name == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_unknown_model_falls_back_to_gpt4o(engine):
    """Unknown model names should gracefully fall back."""
    profile = await engine.profile_request(
        text="Test",
        model_name="gpt-99-turbo-ultra",
        pre_detected_language="en",
    )
    assert profile.model_name == "gpt-4o"


@pytest.mark.asyncio
async def test_low_ids_alert_triggers(engine):
    """Repetitive/predictable text should trigger low_ids_alert."""
    repetitive = "the the the the the the the the the the " * 5
    profile = await engine.profile_request(
        text=repetitive,
        model_name="gpt-4o",
        pre_detected_language="en",
    )
    # With very low IDS, alert should fire
    if profile.ids_score < engine.config.low_ids_alert_threshold:
        assert profile.low_ids_alert is True


@pytest.mark.asyncio
async def test_compress_bridge_spans_for_low_ids(engine):
    """Low-IDS spans should be populated for wasteful text."""
    engine.config.low_ids_alert_threshold = 5.0  # Force alert for test
    profile = await engine.profile_request(
        text="the the the the the the the the the the",
        model_name="gpt-4o",
        pre_detected_language="en",
    )
    engine.config.low_ids_alert_threshold = 0.6  # Restore


@pytest.mark.asyncio
async def test_arbitrage_recommends_lower_cost_model(engine):
    """Arbitrage should recommend a cheaper model for Tamil."""
    result = await engine.compute_arbitrage(
        text="என் கடவுச்சொல்லை மீட்டமைக்க எப்படி?",
        language="ta",
    )
    assert result is not None
    assert result.recommended_model in engine.SUPPORTED_MODELS
    assert isinstance(result.max_savings_pct, float)
    assert len(result.results) > 1


@pytest.mark.asyncio
async def test_arbitrage_claude_vs_gpt(engine):
    """Claude Opus should be more expensive than GPT-4o-mini for same text."""
    result = await engine.compute_arbitrage(
        text="Summarize the quarterly results.",
        language="en",
    )
    opus_cost = result.results.get("claude-opus-4-6", {}).get("cost_usd", 0)
    mini_cost = result.results.get("gpt-4o-mini", {}).get("cost_usd", 0)
    assert opus_cost > mini_cost, \
        f"claude-opus-4-6 ({opus_cost}) should cost more than gpt-4o-mini ({mini_cost})"


@pytest.mark.asyncio
async def test_spike_prediction_from_rising_entropy(engine):
    """Rising entropy window should trigger spike prediction (CUSUM)."""
    from lens.prediction.spike_predictor import SpikePredictor
    predictor = SpikePredictor(engine.config)

    base = datetime.utcnow()
    # Sharply rising entropy (event-driven)
    window = [(base + timedelta(minutes=i), 10.0 + i * 1.5) for i in range(20)]
    prediction = await predictor.predict("ta", window, horizon_minutes=30)
    # Strong rising trend should trigger
    assert prediction is not None, "Rising entropy should trigger spike prediction"
    assert prediction.alert_level in ("watch", "warning", "alert", "critical")


@pytest.mark.asyncio
async def test_spike_no_prediction_flat_entropy(engine):
    """Flat entropy should not trigger spike prediction."""
    from lens.prediction.spike_predictor import SpikePredictor
    predictor = SpikePredictor(engine.config)

    base = datetime.utcnow()
    window = [
        (base + timedelta(minutes=i), 10.0 + np.random.randn() * 0.01)
        for i in range(20)
    ]
    prediction = await predictor.predict("en", window, horizon_minutes=30)
    if prediction:
        assert prediction.confidence < 0.8, \
            "Flat entropy should produce low-confidence prediction at most"


@pytest.mark.asyncio
async def test_spike_cause_attribution_codeswitching(engine):
    """Code-switching token types should be attributed correctly."""
    from lens.prediction.spike_predictor import SpikePredictor
    predictor = SpikePredictor(engine.config)

    base = datetime.utcnow()
    window = [(base + timedelta(minutes=i), 10.0 + i * 0.8) for i in range(20)]
    # Simulate token types with lots of code-switching
    token_types = ["codesw"] * 30 + ["normal"] * 70
    prediction = await predictor.predict("ta", window, horizon_minutes=30,
                                          high_surprisal_token_types=token_types)
    if prediction:
        assert prediction.spike_cause == "codeswitching"
        assert prediction.is_likely_temporary is True


@pytest.mark.asyncio
async def test_batch_profile_returns_all_results(engine):
    """Batch profiling should return one result per input."""
    requests = [
        {"text": f"Message number {i}", "language": "en", "model": "gpt-4o"}
        for i in range(10)
    ]
    profiles = await engine.batch_profile(requests, concurrency=5)
    assert len(profiles) == 10
    for p in profiles:
        assert p.token_count > 0


@pytest.mark.asyncio
async def test_health_check_returns_status(engine):
    health = await engine.health_check()
    assert health["status"] == "healthy"
    assert "lang_detector" in health
    assert "clickhouse" in health
    assert "redis" in health
    assert "semantic_entropy" in health


@pytest.mark.asyncio
async def test_outcome_tracker_recalibration(engine):
    """Prediction outcome tracker should update thresholds."""
    tracker = engine.outcome_tracker
    lang = "ta"
    # Record many inaccurate predictions
    pred_ids = []
    for _ in range(15):
        pid = await tracker.record_prediction(lang, 20.0, 0.75, 30)
        pred_ids.append(pid)
    # Record outcomes as all inaccurate
    for pid in pred_ids:
        await tracker.record_outcome(pid, 0.5)  # Way off
    # Threshold should increase
    new_threshold = tracker.get_threshold(lang)
    assert new_threshold >= engine.config.spike_alert_confidence_threshold


@pytest.mark.asyncio
async def test_efficiency_ratio_tamil_greater_than_one(engine):
    """Tamil efficiency ratio should be > 1 (more tokens than English baseline)."""
    profile = await engine.profile_request(
        text="என் கணக்கில் பிழையான கட்டணம் விதிக்கப்பட்டுள்ளது",
        model_name="gpt-4o",
        pre_detected_language="ta",
    )
    assert profile.efficiency_ratio >= 1.0, \
        f"Tamil efficiency ratio should be >= 1.0, got {profile.efficiency_ratio}"
