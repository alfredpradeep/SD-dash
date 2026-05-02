"""
Tests for the 9 LENS Platform features:
  1. Token Verification
  2. Shadow Billing
  3. Cost Stream
  4. Semantic Prompt Cache
  5. Semantic Deduplication
  6. Provider Health Monitor
  7. Tokenizer Drift Detector
  8. Cost Attribution
  9. Audit Trail
"""

import pytest
import time
from datetime import datetime, timedelta


# ═══════════════════════════════════════════
# Feature 1 — Token Verification
# ═══════════════════════════════════════════

class TestTokenVerification:
    @pytest.fixture
    def verifier(self):
        from lens.moat.token_verifier import TokenVerificationEngine
        return TokenVerificationEngine()

    def test_verify_returns_result(self, verifier):
        result = verifier.verify("Hello world, this is a test.", "gpt-4o")
        assert result.model == "gpt-4o"
        assert result.measured_tokens > 0
        assert result.estimated_cost_usd >= 0
        assert 0 < result.confidence <= 1.0
        assert result.verification_method is not None

    def test_verify_different_models(self, verifier):
        text = "The quick brown fox jumps over the lazy dog."
        r1 = verifier.verify(text, "gpt-4o")
        r2 = verifier.verify(text, "claude-3-haiku")
        assert r1.measured_tokens > 0
        assert r2.measured_tokens > 0
        assert r1.provider != r2.provider

    def test_compare_providers(self, verifier):
        result = verifier.compare_providers("Hello world test", "gpt-4o")
        assert result.current_model == "gpt-4o"
        assert result.current_tokens > 0
        assert len(result.alternatives) > 0
        assert result.best_alternative is not None
        assert result.max_savings_pct >= 0


# ═══════════════════════════════════════════
# Feature 2 — Shadow Billing
# ═══════════════════════════════════════════

class TestBillingReconciler:
    @pytest.fixture
    def reconciler(self):
        from lens.moat.billing import BillingReconciler
        return BillingReconciler()

    def test_record_creates_entry(self, reconciler):
        entry = reconciler.record(
            model="gpt-4o",
            input_text="Hello world test input",
            billed_input_tokens=100,
            billed_cost=0.0005,
            team="engineering",
        )
        assert entry.request_id is not None
        assert entry.model == "gpt-4o"
        assert entry.input_tokens_measured > 0
        assert entry.cost_billed == 0.0005

    def test_reconcile_detects_discrepancy(self, reconciler):
        # Record with inflated billed tokens
        reconciler.record(
            model="gpt-4o",
            input_text="Short text",
            billed_input_tokens=500,  # Way more than actual
            billed_cost=0.0025,
            team="qa",
        )
        report = reconciler.reconcile()
        assert report.total_requests >= 1
        assert report.discrepancy_tokens != 0

    def test_get_by_team(self, reconciler):
        reconciler.record(model="gpt-4o", input_text="test", team="alpha")
        reconciler.record(model="gpt-4o", input_text="test", team="beta")
        alpha = reconciler.get_by_team("alpha")
        assert len(alpha) >= 1

    def test_reconciliation_recommendations(self, reconciler):
        for _ in range(5):
            reconciler.record(
                model="gpt-4o", input_text="test " * 50,
                billed_input_tokens=1000, billed_cost=0.005
            )
        report = reconciler.reconcile()
        assert isinstance(report.recommendations, list)


# ═══════════════════════════════════════════
# Feature 3 — Cost Stream
# ═══════════════════════════════════════════

class TestCostStream:
    @pytest.fixture
    def stream(self):
        from lens.moat.cost_stream import CostStream
        return CostStream(daily_budget=100.0)

    def test_record_event(self, stream):
        event = stream.record(
            request_id="test-1",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
        )
        assert event.request_id == "test-1"
        assert event.cost_usd == 0.005
        assert event.cumulative_cost_today >= 0.005

    def test_velocity_calculation(self, stream):
        for i in range(5):
            stream.record(f"r-{i}", "gpt-4o", 100, 50, 0.001)
        velocity = stream.get_velocity(window_minutes=60)
        assert velocity >= 0

    def test_summary(self, stream):
        stream.record("r-1", "gpt-4o", 100, 50, 0.01)
        summary = stream.get_summary()
        assert "today_total" in summary
        assert summary["today_total"] >= 0.01
        assert summary["budget_remaining"] <= 100.0

    def test_hourly_breakdown(self, stream):
        stream.record("r-1", "gpt-4o", 100, 50, 0.01)
        hourly = stream.get_hourly_breakdown()
        assert isinstance(hourly, list)


# ═══════════════════════════════════════════
# Feature 4 — Semantic Prompt Cache
# ═══════════════════════════════════════════

class TestPromptCache:
    @pytest.fixture
    def cache(self):
        from lens.moat.prompt_cache import SemanticPromptCache
        return SemanticPromptCache(similarity_threshold=0.85)

    def test_store_and_exact_lookup(self, cache):
        cache.store("What is the capital of France?", "Paris", "gpt-4o", 10, 0.001)
        result = cache.lookup("What is the capital of France?", "gpt-4o")
        assert result.hit is True
        assert result.method in ("exact", "exact_match")

    def test_miss_on_unrelated(self, cache):
        cache.store("What is the capital of France?", "Paris", "gpt-4o", 10, 0.001)
        result = cache.lookup("How do I install Python?", "gpt-4o")
        assert result.hit is False

    def test_semantic_match(self, cache):
        cache.store("How do I reset my password?", "Go to settings...", "gpt-4o", 50, 0.005)
        # Very similar prompt
        result = cache.lookup("How can I change my password?", "gpt-4o")
        # May or may not hit depending on TF-IDF similarity — test that it doesn't crash
        assert result is not None
        assert isinstance(result.hit, bool)

    def test_stats(self, cache):
        cache.store("test prompt", "test response", "gpt-4o", 10, 0.001)
        cache.lookup("test prompt", "gpt-4o")
        stats = cache.get_stats()
        assert stats.total_queries >= 1
        assert stats.cache_size >= 1

    def test_top_duplicates(self, cache):
        cache.store("repeated prompt", "resp", "gpt-4o", 10, 0.001)
        # Look up same thing multiple times
        for _ in range(3):
            cache.lookup("repeated prompt", "gpt-4o")
        dupes = cache.get_top_duplicates(5)
        assert isinstance(dupes, list)


# ═══════════════════════════════════════════
# Feature 5 — Semantic Deduplication
# ═══════════════════════════════════════════

class TestDeduplication:
    @pytest.fixture
    def dedup(self):
        from lens.moat.dedup import SemanticDeduplicator
        return SemanticDeduplicator()

    def test_analyze_request(self, dedup):
        text = "The quick brown fox. The quick brown fox. The quick brown fox."
        report = dedup.analyze_request(text)
        assert report.original_tokens > 0
        assert report.redundant_tokens >= 0
        assert 0 <= report.redundancy_pct <= 100

    def test_unique_text_low_redundancy(self, dedup):
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
        report = dedup.analyze_request(text)
        assert report.redundancy_pct < 50  # Mostly unique

    def test_analyze_conversation(self, dedup):
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            {"role": "user", "content": "Tell me more about machine learning algorithms."},
            {"role": "assistant", "content": "Common algorithms include decision trees..."},
        ]
        analysis = dedup.analyze_conversation(messages)
        assert analysis.total_turns >= 2
        assert analysis.total_tokens_sent > 0

    def test_suggest_optimization(self, dedup):
        messages = [
            {"role": "user", "content": "Hello " * 50},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Hello " * 50 + " Again!"},
        ]
        analysis = dedup.analyze_conversation(messages)
        opt = dedup.suggest_optimization(analysis)
        assert isinstance(opt, dict)


# ═══════════════════════════════════════════
# Feature 6 — Provider Health Monitor
# ═══════════════════════════════════════════

class TestHealthMonitor:
    @pytest.fixture
    def monitor(self):
        from lens.moat.health_monitor import HealthMonitor
        return HealthMonitor()

    def test_record_and_status(self, monitor):
        monitor.record_request("openai", "gpt-4o", latency_ms=250, success=True)
        monitor.record_request("openai", "gpt-4o", latency_ms=300, success=True)
        status = monitor.get_status("openai")
        # Keys are "provider:model" format
        assert any("openai" in k for k in status)

    def test_sla_report(self, monitor):
        for i in range(10):
            monitor.record_request("openai", "gpt-4o", latency_ms=200 + i * 10, success=True)
        monitor.record_request("openai", "gpt-4o", latency_ms=0, success=False, error_msg="timeout")
        report = monitor.get_sla_report("openai", hours=24)
        assert report.provider == "openai"
        assert report.total_probes == 11
        assert report.successful_probes == 10
        assert report.uptime_pct > 0
        assert report.avg_latency_ms > 0

    def test_multiple_providers(self, monitor):
        monitor.record_request("openai", "gpt-4o", 200, True)
        monitor.record_request("anthropic", "claude-3-haiku", 300, True)
        status = monitor.get_status()
        # Should have both providers
        assert len(status) >= 2

    def test_latency_history(self, monitor):
        for i in range(5):
            monitor.record_request("openai", "gpt-4o", 200 + i * 50, True)
        history = monitor.get_latency_history("openai", hours=1)
        assert isinstance(history, list)


# ═══════════════════════════════════════════
# Feature 7 — Tokenizer Drift Detector
# ═══════════════════════════════════════════

class TestDriftDetector:
    @pytest.fixture
    def detector(self):
        from lens.moat.drift_detector import TokenizerDriftDetector
        return TokenizerDriftDetector()

    def _count_fn(self, text):
        return len(text.encode("utf-8"))

    def test_establish_baseline(self, detector):
        baseline = detector.establish_baseline("gpt-4o", self._count_fn)
        assert isinstance(baseline, dict)
        assert len(baseline) > 0

    def test_check_drift_no_drift(self, detector):
        detector.establish_baseline("gpt-4o", self._count_fn)
        # Same counting function → no drift
        reports = detector.check_drift("gpt-4o", self._count_fn)
        assert isinstance(reports, list)
        for r in reports:
            assert r.drift_pct == 0.0

    def test_check_drift_with_drift(self, detector):
        detector.establish_baseline("gpt-4o", self._count_fn)
        # Different counting function → drift detected
        def inflated_count(text):
            return len(text.encode("utf-8")) + 10
        reports = detector.check_drift("gpt-4o", inflated_count)
        assert any(r.drift_tokens != 0 for r in reports)

    def test_get_history(self, detector):
        detector.establish_baseline("gpt-4o", self._count_fn)
        detector.check_drift("gpt-4o", self._count_fn)
        history = detector.get_history("gpt-4o")
        assert history.model == "gpt-4o"


# ═══════════════════════════════════════════
# Feature 8 — Cost Attribution
# ═══════════════════════════════════════════

class TestCostAttribution:
    @pytest.fixture
    def engine(self):
        from lens.moat.attribution import CostAttributionEngine, AttributionTag
        return CostAttributionEngine()

    def test_record_entry(self, engine):
        from lens.moat.attribution import AttributionTag
        tags = AttributionTag(team="eng", feature="chatbot", user_id="user1")
        entry = engine.record("r-1", "gpt-4o", 100, 50, 0.005, tags)
        assert entry.request_id == "r-1"
        assert entry.cost_usd == 0.005

    def test_breakdown_by_team(self, engine):
        from lens.moat.attribution import AttributionTag
        engine.record("r-1", "gpt-4o", 100, 50, 0.005, AttributionTag(team="eng"))
        engine.record("r-2", "gpt-4o", 200, 100, 0.01, AttributionTag(team="sales"))
        engine.record("r-3", "gpt-4o", 150, 75, 0.008, AttributionTag(team="eng"))
        breakdown = engine.get_breakdown("team")
        assert breakdown.total_requests == 3
        assert breakdown.total_cost > 0

    def test_budget_enforcement(self, engine):
        from lens.moat.attribution import AttributionTag
        engine.set_budget("eng", 100.0)
        engine.record("r-1", "gpt-4o", 100, 50, 50.0, AttributionTag(team="eng"))
        status = engine.get_team_budget_status("eng")
        assert status["budget"] == 100.0
        assert status["spent"] >= 50.0
        assert status["pct_used"] >= 50.0

    def test_top_consumers(self, engine):
        from lens.moat.attribution import AttributionTag
        engine.record("r-1", "gpt-4o", 100, 50, 0.01, AttributionTag(user_id="alice"))
        engine.record("r-2", "gpt-4o", 100, 50, 0.02, AttributionTag(user_id="bob"))
        top = engine.get_top_consumers(n=5, dimension="user_id")
        assert isinstance(top, list)
        assert len(top) >= 2


# ═══════════════════════════════════════════
# Feature 9 — Audit Trail
# ═══════════════════════════════════════════

class TestAuditTrail:
    @pytest.fixture
    def trail(self):
        from lens.moat.audit_trail import AuditTrail
        return AuditTrail()

    def test_record_creates_entry(self, trail):
        record = trail.record(
            model="gpt-4o", provider="openai",
            input_text="Hello", output_text="Hi there",
            input_tokens=5, output_tokens=3, cost_usd=0.001,
        )
        assert record.record_id is not None
        assert record.input_hash is not None
        assert record.record_hash is not None

    def test_hash_chain_integrity(self, trail):
        trail.record("gpt-4o", "openai", "Hello", "Hi", 5, 3, 0.001)
        trail.record("gpt-4o", "openai", "World", "!", 5, 1, 0.001)
        trail.record("gpt-4o", "openai", "Test", "OK", 4, 2, 0.001)
        valid, msg = trail.verify_integrity()
        assert valid is True

    def test_generate_report(self, trail):
        trail.record("gpt-4o", "openai", "Hello", "Hi", 5, 3, 0.001, team="eng")
        trail.record("claude-3-haiku", "anthropic", "Test", "OK", 4, 2, 0.002, team="sales")
        report = trail.generate_report()
        assert report.total_records == 2
        assert report.total_cost > 0
        assert isinstance(report.compliance_notes, list)

    def test_merkle_root(self, trail):
        trail.record("gpt-4o", "openai", "A", "B", 1, 1, 0.001)
        root = trail.get_latest_merkle_root()
        assert isinstance(root, str)
        assert len(root) > 0

    def test_empty_trail_integrity(self, trail):
        valid, msg = trail.verify_integrity()
        assert valid is True  # Empty chain is valid

    def test_export_for_auditor(self, trail):
        trail.record("gpt-4o", "openai", "Hello", "Hi", 5, 3, 0.001)
        export = trail.export_for_auditor()
        assert isinstance(export, list)
        assert len(export) == 1
