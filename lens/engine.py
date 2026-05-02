"""
LEAEEngine — core orchestrator for the Linguistic Entropy Attribution Engine.

Wires together all subsystems:
  entropy/         — token surprisal, IDS, ETR, semantic entropy
  detection/       — language detection, token counting
  prediction/      — CUSUM spike prediction, rolling windows, anomaly detection
  storage/         — ClickHouse time-series, Redis cache
  analytics/       — SQL aggregation, model arbitrage, report generation

All improvements from the 10/10 analysis are integrated here:
  Gap 1 — Real LM surprisal (distilgpt2) via TokenEntropyCalculator
  Gap 2 — Semantic entropy (SemanticEntropyCalculator) — 2D waste matrix
  Gap 3 — CUSUM spike prediction (SpikePredictor)
  Gap 4 — Cross-model arbitrage (ModelArbitrageAdvisor)
  Gap 5 — Spike cause attribution (SpikePredictor._classify_cause)
  Gap 6 — Prediction outcome tracking & recalibration (PredictionOutcomeTracker)
  Addition — COMPRESS bridge signal (low_ids_alert on EntropyProfile)
"""

import asyncio
import time
import uuid
from datetime import datetime
from loguru import logger

from lens.config import Config
from lens.entropy.token_entropy import TokenEntropyCalculator
from lens.entropy.ids_calculator import IDSCalculator
from lens.entropy.etr_calculator import ETRCalculator
from lens.entropy.semantic_entropy import SemanticEntropyCalculator
from lens.entropy.structures import EntropyProfile, SpikePrediction, TokenizerArbitrageResult
from lens.detection.language_detector import LanguageDetector
from lens.detection.token_counter import TokenCounter
from lens.prediction.spike_predictor import SpikePredictor
from lens.prediction.rolling_window import RollingEntropyWindow
from lens.prediction.anomaly_detector import AnomalyDetector
from lens.prediction.outcome_tracker import PredictionOutcomeTracker
from lens.storage.clickhouse_store import ClickHouseStore
from lens.storage.redis_cache import RedisCache
from lens.analytics.aggregator import Aggregator
from lens.analytics.model_arbitrage import ModelArbitrageAdvisor
# Gap 8–16 advanced modules
from lens.entropy.compression_ratio import CompressionAnalyzer
from lens.entropy.adversarial_prober import AdversarialProber
from lens.prediction.wavelet_analyzer import WaveletEntropyAnalyzer
from lens.prediction.transfer_entropy import TransferEntropyAnalyzer
from lens.prediction.mdp_router import MDPRouter, MDPState
# Platform features
from lens.moat.token_verifier import TokenVerificationEngine
from lens.moat.billing import BillingReconciler
from lens.moat.cost_stream import CostStream
from lens.moat.prompt_cache import SemanticPromptCache
from lens.moat.dedup import SemanticDeduplicator
from lens.moat.health_monitor import HealthMonitor
from lens.moat.drift_detector import TokenizerDriftDetector
from lens.moat.attribution import CostAttributionEngine, AttributionTag
from lens.moat.audit_trail import AuditTrail


class LEAEEngine:
    """
    Linguistic Entropy Attribution Engine — production orchestrator.

    Supported models (updated for claude-opus-4-6):
      OpenAI:    gpt-4o, gpt-4o-mini, gpt-3.5-turbo
      Anthropic: claude-opus-4-6, claude-sonnet-4-6, claude-3-5-sonnet, claude-3-haiku
      Meta:      llama-3-8b, llama-3-70b
      Mistral:   mistral-7b
      Google:    gemini-1.5-pro
    """

    SUPPORTED_LANGUAGES = {
        "en", "ta", "hi", "ar", "ja", "zh", "ko", "pt", "es",
        "fr", "de", "id", "ms", "bn", "ur", "te", "ml", "pa", "gu", "mr", "kn",
    }

    SUPPORTED_MODELS = {
        "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
        "claude-opus-4-6", "claude-sonnet-4-6",
        "claude-3-5-sonnet", "claude-3-haiku",
        "llama-3-8b", "llama-3-70b",
        "mistral-7b", "gemini-1.5-pro",
    }

    # Token cost per 1M input tokens (USD)
    MODEL_COSTS: dict[str, float] = {
        "gpt-4o":              5.00,
        "gpt-4o-mini":         0.15,
        "gpt-3.5-turbo":       0.50,
        "claude-opus-4-6":     15.00,
        "claude-sonnet-4-6":   3.00,
        "claude-3-5-sonnet":   3.00,
        "claude-3-haiku":      0.25,
        "llama-3-8b":          0.00,    # Self-hosted
        "llama-3-70b":         0.00,
        "mistral-7b":          0.00,
        "gemini-1.5-pro":      3.50,
    }

    def __init__(self, config: Config):
        self.config = config
        # Entropy subsystem
        self.entropy_calc = TokenEntropyCalculator(config)
        self.ids_calc = IDSCalculator(config)
        self.etr_calc = ETRCalculator(config)
        self.semantic_calc = SemanticEntropyCalculator(config)
        # Detection
        self.lang_detector = LanguageDetector(config)
        self.token_counter = TokenCounter(config)
        # Prediction
        self.spike_predictor = SpikePredictor(config)
        self.rolling_window = RollingEntropyWindow(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.outcome_tracker = PredictionOutcomeTracker(config)
        # Storage
        self.ch_store = ClickHouseStore(config)
        self.cache = RedisCache(config)
        # Analytics
        self.arbitrage_advisor = ModelArbitrageAdvisor(config, self.token_counter, self.etr_calc)
        # Gap 8: Compression-based Kolmogorov complexity
        self.compression_analyzer = CompressionAnalyzer()
        # Gap 10: Wavelet multi-scale entropy
        self.wavelet_analyzer = WaveletEntropyAnalyzer()
        # Gap 11: Transfer entropy cross-language causality
        self.transfer_entropy = TransferEntropyAnalyzer()
        # Gap 14: MDP dynamic model routing
        self.mdp_router = MDPRouter(models=list(self.SUPPORTED_MODELS))
        # Gap 16: Adversarial tokenizer probing
        self.adversarial_prober = AdversarialProber(models=list(self.SUPPORTED_MODELS))
        # Platform features
        self.token_verifier = TokenVerificationEngine()
        self.billing_reconciler = BillingReconciler(verifier=self.token_verifier)
        self.cost_stream = CostStream(daily_budget=1000.0)
        self.prompt_cache = SemanticPromptCache()
        self.deduplicator = SemanticDeduplicator()
        self.provider_health = HealthMonitor()
        self.drift_detector = TokenizerDriftDetector()
        self.cost_attribution = CostAttributionEngine()
        self.audit_trail = AuditTrail()
        logger.info(
            "LEAEEngine initialised. device={} semantic_entropy={} arbitrage={}",
            config.device,
            config.semantic_entropy_enabled,
            config.arbitrage_enabled,
        )

    async def profile_request(
        self,
        text: str,
        model_name: str,
        customer_id: str = "default",
        request_id: str = None,
        pre_detected_language: str = None,
    ) -> EntropyProfile:
        """
        Full entropy profiling of an AI API request.

        Steps:
          1. Language detection (cached)
          2. Token counting — actual + English baseline
          3. Per-token entropy decomposition (Gap 1: LM surprisal)
          4. IDS computation + COMPRESS bridge span detection
          5. ETR computation + inequity ratio
          6. Semantic entropy — 2D waste matrix (Gap 2)
          7. Waste type classification
          8. Cost calculation (actual + waste)
          9. Async storage to ClickHouse (fire-and-forget)
          10. Rolling entropy window update (for Gap 3 spike detection)
          11. Anomaly detection observation
        """
        start_ms = time.monotonic() * 1000
        request_id = request_id or str(uuid.uuid4())

        # Step 1: Language detection
        if pre_detected_language and pre_detected_language in self.SUPPORTED_LANGUAGES:
            language = pre_detected_language
            lang_confidence = 1.0
        else:
            cached_lang = await self.cache.get_language(text)
            if cached_lang:
                language, lang_confidence = cached_lang
            else:
                language, lang_confidence = await self.lang_detector.detect(text)
                await self.cache.set_language(text, language, lang_confidence)

        if model_name not in self.SUPPORTED_MODELS:
            logger.debug("Unknown model '{}' — defaulting to gpt-4o", model_name)
            model_name = "gpt-4o"

        # Step 2: Token counting
        actual_tokens = self.token_counter.count(text, model_name)
        english_baseline_tokens = await self.token_counter.count_english_baseline(
            text, language, model_name
        )
        efficiency_ratio = actual_tokens / max(english_baseline_tokens, 1)

        # Step 3: Per-token entropy decomposition (Gap 1 fix)
        token_records = await self.entropy_calc.decompose(text, model_name, language)
        total_entropy = sum(r.surprisal_bits for r in token_records)
        mean_entropy = total_entropy / max(len(token_records), 1)

        # Step 4: IDS + COMPRESS bridge signal
        ids_score = self.ids_calc.compute(token_records, language)
        low_ids_alert = ids_score < self.config.low_ids_alert_threshold
        low_ids_spans = []
        if low_ids_alert:
            low_ids_spans = self.ids_calc.find_low_ids_spans(token_records, language)

        # Step 5: ETR
        etr_score = self.etr_calc.compute(total_entropy, actual_tokens, language)
        etr_english = self.etr_calc.compute(total_entropy, english_baseline_tokens, "en")
        etr_inequity = self.etr_calc.compute_inequity_ratio(etr_score, etr_english)

        # Step 6: Semantic entropy (Gap 2)
        semantic_entropy_bits = 0.0
        semantic_windows = []
        semantic_ids_score = 0.0
        if self.config.semantic_entropy_enabled:
            semantic_entropy_bits, semantic_windows = await self.semantic_calc.compute(
                text, language
            )
            semantic_ids_score = semantic_entropy_bits / max(actual_tokens * 0.01, 0.001)

        # Step 7: Waste type classification
        waste_type = self.semantic_calc.classify_waste_type(ids_score, semantic_entropy_bits)

        # Step 8: Cost attribution
        cost_per_token = self.MODEL_COSTS.get(model_name, 3.0) / 1_000_000
        cost_usd = actual_tokens * cost_per_token
        baseline_cost = english_baseline_tokens * cost_per_token
        waste_cost_usd = max(0.0, cost_usd - baseline_cost)

        profile = EntropyProfile(
            text=text[:500],
            language=language,
            model_name=model_name,
            token_count=actual_tokens,
            english_baseline_tokens=english_baseline_tokens,
            efficiency_ratio=round(efficiency_ratio, 4),
            total_entropy_bits=round(total_entropy, 4),
            mean_entropy_per_token=round(mean_entropy, 4),
            ids_score=round(ids_score, 4),
            etr_score=round(etr_score, 4),
            etr_english_baseline=round(etr_english, 4),
            etr_inequity_ratio=round(etr_inequity, 4),
            semantic_entropy_bits=round(semantic_entropy_bits, 4),
            semantic_ids_score=round(semantic_ids_score, 4),
            waste_type=waste_type,
            cost_usd=round(cost_usd, 8),
            waste_cost_usd=round(waste_cost_usd, 8),
            low_ids_alert=low_ids_alert,
            low_ids_token_spans=low_ids_spans,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            customer_id=customer_id,
            token_records=token_records,
            semantic_windows=semantic_windows,
        )

        # Step 9–11: Async side effects (fire-and-forget)
        asyncio.create_task(self._store_profile(profile))
        asyncio.create_task(
            self.rolling_window.update(language, total_entropy, datetime.utcnow(), customer_id)
        )
        self.anomaly_detector.observe(language, total_entropy)
        # Gap 11: Feed transfer entropy analyzer
        self.transfer_entropy.observe(language, total_entropy, time.time())

        elapsed_ms = (time.monotonic() * 1000) - start_ms
        logger.debug(
            "profile_request: lang={} tokens={} eff={:.2f}x IDS={:.3f} "
            "ETR={:.2f} sem_H={:.2f} waste_type={} waste=${:.6f} [{:.1f}ms]",
            language, actual_tokens, efficiency_ratio,
            ids_score, etr_score, semantic_entropy_bits,
            waste_type, waste_cost_usd, elapsed_ms,
        )
        return profile

    async def batch_profile(
        self,
        requests: list[dict],
        concurrency: int = 20,
    ) -> list[EntropyProfile]:
        """Profile multiple requests concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(req: dict) -> EntropyProfile:
            async with semaphore:
                return await self.profile_request(
                    text=req["text"],
                    model_name=req.get("model", "gpt-4o"),
                    customer_id=req.get("customer_id", "default"),
                    request_id=req.get("request_id"),
                    pre_detected_language=req.get("language"),
                )

        return await asyncio.gather(*[_one(r) for r in requests])

    async def predict_spikes(
        self,
        languages: list[str] = None,
        horizon_minutes: int = 30,
        customer_id: str = "global",
    ) -> list[SpikePrediction]:
        """
        Predict imminent token cost spikes from entropy velocity (CUSUM).

        Returns predictions sorted by confidence descending.
        Tracks predictions in outcome_tracker for Gap 6 recalibration.
        """
        if languages is None:
            languages = await self.rolling_window.get_all_languages(customer_id)
            if not languages:
                languages = list(self.SUPPORTED_LANGUAGES)

        predictions = []
        for lang in languages:
            window = await self.rolling_window.get(lang, customer_id)
            if window is None or len(window) < 5:
                continue

            # Use recalibrated threshold for this language (Gap 6)
            threshold = self.outcome_tracker.get_threshold(lang)
            original_threshold = self.config.spike_alert_confidence_threshold
            self.config.spike_alert_confidence_threshold = threshold

            prediction = await self.spike_predictor.predict(
                language=lang,
                entropy_window=window,
                horizon_minutes=horizon_minutes,
            )

            self.config.spike_alert_confidence_threshold = original_threshold

            if prediction:
                predictions.append(prediction)
                # Record prediction for outcome tracking (Gap 6)
                if self.config.prediction_tracking_enabled:
                    asyncio.create_task(
                        self.outcome_tracker.record_prediction(
                            language=lang,
                            predicted_increase_pct=prediction.predicted_token_increase_pct,
                            confidence=prediction.confidence,
                            horizon_minutes=horizon_minutes,
                        )
                    )

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

    async def compute_arbitrage(
        self,
        text: str,
        language: str,
        customer_id: str = "default",
    ) -> TokenizerArbitrageResult:
        """Cross-model tokenizer comparison (Gap 4)."""
        if not self.config.arbitrage_enabled:
            return None
        # Check cache
        cached = await self.cache.get_arbitrage(text, language)
        if cached:
            return TokenizerArbitrageResult(**cached)
        result = await self.arbitrage_advisor.compute(text, language, customer_id)
        await self.cache.set_arbitrage(text, language, {
            "text": result.text,
            "language": result.language,
            "results": result.results,
            "recommended_model": result.recommended_model,
            "max_savings_pct": result.max_savings_pct,
            "savings_summary": result.savings_summary,
        })
        return result

    async def get_language_report(
        self,
        language: str,
        customer_id: str,
        period_hours: int = 24,
    ):
        """Generate cost attribution and equity report for a language."""
        agg = Aggregator(self.ch_store)
        return await agg.language_report(language, customer_id, period_hours)

    async def _store_profile(self, profile: EntropyProfile) -> None:
        """Async fire-and-forget ClickHouse write."""
        try:
            await self.ch_store.insert_profile(profile)
        except Exception as e:
            logger.warning("ClickHouse write failed: request_id={} err={}", profile.request_id, e)

    # ── Gap 8: Compression Analysis ──

    async def analyze_compression(self, text: str, model_name: str = "gpt-4o"):
        """Kolmogorov complexity approximation via multi-algorithm compression (Gap 8)."""
        token_count = self.token_counter.count(text, model_name)
        return self.compression_analyzer.analyze(text, token_count, model_name)

    # ── Gap 10: Wavelet Multi-Scale Analysis ──

    async def analyze_wavelet(self, language: str, customer_id: str = "global"):
        """Multi-scale wavelet entropy decomposition for a language (Gap 10)."""
        window = await self.rolling_window.get(language, customer_id)
        if window is None or len(window) < 8:
            return None
        entropy_values = [v for v in window]
        return self.wavelet_analyzer.analyze(entropy_values)

    # ── Gap 11: Transfer Entropy ──

    async def get_causal_network(self, languages: list[str] = None):
        """Compute cross-language causal network via transfer entropy (Gap 11)."""
        return self.transfer_entropy.compute_network(languages)

    # ── Gap 14: MDP Dynamic Routing ──

    async def route_model(
        self,
        language: str,
        text_type: str = "prose",
        entropy_level: str = "medium",
        time_bucket: str = "peak",
        token_volume: str = "moderate",
    ):
        """Get optimal model recommendation via MDP Q-learning (Gap 14)."""
        state = MDPState(
            language=language,
            text_type=text_type,
            entropy_level=entropy_level,
            time_bucket=time_bucket,
            token_volume=token_volume,
        )
        return self.mdp_router.route(state)

    async def get_routing_policy(self):
        """Get the learned MDP routing policy summary (Gap 14)."""
        return self.mdp_router.get_policy_summary()

    # ── Gap 16: Adversarial Probing ──

    async def probe_adversarial(self, text: str, model_name: str = "gpt-4o"):
        """Run adversarial tokenizer probing against a model (Gap 16)."""
        def token_count_fn(t, m):
            return self.token_counter.count(t, m)
        return self.adversarial_prober.probe(text, model_name, token_count_fn)

    async def health_check(self) -> dict:
        ch_health, redis_health, lang_health, tok_health = await asyncio.gather(
            self.ch_store.health(),
            self.cache.health(),
            self.lang_detector.health(),
            self.token_counter.health(),
        )
        sem_health = "enabled" if self.config.semantic_entropy_enabled else "disabled"
        return {
            "status": "healthy",
            "lang_detector": lang_health,
            "token_counter": tok_health,
            "clickhouse": ch_health,
            "redis": redis_health,
            "semantic_entropy": sem_health,
        }
