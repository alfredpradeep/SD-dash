"""
FastAPI router for LENS endpoints.

Routes:
  POST /lens/profile          — Single request entropy + cost attribution
  POST /lens/profile/batch    — Batch profiling (up to 500 items)
  GET  /lens/predict/spikes   — Current spike predictions per language
  POST /lens/report/language  — Per-language cost attribution report
  POST /lens/arbitrage        — Cross-model tokenizer comparison (Gap 4)
  GET  /lens/accuracy         — Spike prediction accuracy report (Gap 6)
  GET  /lens/health           — Health check

  Platform Features:
  POST /lens/platform/verify           — Token verification
  POST /lens/platform/verify/compare   — Cross-provider comparison
  POST /lens/platform/billing/record   — Record billed request
  GET  /lens/platform/billing/reconcile — Billing reconciliation
  POST /lens/platform/cost/record      — Record cost event
  GET  /lens/platform/cost/summary     — Cost stream summary
  POST /lens/platform/cache/lookup     — Semantic cache lookup
  POST /lens/platform/cache/store      — Store in cache
  GET  /lens/platform/cache/stats      — Cache statistics
  POST /lens/platform/dedup/analyze    — Deduplicate request
  POST /lens/platform/dedup/conversation — Conversation dedup
  POST /lens/platform/health/record    — Record provider health
  GET  /lens/platform/health/status    — Provider status
  GET  /lens/platform/health/sla/{provider} — SLA report
  POST /lens/platform/drift/baseline   — Establish tokenizer baseline
  POST /lens/platform/drift/check      — Check for drift
  GET  /lens/platform/drift/history/{model} — Drift history
  POST /lens/platform/attribution/record — Record attributed cost
  GET  /lens/platform/attribution/breakdown/{dimension} — Cost breakdown
  POST /lens/platform/attribution/budget — Set team budget
  GET  /lens/platform/attribution/budget/{team} — Budget status
  POST /lens/platform/audit/record     — Record audit entry
  GET  /lens/platform/audit/integrity  — Verify chain integrity
  GET  /lens/platform/audit/report     — Compliance report
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger
from lens.api.schemas import (
    ProfileRequest, ProfileResponse,
    BatchProfileRequest, BatchProfileResponse,
    SpikePredictionResponse, HealthResponse,
    CostReportRequest, CostReportResponse,
    ArbitrageRequest, ArbitrageResponse, ArbitrageModelResult,
    AccuracyReportResponse, LowIDSSpan, TokenRecord, SemanticWindowRecord,
    # Gap 8–16
    CompressionRequest, CompressionResponse,
    WaveletScaleResponse, WaveletResponse,
    CausalLinkResponse, CausalNetworkResponse,
    RoutingRequest, RoutingResponse, RoutingAlternative, RoutingPolicyResponse,
    AdversarialRequest, AdversarialResultResponse, AdversarialResponse,
    # Platform features
    TokenVerifyRequest, TokenVerifyResponse, ProviderComparisonResponse,
    BillingRecordRequest, BillingRecordResponse, BillingReconciliationResponse,
    CostEventRequest, CostEventResponse, CostSummaryResponse,
    CacheLookupRequest, CacheLookupResponse, CacheStoreRequest, CacheStatsResponse,
    DedupRequest, DedupResponse, ConversationDedupRequest, ConversationDedupResponse,
    HealthRecordRequest, ProviderStatusResponse, ProviderSLAResponse,
    DriftCheckRequest, DriftReportResponse, DriftHistoryResponse,
    AttributionRecordRequest, AttributionBreakdownResponse, BudgetStatusResponse, SetBudgetRequest,
    AuditRecordRequest, AuditRecordResponse, AuditIntegrityResponse, AuditReportResponse,
)
from lens.moat.attribution import AttributionTag

router = APIRouter(prefix="/lens", tags=["LENS — Linguistic Entropy Attribution Engine"])


def get_engine():
    from lens.main import engine_instance
    return engine_instance


@router.post("/profile", response_model=ProfileResponse, status_code=200)
async def profile_request(
    request: ProfileRequest,
    engine=Depends(get_engine),
) -> ProfileResponse:
    """
    Profile a single AI API request for entropy, cost attribution, and waste detection.

    Returns IDS, ETR, semantic entropy (2D waste matrix), equity statement,
    and optional COMPRESS bridge signals.
    """
    try:
        profile = await engine.profile_request(
            text=request.text,
            model_name=request.model_name,
            customer_id=request.customer_id,
            request_id=request.request_id,
            pre_detected_language=request.language,
        )

        waste_pct = (profile.waste_cost_usd / profile.cost_usd * 100) if profile.cost_usd > 0 else 0.0

        # Equity statement from ETR calculator
        equity_stmt = engine.etr_calc.equity_statement(
            language=profile.language,
            etr_language=profile.etr_score,
            etr_english=profile.etr_english_baseline,
            cost_usd=profile.cost_usd,
            waste_cost_usd=profile.waste_cost_usd,
        ) if profile.language != "en" else None

        token_records_resp = None
        if request.include_token_records and profile.token_records:
            token_records_resp = [
                TokenRecord(
                    position=r.position,
                    token_text=r.token_text,
                    surprisal_bits=round(r.surprisal_bits, 4),
                    is_high_entropy=r.is_high_entropy,
                    token_type=r.token_type,
                )
                for r in profile.token_records[:500]  # cap at 500 for response size
            ]

        semantic_windows_resp = None
        if request.include_token_records and profile.semantic_windows:
            semantic_windows_resp = [
                SemanticWindowRecord(
                    window_text=w.window_text,
                    embedding_variance=w.embedding_variance,
                    semantic_entropy_bits=w.semantic_entropy_bits,
                    is_high_semantic_entropy=w.is_high_semantic_entropy,
                )
                for w in profile.semantic_windows[:50]
            ]

        low_ids_spans = [
            LowIDSSpan(start_token=s, end_token=e, ids_score=ids)
            for s, e, ids in profile.low_ids_token_spans
        ]

        arb_resp = None
        if request.include_arbitrage:
            arb = await engine.compute_arbitrage(
                text=request.text,
                language=profile.language,
                customer_id=request.customer_id,
            )
            if arb:
                arb_resp = _arb_to_response(arb, request.model_name)

        return ProfileResponse(
            request_id=profile.request_id,
            language=profile.language,
            model_name=profile.model_name,
            token_count=profile.token_count,
            baseline_tokens=profile.english_baseline_tokens,
            efficiency_ratio=round(profile.efficiency_ratio, 3),
            ids_score=round(profile.ids_score, 4),
            etr_score=round(profile.etr_score, 4),
            etr_inequity_ratio=round(profile.etr_inequity_ratio, 3),
            semantic_entropy_bits=round(profile.semantic_entropy_bits, 4),
            waste_type=profile.waste_type,
            cost_usd=round(profile.cost_usd, 8),
            waste_cost_usd=round(profile.waste_cost_usd, 8),
            waste_pct=round(waste_pct, 2),
            equity_statement=equity_stmt,
            low_ids_alert=profile.low_ids_alert,
            low_ids_spans=low_ids_spans,
            token_records=token_records_resp,
            semantic_windows=semantic_windows_resp,
            arbitrage=arb_resp,
        )
    except Exception as e:
        logger.exception("Error in profile_request")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profile/batch", response_model=BatchProfileResponse)
async def batch_profile(
    request: BatchProfileRequest,
    engine=Depends(get_engine),
) -> BatchProfileResponse:
    """Profile up to 500 requests concurrently."""
    if len(request.items) > 500:
        raise HTTPException(status_code=422, detail="Maximum 500 items per batch")

    try:
        profiles = await engine.batch_profile([
            {
                "text": item.text,
                "model": item.model,
                "customer_id": item.customer_id,
                "request_id": item.request_id,
                "language": item.language,
            }
            for item in request.items
        ])

        results = []
        total_cost = 0.0
        total_waste = 0.0
        langs = set()

        for p in profiles:
            waste_pct = (p.waste_cost_usd / p.cost_usd * 100) if p.cost_usd > 0 else 0.0
            total_cost += p.cost_usd
            total_waste += p.waste_cost_usd
            langs.add(p.language)
            results.append(ProfileResponse(
                request_id=p.request_id,
                language=p.language,
                model_name=p.model_name,
                token_count=p.token_count,
                baseline_tokens=p.english_baseline_tokens,
                efficiency_ratio=round(p.efficiency_ratio, 3),
                ids_score=round(p.ids_score, 4),
                etr_score=round(p.etr_score, 4),
                etr_inequity_ratio=round(p.etr_inequity_ratio, 3),
                semantic_entropy_bits=round(p.semantic_entropy_bits, 4),
                waste_type=p.waste_type,
                cost_usd=round(p.cost_usd, 8),
                waste_cost_usd=round(p.waste_cost_usd, 8),
                waste_pct=round(waste_pct, 2),
                low_ids_alert=p.low_ids_alert,
            ))

        return BatchProfileResponse(
            results=results,
            total_cost_usd=round(total_cost, 8),
            total_waste_usd=round(total_waste, 8),
            languages_detected=sorted(langs),
        )
    except Exception as e:
        logger.exception("Error in batch_profile")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/spikes", response_model=list[SpikePredictionResponse])
async def predict_spikes(
    horizon_minutes: int = Query(default=30, ge=5, le=120),
    language: str = Query(default=None),
    customer_id: str = Query(default="global"),
    engine=Depends(get_engine),
) -> list[SpikePredictionResponse]:
    """Get current spike predictions. Sorted by confidence descending."""
    try:
        languages = [language] if language else None
        predictions = await engine.predict_spikes(
            languages=languages,
            horizon_minutes=horizon_minutes,
            customer_id=customer_id,
        )
        return [SpikePredictionResponse(
            language=p.language,
            entropy_velocity=round(p.current_entropy_velocity, 4),
            predicted_token_increase_pct=round(p.predicted_token_increase_pct, 2),
            confidence=round(p.confidence, 4),
            horizon_minutes=p.horizon_minutes,
            alert_level=p.alert_level,
            spike_cause=p.spike_cause,
            is_likely_temporary=p.is_likely_temporary,
            trigger_reason=p.trigger_reason,
        ) for p in predictions]
    except Exception as e:
        logger.exception("Error in predict_spikes")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/language", response_model=CostReportResponse)
async def language_cost_report(
    request: CostReportRequest,
    engine=Depends(get_engine),
) -> CostReportResponse:
    """Generate per-language cost attribution and equity report."""
    try:
        report = await engine.get_language_report(
            language=request.language,
            customer_id=request.customer_id,
            period_hours=request.period_hours,
        )
        waste_pct = (report.total_waste_cost_usd / report.total_cost_usd * 100) \
            if report.total_cost_usd > 0 else 0.0
        return CostReportResponse(
            language=report.language,
            period_start=report.period_start,
            period_end=report.period_end,
            total_requests=report.total_requests,
            total_tokens=report.total_tokens,
            total_cost_usd=round(report.total_cost_usd, 6),
            total_waste_cost_usd=round(report.total_waste_cost_usd, 6),
            waste_pct=round(waste_pct, 2),
            mean_ids_score=round(report.mean_ids_score, 4),
            mean_etr_score=round(report.mean_etr_score, 4),
            etr_inequity_ratio=round(report.etr_inequity_ratio, 3),
            efficiency_percentile=round(report.efficiency_percentile, 1),
            spike_events=report.spike_events,
            equity_multiplier=round(report.equity_multiplier, 2),
            effective_cost_per_info_unit=round(report.effective_cost_per_info_unit, 8),
            english_cost_per_info_unit=round(report.english_cost_per_info_unit, 8),
            recommendations=report.recommendations,
        )
    except Exception as e:
        logger.exception("Error in language_cost_report")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/arbitrage", response_model=ArbitrageResponse)
async def tokenizer_arbitrage(
    request: ArbitrageRequest,
    engine=Depends(get_engine),
) -> ArbitrageResponse:
    """
    Cross-model tokenizer comparison (Gap 4 fix).

    Shows token count and cost for the same text across all supported models.
    Identifies which model minimises cost for the given language.
    """
    try:
        language = request.language
        if not language:
            language, _ = await engine.lang_detector.detect(request.text)

        arb = await engine.compute_arbitrage(
            text=request.text,
            language=language,
            customer_id=request.customer_id,
        )
        if not arb:
            raise HTTPException(status_code=500, detail="Arbitrage computation failed")

        # Use gpt-4o as "current model" baseline for the response
        return _arb_to_response(arb, "gpt-4o")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in tokenizer_arbitrage")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/accuracy", response_model=AccuracyReportResponse)
async def prediction_accuracy(engine=Depends(get_engine)) -> AccuracyReportResponse:
    """Spike prediction accuracy and recalibration report (Gap 6)."""
    report = engine.outcome_tracker.accuracy_report()
    total = sum(v.get("total_predictions", 0) for v in report.values())
    return AccuracyReportResponse(
        languages=report,
        total_predictions_tracked=total,
    )


@router.get("/health", response_model=HealthResponse)
async def health(engine=Depends(get_engine)) -> HealthResponse:
    status = await engine.health_check()
    return HealthResponse(**status)


# ── Gap 8: Compression Analysis ──

@router.post("/compression", response_model=CompressionResponse, tags=["Gap 8 — Kolmogorov Complexity"])
async def compression_analysis(request: CompressionRequest, engine=Depends(get_engine)):
    """Kolmogorov complexity approximation via multi-algorithm compression."""
    try:
        profile = await engine.analyze_compression(request.text, request.model_name)
        return CompressionResponse(
            raw_bytes=profile.raw_bytes,
            zlib_bytes=profile.zlib_bytes,
            lzma_bytes=profile.lzma_bytes,
            brotli_bytes=profile.brotli_bytes,
            best_compressed_bytes=profile.best_compressed_bytes,
            best_algorithm=profile.best_algorithm,
            token_count=profile.token_count,
            compression_ratio=profile.compression_ratio,
            information_density=profile.information_density,
            tokenizer_inflation=profile.tokenizer_inflation,
            objective_waste_pct=profile.objective_waste_pct,
            kolmogorov_estimate_bits=profile.kolmogorov_estimate_bits,
        )
    except Exception as e:
        logger.exception("Error in compression_analysis")
        raise HTTPException(status_code=500, detail=str(e))


# ── Gap 10: Wavelet Multi-Scale ──

@router.get("/wavelet/{language}", response_model=WaveletResponse, tags=["Gap 10 — Wavelet Decomposition"])
async def wavelet_analysis(
    language: str,
    customer_id: str = Query(default="global"),
    engine=Depends(get_engine),
):
    """Multi-scale wavelet entropy decomposition for spike detection."""
    try:
        result = await engine.analyze_wavelet(language, customer_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {language}")
        return WaveletResponse(
            scales=[WaveletScaleResponse(
                scale_index=s.scale_index,
                scale_name=s.scale_name,
                coefficients=s.coefficients,
                energy=s.energy,
                entropy=s.entropy,
                trend_direction=s.trend_direction,
                trend_strength=s.trend_strength,
                anomaly_detected=s.anomaly_detected,
                anomaly_score=s.anomaly_score,
            ) for s in result.scales],
            dominant_scale=result.dominant_scale,
            overall_trend=result.overall_trend,
            multi_scale_entropy=result.multi_scale_entropy,
            early_warning_score=result.early_warning_score,
            reconstruction=result.reconstruction,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in wavelet_analysis")
        raise HTTPException(status_code=500, detail=str(e))


# ── Gap 11: Transfer Entropy Causal Network ──

@router.get("/causal-network", response_model=CausalNetworkResponse, tags=["Gap 11 — Transfer Entropy"])
async def causal_network(engine=Depends(get_engine)):
    """Cross-language causal prediction network via transfer entropy."""
    try:
        network = await engine.get_causal_network()
        strongest = None
        if network.strongest_link:
            sl = network.strongest_link
            strongest = CausalLinkResponse(
                source_language=sl.source_language,
                target_language=sl.target_language,
                transfer_entropy=sl.transfer_entropy,
                normalized_te=sl.normalized_te,
                lag_minutes=sl.lag_minutes,
                confidence=sl.confidence,
                direction=sl.direction,
                predictive_power=sl.predictive_power,
            )
        return CausalNetworkResponse(
            links=[CausalLinkResponse(
                source_language=l.source_language,
                target_language=l.target_language,
                transfer_entropy=l.transfer_entropy,
                normalized_te=l.normalized_te,
                lag_minutes=l.lag_minutes,
                confidence=l.confidence,
                direction=l.direction,
                predictive_power=l.predictive_power,
            ) for l in network.links],
            strongest_link=strongest,
            hub_language=network.hub_language,
            hub_score=network.hub_score,
            network_density=network.network_density,
            warnings=network.warnings,
        )
    except Exception as e:
        logger.exception("Error in causal_network")
        raise HTTPException(status_code=500, detail=str(e))


# ── Gap 14: MDP Model Routing ──

@router.post("/route", response_model=RoutingResponse, tags=["Gap 14 — MDP Routing"])
async def route_model(request: RoutingRequest, engine=Depends(get_engine)):
    """Get optimal model recommendation via MDP Q-learning policy."""
    try:
        decision = await engine.route_model(
            language=request.language,
            text_type=request.text_type,
            entropy_level=request.entropy_level,
            time_bucket=request.time_bucket,
            token_volume=request.token_volume,
        )
        return RoutingResponse(
            recommended_model=decision.recommended_model,
            expected_cost_per_1k=decision.expected_cost_per_1k,
            expected_quality_score=decision.expected_quality_score,
            confidence=decision.confidence,
            alternatives=[RoutingAlternative(**a) for a in decision.alternatives],
            policy_iteration=decision.policy_iteration,
            exploration_rate=decision.exploration_rate,
        )
    except Exception as e:
        logger.exception("Error in route_model")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/route/policy", response_model=RoutingPolicyResponse, tags=["Gap 14 — MDP Routing"])
async def routing_policy(engine=Depends(get_engine)):
    """Get the learned MDP routing policy summary."""
    try:
        summary = await engine.get_routing_policy()
        return RoutingPolicyResponse(**summary)
    except Exception as e:
        logger.exception("Error in routing_policy")
        raise HTTPException(status_code=500, detail=str(e))


# ── Gap 16: Adversarial Probing ──

@router.post("/adversarial", response_model=AdversarialResponse, tags=["Gap 16 — Adversarial Probing"])
async def adversarial_probe(request: AdversarialRequest, engine=Depends(get_engine)):
    """Run adversarial tokenizer probing to discover vulnerabilities."""
    try:
        report = await engine.probe_adversarial(request.text, request.model_name)
        return AdversarialResponse(
            results=[AdversarialResultResponse(
                original_tokens=r.original_tokens,
                adversarial_tokens=r.adversarial_tokens,
                token_inflation_pct=round(r.token_inflation_pct, 2),
                cost_inflation_pct=round(r.cost_inflation_pct, 2),
                vulnerability_type=r.vulnerability_type,
                severity=r.severity,
                explanation=r.explanation,
                mitigation=r.mitigation,
            ) for r in report.results],
            worst_case_inflation=round(report.worst_case_inflation, 2),
            average_inflation=round(report.average_inflation, 2),
            critical_count=report.critical_count,
            high_count=report.high_count,
            model_vulnerability_score=round(report.model_vulnerability_score, 2),
            recommendations=report.recommendations,
        )
    except Exception as e:
        logger.exception("Error in adversarial_probe")
        raise HTTPException(status_code=500, detail=str(e))


# ---- Helper ----

def _arb_to_response(arb, current_model: str) -> ArbitrageResponse:
    current_cost = arb.results.get(current_model, {}).get("cost_usd", 0)
    model_results = []
    for model_name, data in arb.results.items():
        savings_pct = 0.0
        if current_cost > 0:
            savings_pct = (current_cost - data["cost_usd"]) / current_cost * 100
        model_results.append(ArbitrageModelResult(
            model_name=model_name,
            token_count=data.get("tokens", 0),
            cost_usd=round(data.get("cost_usd", 0), 8),
            etr_score=round(data.get("etr", 0), 4),
            etr_inequity_ratio=round(data.get("inequity", 0), 3),
            savings_vs_current_pct=round(savings_pct, 2),
        ))
    model_results.sort(key=lambda x: x.cost_usd)
    return ArbitrageResponse(
        language=arb.language,
        recommended_model=arb.recommended_model,
        max_savings_pct=round(arb.max_savings_pct, 2),
        savings_summary=arb.savings_summary,
        results=model_results,
    )


# ══════════════════════════════════════════════════════════
# PLATFORM FEATURES — Enterprise Moat
# ══════════════════════════════════════════════════════════


# ── Feature 1: Live Token Verification ──

@router.post("/platform/verify", response_model=TokenVerifyResponse, tags=["Platform — Token Verification"])
async def verify_tokens(request: TokenVerifyRequest, engine=Depends(get_engine)):
    """Independently verify token count for any AI API request."""
    try:
        result = engine.token_verifier.verify(request.text, request.model_name)
        return TokenVerifyResponse(
            text_preview=result.text[:100],
            model=result.model,
            provider=result.provider,
            measured_tokens=result.measured_tokens,
            estimated_cost_usd=round(result.estimated_cost_usd, 8),
            verification_method=result.verification_method,
            confidence=round(result.confidence, 4),
        )
    except Exception as e:
        logger.exception("Error in verify_tokens")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/platform/verify/compare", response_model=ProviderComparisonResponse, tags=["Platform — Token Verification"])
async def compare_providers(request: TokenVerifyRequest, engine=Depends(get_engine)):
    """Compare token costs across all providers for the same text."""
    try:
        result = engine.token_verifier.compare_providers(request.text, request.model_name)
        return ProviderComparisonResponse(
            current_model=result.current_model,
            current_tokens=result.current_tokens,
            current_cost=round(result.current_cost, 8),
            alternatives=result.alternatives,
            best_alternative=result.best_alternative,
            max_savings_pct=round(result.max_savings_pct, 2),
            recommendation=result.recommendation,
        )
    except Exception as e:
        logger.exception("Error in compare_providers")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 2: Shadow Bill Reconciliation ──

@router.post("/platform/billing/record", response_model=BillingRecordResponse, tags=["Platform — Shadow Billing"])
async def record_billing(request: BillingRecordRequest, engine=Depends(get_engine)):
    """Record a billed request for shadow reconciliation."""
    try:
        result = engine.billing_reconciler.record(
            model=request.model,
            input_text=request.input_text,
            output_text=request.output_text,
            billed_input_tokens=request.billed_input_tokens,
            billed_output_tokens=request.billed_output_tokens,
            billed_cost=request.billed_cost,
            team=request.team,
            feature=request.feature,
            user_id=request.user_id,
        )
        return BillingRecordResponse(
            request_id=result.request_id,
            model=result.model,
            input_tokens_measured=result.input_tokens_measured,
            output_tokens_measured=result.output_tokens_measured,
            input_tokens_billed=result.input_tokens_billed,
            output_tokens_billed=result.output_tokens_billed,
            cost_measured=round(result.cost_measured, 8),
            cost_billed=round(result.cost_billed, 8),
            discrepancy_tokens=result.discrepancy_tokens,
            discrepancy_cost=round(result.discrepancy_cost, 8),
        )
    except Exception as e:
        logger.exception("Error in record_billing")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/billing/reconcile", response_model=BillingReconciliationResponse, tags=["Platform — Shadow Billing"])
async def reconcile_billing(engine=Depends(get_engine)):
    """Run shadow bill reconciliation across all recorded requests."""
    try:
        result = engine.billing_reconciler.reconcile()
        return BillingReconciliationResponse(
            period_start=result.period_start,
            period_end=result.period_end,
            total_requests=result.total_requests,
            total_tokens_measured=result.total_tokens_measured,
            total_tokens_billed=result.total_tokens_billed,
            total_cost_measured=round(result.total_cost_measured, 6),
            total_cost_billed=round(result.total_cost_billed, 6),
            discrepancy_tokens=result.discrepancy_tokens,
            discrepancy_cost=round(result.discrepancy_cost, 6),
            discrepancy_pct=round(result.discrepancy_pct, 2),
            overcharge_requests=result.overcharge_requests,
            undercharge_requests=result.undercharge_requests,
            by_model=result.by_model,
            by_team=result.by_team,
            recommendations=result.recommendations,
        )
    except Exception as e:
        logger.exception("Error in reconcile_billing")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 3: Real-Time Cost Stream ──

@router.post("/platform/cost/record", response_model=CostEventResponse, tags=["Platform — Cost Stream"])
async def record_cost_event(request: CostEventRequest, engine=Depends(get_engine)):
    """Record a cost event in the real-time stream."""
    try:
        event = engine.cost_stream.record(
            request_id=str(uuid.uuid4()),
            model=request.model,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            cost_usd=request.cost_usd,
            language=request.language,
            team=request.team,
            feature=request.feature,
        )
        return CostEventResponse(
            request_id=event.request_id,
            timestamp=event.timestamp,
            model=event.model,
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
            cost_usd=round(event.cost_usd, 8),
            cumulative_cost_today=round(event.cumulative_cost_today, 6),
            velocity_per_minute=round(event.velocity_per_minute, 6),
        )
    except Exception as e:
        logger.exception("Error in record_cost_event")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/cost/summary", response_model=CostSummaryResponse, tags=["Platform — Cost Stream"])
async def cost_summary(engine=Depends(get_engine)):
    """Get real-time cost stream summary with velocity and alerts."""
    try:
        summary = engine.cost_stream.get_summary()
        hourly = engine.cost_stream.get_hourly_breakdown()
        return CostSummaryResponse(
            total_cost_today=round(summary.get("today_total", 0), 6),
            total_requests_today=summary.get("total_requests", 0),
            total_tokens_today=summary.get("total_tokens", 0),
            velocity_per_minute=round(summary.get("velocity_per_minute", 0), 6),
            daily_budget=engine.cost_stream._daily_budget if hasattr(engine.cost_stream, '_daily_budget') else summary.get("daily_budget", 1000.0),
            budget_remaining=round(summary.get("budget_remaining", 1000.0), 6),
            budget_pct_used=round(summary.get("budget_percent", 0), 2),
            hourly_breakdown=hourly,
            active_alerts=summary.get("alerts", []),
        )
    except Exception as e:
        logger.exception("Error in cost_summary")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 4: Semantic Prompt Cache ──

@router.post("/platform/cache/lookup", response_model=CacheLookupResponse, tags=["Platform — Prompt Cache"])
async def cache_lookup(request: CacheLookupRequest, engine=Depends(get_engine)):
    """Look up a prompt in the semantic cache."""
    try:
        result = engine.prompt_cache.lookup(request.prompt, request.model)
        return CacheLookupResponse(
            hit=result.hit,
            method=result.method,
            similarity=round(result.similarity, 4) if result.similarity else 0.0,
            tokens_saved=result.tokens_saved,
            cost_saved=round(result.cost_saved, 8) if result.cost_saved else 0.0,
            cached_response=result.entry.response_text if result.hit and result.entry else None,
        )
    except Exception as e:
        logger.exception("Error in cache_lookup")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/platform/cache/store", tags=["Platform — Prompt Cache"])
async def cache_store(request: CacheStoreRequest, engine=Depends(get_engine)):
    """Store a prompt-response pair in the semantic cache."""
    try:
        engine.prompt_cache.store(
            prompt=request.prompt,
            response=request.response,
            model=request.model,
            tokens_used=request.tokens_used,
            cost=request.cost,
        )
        return {"status": "stored", "cache_size": engine.prompt_cache.get_stats().cache_size}
    except Exception as e:
        logger.exception("Error in cache_store")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/cache/stats", response_model=CacheStatsResponse, tags=["Platform — Prompt Cache"])
async def cache_stats(engine=Depends(get_engine)):
    """Get cache performance statistics."""
    try:
        stats = engine.prompt_cache.get_stats()
        duplicates = engine.prompt_cache.get_top_duplicates(10)
        return CacheStatsResponse(
            total_queries=stats.total_queries,
            exact_hits=stats.exact_hits,
            semantic_hits=stats.semantic_hits,
            misses=stats.misses,
            hit_rate=round(stats.hit_rate, 4),
            total_tokens_saved=stats.total_tokens_saved,
            total_cost_saved=round(stats.total_cost_saved, 6),
            cache_size=stats.cache_size,
            avg_similarity_on_hit=round(stats.avg_similarity_on_hit, 4),
            top_duplicates=duplicates,
        )
    except Exception as e:
        logger.exception("Error in cache_stats")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 5: Semantic Deduplication ──

@router.post("/platform/dedup/analyze", response_model=DedupResponse, tags=["Platform — Deduplication"])
async def analyze_dedup(request: DedupRequest, engine=Depends(get_engine)):
    """Analyze a request for semantic redundancy."""
    try:
        result = engine.deduplicator.analyze_request(request.text, request.session_id)
        return DedupResponse(
            original_tokens=result.original_tokens,
            unique_tokens=result.unique_tokens,
            redundant_tokens=result.redundant_tokens,
            redundancy_pct=round(result.redundancy_pct, 2),
            estimated_savings=round(result.estimated_savings, 6),
        )
    except Exception as e:
        logger.exception("Error in analyze_dedup")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/platform/dedup/conversation", response_model=ConversationDedupResponse, tags=["Platform — Deduplication"])
async def analyze_conversation_dedup(request: ConversationDedupRequest, engine=Depends(get_engine)):
    """Analyze multi-turn conversation for cumulative waste."""
    try:
        analysis = engine.deduplicator.analyze_conversation(
            request.messages, request.model, request.cost_per_1m_tokens,
        )
        optimization = engine.deduplicator.suggest_optimization(analysis)
        return ConversationDedupResponse(
            total_turns=analysis.total_turns,
            total_tokens_sent=analysis.total_tokens_sent,
            unique_tokens=analysis.unique_tokens,
            cumulative_waste_tokens=analysis.cumulative_waste_tokens,
            cumulative_waste_pct=round(analysis.cumulative_waste_pct, 2),
            cumulative_waste_cost=round(analysis.cumulative_waste_cost, 6),
            recommendation=analysis.recommendation,
            optimization=optimization,
        )
    except Exception as e:
        logger.exception("Error in analyze_conversation_dedup")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 6: Provider Health Monitor ──

@router.post("/platform/health/record", tags=["Platform — Health Monitor"])
async def record_health_probe(request: HealthRecordRequest, engine=Depends(get_engine)):
    """Record a health observation from a production request."""
    try:
        engine.provider_health.record_request(
            provider=request.provider,
            model=request.model,
            latency_ms=request.latency_ms,
            success=request.success,
            error_msg=request.error_msg,
            tokens_generated=request.tokens_generated,
            generation_time_ms=request.generation_time_ms,
        )
        return {"status": "recorded", "provider": request.provider}
    except Exception as e:
        logger.exception("Error in record_health_probe")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/health/status", response_model=ProviderStatusResponse, tags=["Platform — Health Monitor"])
async def provider_status(
    provider: str = Query(default=None),
    engine=Depends(get_engine),
):
    """Get current provider health status."""
    try:
        status = engine.provider_health.get_status(provider)
        return ProviderStatusResponse(providers=status)
    except Exception as e:
        logger.exception("Error in provider_status")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/health/sla/{provider}", response_model=ProviderSLAResponse, tags=["Platform — Health Monitor"])
async def provider_sla(
    provider: str,
    hours: int = Query(default=24, ge=1, le=720),
    engine=Depends(get_engine),
):
    """Get SLA compliance report for a provider."""
    try:
        report = engine.provider_health.get_sla_report(provider, hours)
        return ProviderSLAResponse(
            provider=report.provider,
            period_hours=report.period_hours,
            total_probes=report.total_probes,
            successful_probes=report.successful_probes,
            uptime_pct=round(report.uptime_pct, 4),
            avg_latency_ms=round(report.avg_latency_ms, 2),
            p50_latency_ms=round(report.p50_latency_ms, 2),
            p95_latency_ms=round(report.p95_latency_ms, 2),
            p99_latency_ms=round(report.p99_latency_ms, 2),
            error_rate_pct=round(report.error_rate_pct, 4),
            latency_trend=report.latency_trend,
            incidents=report.incidents,
        )
    except Exception as e:
        logger.exception("Error in provider_sla")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 7: Tokenizer Drift Detector ──

@router.post("/platform/drift/baseline", tags=["Platform — Drift Detector"])
async def establish_drift_baseline(
    model: str = Query(default="gpt-4o"),
    engine=Depends(get_engine),
):
    """Establish a tokenizer baseline for drift detection."""
    try:
        def count_fn(text):
            return engine.token_counter.count(text, model)
        baseline = engine.drift_detector.establish_baseline(model, count_fn)
        return {"status": "baseline_established", "model": model, "baseline": baseline}
    except Exception as e:
        logger.exception("Error in establish_drift_baseline")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/platform/drift/check", response_model=list[DriftReportResponse], tags=["Platform — Drift Detector"])
async def check_drift(request: DriftCheckRequest, engine=Depends(get_engine)):
    """Check for tokenizer drift against established baseline."""
    try:
        def count_fn(text):
            return engine.token_counter.count(text, request.model)
        reports = engine.drift_detector.check_drift(request.model, count_fn, request.monthly_spend)
        return [DriftReportResponse(
            model=r.model,
            drift_tokens=r.drift_tokens,
            drift_pct=round(r.drift_pct, 4),
            severity=r.severity,
            estimated_monthly_cost_impact=round(r.estimated_monthly_cost_impact, 2),
            details=r.details if hasattr(r, 'details') and r.details else [],
        ) for r in reports]
    except Exception as e:
        logger.exception("Error in check_drift")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/drift/history/{model}", response_model=DriftHistoryResponse, tags=["Platform — Drift Detector"])
async def drift_history(model: str, engine=Depends(get_engine)):
    """Get tokenizer drift history for a model."""
    try:
        history = engine.drift_detector.get_history(model)
        return DriftHistoryResponse(
            model=history.model,
            snapshots=history.snapshots,
            total_drift_pct=round(history.total_drift_pct, 4),
            drift_direction=history.drift_direction,
        )
    except Exception as e:
        logger.exception("Error in drift_history")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 8: Cost Attribution ──

@router.post("/platform/attribution/record", tags=["Platform — Cost Attribution"])
async def record_attribution(request: AttributionRecordRequest, engine=Depends(get_engine)):
    """Record a cost-attributed request."""
    try:
        tags = AttributionTag(
            team=request.team,
            feature=request.feature,
            user_id=request.user_id,
            environment=request.environment,
        )
        entry = engine.cost_attribution.record(
            request_id=str(uuid.uuid4()),
            model=request.model,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            cost_usd=request.cost_usd,
            tags=tags,
            language=request.language,
        )
        return {"status": "recorded", "request_id": entry.request_id, "cost": round(entry.cost_usd, 8)}
    except Exception as e:
        logger.exception("Error in record_attribution")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/attribution/breakdown/{dimension}", response_model=AttributionBreakdownResponse, tags=["Platform — Cost Attribution"])
async def attribution_breakdown(dimension: str, engine=Depends(get_engine)):
    """Get cost breakdown by dimension (team, feature, user_id, model, language)."""
    try:
        if dimension not in ("team", "feature", "user_id", "model", "language"):
            raise HTTPException(status_code=400, detail="Invalid dimension")
        breakdown = engine.cost_attribution.get_breakdown(dimension)
        return AttributionBreakdownResponse(
            dimension=breakdown.dimension,
            period_start=breakdown.period_start,
            period_end=breakdown.period_end,
            total_cost=round(breakdown.total_cost, 6),
            total_requests=breakdown.total_requests,
            total_tokens=breakdown.total_tokens,
            breakdown=breakdown.breakdown,
            top_spender=breakdown.top_spender,
            top_spender_pct=round(breakdown.top_spender_pct, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in attribution_breakdown")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/platform/attribution/budget", tags=["Platform — Cost Attribution"])
async def set_budget(request: SetBudgetRequest, engine=Depends(get_engine)):
    """Set monthly budget for a team."""
    try:
        engine.cost_attribution.set_budget(request.team, request.monthly_budget)
        return {"status": "budget_set", "team": request.team, "monthly_budget": request.monthly_budget}
    except Exception as e:
        logger.exception("Error in set_budget")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/attribution/budget/{team}", response_model=BudgetStatusResponse, tags=["Platform — Cost Attribution"])
async def budget_status(team: str, engine=Depends(get_engine)):
    """Get budget status for a team."""
    try:
        status = engine.cost_attribution.get_team_budget_status(team)
        return BudgetStatusResponse(
            team=team,
            monthly_budget=status.get("budget", 0),
            spent=round(status.get("spent", 0), 6),
            remaining=round(status.get("remaining", 0), 6),
            pct_used=round(status.get("pct_used", 0), 2),
            projected_monthly=round(status.get("projected_monthly", 0), 6),
            on_track=status.get("on_track", True),
        )
    except Exception as e:
        logger.exception("Error in budget_status")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/attribution/top", tags=["Platform — Cost Attribution"])
async def top_consumers(n: int = 5, dimension: str = "user_id", engine=Depends(get_engine)):
    """Get top cost consumers by dimension."""
    try:
        top = engine.cost_attribution.get_top_consumers(n=n, dimension=dimension)
        return top
    except Exception as e:
        logger.exception("Error in top_consumers")
        raise HTTPException(status_code=500, detail=str(e))


# ── Feature 9: Compliance Audit Trail ──

@router.post("/platform/audit/record", response_model=AuditRecordResponse, tags=["Platform — Audit Trail"])
async def record_audit(request: AuditRecordRequest, engine=Depends(get_engine)):
    """Record an immutable audit entry with cryptographic hash chain."""
    try:
        record = engine.audit_trail.record(
            model=request.model,
            provider=request.provider,
            input_text=request.input_text,
            output_text=request.output_text,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            cost_usd=request.cost_usd,
            team=request.team,
            feature=request.feature,
            user_id=request.user_id,
            routing_reason=request.routing_reason,
        )
        return AuditRecordResponse(
            record_id=record.record_id,
            timestamp=record.timestamp,
            input_hash=record.input_hash,
            output_hash=record.output_hash,
            record_hash=record.record_hash,
            merkle_root=record.merkle_root,
        )
    except Exception as e:
        logger.exception("Error in record_audit")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/audit/integrity", response_model=AuditIntegrityResponse, tags=["Platform — Audit Trail"])
async def verify_audit_integrity(engine=Depends(get_engine)):
    """Verify cryptographic integrity of the audit chain."""
    try:
        valid, message = engine.audit_trail.verify_integrity()
        return AuditIntegrityResponse(
            valid=valid,
            message=message,
            total_records=engine.audit_trail.get_record_count(),
            merkle_root=engine.audit_trail.get_latest_merkle_root(),
        )
    except Exception as e:
        logger.exception("Error in verify_audit_integrity")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/platform/audit/report", response_model=AuditReportResponse, tags=["Platform — Audit Trail"])
async def audit_report(engine=Depends(get_engine)):
    """Generate compliance report for auditors."""
    try:
        report = engine.audit_trail.generate_report()
        return AuditReportResponse(
            period_start=report.period_start,
            period_end=report.period_end,
            total_records=report.total_records,
            total_cost=round(report.total_cost, 6),
            chain_integrity=report.chain_integrity,
            merkle_root=report.merkle_root,
            by_model=report.by_model,
            by_team=report.by_team,
            by_feature=report.by_feature,
            compliance_notes=report.compliance_notes,
        )
    except Exception as e:
        logger.exception("Error in audit_report")
        raise HTTPException(status_code=500, detail=str(e))
