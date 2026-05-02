"""
Pydantic request/response models for the LENS FastAPI layer.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ---- Requests ----

class ProfileRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000, description="Input text to profile")
    model_name: str = Field(default="gpt-4o", description="Target AI model name")
    customer_id: str = Field(default="default", description="Customer identifier for cost attribution")
    request_id: Optional[str] = Field(default=None, description="Optional idempotency key")
    language: Optional[str] = Field(default=None, description="Pre-detected language code (skip detection)")
    include_token_records: bool = Field(default=False, description="Include per-token entropy breakdown")
    include_arbitrage: bool = Field(default=False, description="Include cross-model tokenizer comparison")


class BatchProfileItem(BaseModel):
    text: str
    model: str = "gpt-4o"
    customer_id: str = "default"
    request_id: Optional[str] = None
    language: Optional[str] = None


class BatchProfileRequest(BaseModel):
    items: list[BatchProfileItem] = Field(..., max_length=500)
    include_arbitrage: bool = False


class CostReportRequest(BaseModel):
    language: str
    customer_id: str = "default"
    period_hours: int = Field(default=24, ge=1, le=720)


class ArbitrageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    language: Optional[str] = None
    customer_id: str = "default"


# ---- Responses ----

class TokenRecord(BaseModel):
    position: int
    token_text: str
    surprisal_bits: float
    is_high_entropy: bool
    token_type: str


class SemanticWindowRecord(BaseModel):
    window_text: str
    embedding_variance: float
    semantic_entropy_bits: float
    is_high_semantic_entropy: bool


class LowIDSSpan(BaseModel):
    start_token: int
    end_token: int
    ids_score: float


class ProfileResponse(BaseModel):
    request_id: str
    language: str
    model_name: str
    # Token metrics
    token_count: int
    baseline_tokens: int
    efficiency_ratio: float
    # Lexical entropy
    ids_score: float
    etr_score: float
    etr_inequity_ratio: float
    # Semantic entropy (Gap 2)
    semantic_entropy_bits: float
    waste_type: str
    # Cost
    cost_usd: float
    waste_cost_usd: float
    waste_pct: float
    # Equity
    equity_statement: Optional[str] = None
    # COMPRESS bridge signal
    low_ids_alert: bool
    low_ids_spans: list[LowIDSSpan] = []
    # Optional detailed breakdowns
    token_records: Optional[list[TokenRecord]] = None
    semantic_windows: Optional[list[SemanticWindowRecord]] = None
    # Arbitrage (when requested)
    arbitrage: Optional["ArbitrageResponse"] = None


class BatchProfileResponse(BaseModel):
    results: list[ProfileResponse]
    total_cost_usd: float
    total_waste_usd: float
    languages_detected: list[str]


class SpikePredictionResponse(BaseModel):
    language: str
    entropy_velocity: float
    predicted_token_increase_pct: float
    confidence: float
    horizon_minutes: int
    alert_level: str
    spike_cause: str
    is_likely_temporary: Optional[bool]
    trigger_reason: str


class ArbitrageModelResult(BaseModel):
    model_name: str
    token_count: int
    cost_usd: float
    etr_score: float
    etr_inequity_ratio: float
    savings_vs_current_pct: float


class ArbitrageResponse(BaseModel):
    language: str
    recommended_model: str
    max_savings_pct: float
    savings_summary: str
    results: list[ArbitrageModelResult]


class CostReportResponse(BaseModel):
    language: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    total_waste_cost_usd: float
    waste_pct: float
    mean_ids_score: float
    mean_etr_score: float
    etr_inequity_ratio: float
    efficiency_percentile: float
    spike_events: int
    equity_multiplier: float
    effective_cost_per_info_unit: float
    english_cost_per_info_unit: float
    recommendations: list[str]


class HealthResponse(BaseModel):
    status: str
    lang_detector: str
    token_counter: str
    clickhouse: str
    redis: str
    semantic_entropy: str = "unknown"


class AccuracyReportResponse(BaseModel):
    """Prediction outcome recalibration report (Gap 6)."""
    languages: dict[str, dict]
    total_predictions_tracked: int


# ── Gap 8: Compression ──

class CompressionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    model_name: str = "gpt-4o"

class CompressionResponse(BaseModel):
    raw_bytes: int
    zlib_bytes: int
    lzma_bytes: int
    brotli_bytes: Optional[int] = None
    best_compressed_bytes: int
    best_algorithm: str
    token_count: int
    compression_ratio: float
    information_density: float
    tokenizer_inflation: float
    objective_waste_pct: float
    kolmogorov_estimate_bits: float


# ── Gap 10: Wavelet ──

class WaveletScaleResponse(BaseModel):
    scale_index: int
    scale_name: str
    coefficients: list[float]
    energy: float
    entropy: float
    trend_direction: str
    trend_strength: float
    anomaly_detected: bool
    anomaly_score: float

class WaveletResponse(BaseModel):
    scales: list[WaveletScaleResponse]
    dominant_scale: str
    overall_trend: str
    multi_scale_entropy: float
    early_warning_score: float
    reconstruction: list[float]


# ── Gap 11: Transfer Entropy ──

class CausalLinkResponse(BaseModel):
    source_language: str
    target_language: str
    transfer_entropy: float
    normalized_te: float
    lag_minutes: int
    confidence: float
    direction: str
    predictive_power: float

class CausalNetworkResponse(BaseModel):
    links: list[CausalLinkResponse]
    strongest_link: Optional[CausalLinkResponse] = None
    hub_language: str
    hub_score: float
    network_density: float
    warnings: list[str]


# ── Gap 14: MDP Routing ──

class RoutingRequest(BaseModel):
    language: str = "en"
    text_type: str = Field(default="prose", description="code, prose, or mixed")
    entropy_level: str = Field(default="medium", description="low, medium, or high")
    time_bucket: str = Field(default="peak", description="peak, off-peak, or night")
    token_volume: str = Field(default="moderate", description="light, moderate, or heavy")

class RoutingAlternative(BaseModel):
    model: str
    cost: float
    quality: float
    q_value: float

class RoutingResponse(BaseModel):
    recommended_model: str
    expected_cost_per_1k: float
    expected_quality_score: float
    confidence: float
    alternatives: list[RoutingAlternative]
    policy_iteration: int
    exploration_rate: float

class RoutingPolicyResponse(BaseModel):
    policy: dict
    total_iterations: int
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    num_states: int


# ── Gap 16: Adversarial Probing ──

class AdversarialRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    model_name: str = "gpt-4o"

class AdversarialResultResponse(BaseModel):
    original_tokens: int
    adversarial_tokens: int
    token_inflation_pct: float
    cost_inflation_pct: float
    vulnerability_type: str
    severity: str
    explanation: str
    mitigation: str

class AdversarialResponse(BaseModel):
    results: list[AdversarialResultResponse]
    worst_case_inflation: float
    average_inflation: float
    critical_count: int
    high_count: int
    model_vulnerability_score: float
    recommendations: list[str]


# ══════════════════════════════════════════════
# Platform Feature Schemas
# ══════════════════════════════════════════════

# ── Feature 1: Token Verification ──

class TokenVerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)
    model_name: str = "gpt-4o"

class TokenVerifyResponse(BaseModel):
    text_preview: str
    model: str
    provider: str
    measured_tokens: int
    estimated_cost_usd: float
    verification_method: str
    confidence: float

class ProviderComparisonResponse(BaseModel):
    current_model: str
    current_tokens: int
    current_cost: float
    alternatives: list[dict]
    best_alternative: str
    max_savings_pct: float
    recommendation: str


# ── Feature 2: Shadow Billing ──

class BillingRecordRequest(BaseModel):
    model: str = "gpt-4o"
    input_text: str = Field(..., min_length=1)
    output_text: str = ""
    billed_input_tokens: int = 0
    billed_output_tokens: int = 0
    billed_cost: float = 0.0
    team: str = "default"
    feature: str = "default"
    user_id: str = "anonymous"

class BillingRecordResponse(BaseModel):
    request_id: str
    model: str
    input_tokens_measured: int
    output_tokens_measured: int
    input_tokens_billed: int
    output_tokens_billed: int
    cost_measured: float
    cost_billed: float
    discrepancy_tokens: int
    discrepancy_cost: float

class BillingReconciliationResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_tokens_measured: int
    total_tokens_billed: int
    total_cost_measured: float
    total_cost_billed: float
    discrepancy_tokens: int
    discrepancy_cost: float
    discrepancy_pct: float
    overcharge_requests: int
    undercharge_requests: int
    by_model: dict
    by_team: dict
    recommendations: list[str]


# ── Feature 3: Cost Stream ──

class CostEventRequest(BaseModel):
    model: str = "gpt-4o"
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(..., ge=0)
    language: str = "en"
    team: str = "default"
    feature: str = "default"

class CostEventResponse(BaseModel):
    request_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cumulative_cost_today: float
    velocity_per_minute: float

class CostSummaryResponse(BaseModel):
    total_cost_today: float
    total_requests_today: int
    total_tokens_today: int
    velocity_per_minute: float
    daily_budget: float
    budget_remaining: float
    budget_pct_used: float
    hourly_breakdown: list[dict]
    active_alerts: list[dict]


# ── Feature 4: Semantic Prompt Cache ──

class CacheLookupRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-4o"

class CacheLookupResponse(BaseModel):
    hit: bool
    method: Optional[str] = None
    similarity: float = 0.0
    tokens_saved: int = 0
    cost_saved: float = 0.0
    cached_response: Optional[str] = None

class CacheStoreRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    model: str = "gpt-4o"
    tokens_used: int = 0
    cost: float = 0.0

class CacheStatsResponse(BaseModel):
    total_queries: int
    exact_hits: int
    semantic_hits: int
    misses: int
    hit_rate: float
    total_tokens_saved: int
    total_cost_saved: float
    cache_size: int
    avg_similarity_on_hit: float
    top_duplicates: list[dict]


# ── Feature 5: Semantic Deduplication ──

class DedupRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: Optional[str] = None

class DedupResponse(BaseModel):
    original_tokens: int
    unique_tokens: int
    redundant_tokens: int
    redundancy_pct: float
    estimated_savings: float

class ConversationDedupRequest(BaseModel):
    messages: list[dict] = Field(..., min_length=1)
    model: str = "gpt-4o"
    cost_per_1m_tokens: float = 2.50

class ConversationDedupResponse(BaseModel):
    total_turns: int
    total_tokens_sent: int
    unique_tokens: int
    cumulative_waste_tokens: int
    cumulative_waste_pct: float
    cumulative_waste_cost: float
    recommendation: str
    optimization: dict


# ── Feature 6: Provider Health Monitor ──

class HealthRecordRequest(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    latency_ms: float = Field(..., ge=0)
    success: bool = True
    error_msg: str = ""
    tokens_generated: int = 0
    generation_time_ms: float = 0

class ProviderStatusResponse(BaseModel):
    providers: dict

class ProviderSLAResponse(BaseModel):
    provider: str
    period_hours: int
    total_probes: int
    successful_probes: int
    uptime_pct: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate_pct: float
    latency_trend: str
    incidents: list[dict]


# ── Feature 7: Tokenizer Drift Detector ──

class DriftCheckRequest(BaseModel):
    model: str = "gpt-4o"
    monthly_spend: float = 10000.0

class DriftReportResponse(BaseModel):
    model: str
    drift_tokens: int
    drift_pct: float
    severity: str
    estimated_monthly_cost_impact: float
    details: list[dict]

class DriftHistoryResponse(BaseModel):
    model: str
    snapshots: list[dict]
    total_drift_pct: float
    drift_direction: str


# ── Feature 8: Cost Attribution ──

class AttributionRecordRequest(BaseModel):
    model: str = "gpt-4o"
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(..., ge=0)
    team: str = "default"
    feature: str = "default"
    user_id: str = "anonymous"
    environment: str = "production"
    language: str = "en"

class AttributionBreakdownResponse(BaseModel):
    dimension: str
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_requests: int
    total_tokens: int
    breakdown: list
    top_spender: str
    top_spender_pct: float

class BudgetStatusResponse(BaseModel):
    team: str
    monthly_budget: float
    spent: float
    remaining: float
    pct_used: float
    projected_monthly: float
    on_track: bool

class SetBudgetRequest(BaseModel):
    team: str
    monthly_budget: float = Field(..., gt=0)


# ── Feature 9: Audit Trail ──

class AuditRecordRequest(BaseModel):
    model: str = "gpt-4o"
    provider: str = "openai"
    input_text: str = Field(..., min_length=1)
    output_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    team: str = "default"
    feature: str = "default"
    user_id: str = "anonymous"
    routing_reason: str = "default"

class AuditRecordResponse(BaseModel):
    record_id: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    record_hash: str
    merkle_root: str

class AuditIntegrityResponse(BaseModel):
    valid: bool
    message: str
    total_records: int
    merkle_root: str

class AuditReportResponse(BaseModel):
    period_start: datetime
    period_end: datetime
    total_records: int
    total_cost: float
    chain_integrity: bool
    merkle_root: str
    by_model: dict
    by_team: dict
    by_feature: dict
    compliance_notes: list[str]
