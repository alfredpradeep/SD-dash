"""
SHIELD API Schemas.

Pydantic models for all API request/response types.
Supports: batch audit, real-time monitoring, simulator mode,
          bias/hallucination/compliance detection, industry profiles.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════

class ScanDepth(str, Enum):
    """Scan depth levels."""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class SimulatorMode(str, Enum):
    """Simulator modes."""
    DISABLED = "disabled"
    DEMO = "demo"
    BENCHMARK = "benchmark"


# ═══════════════════════════════════════════════════════════════════
# SCAN REQUESTS
# ═══════════════════════════════════════════════════════════════════

class ScanRequest(BaseModel):
    """Full scan request — supports batch audit + simulator modes."""

    # Target endpoint config
    target_endpoint: str = Field(
        ...,
        description="LLM endpoint URL to test (OpenAI-compatible chat/completions)"
    )
    target_api_key: Optional[str] = Field(
        None,
        description="API key for the target endpoint (Bearer token)"
    )
    target_model: Optional[str] = Field(
        None,
        description="Model name at the target endpoint (e.g. 'llama-3.1-8b-instant')"
    )
    architecture: str = Field(
        ...,
        description="Model architecture (e.g., 'translate_sandwich', 'native_multilingual')"
    )

    # Scan scope
    languages: List[str] = Field(
        default=["english"],
        description="Languages to test"
    )
    categories: List[str] = Field(
        default=["financial", "medical", "violence", "pii", "legal"],
        description="Harm categories to test"
    )
    scan_depth: ScanDepth = Field(
        default=ScanDepth.STANDARD,
        description="Scan depth"
    )

    # Detection scope — what to analyze
    detection_scope: List[str] = Field(
        default=["safety"],
        description="What to detect: safety|bias|hallucination|compliance"
    )

    # Industry context
    target_industry: Optional[str] = Field(
        None,
        description="Industry: ai_interviewing|banking|healthcare|appointment_booking|ecommerce|customer_support|insurance"
    )

    # Simulator mode
    simulator_mode: SimulatorMode = Field(
        default=SimulatorMode.DISABLED,
        description="disabled=real endpoint, demo=lightweight synthetic, benchmark=calibrated synthetic"
    )
    simulator_model_profile: Optional[str] = Field(
        None,
        description="Model to simulate: gpt-4o|claude-3-sonnet|llama-3.1-8b|mistral-7b|gemini-1.5-flash"
    )

    # Advanced settings
    pair_iterations: int = Field(
        default=3, ge=1, le=10,
        description="PAIR attack iterations"
    )
    semantic_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Semantic harm detection threshold"
    )
    smoothing_samples: int = Field(
        default=5, ge=1, le=20,
        description="Paraphrase samples for smoothing"
    )
    gcg_enabled: bool = Field(
        default=False,
        description="Enable gradient-based suffix generation"
    )

    # Compliance config
    compliance_regulations: List[str] = Field(
        default=[],
        description="Regulations to check: eeoc|hipaa|fcra|gdpr|ccpa|ada"
    )

    # Bias config
    bias_demographic_groups: List[str] = Field(
        default=["gender", "race_ethnicity", "age"],
        description="Demographic groups to test for bias"
    )

    custom_prompts: Optional[List[str]] = Field(
        None,
        description="Custom attack prompts to include"
    )
    translation_endpoint: Optional[str] = Field(
        None,
        description="Translation service endpoint for cross-lingual testing"
    )


class QuickScanRequest(BaseModel):
    """Quick scan request (reduced probe count)."""

    target_endpoint: str
    target_api_key: Optional[str] = Field(None)
    target_model: Optional[str] = Field(None)
    architecture: str
    languages: List[str] = Field(default=["english"])
    categories: List[str] = Field(default=["financial", "medical", "violence", "pii", "legal"])
    target_industry: Optional[str] = Field(None)
    simulator_mode: SimulatorMode = Field(default=SimulatorMode.DISABLED)
    simulator_model_profile: Optional[str] = Field(None)
    detection_scope: List[str] = Field(default=["safety"])


class SingleProbeRequest(BaseModel):
    """Single probe test request."""

    target_endpoint: str
    target_api_key: Optional[str] = Field(None)
    target_model: Optional[str] = Field(None)
    architecture: str
    probe_text: str
    language: str = Field(default="english")
    category: str = Field(default="financial")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    simulator_mode: SimulatorMode = Field(default=SimulatorMode.DISABLED)
    simulator_model_profile: Optional[str] = Field(None)


# ═══════════════════════════════════════════════════════════════════
# MONITORING REQUESTS
# ═══════════════════════════════════════════════════════════════════

class MonitorStartRequest(BaseModel):
    """Start a real-time monitoring session."""
    target_endpoint: str
    target_api_key: Optional[str] = Field(None)
    target_model: Optional[str] = Field(None)
    model_name: str = Field(default="unknown")
    language: str = Field(default="english")
    industry: Optional[str] = Field(None)
    detection_scope: List[str] = Field(default=["safety", "bias", "hallucination", "compliance"])


class MonitorCheckRequest(BaseModel):
    """Check a single prompt/response pair in real-time."""
    session_id: str
    user_prompt: str
    model_response: str
    language: Optional[str] = Field(None, description="Override language detection")


class MonitorStopRequest(BaseModel):
    """Stop a monitoring session."""
    session_id: str


# ═══════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════

class ProbeResultResponse(BaseModel):
    """Single probe result."""
    probe_text: str
    response: str
    judgment: Dict[str, Any]
    harm_vector: Dict[str, float]
    processing_ms: int


class GapMatrixCell(BaseModel):
    """Gap matrix cell (language x category)."""
    language: str
    category: str
    bypass_rate: float = Field(ge=0.0, le=1.0)
    average_hasd_score: float = Field(ge=0.0, le=1.0)
    sample_count: int


class EngineResult(BaseModel):
    """Result from a SHIELD engine."""
    engine_id: int
    name: str
    status: str
    metrics: Dict[str, Any]
    errors: Optional[List[str]] = None


class BiasAnalysisResult(BaseModel):
    """Bias analysis for a demographic group."""
    demographic_group: str
    metric_name: str
    baseline_rate: float
    comparison_rate: float
    gap: float
    statistical_significance: float
    is_significant: bool
    risk_level: str


class HallucinationResult(BaseModel):
    """Hallucination detection result."""
    claim_text: str
    knowledge_source: str
    supported: bool
    confidence_score: float
    severity: str
    domain: Optional[str] = None


class ComplianceGap(BaseModel):
    """Compliance gap finding."""
    regulation: str
    requirement: str
    status: str  # compliant | gap | violation
    risk_level: str
    remediation: str
    language: Optional[str] = None
    category: Optional[str] = None


class RemediationRecommendation(BaseModel):
    """Remediation recommendation."""
    detection_improvement: str
    response_strategy: str
    training_gap: str
    architecture_change: str
    priority: str
    estimated_effort_hours: int


class RegulatoryReport(BaseModel):
    """Regulatory compliance report."""
    region: str
    regulation: str
    compliance_status: str
    findings: List[str]
    recommendations: List[str]


class BenchmarkComparison(BaseModel):
    """Comparison against benchmarks."""
    model: str
    language: str
    category: str
    current_bypass_rate: float
    baseline_bypass_rate: Optional[float] = None
    percentile: Optional[int] = None
    trend: Optional[str] = None


class ScanResponse(BaseModel):
    """Full scan response — includes all analysis types."""

    scan_id: str = Field(..., description="Unique scan identifier")
    status: str = Field(..., description="Scan status: completed|in_progress|failed")
    gap_matrix: List[GapMatrixCell] = Field(default=[], description="Language x category bypass rates")
    engine_results: List[EngineResult] = Field(default=[], description="Results from each SHIELD engine")
    remediation_roadmap: List[RemediationRecommendation] = Field(default=[], description="Recommended fixes")
    regulatory_report: Optional[RegulatoryReport] = None
    benchmark_comparison: Optional[BenchmarkComparison] = None
    processing_ms: int = Field(..., description="Total processing time")
    error: Optional[str] = None

    # Extended analysis
    bias_analysis: Optional[List[BiasAnalysisResult]] = None
    hallucination_analysis: Optional[List[HallucinationResult]] = None
    compliance_analysis: Optional[List[ComplianceGap]] = None

    # Metadata
    simulated: bool = Field(default=False, description="Whether this scan used simulator mode")
    simulator_model: Optional[str] = Field(None, description="Model profile used for simulation")
    target_industry: Optional[str] = Field(None, description="Industry context used")
    detection_scope: List[str] = Field(default=["safety"], description="What was detected")


class MonitorInteractionResponse(BaseModel):
    """Response for a real-time monitoring check."""
    interaction_id: str
    verdict: str  # SAFE | UNSAFE | RISKY
    confidence: float
    reasoning: str
    processing_ms: int
    bias_flags: List[Dict[str, Any]] = Field(default=[])
    hallucination_flags: List[Dict[str, Any]] = Field(default=[])
    compliance_flags: List[Dict[str, Any]] = Field(default=[])


class HealthResponseItem(BaseModel):
    """Health status for single component."""
    component: str
    healthy: bool
    status: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    overall_status: str
    timestamp: str
    components: List[HealthResponseItem]


# ═══════════════════════════════════════════════════════════════════
# INTERVIEW SIMULATION REQUESTS & RESPONSES
# ═══════════════════════════════════════════════════════════════════

class InterviewExchangeResponse(BaseModel):
    """Single exchange in a simulated interview."""
    turn_number: int
    timestamp_offset_seconds: int
    time_formatted: str
    interviewer_message: str
    candidate_response: str
    ai_system_response: Optional[str] = None
    safety_verdict: str  # SAFE | RISKY | UNSAFE
    confidence: float
    bias_flags: List[str] = []
    compliance_flags: List[str] = []
    reasoning: str
    category: str  # greeting | experience | technical | behavioral | culture | edge_case


class InterviewSimRequest(BaseModel):
    """Request to simulate an interview."""
    language: str = Field(default="english", description="Interview language: english|hindi|tamil|chinese")
    industry: str = Field(default="ai_interviewing", description="Industry context")
    duration_minutes: int = Field(default=20, ge=10, le=60, description="Interview duration")
    candidate_name: Optional[str] = Field(None, description="Optional candidate name")
    role: str = Field(default="Senior Software Engineer", description="Job role")


class InterviewSimResponse(BaseModel):
    """Response from interview simulation."""
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]
    exchanges: List[InterviewExchangeResponse]
