from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class TokenEntropyRecord:
    """Entropy measurement for a single token."""
    token_id: int
    token_text: str
    token_bytes: bytes
    character_count: int
    surprisal_bits: float           # -log2(P(token | context)) — higher = more informative
    is_high_entropy: bool           # True if surprisal > threshold
    position: int                   # Position in sequence
    token_type: str = "normal"      # "normal" | "fragment" | "oov" | "entity" | "codesw"


@dataclass
class SemanticEntropyWindow:
    """Semantic entropy measurement for a sliding sentence window (Gap 2 fix)."""
    window_text: str
    embedding_variance: float       # Variance of sentence embeddings in window
    semantic_entropy_bits: float    # Derived semantic entropy
    is_high_semantic_entropy: bool


@dataclass
class EntropyProfile:
    """Complete entropy analysis of an input text — lexical + semantic (2D waste matrix)."""
    text: str
    language: str
    model_name: str

    # Token metrics
    token_count: int
    english_baseline_tokens: int    # Token count of English translation equivalent
    efficiency_ratio: float         # token_count / english_baseline_tokens

    # Lexical entropy metrics (Layer 1 / Layer 2)
    total_entropy_bits: float       # Sum of all token surprisal values
    mean_entropy_per_token: float   # total_entropy / token_count

    # IDS: Information Density Score
    ids_score: float                # mean entropy per token (language-normalised)

    # ETR: Entropy-Token Ratio
    etr_score: float                # linguistic_entropy / token_count
    etr_english_baseline: float     # ETR for equivalent English content
    etr_inequity_ratio: float       # etr_english / etr_language — higher = more inequitable

    # Semantic entropy (Gap 2 fix — 2nd dimension of waste matrix)
    semantic_entropy_bits: float = 0.0
    semantic_ids_score: float = 0.0             # Semantic info per token
    waste_type: str = "unknown"                 # "lexical" | "semantic" | "combined" | "efficient"

    # Cost attribution
    cost_usd: float = 0.0
    waste_cost_usd: float = 0.0     # Cost attributable to tokenization inefficiency

    # COMPRESS bridge signal (Addition)
    low_ids_alert: bool = False
    low_ids_token_spans: list = field(default_factory=list)  # list of (start, end, ids) tuples

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = ""
    customer_id: str = ""
    token_records: list = field(default_factory=list)   # list[TokenEntropyRecord]
    semantic_windows: list = field(default_factory=list)  # list[SemanticEntropyWindow]


@dataclass
class SpikePrediction:
    """Predicted token cost spike for a language."""
    language: str
    current_entropy_velocity: float     # Rate of entropy change (bits/min)
    predicted_token_increase_pct: float # Estimated % token increase
    confidence: float                   # 0.0–1.0
    horizon_minutes: int                # How many minutes until spike materialises
    trigger_reason: str                 # Human-readable explanation
    alert_level: str                    # "watch" | "warning" | "alert" | "critical"

    # Gap 5 fix — cause attribution
    spike_cause: str = "unknown"        # "new_named_entity" | "domain_term" | "codeswitching" | "event" | "unknown"
    high_surprisal_tokens: list = field(default_factory=list)  # list of token strings driving spike
    is_likely_temporary: Optional[bool] = None  # True for event-driven, False for vocab expansion


@dataclass
class LanguageCostReport:
    """Per-language cost attribution report."""
    language: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    total_waste_cost_usd: float
    mean_ids_score: float
    mean_etr_score: float
    etr_inequity_ratio: float       # vs English baseline
    efficiency_percentile: float    # Where this language sits vs all languages
    spike_events: int               # Number of spike predictions triggered
    recommendations: list = field(default_factory=list)

    # Equity framing (Gap 6 addition)
    effective_cost_per_info_unit: float = 0.0
    english_cost_per_info_unit: float = 0.0
    equity_multiplier: float = 1.0  # How many times more users of this language pay per info unit


@dataclass
class TokenizerArbitrageResult:
    """Cross-model tokenizer comparison for a single text (Gap 4 fix)."""
    text: str
    language: str
    results: dict = field(default_factory=dict)  # model_name -> {tokens, cost_usd, etr, inequity}
    recommended_model: str = ""
    max_savings_pct: float = 0.0
    savings_summary: str = ""


@dataclass
class PredictionOutcome:
    """Tracks a spike prediction vs actual outcome for recalibration (Gap 6 fix)."""
    prediction_id: str
    language: str
    predicted_at: datetime
    predicted_increase_pct: float
    confidence: float
    horizon_minutes: int
    actual_increase_pct: Optional[float] = None
    outcome_recorded_at: Optional[datetime] = None
    was_accurate: Optional[bool] = None     # True if actual within 20% of predicted
