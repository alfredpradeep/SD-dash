-- =============================================================================
-- LENS ClickHouse Schema
-- Stores entropy profiles, spike predictions, model arbitrage results,
-- and prediction outcomes for recalibration.
-- =============================================================================

-- Main entropy profiles table (time-series, partitioned by month)
CREATE TABLE IF NOT EXISTS lens_entropy_profiles
(
    timestamp               DateTime64(3),
    request_id              String,
    customer_id             String,
    language                LowCardinality(String),
    model_name              LowCardinality(String),
    token_count             UInt32,
    baseline_tokens         UInt32,
    efficiency_ratio        Float32,
    total_entropy_bits      Float32,
    mean_entropy            Float32,
    ids_score               Float32,
    etr_score               Float32,
    etr_english             Float32,
    etr_inequity_ratio      Float32,
    -- Gap 2: Semantic entropy
    semantic_entropy_bits   Float32   DEFAULT 0.0,
    semantic_ids_score      Float32   DEFAULT 0.0,
    waste_type              LowCardinality(String) DEFAULT 'unknown',
    -- Cost
    cost_usd                Float64,
    waste_cost_usd          Float64,
    -- COMPRESS bridge
    low_ids_alert           UInt8     DEFAULT 0,
    -- Equity
    equity_multiplier       Float32   DEFAULT 1.0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (customer_id, language, timestamp)
TTL timestamp + INTERVAL 12 MONTH
SETTINGS index_granularity = 8192;

-- Spike predictions log
CREATE TABLE IF NOT EXISTS lens_spike_predictions
(
    timestamp                   DateTime64(3),
    prediction_id               String,
    customer_id                 String,
    language                    LowCardinality(String),
    entropy_velocity            Float32,
    cusum_statistic             Float32   DEFAULT 0.0,
    predicted_token_increase    Float32,
    confidence                  Float32,
    horizon_minutes             UInt16,
    alert_level                 LowCardinality(String),
    spike_cause                 LowCardinality(String)  DEFAULT 'unknown',
    is_likely_temporary         Nullable(UInt8),
    trigger_reason              String
)
ENGINE = MergeTree()
ORDER BY (customer_id, timestamp)
TTL timestamp + INTERVAL 3 MONTH;

-- Prediction outcomes (for recalibration)
CREATE TABLE IF NOT EXISTS lens_prediction_outcomes
(
    prediction_id               String,
    language                    LowCardinality(String),
    predicted_at                DateTime64(3),
    predicted_increase_pct      Float32,
    confidence                  Float32,
    actual_increase_pct         Nullable(Float32),
    was_accurate                Nullable(UInt8),
    outcome_recorded_at         Nullable(DateTime64(3))
)
ENGINE = MergeTree()
ORDER BY (language, predicted_at)
TTL predicted_at + INTERVAL 30 DAY;

-- Model arbitrage results
CREATE TABLE IF NOT EXISTS lens_arbitrage_results
(
    timestamp                   DateTime64(3),
    customer_id                 String,
    language                    LowCardinality(String),
    text_hash                   String,
    recommended_model           LowCardinality(String),
    max_savings_pct             Float32,
    current_model               LowCardinality(String),
    current_tokens              UInt32,
    recommended_tokens          UInt32
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (customer_id, language, timestamp)
TTL timestamp + INTERVAL 3 MONTH;

-- =============================================================================
-- MATERIALIZED VIEWS (pre-aggregated for fast dashboard queries)
-- =============================================================================

-- Hourly cost aggregates per customer/language
CREATE MATERIALIZED VIEW IF NOT EXISTS lens_hourly_agg
ENGINE = SummingMergeTree()
ORDER BY (customer_id, language, model_name, hour)
POPULATE
AS SELECT
    customer_id,
    language,
    model_name,
    toStartOfHour(timestamp)    AS hour,
    sum(token_count)            AS total_tokens,
    sum(cost_usd)               AS total_cost,
    sum(waste_cost_usd)         AS total_waste,
    avg(ids_score)              AS avg_ids,
    avg(etr_score)              AS avg_etr,
    avg(etr_inequity_ratio)     AS avg_etr_inequity,
    avg(efficiency_ratio)       AS avg_efficiency,
    avg(semantic_entropy_bits)  AS avg_semantic_entropy,
    countIf(low_ids_alert = 1)  AS low_ids_alert_count,
    count()                     AS request_count
FROM lens_entropy_profiles
GROUP BY customer_id, language, model_name, hour;

-- Daily equity report per language
CREATE MATERIALIZED VIEW IF NOT EXISTS lens_daily_equity
ENGINE = SummingMergeTree()
ORDER BY (customer_id, language, day)
POPULATE
AS SELECT
    customer_id,
    language,
    toDate(timestamp)           AS day,
    avg(etr_inequity_ratio)     AS avg_inequity_ratio,
    sum(waste_cost_usd)         AS total_waste_cost,
    sum(cost_usd)               AS total_cost,
    count()                     AS requests
FROM lens_entropy_profiles
GROUP BY customer_id, language, day;

-- Waste type distribution (for 2D waste matrix reporting)
CREATE MATERIALIZED VIEW IF NOT EXISTS lens_waste_type_dist
ENGINE = SummingMergeTree()
ORDER BY (customer_id, language, waste_type, hour)
POPULATE
AS SELECT
    customer_id,
    language,
    waste_type,
    toStartOfHour(timestamp)    AS hour,
    count()                     AS request_count,
    sum(cost_usd)               AS total_cost,
    sum(waste_cost_usd)         AS total_waste_cost
FROM lens_entropy_profiles
GROUP BY customer_id, language, waste_type, hour;
