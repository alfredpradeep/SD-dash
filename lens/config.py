from pydantic_settings import BaseSettings
from typing import Optional


class Config(BaseSettings):
    # Runtime
    device: str = "cpu"
    log_level: str = "INFO"

    # Spike prediction
    spike_alert_confidence_threshold: float = 0.65
    rolling_window_minutes: int = 30
    cusum_threshold: float = 4.0          # CUSUM slack parameter (stddevs)
    cusum_drift: float = 0.5              # CUSUM allowable drift

    # Entropy thresholds
    high_entropy_bit_threshold: float = 5.0
    low_ids_alert_threshold: float = 0.6  # IDS below this triggers COMPRESS bridge alert
    high_freq_rank_threshold: int = 1000

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "sentinel"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8003

    # Semantic entropy (Gap 2 fix)
    semantic_entropy_enabled: bool = True
    semantic_entropy_window_size: int = 3   # sentences per window

    # Model arbitrage (Gap 4 fix)
    arbitrage_enabled: bool = True

    # Compress bridge (Addition)
    compress_bridge_enabled: bool = False
    compress_bridge_url: str = "http://localhost:8001"

    # Prediction outcome tracking (Gap 6 fix)
    prediction_tracking_enabled: bool = True
    prediction_outcome_window_hours: int = 1

    class Config:
        env_prefix = "LENS_"
        env_file = ".env"
