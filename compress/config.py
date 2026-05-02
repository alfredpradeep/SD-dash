"""
Configuration for the SCL Compress engine.

All values can be overridden via environment variables prefixed with COMPRESS_.
Example: COMPRESS_DEVICE=cuda COMPRESS_BEAM_WIDTH=16
"""

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # Model paths
    compress_model_path: str = "google/mt5-base"
    amr_model_path: str = "models/amr-spring-base"
    amr_adapter_dir: str = "models/amr-adapters"

    # Device
    device: str = "cpu"   # "cuda" for GPU

    # Beam search
    beam_width: int = 12
    max_candidates: int = 8

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 3600

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    log_level: str = "INFO"

    # Prometheus
    metrics_port: int = 9101

    # Dev mode — runs without heavy ML deps (torch, transformers, etc.)
    dev_mode: bool = False

    class Config:
        env_prefix = "COMPRESS_"
        env_file = ".env"
        extra = "ignore"
