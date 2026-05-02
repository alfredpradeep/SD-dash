"""
ClickHouse time-series writer for LENS entropy profiles.

Writes are async fire-and-forget using asyncio.get_event_loop().run_in_executor.
Schema is defined in storage/schemas.sql.
"""

import asyncio
from datetime import datetime
from loguru import logger
from lens.entropy.structures import EntropyProfile, SpikePrediction, TokenizerArbitrageResult
from lens.config import Config

try:
    import clickhouse_connect
    CH_AVAILABLE = True
except ImportError:
    CH_AVAILABLE = False
    logger.warning("clickhouse-connect not available — storage disabled (dry-run mode)")


class ClickHouseStore:
    """Async ClickHouse writer. Falls back to no-op dry-run if unavailable."""

    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._dry_run = not CH_AVAILABLE

    def _get_client(self):
        if self._dry_run:
            return None
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.config.clickhouse_host,
                port=self.config.clickhouse_port,
                database=self.config.clickhouse_database,
                username=self.config.clickhouse_user,
                password=self.config.clickhouse_password,
                connect_timeout=5,
            )
        return self._client

    def _run_in_executor(self, fn):
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, fn)

    async def ensure_schema(self) -> None:
        """Run schema creation SQL from schemas.sql."""
        import os
        schema_path = os.path.join(os.path.dirname(__file__), "schemas.sql")
        if not os.path.exists(schema_path):
            logger.warning("schemas.sql not found — skipping schema creation")
            return
        with open(schema_path, "r") as f:
            statements = [s.strip() for s in f.read().split(";") if s.strip()]
        if self._dry_run:
            logger.info("Dry-run: would execute {} schema statements", len(statements))
            return
        def _exec():
            client = self._get_client()
            for stmt in statements:
                try:
                    client.command(stmt)
                except Exception as e:
                    logger.debug("Schema statement skipped (may already exist): {}", e)
        await self._run_in_executor(_exec)
        logger.info("Schema ensured in ClickHouse")

    async def insert_profile(self, profile: EntropyProfile) -> None:
        """Insert an entropy profile. Async, non-blocking."""
        if self._dry_run:
            logger.debug("Dry-run insert_profile: lang={} tokens={}", profile.language, profile.token_count)
            return

        def _insert():
            client = self._get_client()
            client.insert(
                "lens_entropy_profiles",
                [[
                    profile.timestamp,
                    profile.request_id,
                    profile.customer_id,
                    profile.language,
                    profile.model_name,
                    profile.token_count,
                    profile.english_baseline_tokens,
                    profile.efficiency_ratio,
                    profile.total_entropy_bits,
                    profile.mean_entropy_per_token,
                    profile.ids_score,
                    profile.etr_score,
                    profile.etr_english_baseline,
                    profile.etr_inequity_ratio,
                    profile.semantic_entropy_bits,
                    profile.semantic_ids_score,
                    profile.waste_type,
                    profile.cost_usd,
                    profile.waste_cost_usd,
                    1 if profile.low_ids_alert else 0,
                ]],
                column_names=[
                    "timestamp", "request_id", "customer_id",
                    "language", "model_name", "token_count",
                    "baseline_tokens", "efficiency_ratio",
                    "total_entropy_bits", "mean_entropy",
                    "ids_score", "etr_score", "etr_english",
                    "etr_inequity_ratio", "semantic_entropy_bits",
                    "semantic_ids_score", "waste_type",
                    "cost_usd", "waste_cost_usd", "low_ids_alert",
                ],
            )
        try:
            await self._run_in_executor(_insert)
        except Exception as e:
            logger.warning("ClickHouse insert_profile failed: {}", e)
            raise

    async def insert_spike_prediction(
        self,
        prediction: SpikePrediction,
        prediction_id: str,
        customer_id: str,
    ) -> None:
        if self._dry_run:
            return

        def _insert():
            client = self._get_client()
            client.insert(
                "lens_spike_predictions",
                [[
                    datetime.utcnow(),
                    prediction_id,
                    customer_id,
                    prediction.language,
                    prediction.current_entropy_velocity,
                    0.0,  # cusum_statistic (add to SpikePrediction if needed)
                    prediction.predicted_token_increase_pct,
                    prediction.confidence,
                    prediction.horizon_minutes,
                    prediction.alert_level,
                    prediction.spike_cause,
                    None if prediction.is_likely_temporary is None else (
                        1 if prediction.is_likely_temporary else 0
                    ),
                    prediction.trigger_reason[:2000],
                ]],
                column_names=[
                    "timestamp", "prediction_id", "customer_id",
                    "language", "entropy_velocity", "cusum_statistic",
                    "predicted_token_increase", "confidence",
                    "horizon_minutes", "alert_level", "spike_cause",
                    "is_likely_temporary", "trigger_reason",
                ],
            )
        try:
            await self._run_in_executor(_insert)
        except Exception as e:
            logger.warning("ClickHouse insert_spike_prediction failed: {}", e)

    async def insert_arbitrage_result(
        self,
        result: "TokenizerArbitrageResult",
        customer_id: str,
        current_model: str,
        text_hash: str,
    ) -> None:
        if self._dry_run:
            return
        current_tokens = result.results.get(current_model, {}).get("tokens", 0)
        rec_tokens = result.results.get(result.recommended_model, {}).get("tokens", 0)

        def _insert():
            client = self._get_client()
            client.insert(
                "lens_arbitrage_results",
                [[
                    datetime.utcnow(), customer_id, result.language, text_hash,
                    result.recommended_model, result.max_savings_pct,
                    current_model, current_tokens, rec_tokens,
                ]],
                column_names=[
                    "timestamp", "customer_id", "language", "text_hash",
                    "recommended_model", "max_savings_pct",
                    "current_model", "current_tokens", "recommended_tokens",
                ],
            )
        try:
            await self._run_in_executor(_insert)
        except Exception as e:
            logger.warning("ClickHouse insert_arbitrage failed: {}", e)

    async def health(self) -> str:
        if self._dry_run:
            return "dry-run (clickhouse-connect not installed)"
        def _ping():
            self._get_client().ping()
        try:
            await self._run_in_executor(_ping)
            return "healthy"
        except Exception:
            return "unhealthy"
