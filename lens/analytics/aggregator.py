"""
ClickHouse SQL aggregation queries for LENS analytics.

Generates per-language cost reports, equity analysis, and waste type breakdowns
from the materialized views and raw tables.
"""

from datetime import datetime, timedelta
from loguru import logger
from lens.entropy.structures import LanguageCostReport
from lens.storage.clickhouse_store import ClickHouseStore


class Aggregator:
    """Runs ClickHouse aggregation queries for LENS analytics."""

    # Language names for report text
    LANG_NAMES: dict[str, str] = {
        "ta": "Tamil", "ml": "Malayalam", "te": "Telugu", "hi": "Hindi",
        "bn": "Bengali", "ar": "Arabic", "ja": "Japanese", "zh": "Chinese",
        "ko": "Korean", "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "pt": "Portuguese",
    }

    def __init__(self, ch_store: ClickHouseStore):
        self.ch = ch_store

    async def language_report(
        self,
        language: str,
        customer_id: str,
        period_hours: int = 24,
    ) -> LanguageCostReport:
        """Generate a full cost attribution report for a language."""
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=period_hours)

        if self.ch._dry_run:
            return self._empty_report(language, customer_id, period_start, period_end)

        def _query():
            client = self.ch._get_client()
            sql = """
                SELECT
                    count()                     AS total_requests,
                    sum(token_count)            AS total_tokens,
                    sum(cost_usd)               AS total_cost,
                    sum(waste_cost_usd)         AS total_waste,
                    avg(ids_score)              AS mean_ids,
                    avg(etr_score)              AS mean_etr,
                    avg(etr_inequity_ratio)     AS mean_inequity,
                    avg(efficiency_ratio)       AS mean_efficiency
                FROM lens_entropy_profiles
                WHERE customer_id = {customer_id:String}
                  AND language    = {language:String}
                  AND timestamp   >= {ts_start:DateTime64}
                  AND timestamp   <  {ts_end:DateTime64}
            """
            result = client.query(sql, parameters={
                "customer_id": customer_id,
                "language": language,
                "ts_start": period_start,
                "ts_end": period_end,
            })
            return result.first_row if result.result_rows else None

        def _spike_count():
            client = self.ch._get_client()
            sql = """
                SELECT count()
                FROM lens_spike_predictions
                WHERE customer_id = {customer_id:String}
                  AND language    = {language:String}
                  AND timestamp   >= {ts_start:DateTime64}
            """
            result = client.query(sql, parameters={
                "customer_id": customer_id,
                "language": language,
                "ts_start": period_start,
            })
            return result.first_row[0] if result.result_rows else 0

        import asyncio
        loop = asyncio.get_event_loop()
        try:
            row = await loop.run_in_executor(None, _query)
            spike_count = await loop.run_in_executor(None, _spike_count)
        except Exception as e:
            logger.warning("Aggregator query failed: {}", e)
            return self._empty_report(language, customer_id, period_start, period_end)

        if not row:
            return self._empty_report(language, customer_id, period_start, period_end)

        total_requests, total_tokens, total_cost, total_waste, \
            mean_ids, mean_etr, mean_inequity, mean_efficiency = row

        # Efficiency percentile: estimate based on mean_efficiency vs known ratios
        efficiency_percentile = self._estimate_percentile(language, float(mean_inequity or 1.0))

        # Equity metrics
        # Cost per information unit = cost / (total_tokens * mean_etr)
        total_info_bits = float(total_tokens or 0) * float(mean_etr or 0)
        eff_cost_per_info = float(total_cost) / max(total_info_bits, 1e-8)
        # English baseline: assume ETR of 18.2 bits/token
        english_info_bits = float(total_tokens or 0) * 18.2
        en_cost_per_info = float(total_cost) / max(english_info_bits, 1e-8)
        equity_multiplier = eff_cost_per_info / max(en_cost_per_info, 1e-12)

        recommendations = self._generate_recommendations(
            language=language,
            mean_ids=float(mean_ids or 0),
            mean_inequity=float(mean_inequity or 1.0),
            waste_pct=(float(total_waste) / max(float(total_cost), 1e-8) * 100),
        )

        return LanguageCostReport(
            language=language,
            period_start=period_start,
            period_end=period_end,
            total_requests=int(total_requests or 0),
            total_tokens=int(total_tokens or 0),
            total_cost_usd=float(total_cost or 0),
            total_waste_cost_usd=float(total_waste or 0),
            mean_ids_score=round(float(mean_ids or 0), 4),
            mean_etr_score=round(float(mean_etr or 0), 4),
            etr_inequity_ratio=round(float(mean_inequity or 1.0), 3),
            efficiency_percentile=efficiency_percentile,
            spike_events=int(spike_count),
            recommendations=recommendations,
            effective_cost_per_info_unit=round(eff_cost_per_info, 10),
            english_cost_per_info_unit=round(en_cost_per_info, 10),
            equity_multiplier=round(equity_multiplier, 2),
        )

    def _empty_report(
        self,
        language: str,
        customer_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> LanguageCostReport:
        return LanguageCostReport(
            language=language,
            period_start=period_start,
            period_end=period_end,
            total_requests=0,
            total_tokens=0,
            total_cost_usd=0.0,
            total_waste_cost_usd=0.0,
            mean_ids_score=0.0,
            mean_etr_score=0.0,
            etr_inequity_ratio=1.0,
            efficiency_percentile=50.0,
            spike_events=0,
            recommendations=["No data available for this period."],
        )

    def _estimate_percentile(self, language: str, inequity_ratio: float) -> float:
        """Estimate efficiency percentile based on ETR inequity ratio."""
        # Higher inequity = lower efficiency percentile
        if inequity_ratio <= 1.2:
            return 90.0
        elif inequity_ratio <= 2.0:
            return 70.0
        elif inequity_ratio <= 3.5:
            return 40.0
        elif inequity_ratio <= 5.0:
            return 20.0
        else:
            return 5.0

    def _generate_recommendations(
        self,
        language: str,
        mean_ids: float,
        mean_inequity: float,
        waste_pct: float,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []
        lang_name = self.LANG_NAMES.get(language, language.upper())

        if mean_inequity > 3.0:
            recs.append(
                f"{lang_name} shows severe tokenization inequity ({mean_inequity:.1f}x). "
                f"Consider routing {lang_name} traffic through the COMPRESS engine to reduce "
                f"token count by semantic code-switching."
            )
        elif mean_inequity > 1.5:
            recs.append(
                f"{lang_name} tokenization is {mean_inequity:.1f}x less efficient than English. "
                f"Run model arbitrage analysis to identify a more cost-effective model."
            )

        if mean_ids < 0.6:
            recs.append(
                f"Mean IDS score {mean_ids:.2f} indicates wasteful prompting patterns. "
                f"Many tokens are highly predictable — consider prompt compression."
            )

        if waste_pct > 40:
            recs.append(
                f"{waste_pct:.0f}% of {lang_name} token spend is attributable to tokenization "
                f"inefficiency. This is recoverable waste — enabling COMPRESS could eliminate "
                f"most of it."
            )

        if not recs:
            recs.append(f"{lang_name} tokenization efficiency is within acceptable range.")

        return recs
