"""
Report builder — generates equity and cost attribution reports as
structured dicts (suitable for JSON API responses or downstream rendering).

The Equity Report is the product's most compelling enterprise output:
  "Your Tamil-speaking users effectively pay 4.8x more per unit of
   information communicated than your English-speaking users."
"""

from datetime import datetime
from lens.entropy.structures import LanguageCostReport


class ReportBuilder:
    """Builds structured LENS equity and cost attribution reports."""

    LANG_NAMES: dict[str, str] = {
        "ta": "Tamil", "ml": "Malayalam", "te": "Telugu", "hi": "Hindi",
        "bn": "Bengali", "ar": "Arabic", "ja": "Japanese", "zh": "Chinese",
        "ko": "Korean", "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "pt": "Portuguese", "id": "Indonesian",
    }

    def build_equity_report(
        self,
        reports: list[LanguageCostReport],
        period_label: str = "last 24 hours",
    ) -> dict:
        """
        Build a multi-language equity report.

        Shows how much more non-English users effectively pay per unit of
        information communicated.
        """
        if not reports:
            return {"error": "No data available"}

        # Find English baseline
        en_report = next((r for r in reports if r.language == "en"), None)
        en_equity = en_report.effective_cost_per_info_unit if en_report else None

        language_rows = []
        for r in sorted(reports, key=lambda x: x.etr_inequity_ratio, reverse=True):
            lang_name = self.LANG_NAMES.get(r.language, r.language.upper())
            row = {
                "language": lang_name,
                "code": r.language,
                "total_requests": r.total_requests,
                "total_cost_usd": r.total_cost_usd,
                "waste_cost_usd": r.total_waste_cost_usd,
                "waste_pct": round(r.total_waste_cost_usd / max(r.total_cost_usd, 1e-9) * 100, 1),
                "etr_inequity_ratio": r.etr_inequity_ratio,
                "equity_multiplier": r.equity_multiplier,
                "mean_ids_score": r.mean_ids_score,
                "spike_events": r.spike_events,
                "equity_statement": (
                    f"{lang_name} users pay {r.equity_multiplier:.1f}x more per unit of "
                    f"information than English users."
                ) if r.equity_multiplier > 1.1 else f"{lang_name}: efficient (at parity with English)",
                "recommendations": r.recommendations,
            }
            language_rows.append(row)

        total_waste = sum(r.total_waste_cost_usd for r in reports)
        total_cost = sum(r.total_cost_usd for r in reports)
        worst_language = max(reports, key=lambda r: r.equity_multiplier, default=None)

        return {
            "report_type": "tokenization_equity_report",
            "generated_at": datetime.utcnow().isoformat(),
            "period": period_label,
            "summary": {
                "total_cost_usd": round(total_cost, 4),
                "total_waste_cost_usd": round(total_waste, 4),
                "overall_waste_pct": round(total_waste / max(total_cost, 1e-9) * 100, 1),
                "languages_analysed": len(reports),
                "worst_language": {
                    "name": self.LANG_NAMES.get(worst_language.language, "") if worst_language else "",
                    "equity_multiplier": worst_language.equity_multiplier if worst_language else 1.0,
                } if worst_language else None,
            },
            "languages": language_rows,
        }

    def build_spike_history_report(
        self,
        spike_records: list[dict],
        period_label: str = "last 7 days",
    ) -> dict:
        """Build a spike history report from stored spike prediction records."""
        by_language: dict[str, list] = {}
        for record in spike_records:
            lang = record.get("language", "unknown")
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(record)

        language_summaries = []
        for lang, events in sorted(by_language.items(), key=lambda x: len(x[1]), reverse=True):
            lang_name = self.LANG_NAMES.get(lang, lang.upper())
            critical = sum(1 for e in events if e.get("alert_level") == "critical")
            causes = {}
            for e in events:
                c = e.get("spike_cause", "unknown")
                causes[c] = causes.get(c, 0) + 1
            top_cause = max(causes, key=lambda k: causes[k]) if causes else "unknown"
            language_summaries.append({
                "language": lang_name,
                "total_spikes": len(events),
                "critical_spikes": critical,
                "dominant_cause": top_cause,
            })

        return {
            "report_type": "spike_history_report",
            "generated_at": datetime.utcnow().isoformat(),
            "period": period_label,
            "total_spike_events": len(spike_records),
            "languages": language_summaries,
        }
