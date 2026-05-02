"""Structured report generator — emits JSON and (optionally) a Markdown brief."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from shield_v2.audit.scanner import AuditReport, ProbeResult


class ReportGenerator:
    """Transforms an AuditReport into customer-facing artifacts."""

    @staticmethod
    def to_json_dict(report: AuditReport) -> Dict[str, Any]:
        base = report.summary()
        base["gap_matrix"] = list(report.gap_matrix.values())
        base["probes"] = [
            {
                "language": r.language,
                "category": r.category,
                "seed_prompt": r.seed_prompt,
                "consensus": r.consensus.as_dict(),
                "per_interaction_risk_usd": r.per_interaction_risk_usd,
                "regulatory_exposure_usd": r.regulatory_exposure_usd,
            }
            for r in report.probe_results
        ]
        return base

    @staticmethod
    def to_markdown(report: AuditReport) -> str:
        s = [
            f"# SHIELD Audit Report — {report.scan_id}",
            "",
            f"- **Status**: {report.status}",
            f"- **Duration**: {report.duration_ms} ms",
            f"- **Languages**: {', '.join(report.languages)}",
            f"- **Categories**: {', '.join(report.categories)}",
            f"- **Total probes**: {report.total_probes}",
            f"- **Bypasses**: {report.bypasses} ({report.bypass_rate:.2%})",
            f"- **Monetary exposure**: ${report.monetary_exposure_usd:,.2f}",
            f"- **Regulatory exposure**: ${report.regulatory_exposure_usd:,.2f}",
            "",
            "## Compliance Attestations",
            "",
        ]
        for k, v in report.compliance_attestations.items():
            mark = "PASS" if v["passed"] else "FAIL"
            s.append(f"- **{v['framework']}**: {mark} (observed {v['observed_bypass_rate']:.2%} vs threshold {v['threshold']:.2%})")

        if report.thin_boundary_languages:
            s.append("")
            s.append("## Thin-Boundary Languages (Manifold Alignment Weak)")
            s.append("")
            for lang in report.thin_boundary_languages:
                s.append(f"- {lang}")

        s.append("")
        s.append("## Safety Gap Matrix")
        s.append("")
        s.append("| Language | Category | Total | Bypasses | Rate | Exposure ($) |")
        s.append("|----------|----------|-------|----------|------|--------------|")
        for cell in report.gap_matrix.values():
            s.append(
                f"| {cell['language']} | {cell['category']} | {cell['total']} | "
                f"{cell['bypasses']} | {cell['bypass_rate']:.2%} | "
                f"${cell['exposure_usd']:,.2f} |"
            )

        return "\n".join(s)
