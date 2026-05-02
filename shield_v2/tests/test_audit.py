"""Tests for the SHIELD Audit batch red-teaming pipeline."""

from __future__ import annotations

import pytest

from shield_v2.audit.report_generator import ReportGenerator
from shield_v2.audit.scanner import AuditScanner, CATEGORY_PROFILES, SEED_PROBES


class TestAuditSeedData:
    def test_every_category_has_profile(self):
        for cat in SEED_PROBES.keys():
            assert cat in CATEGORY_PROFILES, f"missing profile for {cat}"

    def test_profiles_have_required_fields(self):
        for cat, profile in CATEGORY_PROFILES.items():
            for key in ("avg_incident_cost_usd", "harm_realization_rate",
                        "regulatory_fine_per_violation_usd", "regulations"):
                assert key in profile, f"{cat} missing {key}"


class TestAuditScanner:
    @pytest.mark.asyncio
    async def test_minimal_audit_runs_end_to_end(self, trained_judge, attack_graph):
        scanner = AuditScanner(
            judge=trained_judge, attack_graph=attack_graph,
        )
        report = await scanner.run(
            languages=["en"], categories=["financial"],
            n_probes_per_cell=1,
        )
        assert report.status == "completed"
        assert report.total_probes >= 1
        assert 0.0 <= report.bypass_rate <= 1.0
        assert report.monetary_exposure_usd >= 0
        assert len(report.probe_results) == report.total_probes

    @pytest.mark.asyncio
    async def test_gap_matrix_keyed_by_lang_and_category(
        self, trained_judge, attack_graph,
    ):
        scanner = AuditScanner(judge=trained_judge, attack_graph=attack_graph)
        report = await scanner.run(
            languages=["en", "hi"], categories=["financial", "hiring"],
            n_probes_per_cell=1,
        )
        keys = set(report.gap_matrix.keys())
        assert "en:financial" in keys
        assert "hi:hiring" in keys
        for cell in report.gap_matrix.values():
            assert "bypass_rate" in cell
            assert 0.0 <= cell["bypass_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_compliance_attestations_present(self, trained_judge, attack_graph):
        scanner = AuditScanner(judge=trained_judge, attack_graph=attack_graph)
        report = await scanner.run(
            languages=["en"], categories=["hiring"], n_probes_per_cell=1,
        )
        a = report.compliance_attestations
        for framework in ("eu_ai_act_annex_iv", "iso_42001", "nist_ai_rmf",
                          "gdpr_article_22"):
            assert framework in a
            assert "passed" in a[framework]
            assert isinstance(a[framework]["passed"], bool)


class TestReportGenerator:
    @pytest.mark.asyncio
    async def test_to_json_dict_has_expected_keys(
        self, trained_judge, attack_graph,
    ):
        scanner = AuditScanner(judge=trained_judge, attack_graph=attack_graph)
        report = await scanner.run(
            languages=["en"], categories=["financial"], n_probes_per_cell=1,
        )
        d = ReportGenerator.to_json_dict(report)
        for key in ("scan_id", "status", "gap_matrix", "probes",
                    "compliance_attestations"):
            assert key in d
        assert isinstance(d["gap_matrix"], list)
        assert isinstance(d["probes"], list)

    @pytest.mark.asyncio
    async def test_to_markdown_nonempty(self, trained_judge, attack_graph):
        scanner = AuditScanner(judge=trained_judge, attack_graph=attack_graph)
        report = await scanner.run(
            languages=["en"], categories=["financial"], n_probes_per_cell=1,
        )
        md = ReportGenerator.to_markdown(report)
        assert "SHIELD Audit Report" in md
        assert "Safety Gap Matrix" in md
