"""Tests for Pillar 4: CrossLingualManifoldAligner and AntigenicDriftMutator."""

from __future__ import annotations

import numpy as np
import pytest

from shield_v2.core.manifold_alignment import (
    AntigenicDriftMutator,
    CrossLingualManifoldAligner,
    SafetyBoundaryProjection,
)


class TestCrossLingualManifoldAligner:
    def test_initialization(self):
        aligner = CrossLingualManifoldAligner(embed_dim=16)
        assert aligner.d == 16
        assert aligner.projections == {}

    def test_fit_produces_orthogonal_rotation(self, rng):
        d, n = 16, 20
        aligner = CrossLingualManifoldAligner(embed_dim=d, n_stiefel_steps=8)
        src = rng.standard_normal((n, d))
        rot_true = np.linalg.qr(rng.standard_normal((d, d)))[0]
        tgt = src @ rot_true

        rS = rng.standard_normal(d)
        cS = rng.standard_normal(d)
        proj = aligner.fit(
            target_lang="xx",
            source_embeds=src, target_embeds=tgt,
            source_refusal=rS, target_refusal=rS @ rot_true,
            source_compliance=cS, target_compliance=cS @ rot_true,
        )
        # Orthogonality: R^T R ≈ I
        RtR = proj.rotation.T @ proj.rotation
        assert np.allclose(RtR, np.eye(d), atol=1e-5)

    def test_projection_recovers_target(self, rng):
        d, n = 16, 40
        aligner = CrossLingualManifoldAligner(embed_dim=d, n_stiefel_steps=15)
        src = rng.standard_normal((n, d))
        rot_true = np.linalg.qr(rng.standard_normal((d, d)))[0]
        tgt = src @ rot_true

        rS = rng.standard_normal(d); cS = rng.standard_normal(d)
        aligner.fit(
            target_lang="xx",
            source_embeds=src, target_embeds=tgt,
            source_refusal=rS, target_refusal=rS @ rot_true,
            source_compliance=cS, target_compliance=cS @ rot_true,
        )
        reconstruction_error = np.linalg.norm(
            aligner.project(src[0], "xx") - tgt[0]
        )
        # Should be small but not necessarily zero (Stiefel regularisation).
        assert reconstruction_error < 5.0

    def test_fidelity_is_in_unit_interval(self, aligner_with_synthetic):
        for proj in aligner_with_synthetic.projections.values():
            assert 0.0 <= proj.boundary_fidelity <= 1.0
            assert proj.margin > 0

    def test_thin_boundary_threshold(self, aligner_with_synthetic):
        thin = aligner_with_synthetic.thin_boundary_languages(threshold=1.5)
        # Threshold above maximum fidelity -> everything counted as thin
        assert set(thin) == set(aligner_with_synthetic.projections.keys())
        # Threshold below minimum -> empty
        thin_none = aligner_with_synthetic.thin_boundary_languages(threshold=-0.1)
        assert thin_none == []

    def test_unknown_language_raises(self, aligner_with_synthetic, rng):
        with pytest.raises(KeyError):
            aligner_with_synthetic.project(rng.standard_normal(32), "xx_unknown")


class TestSafetyBoundaryProjection:
    def test_safety_score_sign_flips_across_boundary(self, rng):
        d = 16
        r = np.zeros(d); r[0] = 1.0
        c = np.zeros(d); c[0] = -1.0
        proj = SafetyBoundaryProjection(
            source_lang="en", target_lang="xx",
            rotation=np.eye(d),
            refusal_direction=r, compliance_direction=c,
            margin=1.0, boundary_fidelity=1.0, n_pairs=10,
        )
        on_refusal_side = np.zeros(d); on_refusal_side[0] = 1.0
        on_compliance_side = np.zeros(d); on_compliance_side[0] = -1.0
        assert proj.safety_score(on_refusal_side) > 0
        assert proj.safety_score(on_compliance_side) < 0

    def test_project_is_matrix_multiply(self, rng):
        d = 16
        R = np.linalg.qr(rng.standard_normal((d, d)))[0]
        proj = SafetyBoundaryProjection(
            source_lang="en", target_lang="xx",
            rotation=R,
            refusal_direction=rng.standard_normal(d),
            compliance_direction=rng.standard_normal(d),
            margin=1.0, boundary_fidelity=1.0, n_pairs=10,
        )
        x = rng.standard_normal(d)
        assert np.allclose(proj.project(x), x @ R)


class TestAntigenicDriftMutator:
    def _proj(self, rng, d=16):
        r = rng.standard_normal(d); r /= np.linalg.norm(r)
        c = -r
        return SafetyBoundaryProjection(
            source_lang="en", target_lang="xx",
            rotation=np.eye(d),
            refusal_direction=r, compliance_direction=c,
            margin=float(np.linalg.norm(r - c)),
            boundary_fidelity=0.9, n_pairs=10,
        )

    def test_drift_terminates_within_max_steps(self, rng):
        proj = self._proj(rng)
        mutator = AntigenicDriftMutator(max_steps=5, rng_seed=1)
        x0 = 2.0 * proj.refusal_direction + 0.1 * rng.standard_normal(16)
        _, traj = mutator.drift(x0, proj)
        assert len(traj) <= 5

    def test_drift_preserves_semantics_with_strong_anchor(self, rng):
        proj = self._proj(rng)
        strong = AntigenicDriftMutator(
            step_size=0.01, semantic_anchor=5.0, max_steps=20, rng_seed=1,
        )
        x0 = proj.refusal_direction * 1.0
        x_final, _ = strong.drift(x0, proj)
        drift = np.linalg.norm(x_final - x0)
        assert drift < 1.0  # anchor holds

    def test_escape_probability_in_unit_interval(self, rng):
        proj = self._proj(rng)
        mutator = AntigenicDriftMutator(max_steps=4, rng_seed=7)
        x0 = proj.refusal_direction * 2.0
        p = mutator.escape_probability(x0, proj, n_trials=4)
        assert 0.0 <= p <= 1.0

    def test_escape_probability_when_already_on_compliance_side(self, rng):
        proj = self._proj(rng)
        mutator = AntigenicDriftMutator(rng_seed=3)
        x0 = -proj.refusal_direction * 2.0  # starts on compliance side
        assert mutator.escape_probability(x0, proj, n_trials=2) == 1.0

    def test_drift_reduces_safety_score_on_average(self, rng):
        """The whole point — drift should move the point closer to compliance."""
        proj = self._proj(rng)
        mutator = AntigenicDriftMutator(
            step_size=0.3, semantic_anchor=0.0,
            n_candidates=16, max_steps=30, rng_seed=5,
        )
        x0 = proj.refusal_direction * 3.0
        initial = proj.safety_score(x0)
        x_final, _ = mutator.drift(x0, proj)
        final = proj.safety_score(x_final)
        assert final <= initial + 1e-6
