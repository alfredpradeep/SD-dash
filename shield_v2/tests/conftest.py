"""Shared pytest fixtures for the SHIELD v2 test suite."""

from __future__ import annotations

import os
import sys

# Make the project importable regardless of where pytest is invoked from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest

from shield_v2.core.brain_llm import BrainLLM, BudgetConfig, LLMBudgetTracker
from shield_v2.core.distilled_judge import DistilledCrossLingualJudge
from shield_v2.core.manifold_alignment import CrossLingualManifoldAligner
from shield_v2.core.attack_graph import AttackGraph


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def trained_judge() -> DistilledCrossLingualJudge:
    """A distilled judge pre-fit on the synthetic distillation dataset."""
    j = DistilledCrossLingualJudge()
    pairs = DistilledCrossLingualJudge.synthetic_distillation_dataset(n_per_class=10)
    j.fit_from_pairs(pairs)
    return j


@pytest.fixture(scope="session")
def aligner_with_synthetic(rng) -> CrossLingualManifoldAligner:
    """An aligner with a small synthetic projection fit for a handful of languages."""
    d = 32
    n = 30
    aligner = CrossLingualManifoldAligner(embed_dim=d, n_stiefel_steps=10)
    src = rng.standard_normal((n, d))
    src_refusal = rng.standard_normal(d)
    src_compliance = rng.standard_normal(d)
    for lang in ("hi", "ta", "ar"):
        rot = np.linalg.qr(rng.standard_normal((d, d)))[0]
        tgt = src @ rot + 0.01 * rng.standard_normal((n, d))
        aligner.fit(
            target_lang=lang,
            source_embeds=src,
            target_embeds=tgt,
            source_refusal=src_refusal,
            target_refusal=src_refusal @ rot,
            source_compliance=src_compliance,
            target_compliance=src_compliance @ rot,
        )
    return aligner


@pytest.fixture(scope="session")
def attack_graph() -> AttackGraph:
    return AttackGraph.build_standard(target_langs=["en", "hi", "ta", "ar", "zh"])


@pytest.fixture
def offline_brain() -> BrainLLM:
    """A Brain with no API keys — calls fail gracefully to the fallback paths."""
    return BrainLLM(budget=LLMBudgetTracker(BudgetConfig()))
