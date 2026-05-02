"""
Tokenizer Drift Detector

Monitors whether a provider's tokenizer behavior changes over time.
Uses a fixed reference corpus — if token count for the reference changes,
the tokenizer changed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from typing import Callable, Optional
import hashlib


@dataclass
class DriftReport:
    """Report of tokenizer drift detected for a model."""
    model: str
    reference_corpus_hash: str
    baseline_token_count: int
    current_token_count: int
    drift_tokens: int
    drift_pct: float
    detected_at: datetime
    baseline_recorded_at: datetime
    severity: str             # "none", "minor" (<1%), "significant" (1-5%), "major" (>5%)
    estimated_monthly_cost_impact: float
    details: str


@dataclass
class DriftHistory:
    """Historical drift data for a model."""
    model: str
    snapshots: list[dict] = field(default_factory=list)  # [{date, token_count, drift_from_baseline_pct}]
    total_drift_pct: float = 0.0
    drift_direction: str = "stable"      # "inflating" (more tokens), "deflating" (fewer), "stable"


class TokenizerDriftDetector:
    """
    Detects silent tokenizer changes by providers.

    Maintains a fixed reference corpus (multilingual, covering edge cases).
    Periodically tokenizes the corpus and compares against baseline.
    If counts change, the tokenizer was updated.

    For single-provider enterprises: "Your GPT-4o tokenizer changed on
    April 15. Tamil token counts increased 8%. Your monthly cost will
    increase ~$3,200 unless you adjust."
    """

    # Fixed reference corpus — multilingual, covers edge cases
    REFERENCE_CORPUS = {
        "english_prose": "The quick brown fox jumps over the lazy dog. Machine learning models process natural language by tokenizing input text into subword units.",
        "english_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "tamil": "என் கடவுச்சொல்லை மீட்டமைக்க எப்படி? தயவுசெய்து உதவுங்கள்.",
        "hindi": "मेरा पासवर्ड रीसेट करने में मदद करें। कृपया जल्दी करें।",
        "arabic": "كيف أعيد تعيين كلمة المرور الخاصة بي؟ الرجاء المساعدة.",
        "japanese": "パスワードをリセットするにはどうすればよいですか？",
        "chinese": "如何重置我的密码？请帮助我。",
        "mixed_script": "The candidate said என்னுடைய experience is 5 years in Java programming.",
        "numbers_special": "Invoice #INV-2024-78234 total: $12,456.78 due 2024-03-15 (net-30)",
        "unicode_edge": "café résumé naïve Zürich São Paulo — «quotes» …ellipsis",
    }

    def __init__(self):
        self._baselines: dict[str, dict] = {}  # model -> {corpus_key: token_count, hash: ..., recorded_at: ...}
        self._history: dict[str, DriftHistory] = defaultdict(lambda: DriftHistory(model="", snapshots=[], total_drift_pct=0.0, drift_direction="stable"))
        self._corpus_hash = self._compute_corpus_hash()

    @staticmethod
    def _compute_corpus_hash() -> str:
        """Compute SHA-256 hash of the reference corpus."""
        corpus_str = "|".join(f"{k}:{v}" for k, v in sorted(TokenizerDriftDetector.REFERENCE_CORPUS.items()))
        return hashlib.sha256(corpus_str.encode()).hexdigest()

    def establish_baseline(self, model: str, count_fn: Callable[[str], int]) -> dict:
        """
        Establish baseline token counts for reference corpus.

        Args:
            model: Model name (e.g., "gpt-4o")
            count_fn: Function that takes text and returns token count

        Returns:
            Dictionary of {corpus_key: token_count}
        """

        baseline = {}
        for key, text in self.REFERENCE_CORPUS.items():
            try:
                token_count = count_fn(text)
                baseline[key] = token_count
            except Exception as e:
                # Handle tokenizer unavailable gracefully
                baseline[key] = 0

        # Compute total
        total_tokens = sum(baseline.values())

        # Store baseline
        self._baselines[model] = {
            "corpus_hash": self._corpus_hash,
            "recorded_at": datetime.utcnow(),
            "total_tokens": total_tokens,
            "breakdown": baseline
        }

        # Initialize history
        self._history[model] = DriftHistory(
            model=model,
            snapshots=[{
                "date": datetime.utcnow().isoformat(),
                "token_count": total_tokens,
                "drift_from_baseline_pct": 0.0
            }],
            total_drift_pct=0.0,
            drift_direction="stable"
        )

        return baseline

    def check_drift(self, model: str, count_fn: Callable[[str], int],
                   monthly_spend: float = 10000.0) -> list[DriftReport]:
        """
        Check if tokenizer has drifted from baseline.

        Args:
            model: Model name
            count_fn: Function that takes text and returns token count
            monthly_spend: Estimated monthly spend on this model (for impact calculation)

        Returns:
            List of DriftReport objects (one per corpus section with significant drift)
        """

        if model not in self._baselines:
            return []

        baseline_info = self._baselines[model]
        baseline_breakdown = baseline_info["breakdown"]
        baseline_total = baseline_info["total_tokens"]

        # Count current tokens
        current_breakdown = {}
        for key, text in self.REFERENCE_CORPUS.items():
            try:
                token_count = count_fn(text)
                current_breakdown[key] = token_count
            except Exception:
                current_breakdown[key] = 0

        current_total = sum(current_breakdown.values())

        # Analyze drift
        reports = []
        overall_drift_pct = ((current_total - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0

        # Report by section
        for corpus_key in self.REFERENCE_CORPUS.keys():
            baseline_count = baseline_breakdown.get(corpus_key, 0)
            current_count = current_breakdown.get(corpus_key, 0)

            if baseline_count == 0:
                continue

            drift_tokens = current_count - baseline_count
            drift_pct = (drift_tokens / baseline_count) * 100

            # Determine severity
            if abs(drift_pct) < 0.5:
                severity = "none"
            elif abs(drift_pct) < 1.0:
                severity = "minor"
            elif abs(drift_pct) < 5.0:
                severity = "significant"
            else:
                severity = "major"

            # Skip if no drift
            if severity == "none":
                continue

            # Estimate cost impact
            # Assume 50% of requests use this corpus section, cost scales linearly with tokens
            estimated_cost_impact = (monthly_spend * (drift_tokens / baseline_count) * 0.5)

            report = DriftReport(
                model=model,
                reference_corpus_hash=self._corpus_hash,
                baseline_token_count=baseline_count,
                current_token_count=current_count,
                drift_tokens=drift_tokens,
                drift_pct=drift_pct,
                detected_at=datetime.utcnow(),
                baseline_recorded_at=baseline_info["recorded_at"],
                severity=severity,
                estimated_monthly_cost_impact=estimated_cost_impact,
                details=f"Tokenizer for {model} has changed. {corpus_key} section: {baseline_count} -> {current_count} tokens ({drift_pct:+.1f}%)"
            )

            reports.append(report)

        # Update history
        self._history[model].snapshots.append({
            "date": datetime.utcnow().isoformat(),
            "token_count": current_total,
            "drift_from_baseline_pct": overall_drift_pct
        })

        # Determine drift direction
        if overall_drift_pct > 0.5:
            self._history[model].drift_direction = "inflating"
        elif overall_drift_pct < -0.5:
            self._history[model].drift_direction = "deflating"
        else:
            self._history[model].drift_direction = "stable"

        self._history[model].total_drift_pct = overall_drift_pct

        return reports

    def get_history(self, model: str) -> DriftHistory:
        """Get historical drift data for a model."""

        if model in self._history:
            return self._history[model]

        return DriftHistory(
            model=model,
            snapshots=[],
            total_drift_pct=0.0,
            drift_direction="stable"
        )

    def get_all_baselines(self) -> dict:
        """Return all established baselines."""

        result = {}
        for model, baseline_info in self._baselines.items():
            result[model] = {
                "recorded_at": baseline_info["recorded_at"].isoformat(),
                "total_tokens": baseline_info["total_tokens"],
                "corpus_hash": baseline_info["corpus_hash"],
                "breakdown": baseline_info["breakdown"]
            }

        return result

    def reset_baseline(self, model: str):
        """Clear baseline for a model (useful for re-establishing after provider update)."""

        if model in self._baselines:
            del self._baselines[model]

        if model in self._history:
            del self._history[model]
