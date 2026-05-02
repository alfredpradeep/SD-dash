"""
Gap 11 — Transfer Entropy for Cross-Language Causal Prediction.

Measures directional causal information flow between language entropy
time series. Enables cross-language early warning: detect spike in
Tamil, predict upcoming spike in Hindi before it happens.

Based on Schreiber (2000) — Transfer Entropy.
TE(X→Y) = H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from loguru import logger


@dataclass
class CausalLink:
    """Directional causal relationship between two languages."""
    source_language: str
    target_language: str
    transfer_entropy: float      # bits of causal information
    normalized_te: float         # 0-1 normalized
    lag_minutes: int             # optimal time lag
    confidence: float            # statistical significance
    direction: str               # "strong", "moderate", "weak", "none"
    predictive_power: float      # how much source reduces target uncertainty


@dataclass
class CausalNetwork:
    """Full cross-language causal network."""
    links: List[CausalLink]
    strongest_link: Optional[CausalLink]
    hub_language: str            # language that causes most others
    hub_score: float             # how central the hub is
    network_density: float       # fraction of possible links that are significant
    warnings: List[str]          # active cross-language warnings


class TransferEntropyAnalyzer:
    """
    Cross-language causal prediction using transfer entropy.

    Theory: Transfer entropy TE(X→Y) measures how much knowing
    the past of X reduces uncertainty about the future of Y,
    beyond what Y's own past already tells you. If TE(Tamil→Hindi)
    is high, a Tamil entropy spike causally predicts a Hindi spike.

    This enables pre-emptive action: detect spike in one language,
    automatically prepare for correlated languages.
    """

    def __init__(self):
        # Store per-language entropy history
        self._history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # lang -> [(timestamp, entropy)]
        self._causal_cache: Dict[Tuple[str, str], CausalLink] = {}

    def observe(self, language: str, entropy: float, timestamp: float):
        """Record an entropy observation for a language."""
        self._history[language].append((timestamp, entropy))
        # Keep last 1000 observations per language
        if len(self._history[language]) > 1000:
            self._history[language] = self._history[language][-1000:]

    def _discretize(self, values: np.ndarray, n_bins: int = 8) -> np.ndarray:
        """Discretize continuous values into bins for entropy estimation."""
        if len(values) < 2:
            return np.zeros(len(values), dtype=int)
        mn, mx = np.min(values), np.max(values)
        if mx - mn < 1e-10:
            return np.zeros(len(values), dtype=int)
        bins = np.linspace(mn, mx, n_bins + 1)
        return np.clip(np.digitize(values, bins) - 1, 0, n_bins - 1)

    def _estimate_te(self, source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """
        Estimate transfer entropy TE(source → target) using binned estimator.

        TE = sum p(y_{t+1}, y_t, x_t) * log2( p(y_{t+1}|y_t,x_t) / p(y_{t+1}|y_t) )
        """
        n = min(len(source), len(target))
        if n < lag + 10:
            return 0.0

        src = self._discretize(source[:n])
        tgt = self._discretize(target[:n])

        # Build joint counts
        n_bins = int(max(src.max(), tgt.max())) + 1

        # p(y_{t+lag}, y_t, x_t)
        joint_xyz = np.zeros((n_bins, n_bins, n_bins))
        # p(y_{t+lag}, y_t)
        joint_yz = np.zeros((n_bins, n_bins))
        # p(y_t, x_t)
        joint_xz = np.zeros((n_bins, n_bins))
        # p(y_t)
        marginal_z = np.zeros(n_bins)

        count = 0
        for t in range(n - lag):
            yt_next = tgt[t + lag]
            yt = tgt[t]
            xt = src[t]
            joint_xyz[yt_next, yt, xt] += 1
            joint_yz[yt_next, yt] += 1
            joint_xz[yt, xt] += 1
            marginal_z[yt] += 1
            count += 1

        if count < 10:
            return 0.0

        # Normalize
        joint_xyz /= count
        joint_yz /= count
        joint_xz /= count
        marginal_z /= count

        # Compute TE
        te = 0.0
        for y1 in range(n_bins):
            for y0 in range(n_bins):
                for x0 in range(n_bins):
                    p_xyz = joint_xyz[y1, y0, x0]
                    p_yz = joint_yz[y1, y0]
                    p_xz = joint_xz[y0, x0]
                    p_z = marginal_z[y0]

                    if p_xyz > 1e-10 and p_yz > 1e-10 and p_xz > 1e-10 and p_z > 1e-10:
                        te += p_xyz * np.log2((p_xyz * p_z) / (p_yz * p_xz))

        return max(0.0, te)

    def compute_causal_link(self, source_lang: str, target_lang: str,
                            max_lag: int = 5) -> Optional[CausalLink]:
        """Compute transfer entropy between two languages at optimal lag."""
        src_hist = self._history.get(source_lang, [])
        tgt_hist = self._history.get(target_lang, [])

        if len(src_hist) < 20 or len(tgt_hist) < 20:
            return None

        src_vals = np.array([e for _, e in src_hist])
        tgt_vals = np.array([e for _, e in tgt_hist])

        # Find optimal lag
        best_te = 0.0
        best_lag = 1

        for lag in range(1, max_lag + 1):
            te = self._estimate_te(src_vals, tgt_vals, lag)
            if te > best_te:
                best_te = te
                best_lag = lag

        # Compute reverse direction for normalization
        reverse_te = self._estimate_te(tgt_vals, src_vals, best_lag)

        # Normalized TE: how dominant is the forward direction
        total_te = best_te + reverse_te
        normalized = best_te / total_te if total_te > 0 else 0.5

        # Significance: bootstrap test (simplified)
        n_shuffles = 20
        null_dist = []
        for _ in range(n_shuffles):
            shuffled = np.random.permutation(src_vals)
            null_dist.append(self._estimate_te(shuffled, tgt_vals, best_lag))

        null_mean = np.mean(null_dist) if null_dist else 0
        null_std = np.std(null_dist) if null_dist else 1
        null_std = max(null_std, 1e-10)
        z_score = (best_te - null_mean) / null_std
        confidence = min(1.0, max(0.0, 1.0 - np.exp(-z_score)))

        # Classify direction strength
        if best_te > 0.1 and confidence > 0.7:
            direction = "strong"
        elif best_te > 0.05 and confidence > 0.5:
            direction = "moderate"
        elif best_te > 0.02:
            direction = "weak"
        else:
            direction = "none"

        # Estimate lag in minutes (assume ~1 min between observations)
        lag_minutes = best_lag * 5  # rough estimate

        link = CausalLink(
            source_language=source_lang,
            target_language=target_lang,
            transfer_entropy=round(best_te, 6),
            normalized_te=round(normalized, 4),
            lag_minutes=lag_minutes,
            confidence=round(confidence, 4),
            direction=direction,
            predictive_power=round(best_te / (best_te + 0.1), 4),
        )

        self._causal_cache[(source_lang, target_lang)] = link
        return link

    def compute_network(self, languages: Optional[List[str]] = None) -> CausalNetwork:
        """Compute full causal network across all observed languages."""
        if languages is None:
            languages = list(self._history.keys())

        links = []
        hub_scores: Dict[str, float] = defaultdict(float)

        for src in languages:
            for tgt in languages:
                if src == tgt:
                    continue
                link = self.compute_causal_link(src, tgt)
                if link and link.direction != "none":
                    links.append(link)
                    hub_scores[src] += link.transfer_entropy

        # Find hub
        hub_lang = max(hub_scores, key=hub_scores.get) if hub_scores else (languages[0] if languages else "unknown")
        hub_score = hub_scores.get(hub_lang, 0.0)

        # Network density
        max_links = len(languages) * (len(languages) - 1)
        density = len(links) / max_links if max_links > 0 else 0.0

        # Generate warnings
        warnings = []
        for link in sorted(links, key=lambda l: l.transfer_entropy, reverse=True)[:3]:
            if link.direction in ("strong", "moderate"):
                warnings.append(
                    f"{link.source_language.upper()} entropy causally predicts "
                    f"{link.target_language.upper()} with {link.lag_minutes}min lag "
                    f"(TE={link.transfer_entropy:.4f}, conf={link.confidence:.0%})"
                )

        strongest = max(links, key=lambda l: l.transfer_entropy) if links else None

        return CausalNetwork(
            links=links,
            strongest_link=strongest,
            hub_language=hub_lang,
            hub_score=round(hub_score, 4),
            network_density=round(density, 4),
            warnings=warnings,
        )
