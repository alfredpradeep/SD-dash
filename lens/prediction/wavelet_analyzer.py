"""
Gap 10 — Wavelet Multi-Scale Entropy Decomposition.

Decomposes entropy time series into multiple time scales using
discrete wavelet transform, enabling detection of slow-burn spikes
and periodic patterns that single-scale CUSUM misses.

Uses Haar wavelet (simplest, most robust) with scipy or manual fallback.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime
from loguru import logger


@dataclass
class WaveletScale:
    """Analysis at a single time scale."""
    scale_index: int
    scale_name: str          # "15-min", "1-hour", "4-hour", etc.
    coefficients: List[float]
    energy: float            # sum of squared coefficients
    entropy: float           # Shannon entropy of coefficient distribution
    trend_direction: str     # "rising", "falling", "stable"
    trend_strength: float    # 0-1 normalized
    anomaly_detected: bool
    anomaly_score: float     # z-score at this scale


@dataclass
class WaveletDecomposition:
    """Full multi-scale analysis result."""
    scales: List[WaveletScale]
    dominant_scale: str       # which time scale has highest energy
    overall_trend: str        # "accelerating", "decelerating", "stable", "oscillating"
    multi_scale_entropy: float  # cross-scale entropy measure
    early_warning_score: float  # 0-1, combines all scales into single alert
    reconstruction: List[float]  # reconstructed signal


class WaveletEntropyAnalyzer:
    """
    Multi-scale entropy analysis using Haar wavelet decomposition.

    Theory: A single-scale detector (CUSUM) can miss patterns that
    build slowly across multiple time scales. Wavelet decomposition
    separates the entropy signal into:
      - Scale 0: Raw fluctuations (minute-level noise)
      - Scale 1: Short-term patterns (15-30 min)
      - Scale 2: Medium patterns (1-2 hours)
      - Scale 3: Long-term trends (4-8 hours)
      - Scale 4+: Macro trends (daily)

    A spike invisible at scale 0 may be obvious at scale 2.
    """

    SCALE_NAMES = {
        0: "1-5 min (noise)",
        1: "15-30 min (short)",
        2: "1-2 hours (medium)",
        3: "4-8 hours (long)",
        4: "12-24 hours (daily)",
        5: "2-3 days (macro)",
    }

    def _haar_decompose(self, signal: np.ndarray, max_levels: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Manual Haar wavelet decomposition (no scipy dependency)."""
        levels = []
        current = signal.copy().astype(float)

        for level in range(max_levels):
            if len(current) < 4:
                break
            # Pad to even length
            if len(current) % 2 != 0:
                current = np.append(current, current[-1])

            # Haar decomposition
            approx = (current[0::2] + current[1::2]) / np.sqrt(2)
            detail = (current[0::2] - current[1::2]) / np.sqrt(2)
            levels.append((approx, detail))
            current = approx

        return levels

    def _compute_scale_entropy(self, coefficients: np.ndarray) -> float:
        """Shannon entropy of wavelet coefficient distribution."""
        if len(coefficients) < 2:
            return 0.0
        # Normalize to probability distribution
        energy = np.abs(coefficients) ** 2
        total = np.sum(energy)
        if total < 1e-10:
            return 0.0
        probs = energy / total
        probs = probs[probs > 1e-10]  # avoid log(0)
        return float(-np.sum(probs * np.log2(probs)))

    def _detect_trend(self, coefficients: np.ndarray) -> Tuple[str, float]:
        """Detect trend direction and strength in coefficients."""
        if len(coefficients) < 3:
            return "stable", 0.0

        # Linear regression on coefficient magnitudes
        x = np.arange(len(coefficients))
        coeffs_abs = np.abs(coefficients)

        if np.std(coeffs_abs) < 1e-10:
            return "stable", 0.0

        slope = np.polyfit(x, coeffs_abs, 1)[0]
        strength = min(1.0, abs(slope) / (np.std(coeffs_abs) + 1e-10))

        if slope > 0.01:
            return "rising", round(strength, 3)
        elif slope < -0.01:
            return "falling", round(strength, 3)
        return "stable", round(strength, 3)

    def analyze(self, entropy_values: List[float], max_levels: int = 5) -> Optional[WaveletDecomposition]:
        """
        Perform multi-scale wavelet decomposition on entropy time series.

        Args:
            entropy_values: List of entropy measurements (time-ordered)
            max_levels: Maximum decomposition levels

        Returns:
            WaveletDecomposition with per-scale analysis
        """
        if len(entropy_values) < 8:
            return None

        signal = np.array(entropy_values)
        levels = self._haar_decompose(signal, max_levels)

        if not levels:
            return None

        scales = []
        max_energy = 0.0
        dominant_idx = 0

        for i, (approx, detail) in enumerate(levels):
            energy = float(np.sum(detail ** 2))
            entropy = self._compute_scale_entropy(detail)
            trend_dir, trend_str = self._detect_trend(detail)

            # Anomaly detection: z-score of latest coefficient
            if len(detail) >= 3:
                mean_d = np.mean(np.abs(detail[:-1])) if len(detail) > 1 else 0
                std_d = np.std(np.abs(detail[:-1])) if len(detail) > 1 else 1
                std_d = max(std_d, 1e-10)
                z_score = float((np.abs(detail[-1]) - mean_d) / std_d)
                anomaly = z_score > 2.5
            else:
                z_score = 0.0
                anomaly = False

            scale = WaveletScale(
                scale_index=i,
                scale_name=self.SCALE_NAMES.get(i, f"scale-{i}"),
                coefficients=[round(float(c), 4) for c in detail[-min(10, len(detail)):]],
                energy=round(energy, 4),
                entropy=round(entropy, 4),
                trend_direction=trend_dir,
                trend_strength=trend_str,
                anomaly_detected=anomaly,
                anomaly_score=round(z_score, 3),
            )
            scales.append(scale)

            if energy > max_energy:
                max_energy = energy
                dominant_idx = i

        # Overall trend classification
        rising_count = sum(1 for s in scales if s.trend_direction == "rising")
        falling_count = sum(1 for s in scales if s.trend_direction == "falling")

        if rising_count > len(scales) * 0.6:
            overall = "accelerating"
        elif falling_count > len(scales) * 0.6:
            overall = "decelerating"
        elif rising_count > 0 and falling_count > 0:
            overall = "oscillating"
        else:
            overall = "stable"

        # Multi-scale entropy: entropy of energy distribution across scales
        energies = np.array([s.energy for s in scales])
        total_e = np.sum(energies)
        if total_e > 0:
            e_probs = energies / total_e
            e_probs = e_probs[e_probs > 1e-10]
            ms_entropy = float(-np.sum(e_probs * np.log2(e_probs)))
        else:
            ms_entropy = 0.0

        # Early warning score: weighted combination of anomaly scores
        weights = [0.1, 0.25, 0.3, 0.25, 0.1][:len(scales)]
        weight_sum = sum(weights)
        warning = sum(s.anomaly_score * w for s, w in zip(scales, weights)) / weight_sum if weight_sum > 0 else 0
        warning = min(1.0, max(0.0, warning / 5.0))  # normalize to 0-1

        return WaveletDecomposition(
            scales=scales,
            dominant_scale=self.SCALE_NAMES.get(dominant_idx, f"scale-{dominant_idx}"),
            overall_trend=overall,
            multi_scale_entropy=round(ms_entropy, 4),
            early_warning_score=round(warning, 4),
            reconstruction=[round(float(v), 4) for v in signal[-min(20, len(signal)):]],
        )
