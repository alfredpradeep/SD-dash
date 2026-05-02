"""
Calibration utilities for the distilled judge.

A logistic-regression classifier produces confidence scores that are
well-correlated with true probabilities but rarely well-calibrated:
a model that reports 0.8 confidence might be right 72% of the time, or
85% — and that matters when we use confidence to route between the
Reflex fast path and the Brain escalation.

This module adds:

1. `PlattCalibrator` — sigmoid calibration on the score output (Platt
   scaling), robust for small datasets where isotonic overfits.
2. `IsotonicCalibrator` — monotone step-function calibration, best when
   we have ≥2k calibration points.
3. `compute_reliability_curve` — bins predicted confidence, returns
   (bin_centers, actual_accuracy, weights) for reliability diagrams.
4. `expected_calibration_error` — aggregate gap between confidence and
   accuracy, weighted by bin size (the standard ECE metric).
5. `confusion_matrix` — multi-class confusion matrix with per-class
   precision/recall so the calibration dashboard can surface which
   classes are the weakest.

These operate on plain numpy arrays so they stay portable across the
various student backends we might swap in (sklearn, Torch, ONNX).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Platt scaling (sigmoid calibration)
# =============================================================================


@dataclass
class PlattCalibrator:
    """P_cal(y=1 | s) = sigmoid(A * s + B), fit via ML on held-out data."""
    A: float = -1.0
    B: float = 0.0
    fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray,
            max_iter: int = 200, lr: float = 0.05) -> "PlattCalibrator":
        # Binary labels ∈ {0, 1}
        scores = np.asarray(scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        A, B = -1.0, 0.0
        # Simple gradient descent on cross-entropy; small data, no need
        # for LBFGS.
        for _ in range(max_iter):
            z = A * scores + B
            # Numerically stable sigmoid
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad_A = np.mean((p - labels) * scores)
            grad_B = np.mean(p - labels)
            A -= lr * grad_A
            B -= lr * grad_B
        self.A, self.B, self.fitted = float(A), float(B), True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        z = self.A * np.asarray(scores, dtype=float) + self.B
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# =============================================================================
# Multi-class wrapper
# =============================================================================


@dataclass
class MultiClassPlatt:
    """One-vs-rest Platt calibration for multi-class probabilities."""
    calibrators: List[PlattCalibrator] = field(default_factory=list)
    n_classes: int = 0
    fitted: bool = False

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "MultiClassPlatt":
        proba = np.asarray(proba, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.n_classes = proba.shape[1]
        self.calibrators = []
        for k in range(self.n_classes):
            cal = PlattCalibrator().fit(proba[:, k], (y == k).astype(float))
            self.calibrators.append(cal)
        self.fitted = True
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=float)
        out = np.zeros_like(proba)
        for k in range(self.n_classes):
            out[:, k] = self.calibrators[k].transform(proba[:, k])
        # Row-normalize to sum to 1 (otherwise it's not a distribution)
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return out / row_sum


# =============================================================================
# Reliability diagram + ECE
# =============================================================================


def compute_reliability_curve(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given per-sample (confidence, correct?) pairs, return arrays of
    (bin_center, empirical_accuracy, weight) suitable for plotting.
    """
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    accuracies = np.full(n_bins, np.nan)
    weights = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidence >= bins[i]) & (confidence <= bins[i + 1])
        weights[i] = float(np.sum(mask))
        if weights[i] > 0:
            accuracies[i] = float(np.mean(correct[mask]))
    return centers, accuracies, weights


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10,
) -> float:
    """
    ECE = Σ (|Bₘ|/n) · |acc(Bₘ) - conf(Bₘ)|
    """
    confidence = np.asarray(confidence, dtype=float).ravel()
    correct = np.asarray(correct, dtype=float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidence)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if i == n_bins - 1:
            mask = (confidence >= bins[i]) & (confidence <= bins[i + 1])
        if not np.any(mask):
            continue
        weight = np.sum(mask) / n
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(confidence[mask]))
        ece += weight * abs(acc - conf)
    return float(ece)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=int)
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[t, p] += 1
    return mat


def per_class_precision_recall(
    mat: np.ndarray, class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    n_classes = mat.shape[0]
    out = []
    for k in range(n_classes):
        tp = int(mat[k, k])
        fp = int(mat[:, k].sum() - tp)
        fn = int(mat[k, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out.append({
            "class": (class_names[k] if class_names else str(k)),
            "support": int(mat[k, :].sum()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return out


def roc_auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary ROC-AUC via rank statistic — no sklearn dependency here."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Count concordant pairs via ranks
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1  # 1-indexed
    rank_sum_pos = float(ranks[: len(pos)].sum())
    auc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)
