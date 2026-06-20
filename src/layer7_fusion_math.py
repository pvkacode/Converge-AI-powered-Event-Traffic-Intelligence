"""
Layer 7 — M7A: Sensor fusion mathematics (pure, dependency-free of other L7 modules).

Weighted Bayesian (inverse-variance, reliability-scaled) fusion + multi-sensor
consistency. No state, no I/O. Safe with 0 observations (returns a fallback tuple).

ADDITIVE ONLY. This module writes nothing.
"""

from __future__ import annotations

import math

import numpy as np

# conflict-flag thresholds on the normalized dispersion of sensor values
CONFLICT_LOW = 0.10
CONFLICT_HIGH = 0.30


def bayesian_fuse(values, reliabilities, variances):
    """Inverse-variance fusion with reliability weighting.

        w_i = R_i / sigma_i^2
        x_fused = sum(w_i x_i) / sum(w_i)
        sigma_fused^2 = 1 / sum(w_i)
        confidence = sum(w_i) / (sum(w_i) + 1)   (bounded [0,1))

    Returns dict(fused_value, fused_variance, fused_confidence, n, sum_weight).
    With n == 0 returns NaN value, +inf variance, 0 confidence (caller -> fallback).
    """
    x = np.asarray(values, dtype=float)
    R = np.clip(np.asarray(reliabilities, dtype=float), 0.0, 1.0)
    s2 = np.asarray(variances, dtype=float)
    mask = np.isfinite(x) & np.isfinite(s2) & (s2 > 0)
    x, R, s2 = x[mask], R[mask], s2[mask]
    if x.size == 0:
        return {"fused_value": float("nan"), "fused_variance": float("inf"),
                "fused_confidence": 0.0, "n": 0, "sum_weight": 0.0}
    w = R / s2
    sw = float(w.sum())
    if sw <= 0:  # all-zero reliability -> equal weights, low confidence
        w = np.ones_like(x)
        sw = float(w.sum())
        fused = float((w * x).sum() / sw)
        return {"fused_value": fused, "fused_variance": float(np.var(x) + 1.0),
                "fused_confidence": 0.05, "n": int(x.size), "sum_weight": 0.0}
    fused = float((w * x).sum() / sw)
    var = float(1.0 / sw)
    conf = float(sw / (sw + 1.0))
    return {"fused_value": fused, "fused_variance": var, "fused_confidence": conf,
            "n": int(x.size), "sum_weight": sw}


def consistency(values, scale: float = 1.0):
    """Multi-sensor agreement.

        conflict_score = std(values) / max(scale, eps)   (normalized dispersion)
        consensus_ratio = fraction of sensors within 1 std of the mean
        flag in {LOW_CONFLICT, MEDIUM_CONFLICT, HIGH_CONFLICT}
    """
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if x.size <= 1:
        return {"conflict_score": 0.0, "consensus_ratio": 1.0,
                "conflict_flag": "LOW_CONFLICT", "n": int(x.size)}
    sd = float(x.std(ddof=0))
    conflict = sd / max(abs(scale), 1e-9)
    mu = float(x.mean())
    consensus = float(np.mean(np.abs(x - mu) <= sd)) if sd > 0 else 1.0
    flag = ("HIGH_CONFLICT" if conflict >= CONFLICT_HIGH
            else "MEDIUM_CONFLICT" if conflict >= CONFLICT_LOW
            else "LOW_CONFLICT")
    return {"conflict_score": round(conflict, 6), "consensus_ratio": round(consensus, 6),
            "conflict_flag": flag, "n": int(x.size)}


def confidence_penalty(base_confidence: float, penalty: float = 0.5) -> float:
    """Apply a multiplicative confidence penalty (fallback mode)."""
    return float(max(0.0, min(1.0, base_confidence * penalty)))
