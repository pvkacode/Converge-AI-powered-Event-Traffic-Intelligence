"""
Duration Quality Gate for Layer 4.5 — ASTraM.

Additive guardrail: sanitizes raw duration quantiles before Layer 5 consumes them.
Implements monotone sanitization, reliability scoring, fallback blending, and
sanity flags. Does not modify any existing layer or training target.

Layer 5 must consume the scenario-ready bundle exported here, NOT the raw
duration quantile file.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FALLBACK_BLEND_THRESHOLD = 0.50
TAIL_RISK_FLAG_THRESHOLD = 0.50


def sanitize_quantiles(
    p50: np.ndarray,
    p80: np.ndarray,
    p95: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce Q50 <= Q80 <= Q95, then apply a hard safety clamp to Q95.

    Clamp rule:
        Q95_safe = min(Q95_mono, max(10 * Q50_mono, 1440))

    If the post-clamp triple is still non-monotone (edge case where clamping
    pushes Q95 below Q80), re-sort deterministically.

    Returns
    -------
    mono_p50, mono_p80, safe_p95, clamp_flags, crossing_flags
        All arrays are shape (n,) float64.  Flags are int (0/1).
    """
    p50 = np.asarray(p50, dtype=float)
    p80 = np.asarray(p80, dtype=float)
    p95 = np.asarray(p95, dtype=float)

    crossing_flags = ((p50 > p80) | (p80 > p95) | (p50 > p95)).astype(int)

    stacked = np.stack([p50, p80, p95], axis=1)
    mono = np.sort(stacked, axis=1)
    mono_p50 = mono[:, 0]
    mono_p80 = mono[:, 1]
    mono_p95 = mono[:, 2]

    upper_bound = np.maximum(10.0 * mono_p50, 1440.0)
    clamp_flags = (mono_p95 > upper_bound).astype(int)
    safe_p95 = np.minimum(mono_p95, upper_bound)

    still_bad = (safe_p95 < mono_p80) | (safe_p95 < mono_p50)
    if still_bad.any():
        stacked2 = np.stack([mono_p50, mono_p80, safe_p95], axis=1)
        mono2 = np.sort(stacked2, axis=1)
        mono_p50 = np.where(still_bad, mono2[:, 0], mono_p50)
        mono_p80 = np.where(still_bad, mono2[:, 1], mono_p80)
        safe_p95 = np.where(still_bad, mono2[:, 2], safe_p95)

    return mono_p50, mono_p80, safe_p95, clamp_flags, crossing_flags


def compute_duration_reliability(
    calibration_quality: np.ndarray,
    retrieval_confidence: np.ndarray,
    novelty_score_norm: np.ndarray,
    drift_score_norm: np.ndarray,
    support_norm: np.ndarray,
    width_norm: np.ndarray,
) -> np.ndarray:
    """
    Deterministic duration reliability score R_i in [0, 1].

    R = 0.25*Cal + 0.20*Retrieval + 0.20*(1-Novelty)
      + 0.15*(1-Drift) + 0.10*Support + 0.10*(1-Width)

    All six inputs must already be normalised to [0, 1].  The function clips
    each component before combining, so mild out-of-range values are safe.
    """
    def _c(x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)

    R = (
        0.25 * _c(calibration_quality)
        + 0.20 * _c(retrieval_confidence)
        + 0.20 * (1.0 - _c(novelty_score_norm))
        + 0.15 * (1.0 - _c(drift_score_norm))
        + 0.10 * _c(support_norm)
        + 0.10 * (1.0 - _c(width_norm))
    )
    return np.clip(R, 0.0, 1.0)


def apply_fallback_chain(
    feat_df: pd.DataFrame,
    global_fallback: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct per-row fallback quantile vectors using the as-of feature chain:
        cause × corridor  →  cause  →  corridor  →  global

    The as-of feature matrix already carries pre-computed cause/corridor
    quantiles in asof_p50/p80/p95_duration.  We use those directly when
    finite and positive; otherwise we fall back to the global training quantiles.

    Returns fb_p50, fb_p80, fb_p95 (shape (n,) float64).
    """
    n = len(feat_df)
    fb_p50 = np.full(n, global_fallback.get("p50", 60.0))
    fb_p80 = np.full(n, global_fallback.get("p80", 120.0))
    fb_p95 = np.full(n, global_fallback.get("p95", 240.0))

    for col, arr in [
        ("asof_p50_duration", fb_p50),
        ("asof_p80_duration", fb_p80),
        ("asof_p95_duration", fb_p95),
    ]:
        if col in feat_df.columns:
            vals = pd.to_numeric(feat_df[col], errors="coerce").values
            valid = np.isfinite(vals) & (vals > 0)
            arr[valid] = vals[valid]

    return fb_p50, fb_p80, fb_p95


def blend_with_fallback(
    safe_p50: np.ndarray,
    safe_p80: np.ndarray,
    safe_p95: np.ndarray,
    fb_p50: np.ndarray,
    fb_p80: np.ndarray,
    fb_p95: np.ndarray,
    reliability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reliability-weighted blend:
        Q_final = R * Q_safe + (1 - R) * Q_fallback

    When R is low the fallback dominates; when R is high the model
    prediction dominates.  This is the correct production-safe behaviour.

    Returns final_p50, final_p80, final_p95, fallback_blend_flags.
    """
    R = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    one_R = 1.0 - R

    final_p50 = R * safe_p50 + one_R * fb_p50
    final_p80 = R * safe_p80 + one_R * fb_p80
    final_p95 = R * safe_p95 + one_R * fb_p95

    fallback_blend_flags = (one_R > FALLBACK_BLEND_THRESHOLD).astype(int)
    return final_p50, final_p80, final_p95, fallback_blend_flags


def build_duration_sanity_flags(
    crossing_flags: np.ndarray,
    clamp_flags: np.ndarray,
    fallback_blend_flags: np.ndarray,
    tail_risk_prob: np.ndarray,
    tail_risk_threshold: float = TAIL_RISK_FLAG_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute per-row quality flags and human-readable reason codes.

    duration_sanity_flag = 1 only when all checks pass (no crossing, no
    clamping, fallback weight <= 0.5, tail risk <= threshold).

    Returns
    -------
    duration_sanity_flag (int array), tail_risk_flag (int array),
    duration_guard_reason (list of pipe-delimited strings).
    """
    tail_risk_flags = (tail_risk_prob > tail_risk_threshold).astype(int)
    sanity = (
        (crossing_flags == 0)
        & (clamp_flags == 0)
        & (fallback_blend_flags == 0)
        & (tail_risk_flags == 0)
    ).astype(int)

    reasons: list[str] = []
    for i in range(len(crossing_flags)):
        parts: list[str] = []
        if crossing_flags[i]:
            parts.append("raw_quantile_crossing")
        if clamp_flags[i]:
            parts.append("p95_clamped")
        if fallback_blend_flags[i]:
            parts.append("fallback_blended_low_reliability")
        if tail_risk_flags[i]:
            parts.append("tail_risk_high")
        reasons.append("|".join(parts) if parts else "sanitized_ok")

    return sanity, tail_risk_flags, reasons


def build_scenario_ready_duration_bundle(
    final_p50: np.ndarray,
    final_p80: np.ndarray,
    final_p95: np.ndarray,
    reliability: np.ndarray,
    tail_risk_prob: np.ndarray,
    sanity_flags: np.ndarray,
    guard_reasons: list[str],
    event_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Canonical Layer 5 duration input.

    Layer 5 must consume THIS bundle — not layer45_duration_raw_predictions.csv.
    Rows flagged as low-reliability may still be used, but only through the
    sanitized safe_duration_* columns.
    """
    return pd.DataFrame({
        "event_id": event_ids,
        "safe_duration_p50": final_p50,
        "safe_duration_p80": final_p80,
        "safe_duration_p95": final_p95,
        "duration_reliability": reliability,
        "tail_risk_prob": tail_risk_prob,
        "duration_sanity_flag": sanity_flags,
        "duration_guard_reason": guard_reasons,
    })
