"""
Layer 6 – Calibration Posterior Update

For each probability bin j, update:
    theta_j | D  ~  Beta(a0 + s_j,  b0 + f_j)

where
    s_j = successes (actual high-impact events in bin j during Mar–Apr)
    f_j = failures  (non-high-impact events in bin j during Mar–Apr)
    a0, b0 = 1, 1  (uniform / non-informative prior)

Tracks: raw probability, calibrated probability, posterior mean,
        95% credible interval, ECE, and Brier score trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

A0, B0      = 1.0, 1.0          # Beta prior hyperparameters (non-informative)
N_BINS      = 10                 # decile bins
CI_ALPHA    = 0.05               # 95 % credible interval


# ---------------------------------------------------------------------------
# Beta posterior helpers
# ---------------------------------------------------------------------------

def _beta_posterior_mean(a: float, b: float) -> float:
    return a / (a + b)


def _beta_credible_interval(a: float, b: float, alpha: float = CI_ALPHA):
    lo = stats.beta.ppf(alpha / 2, a, b)
    hi = stats.beta.ppf(1 - alpha / 2, a, b)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def _make_bins(n: int = N_BINS) -> list[tuple[float, float]]:
    edges = np.linspace(0.0, 1.0, n + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n)]


def _bin_index(p: float, bins: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= p <= hi:
            return i
    return len(bins) - 1   # edge case: exactly 1.0


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def _compute_ece(df: pd.DataFrame, pred_col: str, true_col: str,
                 bin_col: str) -> float:
    total = len(df)
    ece = 0.0
    for _, grp in df.groupby(bin_col):
        if len(grp) == 0:
            continue
        frac_pos    = grp[true_col].mean()
        mean_pred   = grp[pred_col].mean()
        ece        += abs(frac_pos - mean_pred) * len(grp) / total
    return float(ece)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def update_calibration_posteriors(
    joined_df: pd.DataFrame,
    pred_col: str = "high_impact_prob_calibrated",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    joined_df : feedback actuals merged with Layer 4.5 predictions.
                Must contain columns: ``actual_high_impact``, ``pred_col``.
    pred_col  : which probability column to calibrate.

    Returns
    -------
    DataFrame with one row per bin containing posterior parameters and metrics.
    """
    valid = joined_df[joined_df[pred_col].notna() & joined_df["actual_high_impact"].notna()].copy()

    if len(valid) == 0:
        return pd.DataFrame()

    bins = _make_bins(N_BINS)
    valid["bin_idx"]  = valid[pred_col].apply(lambda p: _bin_index(float(p), bins))
    valid["bin_lo"]   = valid["bin_idx"].map(lambda i: bins[i][0])
    valid["bin_hi"]   = valid["bin_idx"].map(lambda i: bins[i][1])

    # Global ECE and Brier (raw prediction vs. actual)
    brier_raw = float(((valid[pred_col] - valid["actual_high_impact"]) ** 2).mean())

    # Also compute using raw uncalibrated prob if present
    brier_raw_uncal = None
    if "high_impact_prob" in valid.columns:
        brier_raw_uncal = float(
            ((valid["high_impact_prob"] - valid["actual_high_impact"]) ** 2).mean()
        )

    ece_before = _compute_ece(valid, pred_col, "actual_high_impact", "bin_idx")

    records = []
    calibrated_preds = []

    for i, (lo, hi) in enumerate(bins):
        grp = valid[valid["bin_idx"] == i]
        n   = len(grp)
        s   = int(grp["actual_high_impact"].sum()) if n > 0 else 0
        f   = n - s

        a_post = A0 + s
        b_post = B0 + f
        post_mean   = _beta_posterior_mean(a_post, b_post)
        ci_lo, ci_hi = _beta_credible_interval(a_post, b_post)

        mean_raw_pred = float(grp[pred_col].mean()) if n > 0 else (lo + hi) / 2.0
        frac_positive = float(grp["actual_high_impact"].mean()) if n > 0 else float("nan")

        records.append({
            "bin_idx":            i,
            "bin_lo":             lo,
            "bin_hi":             hi,
            "n_events":           n,
            "n_successes":        s,
            "n_failures":         f,
            "mean_raw_pred":      round(mean_raw_pred, 6),
            "fraction_positive":  round(frac_positive, 6) if n > 0 else None,
            "beta_a_prior":       A0,
            "beta_b_prior":       B0,
            "beta_a_posterior":   a_post,
            "beta_b_posterior":   b_post,
            "posterior_mean":     round(post_mean, 6),
            "ci95_lo":            round(ci_lo, 6),
            "ci95_hi":            round(ci_hi, 6),
            "calibration_shift":  round(post_mean - mean_raw_pred, 6),
        })

        # Build calibrated prediction column for ECE-after
        if n > 0:
            calibrated_preds.extend([post_mean] * n)

    result_df = pd.DataFrame(records)

    # ECE after calibration
    if len(calibrated_preds) == len(valid):
        valid["calibrated_post_pred"] = calibrated_preds
        ece_after = _compute_ece(valid, "calibrated_post_pred",
                                 "actual_high_impact", "bin_idx")
        brier_calibrated = float(
            ((valid["calibrated_post_pred"] - valid["actual_high_impact"]) ** 2).mean()
        )
    else:
        ece_after        = float("nan")
        brier_calibrated = float("nan")

    # Attach global metrics as repeated columns so they appear in the CSV
    result_df["ece_before_update"]  = round(ece_before, 6)
    result_df["ece_after_update"]   = round(ece_after, 6)
    result_df["brier_raw"]          = round(brier_raw, 6)
    result_df["brier_calibrated"]   = round(brier_calibrated, 6) if not np.isnan(brier_calibrated) else None
    result_df["brier_raw_uncal"]    = round(brier_raw_uncal, 6) if brier_raw_uncal is not None else None
    result_df["n_feedback_total"]   = len(valid)

    return result_df
