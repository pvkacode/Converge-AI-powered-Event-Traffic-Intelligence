"""
Layer 6 – Hierarchical Bayesian Duration Update

Model:  y_i = log(1 + duration_i)  [minutes → log-minutes]
Hierarchy: global  →  cause  →  cause × corridor
Partial pooling via normal-normal conjugate update.
Exponential forgetting:  w_i = exp(-lambda * delta_t_i)
  lambda = log(2) / half_life_days   (default 30 d)

Fallback order (by sparsity of stratum in the *prior* period):
  stratum  →  cause-only  →  global
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reference date = end of feedback window (day after last event)
REF_DATE         = pd.Timestamp("2024-04-09", tz="Asia/Kolkata")
DEFAULT_HALF_LIFE = 30        # days
SPARSE_N_EFF      = 3         # min effective obs for stratum-level prior
MIN_VAR           = 1e-6      # floor on variance estimates


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _decay_lambda(half_life: float) -> float:
    return np.log(2.0) / half_life


def compute_weights(df: pd.DataFrame, half_life: float = DEFAULT_HALF_LIFE) -> np.ndarray:
    """Exponential forgetting weights relative to REF_DATE."""
    delta_days = (REF_DATE - df["start_local"]).dt.total_seconds() / 86400.0
    lam = _decay_lambda(half_life)
    return np.exp(-lam * delta_days.clip(lower=0).values)


# ---------------------------------------------------------------------------
# Effective sample size (Kish)
# ---------------------------------------------------------------------------

def kish_n_eff(w: np.ndarray) -> float:
    s = w.sum()
    if s < 1e-12:
        return 0.0
    return float(s ** 2 / np.dot(w, w))


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------

def weighted_stats(y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Returns (n_eff, weighted_mean, weighted_variance)."""
    n_eff = kish_n_eff(w)
    if n_eff < 1 or w.sum() < 1e-12:
        return 0.0, float("nan"), float("nan")
    mu_w    = float(np.dot(w, y) / w.sum())
    var_w   = float(np.dot(w, (y - mu_w) ** 2) / w.sum())
    return n_eff, mu_w, max(var_w, MIN_VAR)


# ---------------------------------------------------------------------------
# Normal-normal conjugate posterior
# ---------------------------------------------------------------------------

def normal_normal_posterior(
    mu_prior: float, var_prior: float,
    n_eff: float, mu_data: float, var_data: float,
) -> tuple[float, float]:
    """
    Returns (mu_post, var_post) using normal-normal conjugate update.
    If n_eff < 1 the prior is returned unchanged.
    """
    if n_eff < 1:
        return mu_prior, var_prior
    tau_prior = 1.0 / max(var_prior, MIN_VAR)
    tau_data  = n_eff / max(var_data, MIN_VAR)
    tau_post  = tau_prior + tau_data
    mu_post   = (tau_prior * mu_prior + tau_data * mu_data) / tau_post
    var_post  = 1.0 / tau_post
    return float(mu_post), float(var_post)


# ---------------------------------------------------------------------------
# Credible intervals and back-transformed quantiles
# ---------------------------------------------------------------------------

def credible_interval(mu: float, var: float, z: float = 1.96) -> tuple[float, float]:
    sd = np.sqrt(max(var, MIN_VAR))
    return float(mu - z * sd), float(mu + z * sd)


def log_to_minute_quantiles(mu: float, var: float) -> tuple[float, float, float]:
    """Posterior predictive quantiles in original minute scale."""
    sd  = np.sqrt(max(var, MIN_VAR))
    q50 = float(np.expm1(mu))
    q80 = float(np.expm1(mu + stats.norm.ppf(0.80) * sd))
    q95 = float(np.expm1(mu + stats.norm.ppf(0.95) * sd))
    return q50, q80, q95


# ---------------------------------------------------------------------------
# Prior extraction from the prior period dataset
# ---------------------------------------------------------------------------

def _extract_priors(pr_df: pd.DataFrame) -> tuple[dict, dict, float, float]:
    """
    Returns:
        stratum_priors  – {(cause, corridor): (mu, var, n)}
        cause_priors    – {cause: (mu, var, n)}
        mu_global       – global mean
        var_global      – global variance
    """
    obs = pr_df[pr_df["duration_min"].notna() & ~pr_df["is_censored"]].copy()
    obs["log_dur"] = np.log1p(obs["duration_min"])

    mu_global  = float(obs["log_dur"].mean())
    var_global = float(obs["log_dur"].var())

    cause_priors: dict[str, tuple[float, float, int]] = {}
    for cause, grp in obs.groupby("event_cause"):
        ld = grp["log_dur"]
        cause_priors[cause] = (float(ld.mean()), float(max(ld.var(), MIN_VAR)), len(ld))

    stratum_priors: dict[tuple[str, str], tuple[float, float, int]] = {}
    for (cause, corr), grp in obs.groupby(["event_cause", "corridor_fill"]):
        ld = grp["log_dur"]
        stratum_priors[(cause, corr)] = (float(ld.mean()), float(max(ld.var(), MIN_VAR)), len(ld))

    return stratum_priors, cause_priors, mu_global, var_global


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def update_duration_posteriors(
    prior_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    half_life: float = DEFAULT_HALF_LIFE,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the hierarchical Bayesian duration update.

    Parameters
    ----------
    prior_df    : Nov–Feb events (the historical training period)
    feedback_df : Mar–Apr events (the newly closed feedback batch)
    half_life   : exponential forgetting half-life in days

    Returns
    -------
    summary_df  : per-stratum posterior DataFrame
    global_dict : dict suitable for JSON export (global & stratum-level params)
    """
    stratum_priors, cause_priors, mu_g, var_g = _extract_priors(prior_df)

    # Feedback: restrict to uncensored observations
    fb = feedback_df[feedback_df["duration_min"].notna() & ~feedback_df["is_censored"]].copy()
    fb["log_dur"] = np.log1p(fb["duration_min"])
    fb["w"]       = compute_weights(fb, half_life)

    records: list[dict] = []

    # ---- stratum-level updates (cause × corridor) ----
    for (cause, corr), grp in fb.groupby(["event_cause", "corridor_fill"]):
        y, w = grp["log_dur"].values, grp["w"].values
        n_eff, mu_fb, var_fb = weighted_stats(y, w)

        # Choose prior level
        key = (cause, corr)
        if key in stratum_priors and stratum_priors[key][2] >= SPARSE_N_EFF:
            mu_p, var_p, _ = stratum_priors[key]
            prior_level = "stratum"
        elif cause in cause_priors:
            mu_p, var_p, _ = cause_priors[cause]
            prior_level = "cause"
        else:
            mu_p, var_p = mu_g, var_g
            prior_level = "global"

        mu_post, var_post = normal_normal_posterior(mu_p, var_p, n_eff, mu_fb, var_fb)
        ci_lo, ci_hi      = credible_interval(mu_post, var_post)
        q50, q80, q95     = log_to_minute_quantiles(mu_post, var_post)

        records.append({
            "stratum_cause":          cause,
            "stratum_corridor":       corr,
            "prior_level":            prior_level,
            "prior_mu_log":           round(mu_p, 6),
            "prior_sigma_log":        round(np.sqrt(var_p), 6),
            "n_eff_feedback":         round(n_eff, 2),
            "feedback_mu_log":        round(mu_fb, 6) if not np.isnan(mu_fb) else None,
            "posterior_mu_log":       round(mu_post, 6),
            "posterior_sigma_log":    round(np.sqrt(var_post), 6),
            "ci95_lo_log":            round(ci_lo, 6),
            "ci95_hi_log":            round(ci_hi, 6),
            "posterior_median_min":   round(q50, 2),
            "posterior_p80_min":      round(q80, 2),
            "posterior_p95_min":      round(q95, 2),
        })

    # ---- cause-only rollup for causes with no stratum update ----
    covered_causes = {r["stratum_cause"] for r in records}
    for cause, grp in fb.groupby("event_cause"):
        if cause not in covered_causes:
            y, w = grp["log_dur"].values, grp["w"].values
            n_eff, mu_fb, var_fb = weighted_stats(y, w)
            mu_p, var_p = (cause_priors[cause][:2] if cause in cause_priors else (mu_g, var_g))
            mu_post, var_post = normal_normal_posterior(mu_p, var_p, n_eff, mu_fb, var_fb)
            ci_lo, ci_hi      = credible_interval(mu_post, var_post)
            q50, q80, q95     = log_to_minute_quantiles(mu_post, var_post)
            records.append({
                "stratum_cause":          cause,
                "stratum_corridor":       "ALL",
                "prior_level":            "cause_only",
                "prior_mu_log":           round(mu_p, 6),
                "prior_sigma_log":        round(np.sqrt(var_p), 6),
                "n_eff_feedback":         round(n_eff, 2),
                "feedback_mu_log":        round(mu_fb, 6) if not np.isnan(mu_fb) else None,
                "posterior_mu_log":       round(mu_post, 6),
                "posterior_sigma_log":    round(np.sqrt(var_post), 6),
                "ci95_lo_log":            round(ci_lo, 6),
                "ci95_hi_log":            round(ci_hi, 6),
                "posterior_median_min":   round(q50, 2),
                "posterior_p80_min":      round(q80, 2),
                "posterior_p95_min":      round(q95, 2),
            })

    summary_df = pd.DataFrame(records).sort_values(
        ["stratum_cause", "stratum_corridor"]
    ).reset_index(drop=True)

    # ---- global posterior ----
    y_all, w_all        = fb["log_dur"].values, fb["w"].values
    n_eff_g, mu_fb_g, var_fb_g = weighted_stats(y_all, w_all)
    mu_gp, var_gp       = normal_normal_posterior(mu_g, var_g, n_eff_g, mu_fb_g, var_fb_g)
    ci_lo_g, ci_hi_g    = credible_interval(mu_gp, var_gp)
    q50_g, q80_g, q95_g = log_to_minute_quantiles(mu_gp, var_gp)

    # build per-stratum dict for JSON
    strata_json = []
    for r in records:
        strata_json.append({
            "cause":            r["stratum_cause"],
            "corridor":         r["stratum_corridor"],
            "prior_level":      r["prior_level"],
            "posterior_mu_log": r["posterior_mu_log"],
            "posterior_sigma_log": r["posterior_sigma_log"],
            "ci95_lo_log":      r["ci95_lo_log"],
            "ci95_hi_log":      r["ci95_hi_log"],
            "posterior_median_min": r["posterior_median_min"],
            "posterior_p80_min":    r["posterior_p80_min"],
            "posterior_p95_min":    r["posterior_p95_min"],
            "n_eff_feedback":   r["n_eff_feedback"],
        })

    global_dict: dict = {
        "metadata": {
            "prior_period":    "2023-11-10 to 2024-02-29",
            "feedback_period": "2024-03-01 to 2024-04-08",
            "model":           "hierarchical_normal_normal_conjugate",
            "forgetting_model": "exponential",
            "half_life_days":  half_life,
            "n_strata_updated": len(records),
        },
        "global_prior": {
            "mu_log":    round(mu_g, 6),
            "sigma_log": round(np.sqrt(var_g), 6),
            "n_obs_prior": int(
                len(prior_df[prior_df["duration_min"].notna() & ~prior_df["is_censored"]])
            ),
        },
        "global_posterior": {
            "mu_log":               round(mu_gp, 6),
            "sigma_log":            round(np.sqrt(var_gp), 6),
            "ci95_lo_log":          round(ci_lo_g, 6),
            "ci95_hi_log":          round(ci_hi_g, 6),
            "posterior_median_min": round(q50_g, 2),
            "posterior_p80_min":    round(q80_g, 2),
            "posterior_p95_min":    round(q95_g, 2),
            "n_eff_feedback":       round(n_eff_g, 2),
        },
        "strata": strata_json,
    }

    return summary_df, global_dict
