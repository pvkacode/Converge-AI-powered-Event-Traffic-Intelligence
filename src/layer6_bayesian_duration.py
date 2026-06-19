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
_TRIM_PCT         = 0.05      # fraction trimmed from each tail in robust global estimator


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
    """Parameter quantiles in original minute scale (uses posterior SD on mu only)."""
    sd  = np.sqrt(max(var, MIN_VAR))
    q50 = float(np.expm1(mu))
    q80 = float(np.expm1(mu + stats.norm.ppf(0.80) * sd))
    q95 = float(np.expm1(mu + stats.norm.ppf(0.95) * sd))
    return q50, q80, q95


def predictive_variance(var_post: float, var_obs: float) -> float:
    """Posterior predictive variance: var(mu | D) + var(y | mu)."""
    return max(var_post, MIN_VAR) + max(var_obs, MIN_VAR)


def predictive_sigma(var_post: float, var_obs: float) -> float:
    """Posterior predictive standard deviation for a new log-duration observation."""
    return float(np.sqrt(predictive_variance(var_post, var_obs)))


def obs_variance_from_values(
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> float | None:
    """Weighted observation variance from log-duration values."""
    y = np.asarray(y, dtype=float)
    if len(y) < 1:
        return None
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    _, _, var_w = weighted_stats(y, w)
    return float(var_w) if np.isfinite(var_w) else None


def build_obs_variance_pools(fb: pd.DataFrame) -> dict:
    """
    Build stratum, cause, and global observation-variance pools from feedback.

    Returns dict with keys: global (float|None), cause {str: float}, stratum {(cause,corr): float}
    """
    pools: dict = {"global": None, "cause": {}, "stratum": {}}
    if fb.empty or "log_dur" not in fb.columns:
        return pools

    pools["global"] = obs_variance_from_values(fb["log_dur"].values)
    for cause, grp in fb.groupby("event_cause"):
        c = str(cause)
        pools["cause"][c] = obs_variance_from_values(grp["log_dur"].values)
        if "corridor_fill" in grp.columns:
            for corr, sgrp in grp.groupby("corridor_fill"):
                pools["stratum"][(c, str(corr))] = obs_variance_from_values(
                    sgrp["log_dur"].values
                )
    return pools


def resolve_obs_variance(
    pools: dict,
    cause: str | None = None,
    corridor: str | None = None,
    local_y: np.ndarray | None = None,
    local_w: np.ndarray | None = None,
    prior_period_var: float | None = None,
) -> tuple[float, str, bool]:
    """
    Resolve observation variance with explicit fallback chain.

    Returns (var_obs, source_label, valid).
    """
    if local_y is not None and len(local_y) >= 2:
        v = obs_variance_from_values(local_y, local_w)
        if v is not None and np.isfinite(v):
            return v, "local_feedback", True

    if cause and corridor:
        v = pools.get("stratum", {}).get((cause, corridor))
        if v is not None and np.isfinite(v):
            return v, "pooled_stratum", True

    if cause:
        v = pools.get("cause", {}).get(cause)
        if v is not None and np.isfinite(v):
            return v, "pooled_cause", True

    v = pools.get("global")
    if v is not None and np.isfinite(v):
        return v, "pooled_global", True

    if prior_period_var is not None and np.isfinite(prior_period_var):
        return max(float(prior_period_var), MIN_VAR), "prior_period_sample_var", True

    return MIN_VAR, "min_var_floor", False


def log_to_minute_predictive_quantiles(
    mu: float, var_post: float, var_obs: float,
) -> tuple[float, float, float]:
    """Posterior predictive quantiles in original minute scale."""
    sd = predictive_sigma(var_post, var_obs)
    q50 = float(np.expm1(mu))
    q80 = float(np.expm1(mu + stats.norm.ppf(0.80) * sd))
    q95 = float(np.expm1(mu + stats.norm.ppf(0.95) * sd))
    return q50, q80, q95


# ---------------------------------------------------------------------------
# Robust estimator for the global level
# ---------------------------------------------------------------------------

def _robust_trimmed_stats(
    y: np.ndarray,
    w: np.ndarray,
    trim_pct: float = _TRIM_PCT,
) -> tuple[float, float, int]:
    """
    Trimmed weighted mean and variance for robust global estimation.

    Removes the outer `trim_pct` fraction from each tail of `y`, then computes
    weighted mean and variance on the remaining observations.  This is resistant
    to the extreme values that cause plain weighted-mean to produce NaN when
    log1p(negative_duration) slips through.

    Returns (mu_trim, var_trim, n_trim).
    Falls back to full-set weighted mean when n < 4 or trimming leaves < 2 obs.
    """
    finite = np.isfinite(y) & np.isfinite(w)
    y_v = y[finite]
    w_v = w[finite]
    n = len(y_v)

    if n == 0:
        return float("nan"), float("nan"), 0

    if n < 4:
        mu = float(np.average(y_v, weights=w_v))
        var = float(np.average((y_v - mu) ** 2, weights=w_v))
        return mu, max(var, MIN_VAR), n

    k = max(1, int(np.floor(n * trim_pct)))
    order = np.argsort(y_v)
    keep  = order[k : n - k]

    if len(keep) < 2:
        mu  = float(np.average(y_v, weights=w_v))
        var = float(np.average((y_v - mu) ** 2, weights=w_v))
        return mu, max(var, MIN_VAR), n

    y_t = y_v[keep]
    w_t = w_v[keep]
    mu_t  = float(np.average(y_t, weights=w_t))
    var_t = float(np.average((y_t - mu_t) ** 2, weights=w_t))
    return mu_t, max(var_t, MIN_VAR), len(keep)


# ---------------------------------------------------------------------------
# Prior extraction from the prior period dataset
# ---------------------------------------------------------------------------

def _extract_priors(pr_df: pd.DataFrame) -> tuple[dict, dict, float, float]:
    """
    Returns:
        stratum_priors  – {(cause, corridor): (mu, var, n)}
        cause_priors    – {cause: (mu, var, n)}
        mu_global       – global mean (robust trimmed, from clean rows only)
        var_global      – global variance

    Quality gate: rows with duration_min <= 0 or producing non-finite log1p
    are silently excluded here.  Callers that need the quarantine count should
    use _quarantine_feedback_rows() in layer6_adaptive_learning.py.
    """
    raw = pr_df[pr_df["duration_min"].notna() & ~pr_df["is_censored"]].copy()
    # Strict positive gate — log1p requires duration > -1, we require > 0 for safety
    obs = raw[raw["duration_min"] > 0].copy()
    obs["log_dur"] = np.log1p(obs["duration_min"])
    # Defensive: remove any residual NaN/inf (e.g. very large durations → Inf)
    obs = obs[np.isfinite(obs["log_dur"].values)].copy()

    # Robust global stats via trimmed mean
    log_dur_arr = obs["log_dur"].values
    w_ones      = np.ones(len(log_dur_arr))
    mu_global, var_global, _ = _robust_trimmed_stats(log_dur_arr, w_ones)
    if not np.isfinite(mu_global):
        mu_global = float(np.nanmedian(log_dur_arr)) if len(log_dur_arr) > 0 else 0.0
    if not np.isfinite(var_global) or var_global < MIN_VAR:
        var_global = float(np.nanvar(log_dur_arr)) if len(log_dur_arr) > 0 else MIN_VAR
        var_global = max(var_global, MIN_VAR)

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

    # Feedback: restrict to uncensored, positive-duration observations only.
    # Rows with duration_min <= 0 produce NaN in log1p and are excluded here.
    # The upstream quarantine in layer6_adaptive_learning.py provides the full
    # audit trail; this is a defensive second gate to ensure posterior integrity.
    _raw_fb = feedback_df[
        feedback_df["duration_min"].notna() & ~feedback_df["is_censored"]
    ].copy()
    fb = _raw_fb[_raw_fb["duration_min"] > 0].copy()
    fb["log_dur"] = np.log1p(fb["duration_min"])
    fb = fb[np.isfinite(fb["log_dur"].values)].copy()   # remove any residual non-finite
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

    # ---- global posterior (robust, NaN-safe) ----
    #
    # Use a trimmed-mean estimator on the clean feedback observations.
    # If estimation fails for any reason, fall back to the prior — never NaN.
    y_all   = fb["log_dur"].values
    w_all   = fb["w"].values
    n_clean = int(np.isfinite(y_all).sum())

    global_posterior_valid  = False
    global_posterior_reason = "not_computed"

    if n_clean >= 10:
        mu_fb_g, var_fb_g, n_trim = _robust_trimmed_stats(y_all, w_all)
        n_eff_g = kish_n_eff(w_all[np.isfinite(y_all)])
        if np.isfinite(mu_fb_g) and np.isfinite(var_fb_g) and n_eff_g >= 1:
            mu_gp, var_gp = normal_normal_posterior(mu_g, var_g, n_eff_g,
                                                    mu_fb_g, var_fb_g)
            if np.isfinite(mu_gp) and np.isfinite(var_gp):
                global_posterior_valid  = True
                global_posterior_reason = (
                    f"robust_trimmed_mean; n_clean={n_clean}; n_trimmed={n_trim}"
                )
            else:
                mu_gp, var_gp = mu_g, var_g
                global_posterior_reason = (
                    f"posterior_update_nan_fallback_to_prior; n_clean={n_clean}"
                )
        else:
            mu_gp, var_gp = mu_g, var_g
            global_posterior_reason = (
                f"trimmed_stats_failed_fallback_to_prior; n_clean={n_clean}"
            )
    elif 0 < n_clean < 10:
        mu_gp, var_gp = mu_g, var_g
        global_posterior_reason = (
            f"insufficient_clean_obs_{n_clean}_fallback_to_prior"
        )
    else:
        mu_gp, var_gp = mu_g, var_g
        global_posterior_reason = "no_clean_feedback_fallback_to_prior"

    # Final NaN guard — must never reach JSON as NaN
    if not np.isfinite(mu_gp):
        mu_gp  = mu_g
        var_gp = var_g
        global_posterior_valid  = False
        global_posterior_reason = "final_nan_guard_triggered_using_prior"

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
            "n_clean_feedback": n_clean,
            "n_raw_uncensored": int(len(_raw_fb)),
            "n_quarantined_internal": int(len(_raw_fb)) - n_clean,
        },
        "global_prior": {
            "mu_log":    round(mu_g, 6),
            "sigma_log": round(np.sqrt(var_g), 6),
            "n_obs_prior": int(
                len(prior_df[prior_df["duration_min"].notna() & ~prior_df["is_censored"]])
            ),
        },
        "global_posterior": {
            "mu_log":               round(float(mu_gp), 6),
            "sigma_log":            round(float(np.sqrt(max(var_gp, MIN_VAR))), 6),
            "ci95_lo_log":          round(float(ci_lo_g), 6),
            "ci95_hi_log":          round(float(ci_hi_g), 6),
            "posterior_median_min": round(float(q50_g), 2),
            "posterior_p80_min":    round(float(q80_g), 2),
            "posterior_p95_min":    round(float(q95_g), 2),
            "n_eff_feedback":       round(float(n_eff_g) if "n_eff_g" in dir() else 0.0, 2),
            "global_posterior_valid":  global_posterior_valid,
            "global_posterior_reason": global_posterior_reason,
        },
        "strata": strata_json,
    }

    return summary_df, global_dict
