"""
Layer 6 - Adaptive Learning (main orchestrator)

ADDITIVE module: reads upstream outputs, emits recommendations only.
No Layer 4.5 or Layer 5 source files are ever modified.

Dataset constraint: ASTraM log is Nov 2023-Apr 2024 (historical batch only).
  Prior period   : Nov 2023 - Feb 2024  (training baseline)
  Feedback batch : Mar 2024 - Apr 2024  (newly-closed incidents)

Components (Part 1)
-------------------
1. Hierarchical Bayesian Duration Update   -> layer6_bayesian_duration.py
2. Calibration Posterior Update            -> layer6_calibration_updates.py
3. Drift Detection (PH + PSI + ODS)        -> layer6_drift_detection.py
4. Prototype Trust Update (Beta-Binomial)  -> inline below (uses layer4 outputs)
5. Retrain Trigger Generation              -> layer6_retrain_triggers.py

Components (Part 2)
-------------------
6. Resource Effectiveness Posteriors       -> inline (Bayesian linear regression)
7. Bayesian Model Averaging Weights        -> inline (NLL-weighted model ensemble)
8. Model Health Monitoring                 -> inline (rolling metric tracking)

Outputs
-------
outputs/layer6_posterior_duration_priors.json
outputs/layer6_duration_posterior_summary.csv
outputs/layer6_calibration_posteriors.csv
outputs/layer6_drift_report.csv
outputs/layer6_retrain_triggers.csv
outputs/layer6_prototype_trust_updates.csv
outputs/layer6_feedback_log.csv
outputs/layer6_learning_summary.txt
outputs/layer6_resource_effectiveness_posteriors.json
outputs/layer6_bma_weights.csv
outputs/layer6_model_health_summary.csv
outputs/layer6_versioned_knowledge_base.json
outputs/layer6_recalibration_recommendations.csv
outputs/layer6_active_alerts.csv
outputs/layer6_posterior_uncertainty.csv
outputs/layer6_prototype_diagnostics.csv
outputs/layer6_model_artifacts/
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---- Layer 6 helpers ----
from layer6_feedback_store import (
    load_events,
    split_periods,
    load_state_vector,
    load_high_impact_probs,
    load_cause_tau,
    load_l45_metrics,
    load_violations,
    load_resource_allocation,
    load_cvar_comparison,
    load_opt_metrics,
    load_shadow_prices,
    load_prototypes,
    load_prototype_utilization,
    build_feedback_actuals,
    join_predictions_to_feedback,
)
from layer6_bayesian_duration import update_duration_posteriors
from layer6_calibration_updates import update_calibration_posteriors
from layer6_drift_detection import run_drift_detection
from layer6_retrain_triggers import build_retrain_triggers

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = OUTPUTS_DIR / "layer6_model_artifacts"
OUTPUTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

_NOW_STR = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------------------------------------------------------------------------
# Part 1, Component 4 - Prototype Trust Update (Beta-Binomial)
# ---------------------------------------------------------------------------

ETA              = 0.20     # EMA learning rate
BETA_SCALE       = 10.0     # Beta-Binomial pseudo-count scaling
GOOD_RESID_THRESH = 0.5     # |log_pred - log_actual| < 0.5 -> good outcome
POOR_RESID_THRESH = 1.5     # > 1.5 -> poor outcome


def _update_prototype_trust(
    prototypes_df: pd.DataFrame,
    utilization_df: pd.DataFrame,
    feedback_actuals: pd.DataFrame,
    state_vector: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-prototype Beta-Binomial trust update with EMA.
    Q_p^(t+1) = (1-eta)*Q_p^(t) + eta*R_p, R_p = alpha/(alpha+beta).
    """
    sv_cols = ["event_id", "retrieval_confidence", "duration_p50"]
    sv = state_vector[sv_cols].copy() if all(c in state_vector.columns for c in sv_cols) else pd.DataFrame()

    fb = feedback_actuals.merge(sv, on="event_id", how="left") if not sv.empty else feedback_actuals.copy()
    if "retrieval_confidence" not in fb.columns:
        fb["retrieval_confidence"] = 0.0
    if "duration_p50" not in fb.columns:
        fb["duration_p50"] = np.nan

    fb["log_pred"]   = np.log1p(fb["duration_p50"].clip(lower=0))
    fb["log_actual"] = np.log1p(fb["duration_min"].clip(lower=0))
    fb["residual"]   = (fb["log_pred"] - fb["log_actual"]).abs()
    fb["retrieval_confidence"] = fb["retrieval_confidence"].fillna(0.0)

    records = []
    for _, proto in prototypes_df.iterrows():
        pid     = int(proto["prototype_id"])
        cause   = str(proto.get("event_cause", proto.get("cause", "")))
        corridor = str(proto.get("corridor", "Non-corridor"))
        trust_prior = float(proto.get("mean_trust", proto.get("prototype_quality_score", 0.8)))

        matched = fb[fb["event_cause"] == cause].copy()
        if corridor != "Non-corridor" and len(matched) > 0:
            cm = matched[matched["corridor_fill"] == corridor]
            if len(cm) > 0:
                matched = cm

        alpha0 = trust_prior * BETA_SCALE
        beta0  = (1 - trust_prior) * BETA_SCALE

        if len(matched) == 0:
            records.append({
                "prototype_id": pid, "cause": cause, "corridor": corridor,
                "n_matched_feedback": 0,
                "trust_prior": round(trust_prior, 4),
                "alpha_prior": round(alpha0, 4), "beta_prior": round(beta0, 4),
                "n_successes": 0, "n_failures": 0,
                "alpha_posterior": round(alpha0, 4), "beta_posterior": round(beta0, 4),
                "r_p": round(trust_prior, 4), "trust_updated": round(trust_prior, 4),
                "trust_delta": 0.0,
                "posterior_uncertainty": round(
                    float(np.sqrt(trust_prior * (1 - trust_prior) / (BETA_SCALE + 1))), 4
                ),
                "mean_residual": None,
                "update_source": "no_feedback",
            })
            continue

        n_success = 0.0
        n_failure = 0.0
        residuals = []
        for _, ev in matched.iterrows():
            conf = float(ev["retrieval_confidence"])
            res  = float(ev["residual"]) if not np.isnan(ev["residual"]) else POOR_RESID_THRESH + 1
            residuals.append(res)
            good = res < GOOD_RESID_THRESH
            poor = res > POOR_RESID_THRESH
            confident = conf > 0.7
            if confident and good:
                n_success += 1.0
            elif confident and poor:
                n_failure += 1.0
            elif not confident and good:
                n_success += 0.5

        alpha_post = alpha0 + n_success
        beta_post  = beta0  + n_failure
        r_p        = alpha_post / (alpha_post + beta_post)
        trust_updated = (1 - ETA) * trust_prior + ETA * r_p
        post_unc = float(np.sqrt(
            (alpha_post * beta_post) /
            ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
        ))

        records.append({
            "prototype_id": pid, "cause": cause, "corridor": corridor,
            "n_matched_feedback": len(matched),
            "trust_prior": round(trust_prior, 4),
            "alpha_prior": round(alpha0, 4), "beta_prior": round(beta0, 4),
            "n_successes": round(n_success, 2), "n_failures": round(n_failure, 2),
            "alpha_posterior": round(alpha_post, 4), "beta_posterior": round(beta_post, 4),
            "r_p": round(r_p, 4), "trust_updated": round(trust_updated, 4),
            "trust_delta": round(trust_updated - trust_prior, 4),
            "posterior_uncertainty": round(post_unc, 4),
            "mean_residual": round(float(np.mean(residuals)), 4) if residuals else None,
            "update_source": "beta_binomial_ema",
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Part 2, Component 6 - Resource Effectiveness Posteriors
# ---------------------------------------------------------------------------

_GAMMA_NAMES  = ["gamma_p", "gamma_b", "gamma_t", "gamma_q"]
_GAMMA_PRIOR  = np.array([0.18, 0.10, 0.25, 0.30])   # Layer 5 fixed hyperparams
_GAMMA_SIGMA0 = np.array([0.08, 0.05, 0.10, 0.10])   # prior std dev


def _resource_effectiveness_posteriors(
    layer5_alloc: pd.DataFrame,
    feedback_actuals: pd.DataFrame,
    opt_metrics: pd.DataFrame,
    shadow_prices: pd.DataFrame,
    cvar_comparison: pd.DataFrame,
) -> dict:
    """
    Bayesian linear regression update over Layer 5 resource effectiveness parameters.

    Model: z_i = x_i^T beta + epsilon,  beta ~ N(beta_0, Sigma_0)
    beta = [gamma_p, gamma_b, gamma_t, gamma_q]

    Two evidence paths:
      (A) Event-level: join L5 allocations to feedback actuals, use log-ratio residual.
      (B) Aggregate:   use shadow prices as indirect evidence scaled to gamma-space.

    CAUTION: Evidence is observational / model-derived. Shadow prices come from
    the L5 optimization model, not real-world outcome measurement. Do not interpret
    posterior means as causal lift estimates.
    """
    beta_0  = _GAMMA_PRIOR.copy()
    sigma_0 = _GAMMA_SIGMA0.copy()
    Sigma_0 = np.diag(sigma_0 ** 2)
    mu_post  = beta_0.copy()
    Sigma_post = Sigma_0.copy()
    evidence_path = "prior_only"
    n_obs = 0

    # --- Path A: event-level regression ---
    alloc_cols = ["event_id", "officers_allocated", "barricades_allocated",
                  "tow_trucks_allocated", "qru_allocated"]
    dur_cols = ["event_id", "safe_duration_p50"]

    if all(c in layer5_alloc.columns for c in alloc_cols):
        fb_ids = feedback_actuals[["event_id", "duration_min", "log_duration"]].copy()
        merged = fb_ids.merge(layer5_alloc[alloc_cols], on="event_id", how="inner")

        if len(merged) >= 5:
            evidence_path = "event_level"
            n_obs = len(merged)

            # Outcome: log-duration residual vs baseline (safe_p50 as reference)
            if "safe_duration_p50" in layer5_alloc.columns:
                merged = merged.merge(
                    layer5_alloc[["event_id", "safe_duration_p50"]], on="event_id", how="left"
                )
                log_pred = np.log1p(merged["safe_duration_p50"].clip(lower=0))
                z = (merged["log_duration"].values - log_pred.values)
            else:
                z = merged["log_duration"].values - merged["log_duration"].mean()

            X = merged[["officers_allocated", "barricades_allocated",
                         "tow_trucks_allocated", "qru_allocated"]].values.astype(float)
            X[:, 0] /= 12.0   # per-site cap: officers
            X[:, 1] /= 20.0   # barricades
            X[:, 2] /= 4.0    # tow
            X[:, 3] /= 3.0    # qru

            sigma_noise = max(float(np.std(z)), 0.05)
            Lambda_0 = np.linalg.inv(Sigma_0 + 1e-9 * np.eye(4))
            Lambda_n = Lambda_0 + (1.0 / sigma_noise ** 2) * X.T @ X
            rhs = Lambda_0 @ beta_0 + (1.0 / sigma_noise ** 2) * X.T @ z
            try:
                mu_post  = np.linalg.solve(Lambda_n, rhs)
                Sigma_post = np.linalg.inv(Lambda_n)
            except np.linalg.LinAlgError:
                mu_post  = beta_0.copy()
                Sigma_post = Sigma_0.copy()

    # --- Path B: shadow-price aggregate ---
    if evidence_path == "prior_only" and not shadow_prices.empty:
        evidence_path = "shadow_price_aggregate"
        n_obs = min(len(shadow_prices), 4)

        sp_map: dict[str, float] = {}
        if "resource" in shadow_prices.columns and "marginal_value" in shadow_prices.columns:
            for _, row in shadow_prices.iterrows():
                sp_map[str(row["resource"]).lower().strip()] = float(row["marginal_value"])

        # Estimate scale: shadow_price_k ~ gamma_k * mean_delay_per_site * (1 - mean_eff)
        m = {r["metric"]: r["value"] for _, r in opt_metrics.iterrows()} if not opt_metrics.empty else {}
        n_sites   = float(m.get("n_active_sites", 50))
        raw_del   = float(m.get("expected_total_delay_raw", 7e6))
        # Use delay reduction percentage as effectiveness proxy (not CC satisfaction rate)
        delay_red_pct = float(m.get("expected_delay_reduction_pct", 50.0))
        mean_eff  = delay_red_pct / 100.0
        scale     = (raw_del / max(n_sites, 1)) * (1.0 - mean_eff)

        sp_keys = {"police": 0, "barricades": 1, "tow": 2, "qru": 3}
        for res_name, idx in sp_keys.items():
            if res_name in sp_map and scale > 0:
                obs_gamma    = sp_map[res_name] / scale
                sig_n_k      = 0.50 * beta_0[idx]          # 50% CV noise
                denom        = sigma_0[idx] ** 2 + sig_n_k ** 2
                mu_post[idx] = (sig_n_k ** 2 * beta_0[idx] +
                                sigma_0[idx] ** 2 * obs_gamma) / denom
                Sigma_post[idx, idx] = (sigma_0[idx] ** 2 * sig_n_k ** 2) / denom

    # --- Build output record ---
    posterior_params = []
    for i, name in enumerate(_GAMMA_NAMES):
        sig_post_k = float(np.sqrt(max(Sigma_post[i, i], 1e-10)))
        ci_lo = float(mu_post[i] - 1.96 * sig_post_k)
        ci_hi = float(mu_post[i] + 1.96 * sig_post_k)
        posterior_params.append({
            "parameter":        name,
            "prior_mean":       round(float(beta_0[i]), 4),
            "prior_std":        round(float(sigma_0[i]), 4),
            "posterior_mean":   round(float(mu_post[i]), 4),
            "posterior_std":    round(sig_post_k, 4),
            "ci95_lo":          round(ci_lo, 4),
            "ci95_hi":          round(ci_hi, 4),
            "shift_from_prior": round(float(mu_post[i] - beta_0[i]), 4),
        })

    cvar_red = None
    if not cvar_comparison.empty and "percentage_reduction" in cvar_comparison.columns:
        row = cvar_comparison[cvar_comparison["scope"] == "all_sites"]
        if not row.empty:
            cvar_red = round(float(row["percentage_reduction"].iloc[0]), 4)

    return {
        "metadata": {
            "model":          "bayesian_linear_regression",
            "parameters":     _GAMMA_NAMES,
            "evidence_path":  evidence_path,
            "n_observations": n_obs,
            "generated_at":   _NOW_STR,
        },
        "caution": (
            "Evidence is observational / model-derived. Shadow prices are derived from "
            "the Layer 5 optimization model, not real-world measured outcomes. "
            "Posterior means reflect updated parameter beliefs, not causal effect estimates."
        ),
        "prior":     {"means": beta_0.tolist(), "stds": sigma_0.tolist()},
        "posterior": {"parameters": posterior_params},
        "l5_aggregate_evidence": {
            "cvar_reduction_pct": cvar_red,
            "shadow_prices": {
                k: round(v, 2) for k, v in
                dict(zip(shadow_prices["resource"], shadow_prices["marginal_value"])).items()
            } if not shadow_prices.empty and "resource" in shadow_prices.columns else {},
        },
    }


# ---------------------------------------------------------------------------
# Part 2, Component 7 - Bayesian Model Averaging Weights
# ---------------------------------------------------------------------------

_BMA_MODELS = [
    "duration_catboost",
    "retrieval_estimator",
    "calibration_estimator",
    "corridor_cause_prior",
    "scenario_surrogate",
]

_LOG2PI = float(np.log(2.0 * np.pi))


def _gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    """Mean Gaussian NLL in log-space."""
    sigma = max(sigma, 1e-6)
    return float(np.mean(0.5 * ((y - mu) / sigma) ** 2 + 0.5 * _LOG2PI + np.log(sigma)))


def _compute_bma_weights(
    joined_df: pd.DataFrame,
    dur_summary_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    layer5_alloc: pd.DataFrame,
    l45_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Maintain posterior weights over 5 model components using AIC-style BMA.
    w_m  proportional to  exp(-NLL_m / 2),  then normalize.

    NLL proxies (all in log-duration space or calibrated probability space):
      1. duration_catboost    : mean log-space Gaussian NLL on feedback actuals
      2. retrieval_estimator  : NLL proxy from Layer 4.5 retrieval confidence metrics
      3. calibration_estimator: ECE + Brier score composite
      4. corridor_cause_prior : NLL of feedback actuals under stratum prior distribution
      5. scenario_surrogate   : NLL of feedback actuals under L5 lognormal surrogate
    """
    nll: dict[str, float] = {}

    # 1. Duration CatBoost: log-space Gaussian NLL
    if "log_duration" in joined_df.columns and "duration_p50" in joined_df.columns:
        valid = joined_df[joined_df["duration_p50"].notna() & joined_df["log_duration"].notna()].copy()
        if len(valid) > 0:
            log_pred = np.log1p(valid["duration_p50"].clip(lower=0).values)
            log_act  = valid["log_duration"].values
            resid    = log_act - log_pred
            sigma_dur = max(float(np.std(resid)), 0.1)
            nll["duration_catboost"] = _gaussian_nll(log_act, log_pred, sigma_dur)
        else:
            nll["duration_catboost"] = 3.0
    else:
        nll["duration_catboost"] = 3.0

    # 2. Retrieval estimator: use L4.5 holdout planned-event RMSLE as proxy
    ret_rmsle = None
    if not l45_metrics_df.empty:
        mask = (l45_metrics_df["subset"] == "holdout_planned") & \
               (l45_metrics_df["metric"] == "rmsle")
        if mask.any():
            ret_rmsle = float(l45_metrics_df[mask]["value"].iloc[0])
    nll["retrieval_estimator"] = ret_rmsle if ret_rmsle is not None else 2.5

    # 3. Calibration estimator: ECE + Brier composite (lower = better)
    if not cal_df.empty and "ece_before_update" in cal_df.columns:
        ece   = float(cal_df["ece_before_update"].iloc[0])
        brier = float(cal_df["brier_raw"].iloc[0]) if "brier_raw" in cal_df.columns else 0.1
        nll["calibration_estimator"] = ece * 5.0 + brier * 2.0
    else:
        nll["calibration_estimator"] = 2.0

    # 4. Corridor-cause prior: NLL of feedback actuals under stratum prior N(mu_prior, sigma_prior)
    if (not dur_summary_df.empty and "log_duration" in joined_df.columns
            and "event_cause" in joined_df.columns):
        prior_map = {
            (r["stratum_cause"], r["stratum_corridor"]): (r["prior_mu_log"], r["prior_sigma_log"])
            for _, r in dur_summary_df.iterrows()
            if r["stratum_corridor"] != "ALL"
        }
        cause_map = {
            r["stratum_cause"]: (r["prior_mu_log"], r["prior_sigma_log"])
            for _, r in dur_summary_df[dur_summary_df["stratum_corridor"] == "ALL"].iterrows()
        }
        global_mu  = float(dur_summary_df["prior_mu_log"].mean())
        global_sig = max(float(dur_summary_df["prior_sigma_log"].mean()), 0.1)

        nll_prior_vals = []
        for _, row in joined_df.iterrows():
            y_i  = row.get("log_duration")
            if y_i is None or np.isnan(float(y_i)):
                continue
            cause  = str(row.get("event_cause", ""))
            corr   = str(row.get("corridor_fill", ""))
            if (cause, corr) in prior_map:
                mu_p, sig_p = prior_map[(cause, corr)]
            elif cause in cause_map:
                mu_p, sig_p = cause_map[cause]
            else:
                mu_p, sig_p = global_mu, global_sig
            sig_p = max(sig_p, 0.01)
            nll_prior_vals.append(
                0.5 * ((float(y_i) - mu_p) / sig_p) ** 2 + 0.5 * _LOG2PI + np.log(sig_p)
            )
        nll["corridor_cause_prior"] = float(np.mean(nll_prior_vals)) if nll_prior_vals else 3.0
    else:
        nll["corridor_cause_prior"] = 3.0

    # 5. Scenario surrogate: NLL of feedback actuals under L5 lognormal fit
    if (not layer5_alloc.empty and "log_duration" in joined_df.columns
            and "safe_duration_p50" in layer5_alloc.columns
            and "safe_duration_p95" in layer5_alloc.columns):
        sv_df = layer5_alloc[["event_id", "safe_duration_p50", "safe_duration_p95"]].copy()
        sv_df = sv_df[sv_df["safe_duration_p50"] > 0]
        merged_surr = joined_df[["event_id", "log_duration"]].merge(sv_df, on="event_id", how="inner")
        if len(merged_surr) > 0:
            mu_surr   = np.log(merged_surr["safe_duration_p50"].clip(lower=0.01))
            p95_safe  = merged_surr["safe_duration_p95"].clip(lower=merged_surr["safe_duration_p50"] + 0.01)
            sig_surr  = (np.log(p95_safe) - mu_surr) / 1.645
            sig_surr  = sig_surr.clip(lower=0.05)
            nll_vals  = 0.5 * ((merged_surr["log_duration"].values - mu_surr.values) /
                                sig_surr.values) ** 2 + np.log(sig_surr.values)
            nll["scenario_surrogate"] = float(np.mean(nll_vals))
        else:
            nll["scenario_surrogate"] = 3.5
    else:
        nll["scenario_surrogate"] = 3.5

    # Compute BMA weights: w_m ∝ exp(-NLL_m / 2)
    nll_arr = np.array([nll.get(m, 3.0) for m in _BMA_MODELS], dtype=float)
    # Replace NaN with median of valid values (or fallback 3.0)
    valid_nll = nll_arr[~np.isnan(nll_arr)]
    fallback_nll = float(np.median(valid_nll)) if len(valid_nll) > 0 else 3.0
    nll_arr = np.where(np.isnan(nll_arr), fallback_nll, nll_arr)
    # Clip to prevent overflow / underflow
    nll_arr = np.clip(nll_arr, -10, 50)
    log_w_unnorm = -0.5 * nll_arr
    log_w_unnorm -= log_w_unnorm.max()        # numerically stable softmax
    w_unnorm = np.exp(log_w_unnorm)
    w_norm   = w_unnorm / w_unnorm.sum()

    # Prior weights (uniform, since this is the first batch)
    w_prior = np.ones(len(_BMA_MODELS)) / len(_BMA_MODELS)

    # Posterior weight uncertainty: entropy of weight distribution
    entropy = float(-np.sum(w_norm * np.log(w_norm + 1e-12)))

    # Rolling drift: KL(current || uniform)
    kl_from_uniform = float(np.sum(w_norm * np.log(w_norm / w_prior + 1e-12)))

    rows = []
    for i, m in enumerate(_BMA_MODELS):
        raw_nll = nll.get(m, 3.0)
        rows.append({
            "model":              m,
            "nll_proxy_raw":      round(float(raw_nll), 6) if not np.isnan(raw_nll) else None,
            "nll_proxy_used":     round(float(nll_arr[i]), 6),
            "weight_prior":       round(float(w_prior[i]), 6),
            "weight_posterior":   round(float(w_norm[i]), 6),
            "weight_drift":       round(float(w_norm[i] - w_prior[i]), 6),
            "log_weight":         round(float(log_w_unnorm[i]), 6),
            "posterior_entropy":  round(entropy, 6),
            "kl_from_uniform":    round(kl_from_uniform, 6),
            "generated_at":       _NOW_STR,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2, Component 8 - Model Health Monitoring
# ---------------------------------------------------------------------------

def _model_health_summary(
    feedback_actuals: pd.DataFrame,
    joined_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    dur_summary_df: pd.DataFrame,
    trust_df: pd.DataFrame,
    triggers_df: pd.DataFrame,
    l45_metrics_df: pd.DataFrame,
    cvar_comparison: pd.DataFrame,
    opt_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Track rolling health metrics across the prior holdout and the feedback batch.
    Raises urgency when metrics degrade even if drift tests are borderline.
    """
    rows = []

    def _l45(subset: str, task: str, metric: str) -> float | None:
        if l45_metrics_df.empty:
            return None
        mask = (l45_metrics_df["subset"] == subset) & \
               (l45_metrics_df["task"] == task) & \
               (l45_metrics_df["metric"] == metric)
        if mask.any():
            return float(l45_metrics_df[mask]["value"].iloc[0])
        return None

    # ---- Duration metrics ----
    holdout_rmse   = _l45("holdout_sanitized", "duration", "rmse")
    holdout_mae    = _l45("holdout_sanitized", "duration", "mae")
    holdout_medae  = _l45("holdout_sanitized", "duration", "median_ae")
    holdout_rmsle  = _l45("holdout_sanitized", "duration", "rmsle")

    if "log_duration" in joined_df.columns and "duration_p50" in joined_df.columns:
        valid = joined_df[joined_df["duration_p50"].notna() & joined_df["log_duration"].notna()]
        if len(valid) > 0:
            act = valid["duration_min"].clip(lower=0).values
            pred = valid["duration_p50"].clip(lower=0).values
            fb_rmse  = float(np.sqrt(np.mean((act - pred) ** 2)))
            fb_mae   = float(np.mean(np.abs(act - pred)))
            fb_medae = float(np.median(np.abs(act - pred)))
            log_act  = np.log1p(act)
            log_pred = np.log1p(pred)
            fb_rmsle = float(np.sqrt(np.mean((log_act - log_pred) ** 2)))
        else:
            fb_rmse = fb_mae = fb_medae = fb_rmsle = None
    else:
        fb_rmse = fb_mae = fb_medae = fb_rmsle = None

    for metric_name, holdout_val, fb_val, higher_is_worse, warn_thresh, crit_thresh in [
        ("duration_rmse",   holdout_rmse,   fb_rmse,   True, 0.20, 0.50),
        ("duration_mae",    holdout_mae,    fb_mae,    True, 0.20, 0.50),
        ("duration_medae",  holdout_medae,  fb_medae,  True, 0.20, 0.50),
        ("duration_rmsle",  holdout_rmsle,  fb_rmsle,  True, 0.15, 0.40),
    ]:
        relative_change = None
        if holdout_val is not None and fb_val is not None and holdout_val > 0:
            relative_change = (fb_val - holdout_val) / holdout_val

        status = "healthy"
        if relative_change is not None:
            if higher_is_worse and relative_change > crit_thresh:
                status = "critical"
            elif higher_is_worse and relative_change > warn_thresh:
                status = "warning"

        rows.append({
            "metric_group":    "duration",
            "metric":          metric_name,
            "holdout_value":   round(holdout_val, 4) if holdout_val is not None else None,
            "feedback_value":  round(fb_val, 4) if fb_val is not None else None,
            "relative_change": round(relative_change, 4) if relative_change is not None else None,
            "status":          status,
            "note":            "",
        })

    # ---- Probability metrics ----
    holdout_brier = _l45("holdout", "high_impact", "brier")
    holdout_ece   = _l45("holdout", "high_impact", "ece")

    fb_brier = float(cal_df["brier_raw"].iloc[0]) if not cal_df.empty and "brier_raw" in cal_df.columns else None
    fb_ece   = float(cal_df["ece_before_update"].iloc[0]) if not cal_df.empty else None

    for metric_name, holdout_val, fb_val, warn_t, crit_t in [
        ("high_impact_brier", holdout_brier, fb_brier, 0.15, 0.30),
        ("high_impact_ece",   holdout_ece,   fb_ece,   0.05, 0.10),
    ]:
        relative_change = None
        if holdout_val is not None and fb_val is not None and abs(holdout_val) > 1e-9:
            relative_change = (fb_val - holdout_val) / (abs(holdout_val) + 1e-9)
        status = "healthy"
        if fb_val is not None:
            if fb_val > crit_t:
                status = "critical"
            elif fb_val > warn_t:
                status = "warning"
        rows.append({
            "metric_group":   "probability_calibration",
            "metric":          metric_name,
            "holdout_value":   round(holdout_val, 6) if holdout_val is not None else None,
            "feedback_value":  round(fb_val, 6) if fb_val is not None else None,
            "relative_change": round(relative_change, 4) if relative_change is not None else None,
            "status":          status,
            "note":            "",
        })

    # ---- Layer 5 CVaR improvement ----
    if not cvar_comparison.empty and "percentage_reduction" in cvar_comparison.columns:
        row = cvar_comparison[cvar_comparison["scope"] == "all_sites"]
        cvar_pct = float(row["percentage_reduction"].iloc[0]) if not row.empty else None
    else:
        cvar_pct = None

    m_map = ({r["metric"]: float(r["value"]) for _, r in opt_metrics.iterrows()}
             if not opt_metrics.empty else {})
    cc_sat = m_map.get("chance_constraint_satisfaction_mean")

    for metric_name, val, warn_t, crit_t, higher_is_worse in [
        ("cvar_reduction_pct",    cvar_pct, 0.30, 0.20, False),
        ("cc_satisfaction_mean",  cc_sat,   0.90, 0.85, False),
    ]:
        status = "healthy"
        if val is not None:
            if higher_is_worse and val > crit_t:
                status = "critical"
            elif higher_is_worse and val > warn_t:
                status = "warning"
            elif not higher_is_worse and val < crit_t:
                status = "critical"
            elif not higher_is_worse and val < warn_t:
                status = "warning"
        rows.append({
            "metric_group":   "layer5_optimization",
            "metric":          metric_name,
            "holdout_value":   None,
            "feedback_value":  round(val, 4) if val is not None else None,
            "relative_change": None,
            "status":          status,
            "note":            "L5 optimization metrics (no holdout baseline)",
        })

    # ---- Drift scores ----
    if not drift_df.empty:
        for _, dr in drift_df.iterrows():
            status = "critical" if dr["severity"] == "critical" else \
                     "warning"  if dr["severity"] == "moderate" else "healthy"
            rows.append({
                "metric_group":   "drift",
                "metric":          f"{dr['test']}_{dr['variable']}",
                "holdout_value":   None,
                "feedback_value":  round(float(dr["score"]), 4),
                "relative_change": None,
                "status":          status,
                "note":            str(dr["detail"])[:120],
            })

    # ---- Posterior uncertainty width ----
    if not dur_summary_df.empty and "ci95_hi_log" in dur_summary_df.columns:
        ci_widths = (dur_summary_df["ci95_hi_log"] - dur_summary_df["ci95_lo_log"]).dropna()
        mean_width = float(ci_widths.mean()) if len(ci_widths) > 0 else None
        status = "warning" if (mean_width is not None and mean_width > 2.0) else "healthy"
        rows.append({
            "metric_group":   "posterior_uncertainty",
            "metric":          "mean_ci95_width_log",
            "holdout_value":   None,
            "feedback_value":  round(mean_width, 4) if mean_width is not None else None,
            "relative_change": None,
            "status":          status,
            "note":            f"Mean 95% CI width in log-duration space across {len(ci_widths)} strata",
        })

    # ---- Prototype trust stability ----
    if not trust_df.empty:
        mean_delta   = float(trust_df["trust_delta"].abs().mean())
        n_degraded   = int((trust_df["trust_updated"] < 0.5).sum())
        trust_status = "critical" if n_degraded > 3 else ("warning" if n_degraded > 0 else "healthy")
        rows.append({
            "metric_group":   "prototype_trust",
            "metric":          "mean_abs_trust_delta",
            "holdout_value":   None,
            "feedback_value":  round(mean_delta, 4),
            "relative_change": None,
            "status":          trust_status,
            "note":            f"{n_degraded} prototypes degraded below 0.5 trust",
        })

    # ---- Retrain trigger frequency ----
    if not triggers_df.empty:
        n_crit = int((triggers_df["severity"] == "critical").sum())
        n_mod  = int((triggers_df["severity"] == "moderate").sum())
        trig_status = "critical" if n_crit > 5 else ("warning" if n_crit > 0 else "healthy")
        rows.append({
            "metric_group":   "retrain_triggers",
            "metric":          "n_critical_triggers",
            "holdout_value":   None,
            "feedback_value":  float(n_crit),
            "relative_change": None,
            "status":          trig_status,
            "note":            f"{n_crit} critical + {n_mod} moderate triggers in this batch",
        })

    df = pd.DataFrame(rows)
    # Overall urgency: escalate if health metrics degrade even when drift tests are borderline
    n_critical = int((df["status"] == "critical").sum())
    n_warning  = int((df["status"] == "warning").sum())
    if n_critical >= 3:
        overall = "CRITICAL"
    elif n_critical >= 1 or n_warning >= 4:
        overall = "WARNING"
    else:
        overall = "HEALTHY"
    df["overall_health"] = overall
    return df


# ---------------------------------------------------------------------------
# Feedback log helper
# ---------------------------------------------------------------------------

def _build_feedback_log(
    feedback_actuals: pd.DataFrame,
    joined_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "event_id", "event_cause", "corridor_fill",
        "duration_min", "log_duration", "actual_high_impact", "start_local",
    ]
    for opt in ["duration_p50", "high_impact_prob",
                "high_impact_prob_calibrated", "retrieval_confidence"]:
        if opt in joined_df.columns:
            cols.append(opt)

    available = [c for c in cols if c in joined_df.columns]
    log = joined_df[available].copy()
    log["prior_period"]    = "Nov2023-Feb2024"
    log["feedback_period"] = "Mar2024-Apr2024"
    if "duration_p50" in log.columns:
        log["log_pred_p50"]  = np.log1p(log["duration_p50"].clip(lower=0))
        log["log_residual"]  = (log["log_pred_p50"] - log["log_duration"]).abs()
    log["processed_at"] = _NOW_STR
    return log.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Auxiliary output writers
# ---------------------------------------------------------------------------

def _write_versioned_knowledge_base(
    global_dict: dict,
    bma_df: pd.DataFrame,
    health_df: pd.DataFrame,
    triggers_df: pd.DataFrame,
    trust_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    effectiveness_dict: dict,
    out_path: Path,
) -> None:
    """Snapshot of all current posterior beliefs, suitable for versioning."""
    n_crit = int((triggers_df["severity"] == "critical").sum()) if not triggers_df.empty else 0
    overall_health = str(health_df["overall_health"].iloc[0]) if not health_df.empty else "UNKNOWN"

    kb = {
        "version":        "layer6_part2",
        "generated_at":   _NOW_STR,
        "feedback_period": "2024-03-01 to 2024-04-08",
        "prior_period":    "2023-11-10 to 2024-02-29",
        "global_duration_posterior": global_dict.get("global_posterior", {}),
        "bma_weights": {
            row["model"]: {
                "weight": row["weight_posterior"],
                "nll_proxy": row["nll_proxy_used"],
            }
            for _, row in bma_df.iterrows()
        } if not bma_df.empty else {},
        "calibration_summary": {
            "ece_before": float(cal_df["ece_before_update"].iloc[0]) if not cal_df.empty else None,
            "ece_after":  float(cal_df["ece_after_update"].iloc[0])  if not cal_df.empty else None,
            "brier_raw":  float(cal_df["brier_raw"].iloc[0])          if not cal_df.empty else None,
        },
        "effectiveness_posteriors": {
            p["parameter"]: {
                "posterior_mean": p["posterior_mean"],
                "posterior_std":  p["posterior_std"],
            }
            for p in effectiveness_dict.get("posterior", {}).get("parameters", [])
        },
        "prototype_trust_summary": {
            "mean_trust_prior":   float(trust_df["trust_prior"].mean())   if not trust_df.empty else None,
            "mean_trust_updated": float(trust_df["trust_updated"].mean()) if not trust_df.empty else None,
            "n_degraded":         int((trust_df["trust_updated"] < 0.5).sum()) if not trust_df.empty else 0,
        },
        "health": {
            "overall":         overall_health,
            "n_critical_metrics": int((health_df["status"] == "critical").sum()) if not health_df.empty else 0,
        },
        "active_alerts": {
            "n_critical": n_crit,
            "n_total":    len(triggers_df) if not triggers_df.empty else 0,
        },
        "governance": (
            "Layer 6 is ADDITIVE only. No upstream files were modified. "
            "All recommendations are in outputs/layer6_retrain_triggers.csv."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, default=str)


def _write_recalibration_recommendations(cal_df: pd.DataFrame, out_path: Path) -> None:
    """Per-bin calibration priority ranking with recommended action."""
    if cal_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    rec = cal_df[["bin_idx", "bin_lo", "bin_hi", "n_events", "n_successes",
                   "mean_raw_pred", "fraction_positive", "posterior_mean",
                   "ci95_lo", "ci95_hi", "calibration_shift"]].copy()
    rec["abs_shift"] = rec["calibration_shift"].abs()
    rec["priority"] = rec["abs_shift"].rank(ascending=False).astype(int)
    rec["recommendation"] = rec.apply(
        lambda r: (
            f"HIGH: Posterior mean {r['posterior_mean']:.3f} vs raw {r['mean_raw_pred']:.3f} "
            f"(shift {r['calibration_shift']:+.3f}). Recalibrate bin."
            if r["abs_shift"] > 0.10 else
            f"LOW: Shift {r['calibration_shift']:+.3f} — monitor only."
        ),
        axis=1,
    )
    rec.sort_values("priority").to_csv(out_path, index=False)


def _write_active_alerts(triggers_df: pd.DataFrame, health_df: pd.DataFrame,
                         out_path: Path) -> None:
    """Priority-ranked active alerts combining retrain triggers and health flags."""
    alerts = []
    if not triggers_df.empty:
        for _, row in triggers_df[triggers_df["severity"].isin(["critical", "moderate"])].iterrows():
            alerts.append({
                "alert_id":    row["trigger_id"],
                "source":      row["source_module"],
                "severity":    row["severity"],
                "affected_layer": row["affected_layer"],
                "description": str(row["recommendation"])[:200],
                "generated_at": row.get("generated_at", _NOW_STR),
            })
    if not health_df.empty:
        for _, row in health_df[health_df["status"].isin(["critical", "warning"])].iterrows():
            alerts.append({
                "alert_id":    f"HEALTH_{row['metric'].upper()}",
                "source":      "layer6_model_health",
                "severity":    row["status"],
                "affected_layer": row["metric_group"],
                "description": f"Health metric '{row['metric']}' is {row['status'].upper()}. "
                               f"Feedback value: {row['feedback_value']}.",
                "generated_at": _NOW_STR,
            })

    df = pd.DataFrame(alerts)
    if not df.empty:
        _sev = {"critical": 3, "warning": 2, "moderate": 2, "info": 1}
        df["_sev_rank"] = df["severity"].map(_sev).fillna(0)
        df = df.sort_values("_sev_rank", ascending=False).drop(columns=["_sev_rank"])
    df.to_csv(out_path, index=False)


def _write_posterior_uncertainty(dur_summary_df: pd.DataFrame, out_path: Path) -> None:
    """Per-stratum posterior uncertainty widths and stability flags."""
    if dur_summary_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    df = dur_summary_df[[
        "stratum_cause", "stratum_corridor", "prior_level",
        "posterior_mu_log", "posterior_sigma_log", "ci95_lo_log", "ci95_hi_log",
        "n_eff_feedback",
    ]].copy()
    df["ci95_width_log"]  = df["ci95_hi_log"] - df["ci95_lo_log"]
    df["uncertainty_flag"] = (df["posterior_sigma_log"] > 1.5).map({True: "high", False: "ok"})
    df.to_csv(out_path, index=False)


def _write_prototype_diagnostics(trust_df: pd.DataFrame, out_path: Path) -> None:
    """Extended prototype diagnostics with residuals and trust trajectory."""
    if trust_df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    df = trust_df.copy()
    df["trust_direction"] = df["trust_delta"].apply(
        lambda d: "improved" if d > 0.01 else ("degraded" if d < -0.01 else "stable")
    )
    df["health_flag"] = df["trust_updated"].apply(
        lambda t: "critical" if t < 0.3 else ("warning" if t < 0.5 else "ok")
    )
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary(
    prior_df, feedback_df, feedback_actuals,
    dur_df, cal_df, drift_df, trust_df, triggers_df,
    health_df, bma_df, eff_dict,
    out_path: Path,
) -> str:
    n_prior      = len(prior_df)
    n_feedback   = len(feedback_df)
    n_obs_fb     = len(feedback_actuals)
    n_strata     = len(dur_df)
    n_cal_events = int(cal_df["n_feedback_total"].iloc[0]) if not cal_df.empty else 0
    n_triggers   = len(triggers_df)
    n_critical   = int((triggers_df["severity"] == "critical").sum())
    n_drift_alert = int(drift_df["alert"].sum()) if not drift_df.empty else 0
    ece_b = float(cal_df["ece_before_update"].iloc[0]) if not cal_df.empty else float("nan")
    ece_a = float(cal_df["ece_after_update"].iloc[0])  if not cal_df.empty else float("nan")
    trust_mean_prior   = float(trust_df["trust_prior"].mean())   if not trust_df.empty else float("nan")
    trust_mean_updated = float(trust_df["trust_updated"].mean()) if not trust_df.empty else float("nan")
    overall_health = str(health_df["overall_health"].iloc[0]) if not health_df.empty else "UNKNOWN"

    lines = [
        "=" * 70,
        "LAYER 6 ADAPTIVE LEARNING -- LEARNING SUMMARY",
        f"Generated: {_NOW_STR}",
        "=" * 70,
        "",
        "DATASET",
        f"  Prior period    : Nov 2023 - Feb 2024  ({n_prior:,} events)",
        f"  Feedback batch  : Mar 2024 - Apr 2024  ({n_feedback:,} events)",
        f"  Uncensored obs  : {n_obs_fb:,} events used for posterior updates",
        "",
        "COMPONENT 1 -- Hierarchical Bayesian Duration Update",
        f"  Strata updated         : {n_strata}",
        f"  Forgetting half-life   : 30 days (exponential)",
        f"  Model                  : Normal-Normal conjugate, partial pooling",
        f"  Fallback hierarchy     : stratum -> cause -> global",
        "",
        "COMPONENT 2 -- Calibration Posterior Update",
        f"  Feedback events used   : {n_cal_events}",
        f"  Probability bins       : 10 (decile)",
        f"  ECE before update      : {ece_b:.4f}",
        f"  ECE after update       : {ece_a:.4f}",
        "",
        "COMPONENT 3 -- Drift Detection",
        f"  Tests run              : Page-Hinkley, PSI, ODS (weekly), Mean-shift z-test",
        f"  Alerts fired           : {n_drift_alert}",
    ]
    for _, row in drift_df[drift_df["alert"] == True].iterrows():  # noqa: E712
        lines.append(f"    [{row['severity'].upper():8s}] {row['test']} on {row['variable']}: "
                     f"score={row['score']:.4f}")

    lines += [
        "",
        "COMPONENT 4 -- Prototype Trust Update (Beta-Binomial + EMA)",
        f"  Prototypes evaluated   : {len(trust_df)}",
        f"  Mean trust (prior)     : {trust_mean_prior:.4f}",
        f"  Mean trust (updated)   : {trust_mean_updated:.4f}",
    ]
    degraded = trust_df[trust_df["trust_updated"] < 0.5] if not trust_df.empty else pd.DataFrame()
    if len(degraded) > 0:
        lines.append(f"  Degraded prototypes    : {len(degraded)}")

    lines += [
        "",
        "COMPONENT 6 -- Resource Effectiveness Posteriors",
        f"  Evidence path          : {eff_dict.get('metadata', {}).get('evidence_path', 'n/a')}",
        f"  Observations           : {eff_dict.get('metadata', {}).get('n_observations', 0)}",
    ]
    for p in eff_dict.get("posterior", {}).get("parameters", []):
        lines.append(f"  {p['parameter']:10s} : prior={p['prior_mean']:.3f} -> "
                     f"post={p['posterior_mean']:.3f} (shift {p['shift_from_prior']:+.3f})")

    if not bma_df.empty:
        lines += ["", "COMPONENT 7 -- Bayesian Model Averaging Weights"]
        for _, brow in bma_df.sort_values("weight_posterior", ascending=False).iterrows():
            lines.append(f"  {brow['model']:28s}: w={brow['weight_posterior']:.4f} "
                         f"NLL={brow['nll_proxy_used']:.3f}")
        lines.append(f"  Posterior entropy      : {bma_df['posterior_entropy'].iloc[0]:.4f}")
        lines.append(f"  KL from uniform        : {bma_df['kl_from_uniform'].iloc[0]:.4f}")

    n_health_crit = int((health_df["status"] == "critical").sum()) if not health_df.empty else 0
    n_health_warn = int((health_df["status"] == "warning").sum())  if not health_df.empty else 0
    lines += [
        "",
        "COMPONENT 8 -- Model Health Monitoring",
        f"  Overall health         : {overall_health}",
        f"  Critical metrics       : {n_health_crit}",
        f"  Warning metrics        : {n_health_warn}",
    ]
    if not health_df.empty:
        for _, hr in health_df[health_df["status"] != "healthy"].iterrows():
            v = hr["feedback_value"]
            v_str = f"{v:.4f}" if v is not None and not np.isnan(float(v)) else "n/a"
            lines.append(f"    [{hr['status'].upper():8s}] {hr['metric']}: {v_str}")

    lines += [
        "",
        "RETRAIN TRIGGERS",
        f"  Total triggers         : {n_triggers}",
        f"  Critical               : {n_critical}",
        f"  Moderate               : {int((triggers_df['severity'] == 'moderate').sum())}",
        f"  Info                   : {int((triggers_df['severity'] == 'info').sum())}",
        "",
        "TOP RECOMMENDATIONS",
    ]
    top = triggers_df[triggers_df["severity"].isin(["critical", "moderate"])].head(5)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        lines.append(f"  {i}. [{row['severity'].upper()}] {row['trigger_id']}")
        wrapped = textwrap.fill(row["recommendation"], width=66,
                                initial_indent="     ", subsequent_indent="     ")
        lines.append(wrapped)

    lines += [
        "",
        "GOVERNANCE",
        "  Layer 6 is ADDITIVE only. No upstream files were modified.",
        "  All recommendations are in outputs/layer6_retrain_triggers.csv.",
        "=" * 70,
    ]

    report = "\n".join(lines)
    out_path.write_text(report, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("[L6] Loading data...")
    events_df = load_events()
    prior_df, feedback_df = split_periods(events_df)
    print(f"[L6]   Prior : {len(prior_df):,} events | Feedback : {len(feedback_df):,} events")

    state_vector    = load_state_vector()
    hi_probs        = load_high_impact_probs()
    cause_tau_df    = load_cause_tau()
    violations_df   = load_violations()
    layer5_alloc    = load_resource_allocation()
    cvar_comparison = load_cvar_comparison()
    opt_metrics     = load_opt_metrics()
    shadow_prices   = load_shadow_prices()
    prototypes_df   = load_prototypes()
    utilization_df  = load_prototype_utilization()
    l45_metrics_df  = load_l45_metrics()

    # Feedback actuals: uncensored Mar-Apr with ground-truth high-impact labels
    feedback_actuals = build_feedback_actuals(feedback_df, cause_tau_df)
    print(f"[L6]   Uncensored feedback obs: {len(feedback_actuals):,}")

    prior_df["log_duration"] = np.log1p(prior_df["duration_min"].clip(lower=0))
    joined_df = join_predictions_to_feedback(feedback_actuals, state_vector, hi_probs)

    # =========================================================================
    # COMPONENT 1 -- Hierarchical Bayesian Duration Update
    # =========================================================================
    print("[L6] Component 1: Hierarchical Bayesian Duration Update...")
    dur_summary_df, global_dict = update_duration_posteriors(prior_df, feedback_df)
    print(f"[L6]   {len(dur_summary_df)} strata updated.")
    dur_summary_df.to_csv(OUTPUTS_DIR / "layer6_duration_posterior_summary.csv", index=False)
    with open(OUTPUTS_DIR / "layer6_posterior_duration_priors.json", "w", encoding="utf-8") as f:
        json.dump(global_dict, f, indent=2)
    print("[L6]   -> layer6_duration_posterior_summary.csv")
    print("[L6]   -> layer6_posterior_duration_priors.json")

    # =========================================================================
    # COMPONENT 2 -- Calibration Posterior Update
    # =========================================================================
    print("[L6] Component 2: Calibration Posterior Update...")
    cal_df = update_calibration_posteriors(joined_df)
    _cal_empty = pd.DataFrame([{"ece_before_update": float("nan"),
                                 "ece_after_update": float("nan"),
                                 "n_feedback_total": 0}])
    if cal_df.empty:
        print("[L6]   WARNING: No calibration data -- skipping.")
        cal_df = _cal_empty
    else:
        cal_df.to_csv(OUTPUTS_DIR / "layer6_calibration_posteriors.csv", index=False)
        print(f"[L6]   ECE before={cal_df['ece_before_update'].iloc[0]:.4f} "
              f"-> after={cal_df['ece_after_update'].iloc[0]:.4f}")
        print("[L6]   -> layer6_calibration_posteriors.csv")

    # =========================================================================
    # COMPONENT 3 -- Drift Detection
    # =========================================================================
    print("[L6] Component 3: Drift Detection...")
    extra_features = [f for f in ["trust_score", "iso_anomaly_score", "hour_local"]
                      if f in prior_df.columns and f in feedback_actuals.columns]
    drift_df = run_drift_detection(prior_df, feedback_actuals, features=extra_features)
    print(f"[L6]   {int(drift_df['alert'].sum())} alerts across {len(drift_df)} tests.")
    drift_df.to_csv(OUTPUTS_DIR / "layer6_drift_report.csv", index=False)
    print("[L6]   -> layer6_drift_report.csv")

    # =========================================================================
    # COMPONENT 4 -- Prototype Trust Update
    # =========================================================================
    print("[L6] Component 4: Prototype Trust Update...")
    trust_df = _update_prototype_trust(
        prototypes_df, utilization_df, feedback_actuals, state_vector
    )
    print(f"[L6]   {len(trust_df)} prototypes | "
          f"{int((trust_df['trust_updated'] < 0.5).sum())} degraded.")
    trust_df.to_csv(OUTPUTS_DIR / "layer6_prototype_trust_updates.csv", index=False)
    print("[L6]   -> layer6_prototype_trust_updates.csv")

    # =========================================================================
    # COMPONENT 5 -- Retrain Triggers
    # =========================================================================
    print("[L6] Component 5: Retrain Triggers...")
    triggers_df = build_retrain_triggers(
        drift_df=drift_df, cal_df=cal_df, dur_df=dur_summary_df,
        trust_df=trust_df, violations_df=violations_df,
    )
    print(f"[L6]   {len(triggers_df)} triggers "
          f"({int((triggers_df['severity'] == 'critical').sum())} critical).")
    triggers_df.to_csv(OUTPUTS_DIR / "layer6_retrain_triggers.csv", index=False)
    print("[L6]   -> layer6_retrain_triggers.csv")

    # =========================================================================
    # COMPONENT 6 -- Resource Effectiveness Posteriors
    # =========================================================================
    print("[L6] Component 6: Resource Effectiveness Posteriors...")
    eff_dict = _resource_effectiveness_posteriors(
        layer5_alloc=layer5_alloc,
        feedback_actuals=feedback_actuals,
        opt_metrics=opt_metrics,
        shadow_prices=shadow_prices,
        cvar_comparison=cvar_comparison,
    )
    with open(OUTPUTS_DIR / "layer6_resource_effectiveness_posteriors.json",
              "w", encoding="utf-8") as f:
        json.dump(eff_dict, f, indent=2)
    path_used = eff_dict["metadata"]["evidence_path"]
    print(f"[L6]   Evidence path: {path_used}")
    for p in eff_dict["posterior"]["parameters"]:
        print(f"[L6]   {p['parameter']:10s}: prior={p['prior_mean']:.3f} "
              f"-> post={p['posterior_mean']:.3f}")
    print("[L6]   -> layer6_resource_effectiveness_posteriors.json")

    # =========================================================================
    # COMPONENT 7 -- Bayesian Model Averaging Weights
    # =========================================================================
    print("[L6] Component 7: Bayesian Model Averaging...")
    bma_df = _compute_bma_weights(
        joined_df=joined_df,
        dur_summary_df=dur_summary_df,
        cal_df=cal_df,
        layer5_alloc=layer5_alloc,
        l45_metrics_df=l45_metrics_df,
    )
    bma_df.to_csv(OUTPUTS_DIR / "layer6_bma_weights.csv", index=False)
    print("[L6]   BMA weights:")
    for _, row in bma_df.sort_values("weight_posterior", ascending=False).iterrows():
        print(f"[L6]     {row['model']:28s}: w={row['weight_posterior']:.4f} "
              f"(NLL={row['nll_proxy_used']:.3f})")
    print("[L6]   -> layer6_bma_weights.csv")

    # =========================================================================
    # COMPONENT 8 -- Model Health Monitoring
    # =========================================================================
    print("[L6] Component 8: Model Health Monitoring...")
    health_df = _model_health_summary(
        feedback_actuals=feedback_actuals,
        joined_df=joined_df,
        cal_df=cal_df,
        drift_df=drift_df,
        dur_summary_df=dur_summary_df,
        trust_df=trust_df,
        triggers_df=triggers_df,
        l45_metrics_df=l45_metrics_df,
        cvar_comparison=cvar_comparison,
        opt_metrics=opt_metrics,
    )
    overall = str(health_df["overall_health"].iloc[0]) if not health_df.empty else "UNKNOWN"
    n_crit_h = int((health_df["status"] == "critical").sum())
    n_warn_h = int((health_df["status"] == "warning").sum())
    print(f"[L6]   Overall health: {overall} ({n_crit_h} critical, {n_warn_h} warning metrics)")
    health_df.to_csv(OUTPUTS_DIR / "layer6_model_health_summary.csv", index=False)
    print("[L6]   -> layer6_model_health_summary.csv")

    # =========================================================================
    # Feedback Log
    # =========================================================================
    print("[L6] Writing feedback log...")
    feedback_log = _build_feedback_log(feedback_actuals, joined_df, prior_df, feedback_df)
    feedback_log.to_csv(OUTPUTS_DIR / "layer6_feedback_log.csv", index=False)
    print(f"[L6]   {len(feedback_log)} records -> layer6_feedback_log.csv")

    # =========================================================================
    # Auxiliary outputs
    # =========================================================================
    print("[L6] Writing auxiliary outputs...")

    _write_versioned_knowledge_base(
        global_dict=global_dict, bma_df=bma_df, health_df=health_df,
        triggers_df=triggers_df, trust_df=trust_df, cal_df=cal_df,
        effectiveness_dict=eff_dict,
        out_path=OUTPUTS_DIR / "layer6_versioned_knowledge_base.json",
    )
    print("[L6]   -> layer6_versioned_knowledge_base.json")

    _write_recalibration_recommendations(cal_df, OUTPUTS_DIR / "layer6_recalibration_recommendations.csv")
    print("[L6]   -> layer6_recalibration_recommendations.csv")

    _write_active_alerts(triggers_df, health_df, OUTPUTS_DIR / "layer6_active_alerts.csv")
    print("[L6]   -> layer6_active_alerts.csv")

    _write_posterior_uncertainty(dur_summary_df, OUTPUTS_DIR / "layer6_posterior_uncertainty.csv")
    print("[L6]   -> layer6_posterior_uncertainty.csv")

    _write_prototype_diagnostics(trust_df, OUTPUTS_DIR / "layer6_prototype_diagnostics.csv")
    print("[L6]   -> layer6_prototype_diagnostics.csv")

    # Model artifacts directory (JSON snapshots for versioning)
    with open(ARTIFACTS_DIR / "duration_posteriors.json", "w", encoding="utf-8") as f:
        json.dump(global_dict, f, indent=2)
    with open(ARTIFACTS_DIR / "effectiveness_posteriors.json", "w", encoding="utf-8") as f:
        json.dump(eff_dict, f, indent=2)
    with open(ARTIFACTS_DIR / "bma_weights.json", "w", encoding="utf-8") as f:
        json.dump(bma_df.to_dict(orient="records") if not bma_df.empty else [], f, indent=2)
    with open(ARTIFACTS_DIR / "health_snapshot.json", "w", encoding="utf-8") as f:
        snap = health_df.to_dict(orient="records") if not health_df.empty else []
        json.dump({"overall": overall, "metrics": snap, "generated_at": _NOW_STR}, f, indent=2)
    with open(ARTIFACTS_DIR / "calibration_posteriors.json", "w", encoding="utf-8") as f:
        json.dump(cal_df.to_dict(orient="records") if not cal_df.empty else [], f, indent=2)
    print(f"[L6]   -> layer6_model_artifacts/ ({len(list(ARTIFACTS_DIR.glob('*.json')))} files)")

    # =========================================================================
    # Learning Summary
    # =========================================================================
    print("[L6] Writing learning summary...")
    report = _write_summary(
        prior_df=prior_df, feedback_df=feedback_df, feedback_actuals=feedback_actuals,
        dur_df=dur_summary_df, cal_df=cal_df, drift_df=drift_df,
        trust_df=trust_df, triggers_df=triggers_df,
        health_df=health_df, bma_df=bma_df, eff_dict=eff_dict,
        out_path=OUTPUTS_DIR / "layer6_learning_summary.txt",
    )
    print("[L6]   -> layer6_learning_summary.txt")

    print("\n" + "=" * 60)
    print("Layer 6 complete.")
    print("=" * 60)
    safe_report = report.encode("ascii", errors="replace").decode("ascii")
    print(safe_report)


if __name__ == "__main__":
    main()
