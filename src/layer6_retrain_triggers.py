"""
Layer 6 – Retrain Trigger Generator

Aggregates signals from:
  • Drift detection (PH / PSI / ODS)
  • Calibration posteriors (ECE, Brier, bin-level miscalibration)
  • Bayesian duration update (large posterior shifts, sparse strata)
  • Prototype trust (degraded trust scores)
  • Layer 5 constraint violations

and emits structured recommendation records to
outputs/layer6_retrain_triggers.csv.

GOVERNANCE: This module only *recommends* retraining.
It never modifies Layer 4.5 or Layer 5 source / output files.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Severity constants
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"none": 0, "info": 1, "moderate": 2, "critical": 3}

_NOW = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _sev(label: str) -> int:
    return _SEVERITY_RANK.get(label, 0)


# ---------------------------------------------------------------------------
# Individual signal extractors
# ---------------------------------------------------------------------------

def _triggers_from_drift(drift_df: pd.DataFrame) -> list[dict]:
    """One trigger per drift test that fired an alert."""
    triggers = []
    for _, row in drift_df[drift_df["alert"] == True].iterrows():  # noqa: E712
        triggers.append({
            "trigger_id":      f"DRIFT_{row['test'].upper()}_{row['variable'].upper()}",
            "source_module":   "layer6_drift_detection",
            "signal_type":     "drift",
            "variable":        row["variable"],
            "test":            row["test"],
            "score":           row["score"],
            "threshold":       row["threshold"],
            "severity":        row["severity"],
            "recommendation":  row["recommendation"],
            "affected_layer":  "Layer4.5",
            "action":          "retrain_duration_model",
            "generated_at":    _NOW,
        })
    return triggers


def _triggers_from_calibration(cal_df: pd.DataFrame) -> list[dict]:
    """Triggers based on ECE degradation and bin-level miscalibration."""
    if cal_df.empty:
        return []

    triggers = []
    ece_before = float(cal_df["ece_before_update"].iloc[0])
    ece_after  = float(cal_df["ece_after_update"].iloc[0])
    brier_raw  = float(cal_df["brier_raw"].iloc[0])

    # ECE-based trigger
    if ece_before > 0.05:
        triggers.append({
            "trigger_id":    "CALIB_ECE_HIGH",
            "source_module": "layer6_calibration_updates",
            "signal_type":   "calibration",
            "variable":      "high_impact_prob_calibrated",
            "test":          "ece",
            "score":         round(ece_before, 6),
            "threshold":     0.05,
            "severity":      "critical" if ece_before > 0.10 else "moderate",
            "recommendation": (
                f"ECE={ece_before:.4f} on feedback batch. "
                "Isotonic or Platt recalibration recommended for high-impact classifier."
            ),
            "affected_layer":  "Layer4.5",
            "action":          "recalibrate_classifier",
            "generated_at":    _NOW,
        })

    # Brier score trigger
    if brier_raw > 0.15:
        triggers.append({
            "trigger_id":    "CALIB_BRIER_HIGH",
            "source_module": "layer6_calibration_updates",
            "signal_type":   "calibration",
            "variable":      "high_impact_prob",
            "test":          "brier_score",
            "score":         round(brier_raw, 6),
            "threshold":     0.15,
            "severity":      "moderate",
            "recommendation": (
                f"Brier score={brier_raw:.4f} on feedback batch. "
                "Review feature importance and consider model refit."
            ),
            "affected_layer":  "Layer4.5",
            "action":          "retrain_high_impact_classifier",
            "generated_at":    _NOW,
        })

    # Bin-level: flag bins where |calibration_shift| > 0.10
    bad_bins = cal_df[cal_df["calibration_shift"].abs() > 0.10]
    for _, row in bad_bins.iterrows():
        triggers.append({
            "trigger_id":    f"CALIB_BIN_{int(row['bin_idx'])}_SHIFT",
            "source_module": "layer6_calibration_updates",
            "signal_type":   "calibration_bin",
            "variable":      f"bin_{int(row['bin_idx'])} [{row['bin_lo']:.1f},{row['bin_hi']:.1f}]",
            "test":          "beta_posterior_shift",
            "score":         round(abs(row["calibration_shift"]), 6),
            "threshold":     0.10,
            "severity":      "moderate",
            "recommendation": (
                f"Bin [{row['bin_lo']:.1f},{row['bin_hi']:.1f}]: "
                f"posterior mean={row['posterior_mean']:.3f}, "
                f"raw pred={row['mean_raw_pred']:.3f} "
                f"(shift={row['calibration_shift']:+.3f}). "
                "Posterior updated; consider recalibration pass."
            ),
            "affected_layer":  "Layer4.5",
            "action":          "recalibrate_bin",
            "generated_at":    _NOW,
        })

    return triggers


def _triggers_from_duration(dur_df: pd.DataFrame) -> list[dict]:
    """Triggers from Bayesian duration posteriors with large shifts."""
    if dur_df.empty:
        return []

    triggers = []
    dur_df = dur_df.copy()
    dur_df["log_shift"] = dur_df["posterior_mu_log"] - dur_df["prior_mu_log"]
    dur_df["sigma_shift"] = dur_df["log_shift"].abs() / dur_df["prior_sigma_log"].clip(lower=1e-6)

    large_shift = dur_df[dur_df["sigma_shift"] > 1.5]
    for _, row in large_shift.iterrows():
        triggers.append({
            "trigger_id":    (
                f"DUR_SHIFT_{row['stratum_cause'].upper()}_"
                f"{str(row['stratum_corridor']).replace(' ', '_').upper()[:20]}"
            ),
            "source_module": "layer6_bayesian_duration",
            "signal_type":   "duration_posterior_shift",
            "variable":      f"{row['stratum_cause']} × {row['stratum_corridor']}",
            "test":          "posterior_sigma_shift",
            "score":         round(row["sigma_shift"], 4),
            "threshold":     1.5,
            "severity":      "critical" if row["sigma_shift"] > 3.0 else "moderate",
            "recommendation": (
                f"Stratum '{row['stratum_cause']} × {row['stratum_corridor']}': "
                f"posterior_mu shifted {row['log_shift']:+.3f} log-units "
                f"({row['sigma_shift']:.1f} sigma). "
                f"Prior fallback level: {row['prior_level']}. "
                "Recommend updating Layer 4.5 duration priors for this stratum."
            ),
            "affected_layer":  "Layer4.5",
            "action":          "update_duration_prior",
            "generated_at":    _NOW,
        })

    # Sparse-stratum trigger
    sparse = dur_df[dur_df["n_eff_feedback"] < 3]
    if len(sparse) > 0:
        causes = sparse["stratum_cause"].unique().tolist()
        triggers.append({
            "trigger_id":    "DUR_SPARSE_STRATA",
            "source_module": "layer6_bayesian_duration",
            "signal_type":   "sparse_stratum",
            "variable":      "stratum coverage",
            "test":          "n_eff",
            "score":         float(sparse["n_eff_feedback"].mean()),
            "threshold":     3.0,
            "severity":      "info",
            "recommendation": (
                f"{len(sparse)} strata had n_eff < 3 in the feedback batch "
                f"(causes: {causes[:5]}). "
                "Fell back to cause or global prior; consider data collection for underrepresented strata."
            ),
            "affected_layer":  "Layer4.5",
            "action":          "collect_more_data",
            "generated_at":    _NOW,
        })

    return triggers


def _triggers_from_trust(trust_df: pd.DataFrame) -> list[dict]:
    """Triggers from degraded prototype trust scores."""
    if trust_df.empty:
        return []

    triggers = []
    degraded = trust_df[trust_df["trust_updated"] < 0.5]
    for _, row in degraded.iterrows():
        triggers.append({
            "trigger_id":    f"TRUST_DEGRADED_P{int(row['prototype_id'])}",
            "source_module": "layer6_adaptive_learning",
            "signal_type":   "prototype_trust",
            "variable":      f"prototype_{int(row['prototype_id'])}",
            "test":          "beta_binomial_trust",
            "score":         round(float(row["trust_updated"]), 4),
            "threshold":     0.5,
            "severity":      "critical" if float(row["trust_updated"]) < 0.3 else "moderate",
            "recommendation": (
                f"Prototype {int(row['prototype_id'])} "
                f"({row.get('cause', 'unknown')} / {row.get('corridor', 'unknown')}): "
                f"trust degraded from {row['trust_prior']:.3f} to {row['trust_updated']:.3f}. "
                "Review prototype representativeness; consider refreshing with recent events."
            ),
            "affected_layer":  "Layer4",
            "action":          "refresh_prototype",
            "generated_at":    _NOW,
        })

    return triggers


def _triggers_from_l5_violations(violations_df: pd.DataFrame) -> list[dict]:
    """Trigger if Layer 5 chance-constraint violations persist in feedback period."""
    if violations_df.empty:
        return []

    violated = violations_df[violations_df["violation_flag"] == 1]
    if violated.empty:
        return []

    return [{
        "trigger_id":    "L5_CHANCE_CONSTRAINT_VIOLATION",
        "source_module": "layer5_chance_constraint_violations",
        "signal_type":   "constraint_violation",
        "variable":      "chance_constraint_satisfaction",
        "test":          "realized_satisfaction_rate",
        "score":         round(float(violated["violation_margin"].mean()), 4),
        "threshold":     0.0,
        "severity":      "critical",
        "recommendation": (
            f"{len(violated)} event(s) violated chance constraints "
            f"(mean margin={violated['violation_margin'].mean():.3f}). "
            "Layer 5 epsilon parameters or risk tiers may need adjustment. "
            "Review duration quantile accuracy for critical-tier events."
        ),
        "affected_layer":  "Layer5",
        "action":          "review_risk_tiers",
        "generated_at":    _NOW,
    }]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_retrain_triggers(
    drift_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    dur_df: pd.DataFrame,
    trust_df: pd.DataFrame,
    violations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate all signals into a sorted retrain-trigger table.

    Parameters
    ----------
    drift_df      : output of layer6_drift_detection.run_drift_detection
    cal_df        : output of layer6_calibration_updates.update_calibration_posteriors
    dur_df        : output of layer6_bayesian_duration.update_duration_posteriors
    trust_df      : prototype trust updates (from layer6_adaptive_learning)
    violations_df : layer5_chance_constraint_violations

    Returns
    -------
    DataFrame written to outputs/layer6_retrain_triggers.csv
    """
    triggers: list[dict] = []
    triggers.extend(_triggers_from_drift(drift_df))
    triggers.extend(_triggers_from_calibration(cal_df))
    triggers.extend(_triggers_from_duration(dur_df))
    triggers.extend(_triggers_from_trust(trust_df))
    triggers.extend(_triggers_from_l5_violations(violations_df))

    if not triggers:
        triggers.append({
            "trigger_id":      "NO_TRIGGERS",
            "source_module":   "layer6_retrain_triggers",
            "signal_type":     "summary",
            "variable":        "all",
            "test":            "aggregate",
            "score":           0.0,
            "threshold":       0.0,
            "severity":        "none",
            "recommendation":  "No retrain signals detected. Pipeline appears stable.",
            "affected_layer":  "none",
            "action":          "monitor",
            "generated_at":    _NOW,
        })

    df = pd.DataFrame(triggers)
    df["severity_rank"] = df["severity"].map(_SEVERITY_RANK).fillna(0).astype(int)
    df = df.sort_values(["severity_rank", "trigger_id"], ascending=[False, True])
    df = df.drop(columns=["severity_rank"]).reset_index(drop=True)
    return df
