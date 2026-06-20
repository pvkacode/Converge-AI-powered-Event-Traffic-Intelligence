"""
Layer 6 – Feedback Store
Loads and exposes prior (Nov 2023–Feb 2024) and feedback batch (Mar–Apr 2024)
datasets plus all upstream Layer 4.5 / Layer 5 outputs for other L6 modules.
No upstream files are ever modified here.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Canonical period boundaries (tz-aware, IST = UTC+05:30)
_TZ           = "Asia/Kolkata"
PRIOR_START   = pd.Timestamp("2023-11-01", tz=_TZ)
PRIOR_END     = pd.Timestamp("2024-03-01", tz=_TZ)   # exclusive
FEEDBACK_START = pd.Timestamp("2024-03-01", tz=_TZ)
FEEDBACK_END   = pd.Timestamp("2024-04-09", tz=_TZ)  # inclusive upper bound


# ---------------------------------------------------------------------------
# Core data
# ---------------------------------------------------------------------------

def load_events(tz: str = _TZ) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "events_clean.parquet")
    df["start_local"] = pd.to_datetime(df["start_local"], utc=True).dt.tz_convert(tz)
    return df


def split_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior    = df[(df["start_local"] >= PRIOR_START) & (df["start_local"] < PRIOR_END)].copy()
    feedback = df[(df["start_local"] >= FEEDBACK_START) & (df["start_local"] < FEEDBACK_END)].copy()
    return prior, feedback


# ---------------------------------------------------------------------------
# Layer 4.5 outputs
# ---------------------------------------------------------------------------

def load_state_vector() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer45_operational_state_vector_normalized.csv")


def load_high_impact_probs() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer45_high_impact_probabilities.csv")


def load_scenario_duration() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer45_scenario_ready_duration.csv")


def load_cause_tau() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer45_cause_tau_thresholds.csv")


def load_duration_quality() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer45_duration_quality.csv")


def load_l45_metrics() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer45_metrics.csv")


# ---------------------------------------------------------------------------
# Layer 5 outputs
# ---------------------------------------------------------------------------

def load_resource_allocation() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer5_resource_allocation.csv")


def load_cvar_comparison() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer5_pre_post_cvar_comparison.csv")


def load_violations() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer5_chance_constraint_violations.csv")


def load_opt_metrics() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer5_optimization_metrics.csv")


def load_shadow_prices() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer5_shadow_prices.csv")


# ---------------------------------------------------------------------------
# Layer 4 prototype outputs
# ---------------------------------------------------------------------------

def load_prototypes() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer4_planned_event_prototypes.csv")


def load_prototype_utilization() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "layer4_prototype_utilization.csv")


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

def build_feedback_actuals(
    feedback_df: pd.DataFrame,
    tau_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For feedback-period events with observed (uncensored) durations,
    derive actual high-impact labels and log-duration.
    """
    tau_map    = dict(zip(tau_df["event_cause"], tau_df["tau_p75_min"]))
    global_tau = float(tau_df["global_fallback_tau"].iloc[0])

    obs = feedback_df[
        ~feedback_df["is_censored"] & feedback_df["duration_min"].notna()
    ].copy()
    obs["tau_c"]            = obs["event_cause"].map(tau_map).fillna(global_tau)
    obs["actual_high_impact"] = (obs["duration_min"] > obs["tau_c"]).astype(int)
    obs["log_duration"]     = np.log1p(obs["duration_min"])
    return obs


def join_predictions_to_feedback(
    feedback_actuals: pd.DataFrame,
    state_vector: pd.DataFrame,
    hi_probs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach Layer 4.5 predictions (high-impact prob, duration quantiles)
    to feedback actuals so calibration and residual computation can be done
    in one place.
    """
    sv = state_vector[["event_id", "start_local",
                        "duration_pred", "duration_p50", "duration_p80",
                        "high_impact_prob", "high_impact_prob_calibrated",
                        "retrieval_confidence", "trust_score"]].copy()
    hip = hi_probs[["event_id", "high_impact_prob", "high_impact_prob_calibrated"]].copy()

    merged = (
        feedback_actuals
        .merge(sv, on="event_id", how="left", suffixes=("", "_sv"))
        .merge(hip, on="event_id", how="left", suffixes=("", "_hip"))
    )
    # prefer state-vector prob; fall back to hi_probs CSV
    for col in ["high_impact_prob", "high_impact_prob_calibrated"]:
        merged[col] = merged[col].combine_first(merged[f"{col}_hip"])
    for c in merged.columns:
        if c.endswith("_hip"):
            merged.drop(columns=[c], inplace=True)
    return merged
