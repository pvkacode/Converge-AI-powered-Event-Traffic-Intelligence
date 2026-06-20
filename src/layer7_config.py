"""
Layer 7 — Operational Intelligence: shared configuration (M1).

Single source of fixed constants and path resolution for all Layer 7 modules.
Layer 7 is ADDITIVE ONLY. Nothing here learns or fits a parameter; every value
is a documented operational constant. No Layer 1-6 file is read for config.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FRONT = OUT / "frontend"

# Namespace guard: Layer 7 may only ever write to paths matching these prefixes.
ALLOWED_WRITE_PREFIXES = ("layer7_",)
ALLOWED_WRITE_DIRS = (OUT, FRONT)

# ----------------------------------------------------------------------------- inputs
# Canonical L4.5 normalized Joint Operational State Vector (per-event predictive state).
JOSV_NORMALIZED = "layer45_operational_state_vector_normalized.csv"

LAYER5_REQUIRED = [
    "layer5_resource_allocation.csv",
    "layer5_frontend_export.csv",
    "layer5_optimization_metrics.csv",
    "layer5_pre_post_cvar_comparison.csv",
    "layer5_shadow_prices.csv",
    "layer5_chance_constraint_violations.csv",
]

LAYER6_REQUIRED = [
    "layer6_active_alerts.csv",
    "layer6_retrain_triggers.csv",
    "layer6_drift_report.csv",
    "layer6_model_health_summary.csv",
    "layer6_monitoring_diagnostics.csv",
]

# Extra input the Operational State Engine needs (L4.5). Treated as required for
# Phase 2 but audited separately so Phase 1 input-audit stays L5/L6-focused.
STATE_INPUTS = [JOSV_NORMALIZED]

# Minimal critical columns used for missing-column diagnostics (NOT full schemas;
# upstream may add columns freely — we only check what Layer 7 actually consumes).
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "layer5_resource_allocation.csv": ["event_id", "robustness_score", "service_tier"],
    "layer5_frontend_export.csv": ["event_id", "baseline_cvar", "optimized_cvar"],
    "layer5_optimization_metrics.csv": ["metric", "value"],
    "layer5_pre_post_cvar_comparison.csv": ["scope", "alpha", "percentage_reduction"],
    "layer5_shadow_prices.csv": ["resource", "marginal_value"],
    "layer5_chance_constraint_violations.csv": [
        "event_id", "violation_flag", "violation_margin", "service_tier",
    ],
    "layer6_active_alerts.csv": ["alert_id", "severity", "affected_layer", "generated_at"],
    "layer6_retrain_triggers.csv": [
        "trigger_id", "severity", "variable", "affected_layer", "generated_at",
    ],
    "layer6_drift_report.csv": ["test", "variable", "severity", "alert", "retrain_urgency"],
    "layer6_model_health_summary.csv": [
        "metric_group", "metric", "status", "overall_health",
    ],
    "layer6_monitoring_diagnostics.csv": ["diagnostic_group", "metric", "value", "flag"],
    JOSV_NORMALIZED: [
        "event_id", "tail_risk_prob_z", "fragility_signal_z", "obi_signal_z",
        "drift_score_z", "novelty_score_z", "duration_reliability",
    ],
}

# ----------------------------------------------------------------------------- freshness
STALE_HOURS = 48.0  # input older than this (vs run time) is flagged stale

# ----------------------------------------------------------------------------- ORS (Phase 2)
# Operational Risk Score logit weights (fixed; mandated by M1 spec).
ORS_WEIGHTS = {
    "tail_risk_prob_z": 0.30,
    "fragility_signal_z": 0.20,
    "obi_signal_z": 0.15,
    "drift_score_z": 0.15,
    "novelty_score_z": 0.10,
    "critical_alert_indicator": 0.10,
}
# Discount coefficients: well-protected (high L5 robustness) and high-reliability
# sites are less operationally urgent.
ORS_ROBUSTNESS_DISCOUNT = 0.50   # ORS *= (1 - 0.50 * robustness_score)
ORS_RELIABILITY_DISCOUNT = 0.30  # ORS *= (1 - 0.30 * (1 - duration_reliability))

# PATCH F-001: ORS uses PERCENTILE-RANK normalization of each feature before
# weighting (replaces raw L4.5 z-scores, which let tail_risk_prob_z dominate ~94%).
# Each feature -> within-population percentile, centred to [-0.5, 0.5], then scaled
# so the weighted logit spans a meaningful sigmoid range.
ORS_PCT_SCALE = 6.0

# Tier percentile cut-points (on discounted ORS rank). Mirrors L5 tier philosophy.
ORS_TIER_CUTS = [
    ("Emergency", 0.92),  # top 8%
    ("Critical", 0.80),   # next 12%
    ("Elevated", 0.60),   # next 20%
    ("Normal", 0.00),     # remainder
]

# ----------------------------------------------------------------------------- ASS (Phase 3)
# Base severity map (mandated). Unknown labels fall back to info with a warning.
BASE_SEVERITY = {
    "info": 0.25,
    "warning": 0.50,
    "moderate": 0.65,
    "critical": 0.90,
}
SEVERITY_FALLBACK = 0.25
CORROBORATION_PER_SOURCE = 0.15   # corroboration_factor = 1 + 0.15 * n_sources
RECENCY_HALFLIFE_HOURS = 24.0     # recency_factor = exp(-age_hours / 24)

# Alert priority cut-points on Alert Severity Score (ASS).
# (Retained for backward compatibility; superseded by ASS_PRIORITY_PCT_CUTS below.)
ASS_PRIORITY_CUTS = [
    ("P1", 0.85),
    ("P2", 0.60),
    ("P3", 0.40),
    ("P4", 0.00),
]

# PATCH F-002: ASS stays MULTIPLICATIVE (base x corroboration x recency) but is now
# bounded to [0,1] by dividing by its theoretical maximum (monotone rescale, ordering
# preserved). Max corroboration assumes up to 4 independent source feeds.
ASS_MAX_SOURCES = 4
# PATCH F-003: priority is assigned by PERCENTILE of the bounded ASS distribution
# (quantile-based), so P1 is reserved for the genuine top band instead of every
# corroborated critical. P1=top 15%, P2=55-85, P3=25-55, P4=bottom 25%.
ASS_PRIORITY_PCT_CUTS = [
    ("P1", 0.85),
    ("P2", 0.55),
    ("P3", 0.25),
    ("P4", 0.00),
]

# PATCH F-004: override impact tiers are quantile-based on the absolute_ois
# distribution (the fixed-anchor absolute_ois never reaches 0.5, so static cuts were
# unreachable). Critical=top 10%, High=next 20%, Moderate=next 30%, Low=remainder.
OIS_IMPACT_PCT_CUTS = [
    ("Critical", 0.90),
    ("High", 0.70),
    ("Moderate", 0.40),
    ("Low", 0.00),
]
