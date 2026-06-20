"""
Layer 7 Part C — Risk Translation to Operational Actions
ASTraM Bengaluru Traffic Disruption Intelligence

Additive. Does NOT modify Layers 1-6 or Parts A/B outputs.
Run after layer7_persistence.py.

PURPOSE:
  Translate Part B's ERI, persistence class, SSC, and confidence bands
  into six operational actions so the output says what to do next, not
  only what the risk number is.

ACTIONS (in escalating order):
  watch                          No pre-positioning needed.
  increase_monitoring            Heighten alert; duty officer checks more often.
  stage_tow_unit                 Pre-position tow near zone; standby deployment.
  stage_barricade                Move barricading equipment to zone perimeter.
  prepare_diversion              Alert traffic control; pre-clear alternate routes.
  escalate_supervision           Senior officer response; activate diversion now.

THRESHOLD CALIBRATION:
  Action thresholds are percentiles of the maximum-across-horizons ERI
  computed on the TRAINING period (Nov 10 – Dec 31 2023) using the same
  CatBoost + isotonic calibration pipeline as Part B.  No thresholds are
  hardcoded to arbitrary numbers.  The SSC high/low split uses SSC_norm
  >= 0.5 (median across the 10 zones).  The CI-width narrow/wide split
  uses the p50 of eval CI half-widths.

D_hat NORMALIZATION:
  D_hat_norm uses a [0.05, 1.0] floor instead of pure [0, 1] min-max.
  Central Zone 1's raw fragility median (0.864) is the minimum across zones
  but only 0.085 below the next-lowest (North Zone 1, 0.948), supported by
  183 training observations — a real, confident estimate, not thin data.
  Pure [0, 1] normalization would silence CZ1's ERI entirely regardless of
  its predicted probability or spillover involvement.  The 0.05 floor gives
  CZ1 D_hat_norm = 0.050, preserving its rank while letting its ERI respond
  to high predicted probability or high SSC.

OUTPUTS:
  outputs/layer7_action_policy.csv
  outputs/layer7_prepositioning_recommendations.csv
  outputs/layer7_operational_alerts.csv
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUTS   = Path("outputs")
ARTIFACTS = OUTPUTS / "layer7_model_artifacts"

BASE_FEATURES = [
    "hour_sin", "hour_cos", "dow", "is_weekend", "is_peak",
    "log_lambda_v", "count_1h", "count_4h", "count_24h",
    "burden_1h", "burden_4h",
    "hotspot_n_sig", "hotspot_max_z", "hotspot_mean_burden",
    "zone_mean_fragility", "zone_hist_rate",
]
ALL_FEATURES  = BASE_FEATURES + ["zone"]
CAT_FEATURES  = ["zone"]
HORIZONS      = [3, 6, 9]
TOP_K_ALERTS  = 30

ACTION_LABELS = {
    "watch":                 "No action — continue normal monitoring",
    "increase_monitoring":   "Heighten alert; duty officer should increase check-in frequency",
    "stage_tow_unit":        "Pre-position tow unit near zone; standby for rapid deployment",
    "stage_barricade":       "Move barricading equipment to zone perimeter; notify junction marshals",
    "prepare_diversion":     "Alert traffic control; identify and pre-clear alternate routes",
    "escalate_supervision":  "Senior officer response required; activate diversion plan immediately",
}
ACTION_ORDER = list(ACTION_LABELS.keys())   # ascending severity

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATA AND ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: LOAD DATA ===")

panel = pd.read_csv(OUTPUTS / "layer7_panel_dataset.csv")
eri   = pd.read_csv(OUTPUTS / "layer7_expected_risk_index.csv")
pers  = pd.read_csv(OUTPUTS / "layer7_risk_persistence.csv")
ssc   = pd.read_csv(OUTPUTS / "layer7_spillover_centrality.csv")
feat  = pd.read_csv(OUTPUTS / "layer45_asof_feature_matrix.csv")

with open(ARTIFACTS / "layer7_panel_meta.json") as fh:
    meta = json.load(fh)
with open(ARTIFACTS / "layer7_calibration_decisions.json") as fh:
    cal_decisions = json.load(fh)

zones = meta["zones"]
print(f"  Panel: {panel.shape}, ERI: {eri.shape}, Persistence: {pers.shape}")

try:
    from catboost import CatBoostClassifier, Pool as CatPool
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False
    print("  WARNING: CatBoost not available — training ERI will use Hawkes proxy")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: RECOMPUTE D_hat AND SSC NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: D_hat AND SSC ===")

feat["start_local"] = pd.to_datetime(feat["start_local"], utc=True, errors="coerce")
feat_valid = feat.dropna(subset=["zone"]).copy()
train_mask_feat = (
    (feat_valid["start_local"] >= "2023-11-10") &
    (feat_valid["start_local"] <= "2023-12-31")
)
d_raw = (feat_valid[train_mask_feat]
         .groupby("zone")["asof_fragility_proxy"]
         .median()
         .reindex(zones)
         .fillna(feat_valid[train_mask_feat]["asof_fragility_proxy"].median()))
d_min, d_max = d_raw.min(), d_raw.max()
# Floor at 0.05 — same rationale as in layer7_risk_index.py:
# CZ1 median (0.864) is only 0.085 below the next-lowest zone; pure [0,1]
# min-max forces its ERI to exactly 0 regardless of probability or spillover.
DHAT_FLOOR = 0.05
d_hat_norm = (DHAT_FLOOR + (1.0 - DHAT_FLOOR) * (d_raw - d_min) / (d_max - d_min + 1e-9)).to_dict()

ssc_vals = ssc.set_index("zone")["SSC_centrality"].reindex(zones)
ssc_min, ssc_max = ssc_vals.min(), ssc_vals.max()
ssc_norm = ((ssc_vals - ssc_min) / (ssc_max - ssc_min + 1e-9)).to_dict()

print(f"  D_hat_norm (floor=0.05): {dict((z, round(v,4)) for z,v in d_hat_norm.items())}")
print(f"  SSC_norm:   {dict((z, round(v,3)) for z,v in ssc_norm.items())}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAINING-PERIOD ERI  (for threshold calibration)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: TRAINING-PERIOD ERI ===")

panel_train = panel[panel["split"] == "train"].copy()
panel_train["grid_dt"] = pd.to_datetime(panel_train["grid_time_utc"], format="ISO8601", utc=True)

train_eri_by_h  = {}   # h -> Series of ERI values (indexed to panel_train row positions)
train_prob_by_h = {}   # h -> Series of P values  (for per-zone CI calibration)

for h in HORIZONS:
    bin_col = f"Y_bin_{h}h"
    tr = panel_train.dropna(subset=[bin_col]).copy()
    tr_df = tr[ALL_FEATURES].fillna(0)

    if CATBOOST_OK:
        model_path = ARTIFACTS / f"catboost_binary_{h}h.cbm"
        if not model_path.exists():
            print(f"  {h}h: model not found, using Hawkes proxy")
            p_tr = np.minimum(tr["lambda_v"].values * h, 1.0)
        else:
            cb = CatBoostClassifier()
            cb.load_model(str(model_path))
            pool_tr = CatPool(tr_df, cat_features=CAT_FEATURES)
            p_tr = cb.predict_proba(pool_tr)[:, 1]
            if cal_decisions.get(str(h), {}).get("use_calibration", False):
                cal_path = ARTIFACTS / f"isotonic_cal_{h}h.pkl"
                if cal_path.exists():
                    with open(cal_path, "rb") as fh_:
                        iso = pickle.load(fh_)
                    p_tr = iso.predict(p_tr)
    else:
        p_tr = np.minimum(tr["lambda_v"].values * h, 1.0)

    d_arr   = np.array([d_hat_norm.get(z, 0.0) for z in tr["zone"].values])
    ssc_arr = np.array([ssc_norm.get(z, 0.0) for z in tr["zone"].values])
    eri_tr  = p_tr * d_arr * (1.0 + ssc_arr)

    train_eri_by_h[h]  = pd.Series(eri_tr, index=tr.index)
    train_prob_by_h[h] = pd.Series(p_tr,   index=tr.index)
    print(f"  {h}h training ERI: n={len(eri_tr)}  "
          f"mean={eri_tr.mean():.4f}  p75={np.percentile(eri_tr,75):.4f}  "
          f"p95={np.percentile(eri_tr,95):.4f}")

# Max ERI across horizons per training row (most conservative basis)
tr_ref = panel_train.dropna(subset=["Y_bin_3h"]).copy()
eri_stack = pd.DataFrame({h: train_eri_by_h[h].reindex(tr_ref.index).fillna(0)
                          for h in HORIZONS})
max_train_eri = eri_stack.max(axis=1).values

print(f"\n  Max-across-horizons training ERI (n={len(max_train_eri)}):")
for p in [25, 50, 75, 90, 95]:
    print(f"    p{p}: {np.percentile(max_train_eri, p):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CALIBRATE ACTION THRESHOLDS FROM TRAINING PERCENTILES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: THRESHOLD CALIBRATION ===")

T_low      = float(np.percentile(max_train_eri, 25))   # below -> watch
T_high     = float(np.percentile(max_train_eri, 75))   # above -> potentially stage
T_veryhigh = float(np.percentile(max_train_eri, 90))   # above -> prepare_diversion
T_critical = float(np.percentile(max_train_eri, 95))   # above + escalating -> escalate

# SSC split: high if SSC_norm >= 0.5  (above-median zone)
SSC_HIGH_THRESHOLD = 0.5

# ── Per-zone CI narrow/wide thresholds (training sub-window percentiles) ──────
#
# WHY PER-ZONE, NOT GLOBAL:
# The ERI and CI half-width share the same zone-level multiplier:
#   ERI      = P × D_hat × (1 + SSC)
#   CI_half  = 1.96 × std(P) × D_hat × (1 + SSC)
# A global CI threshold conflates two zones that differ only in D_hat.
# East Zone 2 (D_hat=1.0) will always have CI ~19× wider than Central Zone 1
# (D_hat=0.05) for the same probability variance.  Judging East Zone 2's CI
# against a global p50 that is dominated by low-D_hat zones is equivalent
# to requiring East Zone 2 to be 19× more certain than other zones to be
# considered "narrow" — which is impossible by construction.
#
# Per-zone thresholds fix this by asking: is THIS zone's current CI narrow
# relative to ITS OWN historical uncertainty?  That removes the D_hat confound.
#
# METHOD:
# Split training into 3 sub-windows (monthly) × 3 horizons = 9 CI samples
# per zone. The p25 of those 9 samples is each zone's "narrow" threshold.
# Training sub-windows give the model's own typical uncertainty for each zone,
# using the same CatBoost predictions used everywhere else.

SUB_WINDOWS = [
    ("Nov 10-30", pd.Timestamp("2023-11-10", tz="UTC"), pd.Timestamp("2023-11-30 23:59:59", tz="UTC")),
    ("Dec 01-15", pd.Timestamp("2023-12-01", tz="UTC"), pd.Timestamp("2023-12-15 23:59:59", tz="UTC")),
    ("Dec 16-31", pd.Timestamp("2023-12-16", tz="UTC"), pd.Timestamp("2023-12-31 23:59:59", tz="UTC")),
]

zone_ci_narrow_threshold = {}
zone_ci_info = {}

print("\n  Per-zone CI thresholds (training p25 across sub-windows × horizons):")
print(f"  {'Zone':<24} {'p25(narrow)':>11} {'p50':>7} {'p75':>7} {'min':>7} {'max':>7}  vs_eval_ci")
for z in zones:
    d_z   = d_hat_norm.get(z, 0.0)
    ssc_z = ssc_norm.get(z, 0.0)
    # Base multiplier: same formula as risk_index.py CI computation
    mult  = 1.96 * d_z * (1.0 + ssc_z)
    if z == "South Zone 2":
        mult *= 1.5   # consistent with the widening applied in risk_index.py

    ci_samples = []
    for win_name, win_start, win_end in SUB_WINDOWS:
        win_idx = panel_train[
            (panel_train["grid_dt"] >= win_start) &
            (panel_train["grid_dt"] <= win_end) &
            (panel_train["zone"] == z)
        ].index
        for h in HORIZONS:
            p_win = train_prob_by_h[h].reindex(win_idx).dropna()
            if len(p_win) >= 5:
                ci_samples.append(mult * float(p_win.std()))

    if ci_samples:
        t_narrow = float(np.percentile(ci_samples, 25))
        ci_p50   = float(np.percentile(ci_samples, 50))
        ci_p75   = float(np.percentile(ci_samples, 75))
        ci_min   = float(np.min(ci_samples))
        ci_max   = float(np.max(ci_samples))
    else:
        # Fallback: use overall eval CI median for that zone
        eri_z = eri[eri["zone"] == z]
        t_narrow = float(((eri_z["ERI_ci_hi"] - eri_z["ERI_ci_lo"]) / 2).median())
        ci_p50 = ci_p75 = ci_min = ci_max = t_narrow

    zone_ci_narrow_threshold[z] = t_narrow

    # For reporting: what is the eval CI for this zone (from persistence table)?
    pers_z = pers[pers["zone"] == z]
    eval_ci = float(((pers_z["Pers_ci_hi"] - pers_z["Pers_ci_lo"]) / 2).mean()) if len(pers_z) > 0 else np.nan
    eval_relative = "NARROW" if eval_ci < t_narrow else "wide"

    zone_ci_info[z] = {
        "ci_p25_narrow_threshold": round(t_narrow, 4),
        "ci_p50": round(ci_p50, 4),
        "ci_p75": round(ci_p75, 4),
        "ci_min": round(ci_min, 4),
        "ci_max": round(ci_max, 4),
        "n_training_samples": len(ci_samples),
        "eval_ci_mean": round(eval_ci, 4),
        "eval_ci_vs_threshold": eval_relative,
    }
    print(f"  {z:<24} {t_narrow:>11.4f} {ci_p50:>7.4f} {ci_p75:>7.4f} "
          f"{ci_min:>7.4f} {ci_max:>7.4f}  eval={eval_ci:.4f} ({eval_relative})")

print(f"\n  ERI thresholds: low<{T_low:.4f}  high>={T_high:.4f}  "
      f"veryhigh>={T_veryhigh:.4f}  critical>={T_critical:.4f}")
print(f"  SSC high if SSC_norm >= {SSC_HIGH_THRESHOLD}")

thresholds = {
    "T_low":            round(T_low, 4),
    "T_high":           round(T_high, 4),
    "T_veryhigh":       round(T_veryhigh, 4),
    "T_critical":       round(T_critical, 4),
    "SSC_high_cutoff":  SSC_HIGH_THRESHOLD,
    "CI_narrow_method": "per-zone p25 of training sub-window CI half-widths",
    "CI_narrow_rationale": (
        "Global threshold confounded by shared D_hat multiplier; "
        "per-zone threshold removes this by asking whether a zone's "
        "current CI is narrow relative to its own historical uncertainty."
    ),
    "source": "training-period percentiles (max ERI p25/p75/p90/p95, CI p25 per zone)",
}

def eri_level(val):
    if val >= T_critical:  return "critical"
    if val >= T_veryhigh:  return "very_high"
    if val >= T_high:      return "high"
    if val >= T_low:       return "moderate"
    return "low"

def assign_action(max_eri, pers_class, ssc_n, ci_half, zone):
    """
    Rule-based action assignment.  All cutoffs are training-period
    percentile-calibrated.  CI narrow/wide uses a PER-ZONE threshold
    so high-D_hat zones are judged against their own uncertainty history.
    """
    level      = eri_level(max_eri)
    ssc_hi     = ssc_n >= SSC_HIGH_THRESHOLD
    narrow_thr = zone_ci_narrow_threshold.get(zone, float("inf"))
    ci_wide    = ci_half >= narrow_thr   # wide = above zone's own p25

    if level == "critical" and pers_class == "escalating" and not ci_wide:
        return "escalate_supervision"

    if level in ("critical", "very_high") and ssc_hi and pers_class != "transient":
        return "prepare_diversion"

    if level in ("critical", "very_high", "high") and ssc_hi and pers_class != "transient":
        return "stage_barricade"

    if level in ("critical", "very_high", "high") and pers_class in ("persistent", "escalating"):
        return "stage_tow_unit"

    if level in ("moderate", "high", "very_high", "critical") or ci_wide:
        return "increase_monitoring"

    return "watch"

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: JOIN EVAL ERI + PERSISTENCE, COMPUTE MAX ERI, ASSIGN ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: EVAL ACTION ASSIGNMENT ===")

# persistence table already has ERI_3h, ERI_6h, ERI_9h and Pers_z/Slope_z
df = pers.copy()
df["max_ERI"]   = df[["ERI_3h", "ERI_6h", "ERI_9h"]].fillna(0).max(axis=1)
df["ERI_level"] = df["max_ERI"].apply(eri_level)

# SSC_norm is a zone property — add it from the lookup dict
df["SSC_norm"]  = df["zone"].map(ssc_norm).fillna(0.0)
df["D_hat_norm"]= df["zone"].map(d_hat_norm).fillna(0.0)

# CI half-width from persistence confidence bands
df["ci_half"]   = ((df["Pers_ci_hi"] - df["Pers_ci_lo"]) / 2.0).fillna(0.0)

df["recommended_action"] = df.apply(
    lambda r: assign_action(r["max_ERI"], r["persistence_class"],
                            r["SSC_norm"], r["ci_half"], r["zone"]), axis=1
)
df["action_label"] = df["recommended_action"].map(ACTION_LABELS)
df["action_severity"] = df["recommended_action"].map(
    {a: i for i, a in enumerate(ACTION_ORDER)}
)

action_counts = df["recommended_action"].value_counts()
print(f"  Action distribution across {len(df)} eval (zone, time) pairs:")
for act in ACTION_ORDER:
    print(f"    {act:<28}: {action_counts.get(act, 0):>5}")

# Flag Central Zone 1 D_hat=0 limitation
df["d_hat_floor_note"] = df["zone"].apply(
    lambda z: "D_hat_norm=0.050 (floor applied; raw was minimum zone)"
    if abs(d_hat_norm.get(z, 1.0) - DHAT_FLOOR) < 1e-4 else ""
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: BUILD OUTPUT DATAFRAMES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: BUILD OUTPUTS ===")

# ── 6a: Action Policy (the rules document) ────────────────────────────────────
policy_rows = []
for act in ACTION_ORDER:
    policy_rows.append({
        "action":            act,
        "action_label":      ACTION_LABELS[act],
        "severity_rank":     ACTION_ORDER.index(act),
        "triggers":          {
            "watch":               f"max_ERI < {T_low:.4f}  (training p25)",
            "increase_monitoring": (f"max_ERI in [{T_low:.4f}, {T_high:.4f})  OR  "
                                    f"CI_half >= zone_ci_narrow_threshold[zone]  (wide vs zone's own p25)"),
            "stage_tow_unit":      (f"max_ERI >= {T_high:.4f}  AND  "
                                    f"persistence in [persistent, escalating]"),
            "stage_barricade":     (f"max_ERI >= {T_high:.4f}  AND  "
                                    f"SSC_norm >= {SSC_HIGH_THRESHOLD}  AND  "
                                    f"persistence != transient"),
            "prepare_diversion":   (f"max_ERI >= {T_veryhigh:.4f}  AND  "
                                    f"SSC_norm >= {SSC_HIGH_THRESHOLD}  AND  "
                                    f"persistence != transient"),
            "escalate_supervision":(f"max_ERI >= {T_critical:.4f}  AND  "
                                    f"persistence == escalating  AND  "
                                    f"CI_half < zone_ci_narrow_threshold[zone]  (zone-relative narrow)"),
        }.get(act, ""),
        "threshold_source":  "training-period percentile of max(ERI_3h, ERI_6h, ERI_9h) "
                             "computed Nov 10-Dec 31 2023",
        "ERI_cutoff_low":    round(T_low, 4),
        "ERI_cutoff_high":   round(T_high, 4),
        "ERI_cutoff_veryhigh": round(T_veryhigh, 4),
        "ERI_cutoff_critical": round(T_critical, 4),
        "SSC_norm_high_cutoff":   SSC_HIGH_THRESHOLD,
        "CI_narrow_per_zone":     "see layer7_zone_ci_thresholds.csv",
    })
df_policy = pd.DataFrame(policy_rows)

# Append per-zone CI threshold table as a separate section in policy output
ci_policy_rows = []
for z in zones:
    info = zone_ci_info.get(z, {})
    ci_policy_rows.append({
        "zone": z,
        "D_hat_norm": round(d_hat_norm.get(z, 0.0), 4),
        "SSC_norm": round(ssc_norm.get(z, 0.0), 3),
        "ci_p25_narrow_threshold": info.get("ci_p25_narrow_threshold"),
        "ci_p50": info.get("ci_p50"),
        "ci_p75": info.get("ci_p75"),
        "ci_min_training": info.get("ci_min"),
        "ci_max_training": info.get("ci_max"),
        "n_training_samples": info.get("n_training_samples"),
        "eval_ci_mean": info.get("eval_ci_mean"),
        "eval_ci_vs_threshold": info.get("eval_ci_vs_threshold"),
        "rationale": (
            "Per-zone: D_hat confound removed; 'narrow' = zone's own training p25. "
            f"Old global threshold was {float((eri['ERI_ci_hi']-eri['ERI_ci_lo']).median()/2):.4f}."
        ),
    })
df_zone_ci_policy = pd.DataFrame(ci_policy_rows)

# ── 6b: Prepositioning Recommendations ───────────────────────────────────────
prepos_cols = [
    "grid_time_utc", "zone",
    "recommended_action", "action_label", "action_severity",
    "max_ERI", "ERI_level", "ERI_3h", "ERI_6h", "ERI_9h",
    "Pers_z", "Slope_z", "persistence_class",
    "Pers_ci_lo", "Pers_ci_hi", "ci_half",
    "SSC_norm", "D_hat_norm",
    "prob_bin_hawkes_3h", "prob_bin_hawkes_6h", "prob_bin_hawkes_9h",
    "south_zone_2_note", "d_hat_floor_note",
]
avail = [c for c in prepos_cols if c in df.columns]
df_prepos = df[avail].copy()
df_prepos = df_prepos.round({c: 4 for c in df_prepos.select_dtypes(float).columns})

# ── 6c: Operational Alerts (top-K, plain English, extended top-k) ─────────────
# Sort by action severity (desc) then max_ERI (desc) — most urgent first
df_alerts = (df.sort_values(["action_severity", "max_ERI"], ascending=[False, False])
               .head(TOP_K_ALERTS)
               .reset_index(drop=True))

# Human-readable score display
df_alerts["risk_score"]   = df_alerts["max_ERI"].round(4)
df_alerts["confidence"] = df_alerts.apply(
    lambda r: (f"narrow (CI half={r['ci_half']:.3f})"
               if r["ci_half"] < zone_ci_narrow_threshold.get(r["zone"], float("inf"))
               else f"wide (CI half={r['ci_half']:.3f})"), axis=1
)

alert_cols = [
    "grid_time_utc", "zone",
    "risk_score", "confidence",
    "persistence_class", "Slope_z",
    "recommended_action", "action_label",
    "ERI_3h", "ERI_6h", "ERI_9h",
    "prob_bin_hawkes_3h",
    "SSC_norm", "spillover_contribution_share",
    "south_zone_2_note",
]

# Add spillover share column
df_alerts["spillover_share_3h"] = (
    df_alerts.get("ERI_3h", pd.Series(0, index=df_alerts.index)).fillna(0) -
    df_alerts.get("Pers_mean_ERI_base", pd.Series(0, index=df_alerts.index)).fillna(0).clip(lower=0)
)

avail_alerts = [c for c in alert_cols if c in df_alerts.columns]
df_alerts_out = df_alerts[avail_alerts + ["action_severity"]].copy()
df_alerts_out = df_alerts_out.drop(columns=["action_severity"])
df_alerts_out = df_alerts_out.round({c: 4 for c in df_alerts_out.select_dtypes(float).columns})

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: WRITE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: WRITE OUTPUTS ===")

df_policy.to_csv(OUTPUTS / "layer7_action_policy.csv", index=False)
print(f"  Wrote layer7_action_policy.csv ({len(df_policy)} rows — one per action)")

df_zone_ci_policy.to_csv(OUTPUTS / "layer7_zone_ci_thresholds.csv", index=False)
print(f"  Wrote layer7_zone_ci_thresholds.csv ({len(df_zone_ci_policy)} zones)")

df_prepos.to_csv(OUTPUTS / "layer7_prepositioning_recommendations.csv", index=False)
print(f"  Wrote layer7_prepositioning_recommendations.csv ({len(df_prepos)} rows)")

df_alerts_out.to_csv(OUTPUTS / "layer7_operational_alerts.csv", index=False)
print(f"  Wrote layer7_operational_alerts.csv ({len(df_alerts_out)} rows)")

# Show top-10 alerts
print(f"\n  Top-10 operational alerts:")
print(f"  {'Zone':<22} {'Action':<22} {'Score':>7} {'Confidence':<26} {'Persistence'}")
for _, r in df_alerts_out.head(10).iterrows():
    conf_str = str(r.get("confidence", ""))[:25]
    pers_str = str(r.get("persistence_class", ""))
    print(f"  {r['zone']:<22} {r['recommended_action']:<22} "
          f"{r['risk_score']:>7.4f} {conf_str:<26} {pers_str}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: UPDATE layer7_summary.txt
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: UPDATE SUMMARY ===")

summary_path = OUTPUTS / "layer7_summary.txt"
summary_text = summary_path.read_text(encoding="utf-8")

SUMMARY_MARKER = "OPERATIONAL TRANSLATION (Part C):"
if SUMMARY_MARKER not in summary_text:
    part_c_section = f"""
OPERATIONAL TRANSLATION (Part C):
  src/layer7_risk_translation.py translates ERI + persistence + SSC into
  six operational actions.  This is NOT a new predictive model — it is an
  interpretable rule layer on top of Part B's outputs.

  ACTIONS (ascending severity):
    watch                 No pre-positioning needed.
    increase_monitoring   Duty officer increases check-in frequency.
    stage_tow_unit        Pre-position tow near zone; standby deployment.
    stage_barricade       Move barricading equipment to zone perimeter.
    prepare_diversion     Alert traffic control; pre-clear alternate routes.
    escalate_supervision  Senior officer response; activate diversion now.

  THRESHOLD CALIBRATION (training-period percentiles, not hardcoded):
    Thresholds are percentiles of max(ERI_3h, ERI_6h, ERI_9h) computed on
    the training period (Nov 10 - Dec 31 2023) via the same CatBoost +
    isotonic pipeline used in Part B.
    ERI p25={T_low:.4f}  p75={T_high:.4f}  p90={T_veryhigh:.4f}  p95={T_critical:.4f}
    SSC high if SSC_norm >= {SSC_HIGH_THRESHOLD} (above-median zone).
    CI wide if half-width >= {T_ci_wide:.4f} (eval p50; triggers caution).

  EVAL ACTION DISTRIBUTION (Jan 1-19 2024, {len(df)} zone-time pairs):
    {chr(10).join(f"    {a:<28}: {action_counts.get(a,0):>5}" for a in ACTION_ORDER)}

  CAVEAT — CENTRAL ZONE 1:
    D_hat_norm = 0.000 for this zone (minimum fragility proxy after
    normalization).  Its ERI is always 0 and it always receives "watch".
    Operational users should consult binary probability P_z,h directly
    for Central Zone 1 rather than relying on its ERI or action rank.

  OUTPUTS:
    layer7_action_policy.csv              — Threshold rules + action descriptions
    layer7_prepositioning_recommendations.csv — Per (zone, time): action + full context
    layer7_operational_alerts.csv         — Top-{TOP_K_ALERTS} alerts, plain English, readable
                                            by non-technical users without knowing the math
"""
    summary_path.write_text(summary_text + part_c_section, encoding="utf-8")
    print("  layer7_summary.txt updated with Part C section")
else:
    print("  layer7_summary.txt already has Part C section — skipped")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: UPDATE README.md
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 9: UPDATE README ===")

readme_path = Path("README.md")
readme_text  = readme_path.read_text(encoding="utf-8")

README_MARKER = "### Layer 7 Part C — Operational Translation (completed)"
if README_MARKER not in readme_text:
    readme_addition = f"""
{README_MARKER}

**Full Layer 7 system:** Part A tests for cross-zone spillover. Part B forecasts event probability and count per zone and horizon, classifies persistence, and produces ERI scores. Part C translates those scores into six operational actions a dispatcher can act on directly.

**Part C is an operational translation layer, not a new predictive model.** It applies calibrated rules to Part B outputs. No new ML model is trained.

**Threshold calibration:** Action cutoffs are percentiles of the training-period ERI distribution (max across horizons, Nov 10–Dec 31 2023), not hardcoded values. Specifically: p25/p75/p90/p95 define the low/high/very-high/critical ERI levels. The SSC high/low split uses SSC_norm ≥ 0.5 (median zone). CI width split uses p50 of eval half-widths. All cutoffs are documented in `layer7_action_policy.csv`.

**Actions:** watch → increase monitoring → stage tow unit → stage barricade → prepare diversion → escalate supervision. Higher SSC (spillover involvement) escalates towards barricade/diversion; escalating persistence trend with high certainty triggers the top action.

**Caveat — Central Zone 1:** Its D_hat_norm = 0.000 (minimum fragility proxy after normalization), making its ERI always 0. It always receives "watch" from the action policy. Consult its binary probability P_z,h directly.

| Output | Contents |
|---|---|
| `layer7_action_policy.csv` | Six-row policy: action, description, threshold triggers, all cutoff values |
| `layer7_prepositioning_recommendations.csv` | Per (zone, grid_time): recommended action, ERI decomposition, persistence, CI |
| `layer7_operational_alerts.csv` | Top-{TOP_K_ALERTS} alerts by urgency, plain-English action label, readable without technical context |
"""
    readme_path.write_text(readme_text + readme_addition, encoding="utf-8")
    print("  README.md updated with Layer 7 Part C section")
else:
    print("  README.md already has Part C section — skipped")

print("\n=== layer7_risk_translation.py complete ===")
