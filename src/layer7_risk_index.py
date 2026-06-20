"""
Layer 7 Risk Index — ERI, Calibration, SHAP
ASTraM Bengaluru Traffic Disruption Intelligence

Additive. Does NOT modify Layers 1-6, Part A, or previous sub-part outputs.
Run after layer7_zone_forecast.py.

ERI formula:
  ERI_base_z,h = P_z,h * D_hat_z_norm
  ERI_z,h      = ERI_base_z,h * (1 + SSC_norm_z)

D_hat_z = median(asof_fragility_proxy) for zone z, training period only.
  asof_fragility_proxy is Layer 4.5's as-of-safe impact indicator.
  Normalized to [0,1] across zones.

SSC_norm_z = (SSC_z - SSC_min) / (SSC_max - SSC_min)
  SSC from Part A outputs. Enters ERI only here, never as a model feature.

Confidence bands:
  Per (zone, horizon): 1.96 * std(P_z,h) across eval grid points.
  South Zone 2: band widened by factor 1.5 (KS p=0.021 in Part B;
  weakest spillover-model calibration among the 10 zones).

Outputs:
  outputs/layer7_metrics.csv
  outputs/layer7_reliability_diagram.csv
  outputs/layer7_spatial_calibration.csv
  outputs/layer7_shap_summary.csv
  outputs/layer7_expected_risk_index.csv
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score)

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUTS   = Path("outputs")
ARTIFACTS = OUTPUTS / "layer7_model_artifacts"
SOUTH_Z2  = "South Zone 2"
SZ2_CI_FACTOR = 1.5   # documented CI widening for South Zone 2
N_BINS    = 10        # reliability diagram bins
N_BOOTSTRAP = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: LOAD DATA ===")

pred = pd.read_csv(OUTPUTS / "layer7_predictions.csv")
base = pd.read_csv(OUTPUTS / "layer7_baseline_predictions.csv")
panel = pd.read_csv(OUTPUTS / "layer7_panel_dataset.csv")
ssc  = pd.read_csv(OUTPUTS / "layer7_spillover_centrality.csv")
feat = pd.read_csv(OUTPUTS / "layer45_asof_feature_matrix.csv")

with open(ARTIFACTS / "layer7_panel_meta.json") as fh:
    meta = json.load(fh)
zones = meta["zones"]

# Resolve actual target per row (each row carries only its own horizon's target)
def resolve_target(df, col_prefix):
    return df.apply(lambda r: r.get(f"{col_prefix}_{int(r.horizon_h)}h", np.nan), axis=1)

pred["Y_bin_actual"]   = resolve_target(pred, "Y_bin")
pred["Y_count_actual"] = resolve_target(pred, "Y_count")
base["Y_bin_actual"]   = resolve_target(base, "Y_bin")
base["Y_count_actual"] = resolve_target(base, "Y_count")

pred = pred.dropna(subset=["Y_bin_actual"]).copy()
base = base.dropna(subset=["Y_bin_actual"]).copy()

print(f"  Predictions: {pred.shape}, Baseline: {base.shape}")
print(f"  SSC zones: {len(ssc)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: D_hat AND SSC NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: D_hat AND SSC ===")

feat["start_local"] = pd.to_datetime(feat["start_local"], utc=True, errors="coerce")
feat_valid = feat.dropna(subset=["zone"]).copy()
train_mask = (
    (feat_valid["start_local"] >= "2023-11-10") &
    (feat_valid["start_local"] <= "2023-12-31")
)

# D_hat = median asof_fragility_proxy per zone during training (as-of-safe)
d_raw = (feat_valid[train_mask]
         .groupby("zone")["asof_fragility_proxy"]
         .median()
         .reindex(zones)
         .fillna(feat_valid[train_mask]["asof_fragility_proxy"].median()))
d_min, d_max = d_raw.min(), d_raw.max()
d_hat_norm = (d_raw - d_min) / (d_max - d_min + 1e-9)
print(f"  D_hat (normalized):")
for z in zones:
    print(f"    {z}: raw={d_raw.get(z, np.nan):.3f}  norm={d_hat_norm.get(z, np.nan):.3f}")

# SSC normalization
ssc_vals = ssc.set_index("zone")["SSC_centrality"].reindex(zones)
ssc_min, ssc_max = ssc_vals.min(), ssc_vals.max()
ssc_norm = (ssc_vals - ssc_min) / (ssc_max - ssc_min + 1e-9)
print(f"\n  SSC_norm range: [{ssc_norm.min():.4f}, {ssc_norm.max():.4f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPUTE ERI WITH CONFIDENCE BANDS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: ERI COMPUTATION ===")

eri_rows = []
for h in [3, 6, 9]:
    ph = pred[pred["horizon_h"] == h].copy()
    for z in zones:
        zh = ph[ph["zone"] == z]
        if len(zh) == 0:
            continue
        p_vals = zh["prob_bin_hawkes"].values
        p_mean = float(p_vals.mean())
        p_std  = float(p_vals.std()) if len(p_vals) > 1 else 0.0

        d_z   = float(d_hat_norm.get(z, 0.0))
        ssc_z = float(ssc_norm.get(z, 0.0))

        # Per-grid-point ERI
        eri_base_vals = p_vals * d_z
        eri_vals      = eri_base_vals * (1.0 + ssc_z)

        # Confidence band: 1.96 * d_z * std(P), widened for South Zone 2
        ci_half_base = 1.96 * d_z * p_std
        ci_half_eri  = ci_half_base * (1.0 + ssc_z)
        if z == SOUTH_Z2:
            ci_half_eri *= SZ2_CI_FACTOR
            ci_note = f"CI widened x{SZ2_CI_FACTOR} (KS p=0.021, weakest spillover calibration)"
        else:
            ci_note = ""

        for i, idx in enumerate(zh.index):
            row = zh.loc[idx]
            eri_rows.append({
                "grid_time_utc": row["grid_time_utc"],
                "zone": z,
                "horizon_h": h,
                "prob_bin_hawkes": round(float(p_vals[i]), 6),
                "D_hat_norm": round(d_z, 6),
                "SSC_norm": round(ssc_z, 6),
                "ERI_base": round(float(eri_base_vals[i]), 6),
                "ERI": round(float(eri_vals[i]), 6),
                "ERI_ci_lo": round(max(0.0, float(eri_vals[i]) - ci_half_eri), 6),
                "ERI_ci_hi": round(float(eri_vals[i]) + ci_half_eri, 6),
                "SSC_adjustment": round(float(eri_vals[i]) - float(eri_base_vals[i]), 6),
                "local_activity_contribution": round(float(eri_base_vals[i]), 6),
                "spillover_contribution": round(float(eri_vals[i]) - float(eri_base_vals[i]), 6),
                "Y_bin_actual": int(row["Y_bin_actual"]),
                "Y_count_actual": int(row["Y_count_actual"]) if not np.isnan(row["Y_count_actual"]) else None,
                "count_forecast": round(float(row["count_forecast"]), 4),
                "ci_note": ci_note,
            })

df_eri = pd.DataFrame(eri_rows)
print(f"  ERI table: {df_eri.shape}")
print(f"  ERI range: [{df_eri['ERI'].min():.4f}, {df_eri['ERI'].max():.4f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: HAWKES-ONLY BASELINE (logistic regression on log_lambda_v + zone)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: HAWKES-ONLY BASELINE ===")

panel_train = panel[panel["split"] == "train"].copy()
panel_eval  = panel[panel["split"] == "eval"].copy()
panel_eval["grid_time_utc"] = panel_eval["grid_time_utc"].astype(str)

hawkes_preds = {}
for h in [3, 6, 9]:
    bin_col = f"Y_bin_{h}h"
    tr = panel_train.dropna(subset=[bin_col]).copy()
    ev = panel_eval.dropna(subset=[bin_col]).copy()

    X_tr_hk = np.column_stack([
        tr["log_lambda_v"].fillna(-18).values,
        pd.get_dummies(tr["zone"]).values
    ])
    X_ev_hk = np.column_stack([
        ev["log_lambda_v"].fillna(-18).values,
        pd.get_dummies(ev["zone"]).reindex(
            columns=pd.get_dummies(tr["zone"]).columns, fill_value=0
        ).values
    ])
    y_tr = tr[bin_col].values.astype(int)

    lr_hk = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    lr_hk.fit(X_tr_hk, y_tr)
    prob_hk = lr_hk.predict_proba(X_ev_hk)[:, 1]

    ev = ev.copy()
    ev["prob_hawkes_only"] = prob_hk
    ev["Y_bin_actual"] = ev[bin_col]
    ev["horizon_h"] = h
    hawkes_preds[h] = ev[["grid_time_utc", "zone", "horizon_h", "prob_hawkes_only", "Y_bin_actual"]]
    print(f"  {h}h Hawkes-only LogReg: train={len(tr)}, eval={len(ev)}")

df_hk = pd.concat(hawkes_preds.values(), ignore_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CALIBRATION METRICS (global + per-zone)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: CALIBRATION METRICS ===")

def ece(y_true, y_prob, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece_val += (mask.sum() / n) * abs(acc - conf)
    return ece_val

def crps_point(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))

metrics_rows = []
spatial_rows = []

rng_bs = np.random.default_rng(42)

for h in [3, 6, 9]:
    ph_pred = pred[pred["horizon_h"] == h].copy()
    ph_base = base[base["horizon_h"] == h].copy()
    ph_hk   = df_hk[df_hk["horizon_h"] == h].copy()

    # Merge Hawkes-only probs into main eval set
    ph_merged = ph_pred.merge(
        ph_hk[["grid_time_utc", "zone", "prob_hawkes_only"]],
        on=["grid_time_utc", "zone"], how="left"
    )
    ph_merged["baseline_prob"] = ph_base["baseline_prob_bin"].values[:len(ph_merged)]

    y_true = ph_merged["Y_bin_actual"].values.astype(int)
    y_cnt  = ph_merged["Y_count_actual"].values.astype(float)

    for model_name, p_col in [
        ("catboost_calibrated", "prob_bin_hawkes"),
        ("catboost_raw",        "prob_bin_raw"),
        ("hawkes_only",         "prob_hawkes_only"),
        ("historical_rate",     "baseline_prob"),
    ]:
        y_pred = ph_merged[p_col].fillna(0.5).values
        valid  = np.isfinite(y_pred) & np.isfinite(y_true.astype(float))
        yp, yt = y_pred[valid], y_true[valid]
        if len(np.unique(yt)) < 2 or len(yp) == 0:
            continue
        row = {
            "horizon_h": h, "model": model_name,
            "n_eval": int(valid.sum()),
            "pos_rate": round(float(yt.mean()), 4),
            "auc":     round(roc_auc_score(yt, yp), 4),
            "brier":   round(brier_score_loss(yt, yp), 4),
            "logloss": round(log_loss(yt, yp), 4),
            "ece":     round(ece(yt, yp), 4),
        }
        if model_name == "catboost_calibrated":
            cnt_pred = ph_merged["count_forecast"].fillna(0).values[valid]
            row["count_mae"]  = round(float(np.mean(np.abs(cnt_pred - y_cnt[valid]))), 4)
            row["count_rmse"] = round(float(np.sqrt(np.mean((cnt_pred - y_cnt[valid])**2))), 4)
            row["count_crps"] = row["count_mae"]   # CRPS = MAE for point forecasts
        metrics_rows.append(row)

    # Per-zone calibration (catboost_calibrated)
    for z in zones:
        zm = ph_merged[ph_merged["zone"] == z]
        if len(zm) == 0:
            continue
        y_z   = zm["Y_bin_actual"].values.astype(int)
        p_z   = zm["prob_bin_hawkes"].values
        cnt_z = zm["count_forecast"].values
        n_z   = zm["Y_count_actual"].values.astype(float)
        valid_z = np.isfinite(p_z) & np.isfinite(y_z.astype(float))
        yp_z, yt_z = p_z[valid_z], y_z[valid_z]
        row_z = {
            "horizon_h": h, "zone": z,
            "n_eval": int(valid_z.sum()),
            "pos_rate": round(float(yt_z.mean()), 4),
            "brier":  round(brier_score_loss(yt_z, yp_z), 4) if len(np.unique(yt_z)) > 1 else None,
            "auc":    round(roc_auc_score(yt_z, yp_z), 4)    if len(np.unique(yt_z)) > 1 else None,
            "logloss": round(log_loss(yt_z, yp_z), 4)        if len(np.unique(yt_z)) > 1 else None,
            "ece":    round(ece(yt_z, yp_z), 4),
            "count_mae":  round(float(np.mean(np.abs(cnt_z[valid_z] - n_z[valid_z]))), 4),
            "count_rmse": round(float(np.sqrt(np.mean((cnt_z[valid_z] - n_z[valid_z])**2))), 4),
            "south_zone_2_note": ("KS p=0.021; widened CI applied" if z == SOUTH_Z2 else ""),
        }
        spatial_rows.append(row_z)

df_metrics  = pd.DataFrame(metrics_rows)
df_spatial  = pd.DataFrame(spatial_rows)
print(f"  Global metrics: {df_metrics.shape}")
print(f"  Spatial metrics: {df_spatial.shape}")

# Print summary
for h in [3, 6, 9]:
    sub = df_metrics[df_metrics["horizon_h"] == h][["model","auc","brier","ece"]].set_index("model")
    print(f"\n  {h}h calibration:")
    print(sub.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RELIABILITY DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: RELIABILITY DIAGRAM ===")

diag_rows = []
bins_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])

for h in [3, 6, 9]:
    ph = pred[pred["horizon_h"] == h].copy()
    y_true = ph["Y_bin_actual"].values.astype(int)

    for model_name, p_vals in [
        ("catboost_calibrated", ph["prob_bin_hawkes"].values),
        ("catboost_raw",        ph["prob_bin_raw"].values),
    ]:
        for b_i in range(N_BINS):
            lo, hi = bins_edges[b_i], bins_edges[b_i+1]
            mask = (p_vals >= lo) & (p_vals < hi)
            if mask.sum() == 0:
                continue
            diag_rows.append({
                "horizon_h": h,
                "model": model_name,
                "bin_center": round(bin_centers[b_i], 3),
                "mean_predicted_prob": round(float(p_vals[mask].mean()), 4),
                "fraction_positive": round(float(y_true[mask].mean()), 4),
                "n_in_bin": int(mask.sum()),
            })

df_diag = pd.DataFrame(diag_rows)
print(f"  Reliability diagram: {df_diag.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SHAP SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: SHAP SUMMARY ===")

shap_rows = []

try:
    from catboost import CatBoostClassifier, Pool as CatPool

    panel_eval_df = panel[panel["split"] == "eval"].copy()
    FEATURES_NUM = [
        "hour_sin", "hour_cos", "dow", "is_weekend", "is_peak",
        "log_lambda_v", "count_1h", "count_4h", "count_24h",
        "burden_1h", "burden_4h",
        "hotspot_n_sig", "hotspot_max_z", "hotspot_mean_burden",
        "zone_mean_fragility", "zone_hist_rate",
    ]
    ALL_FEATURES = FEATURES_NUM + ["zone"]

    for h in [3, 6, 9]:
        bin_col = f"Y_bin_{h}h"
        ev_h = panel_eval_df.dropna(subset=[bin_col]).copy()
        if len(ev_h) == 0:
            continue

        model_path = ARTIFACTS / f"catboost_binary_{h}h.cbm"
        if not model_path.exists():
            print(f"  {h}h: model file not found, skipping SHAP")
            continue

        cb = CatBoostClassifier()
        cb.load_model(str(model_path))

        X_ev_df = ev_h[ALL_FEATURES].fillna(0)
        pool_ev = CatPool(X_ev_df, cat_features=["zone"])

        try:
            shap_vals = cb.get_feature_importance(data=pool_ev, type="ShapValues")
            # shap_vals shape: (n_samples, n_features + 1) — last col is bias
            feature_names = cb.feature_names_
            if feature_names is None:
                feature_names = ALL_FEATURES
            mean_abs_shap = np.abs(shap_vals[:, :-1]).mean(axis=0)
            for fi, fname in enumerate(feature_names):
                shap_rows.append({
                    "horizon_h": h,
                    "feature": fname,
                    "mean_abs_shap": round(float(mean_abs_shap[fi]), 6),
                    "rank": 0,  # filled below
                })
            print(f"  {h}h SHAP computed for {len(feature_names)} features")
        except Exception as e:
            print(f"  {h}h SHAP failed: {e} — using built-in feature importance")
            fi_vals = cb.get_feature_importance()
            fi_names = cb.feature_names_ or ALL_FEATURES
            for fi, fname in enumerate(fi_names):
                shap_rows.append({
                    "horizon_h": h,
                    "feature": fname,
                    "mean_abs_shap": round(float(fi_vals[fi]) / 100.0, 6),
                    "rank": 0,
                })

except ImportError:
    print("  CatBoost not available for SHAP — writing empty placeholder")

# Rank within each horizon
if shap_rows:
    df_shap = pd.DataFrame(shap_rows)
    df_shap["rank"] = df_shap.groupby("horizon_h")["mean_abs_shap"].rank(
        ascending=False, method="min").astype(int)
    df_shap = df_shap.sort_values(["horizon_h", "rank"])
else:
    df_shap = pd.DataFrame(columns=["horizon_h", "feature", "mean_abs_shap", "rank"])

print(f"  SHAP rows: {len(df_shap)}")
if len(df_shap) > 0:
    top3 = df_shap[df_shap["horizon_h"] == 3].nsmallest(3, "rank")
    print(f"  Top 3 features (3h): {top3[['feature','mean_abs_shap']].values.tolist()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: SAVE ===")

df_eri.to_csv(OUTPUTS / "layer7_expected_risk_index.csv", index=False)
print(f"  Wrote layer7_expected_risk_index.csv ({len(df_eri)} rows)")

df_metrics.to_csv(OUTPUTS / "layer7_metrics.csv", index=False)
print(f"  Wrote layer7_metrics.csv ({len(df_metrics)} rows)")

df_spatial.to_csv(OUTPUTS / "layer7_spatial_calibration.csv", index=False)
print(f"  Wrote layer7_spatial_calibration.csv ({len(df_spatial)} rows)")

df_diag.to_csv(OUTPUTS / "layer7_reliability_diagram.csv", index=False)
print(f"  Wrote layer7_reliability_diagram.csv ({len(df_diag)} rows)")

df_shap.to_csv(OUTPUTS / "layer7_shap_summary.csv", index=False)
print(f"  Wrote layer7_shap_summary.csv ({len(df_shap)} rows)")

print("\n=== layer7_risk_index.py complete ===")
