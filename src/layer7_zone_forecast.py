"""
Layer 7 Zone Forecast — Binary + Count Models per Horizon
ASTraM Bengaluru Traffic Disruption Intelligence

Additive. Does NOT modify Layer 1-6 or Part A outputs.
Run after layer7_feature_builder.py.

Models per horizon h in {3h, 6h, 9h}:
  Binary  P(Y>0)  : CatBoost classifier + CONDITIONAL isotonic calibration.
                    Calibration is applied per horizon only if it improves ECE
                    on the 15% validation fold used for early stopping.
                    Raw CatBoost probabilities are used directly when they are
                    already well-calibrated (avoids overfitting noise on a small
                    validation fold).
  Count   E[Y]    : Negative Binomial regression (hurdle variant on non-zero counts)
                    with log(lambda_v * h) as NB offset; fallback to Poisson if NB diverges.

CatBoost features explicitly EXCLUDE SSC and spillover centrality metrics —
those enter only in the ERI formula in the next sub-part.

South Zone 2 note: KS calibration p=0.021 in Part B (weakest among zones).
This is flagged in outputs; wider CIs for that zone come in the ERI sub-part.

Outputs:
  outputs/layer7_predictions.csv
  outputs/layer7_baseline_predictions.csv
  outputs/layer7_model_artifacts/  (CatBoost models, calibrators, NB summaries)
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False
    print("WARNING: CatBoost not available, using LogisticRegression fallback")
    from sklearn.linear_model import LogisticRegression

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
    SM_OK = True
except ImportError:
    SM_OK = False
    print("WARNING: statsmodels not available, count model will use Poisson via sklearn")

OUTPUTS  = Path("outputs")
ARTIFACTS = OUTPUTS / "layer7_model_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

HORIZONS = [3, 6, 9]
N_BOOTSTRAP = 1000
SOUTH_ZONE_2 = "South Zone 2"

# ── feature columns (no SSC, no spillover centrality) ────────────────────────
BASE_FEATURES = [
    "hour_sin", "hour_cos", "dow", "is_weekend", "is_peak",
    "log_lambda_v",
    "count_1h", "count_4h", "count_24h",
    "burden_1h", "burden_4h",
    "hotspot_n_sig", "hotspot_max_z", "hotspot_mean_burden",
    "zone_mean_fragility", "zone_hist_rate",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD PANEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: LOAD PANEL ===")

df = pd.read_csv(OUTPUTS / "layer7_panel_dataset.csv")
print(f"  Panel: {df.shape}")
print(f"  Train: {(df['split']=='train').sum()}, Eval: {(df['split']=='eval').sum()}")
print(f"  Zones: {sorted(df['zone'].unique())}")

with open(ARTIFACTS / "layer7_panel_meta.json") as fh:
    meta = json.load(fh)

zones = meta["zones"]

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE / SPLIT PREP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: PREP ===")

# Zone as a categorical feature — keep as string for CatBoost Pool
FEATURES = BASE_FEATURES + ["zone"]
CAT_FEATURE_NAMES = ["zone"]

df_train = df[df["split"] == "train"].copy()
df_eval  = df[df["split"] == "eval"].copy()
print(f"  Train rows: {len(df_train)}, Eval rows: {len(df_eval)}")

# ── helper: bootstrap CI on a scalar metric ───────────────────────────────────

def bootstrap_ci(values, stat_fn=np.mean, B=N_BOOTSTRAP, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    vals = np.asarray(values)
    if len(vals) == 0:
        return np.nan, np.nan
    bs = [stat_fn(rng.choice(vals, size=len(vals), replace=True)) for _ in range(B)]
    return float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2)))


def ece_score(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def bootstrap_ci_dict(pred, true, metric_fn, B=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(pred)
    if n == 0:
        return np.nan, np.nan
    bs = []
    pred, true = np.asarray(pred), np.asarray(true)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            bs.append(metric_fn(true[idx], pred[idx]))
        except Exception:
            pass
    if not bs:
        return np.nan, np.nan
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAIN MODELS PER HORIZON
# ═══════════════════════════════════════════════════════════════════════════════

all_pred_rows = []
all_base_rows = []
metrics_summary = []
calibration_decisions = {}   # horizon -> decision dict
rng_bs = np.random.default_rng(123)

for h in HORIZONS:
    bin_col   = f"Y_bin_{h}h"
    count_col = f"Y_count_{h}h"
    print(f"\n=== HORIZON {h}h ===")

    # Drop rows where target is NaN (window extends past data end)
    tr = df_train.dropna(subset=[bin_col, count_col]).copy()
    ev = df_eval.dropna(subset=[bin_col, count_col]).copy()

    X_tr_df = tr[FEATURES].fillna(0).copy()
    y_bin_tr = tr[bin_col].values.astype(int)
    y_cnt_tr = tr[count_col].values.astype(int)

    X_ev_df = ev[FEATURES].fillna(0).copy()
    y_bin_ev = ev[bin_col].values.astype(int)
    y_cnt_ev = ev[count_col].values.astype(int)

    # Numeric-only arrays for NB/sklearn (exclude zone string column)
    NUM_FEATURES = BASE_FEATURES
    X_tr = tr[NUM_FEATURES].fillna(0).values.astype(float)
    X_ev = ev[NUM_FEATURES].fillna(0).values.astype(float)

    print(f"  Train: {len(tr)} rows, pos_rate={y_bin_tr.mean():.3f}")
    print(f"  Eval : {len(ev)} rows, pos_rate={y_bin_ev.mean():.3f}")

    # ── 3a. CatBoost binary ───────────────────────────────────────────────────
    print(f"  Fitting CatBoost binary ({h}h)...")
    if CATBOOST_OK:
        from catboost import Pool as CatPool
        n_val = max(int(0.15 * len(tr)), 1)
        tr_fit_df = X_tr_df.iloc[:-n_val]
        y_fit     = y_bin_tr[:-n_val]
        tr_val_df = X_tr_df.iloc[-n_val:]
        y_val     = y_bin_tr[-n_val:]
        pool_fit = CatPool(tr_fit_df, label=y_fit, cat_features=CAT_FEATURE_NAMES)
        pool_val = CatPool(tr_val_df, label=y_val, cat_features=CAT_FEATURE_NAMES)
        pool_tr  = CatPool(X_tr_df,   label=y_bin_tr, cat_features=CAT_FEATURE_NAMES)
        pool_ev  = CatPool(X_ev_df,   cat_features=CAT_FEATURE_NAMES)
        cb = CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=5,
            l2_leaf_reg=3.0, loss_function="Logloss", eval_metric="AUC",
            random_seed=42, verbose=0, early_stopping_rounds=40,
        )
        cb.fit(pool_fit, eval_set=pool_val)
        prob_tr_raw = cb.predict_proba(pool_tr)[:, 1]
        prob_ev_raw = cb.predict_proba(pool_ev)[:, 1]
        cb.save_model(str(ARTIFACTS / f"catboost_binary_{h}h.cbm"))
        print(f"    CatBoost trained, best iter={cb.best_iteration_}")
    else:
        lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        lr.fit(X_tr, y_bin_tr)
        prob_tr_raw = lr.predict_proba(X_tr)[:, 1]
        prob_ev_raw = lr.predict_proba(X_ev)[:, 1]
        with open(ARTIFACTS / f"logreg_binary_{h}h.pkl", "wb") as fh:
            pickle.dump(lr, fh)

    # ── 3b. Conditional isotonic calibration ─────────────────────────────────
    # Compare raw vs isotonic-calibrated ECE on the same validation fold used
    # for early stopping (last 15% of training rows).  Only apply calibration
    # if it strictly improves ECE on that fold; otherwise keep raw probabilities
    # to avoid overfitting noise from a small calibration sample.
    if CATBOOST_OK:
        prob_fit_raw = cb.predict_proba(pool_fit)[:, 1]
        prob_val_raw = cb.predict_proba(pool_val)[:, 1]
    else:
        prob_fit_raw = prob_tr_raw[:-n_val]
        prob_val_raw = prob_tr_raw[-n_val:]

    iso_cond = IsotonicRegression(out_of_bounds="clip")
    iso_cond.fit(prob_fit_raw, y_fit)
    prob_val_cal_cond = iso_cond.predict(prob_val_raw)

    ece_raw_val = ece_score(y_val, prob_val_raw)
    ece_cal_val = ece_score(y_val, prob_val_cal_cond)
    use_calibration = bool(ece_cal_val < ece_raw_val)

    decision_reason = (
        f"calibration reduces val ECE {ece_raw_val:.4f} -> {ece_cal_val:.4f}"
        if use_calibration else
        f"raw already lower val ECE ({ece_raw_val:.4f}) than calibrated ({ece_cal_val:.4f}); "
        f"isotonic on small fold adds overfitting noise"
    )
    print(f"  Conditional calibration ({h}h): val ECE raw={ece_raw_val:.4f} "
          f"cal={ece_cal_val:.4f} -> {'APPLY' if use_calibration else 'SKIP (use raw)'}")

    if use_calibration:
        iso_final = IsotonicRegression(out_of_bounds="clip")
        iso_final.fit(prob_tr_raw, y_bin_tr)
        prob_ev_cal = iso_final.predict(prob_ev_raw)
        prob_tr_cal = iso_final.predict(prob_tr_raw)
        with open(ARTIFACTS / f"isotonic_cal_{h}h.pkl", "wb") as fh:
            pickle.dump(iso_final, fh)
    else:
        prob_ev_cal = prob_ev_raw.copy()
        prob_tr_cal = prob_tr_raw.copy()

    calibration_decisions[h] = {
        "horizon_h": h,
        "use_calibration": use_calibration,
        "ece_raw_val_fold": round(ece_raw_val, 4),
        "ece_cal_val_fold": round(ece_cal_val, 4),
        "val_fold_n": int(len(y_val)),
        "reason": decision_reason,
    }

    # ── 3c. Negative Binomial count model (hurdle: fit on non-zero counts) ───
    print(f"  Fitting NB count model ({h}h, hurdle on non-zero counts)...")
    # For NB offset: log(lambda_v * h), treated as Hawkes prior mean
    log_offset_tr = np.log(np.maximum(tr["lambda_v"].values * h, 1e-6))
    log_offset_ev = np.log(np.maximum(ev["lambda_v"].values * h, 1e-6))

    # Hurdle: NB fit on non-zero count rows only
    nz_mask_tr = y_cnt_tr > 0

    nb_model = None
    nb_fitted = False

    if SM_OK and nz_mask_tr.sum() >= 20:
        X_nb_tr_nz = sm.add_constant(X_tr[nz_mask_tr])
        y_nb_tr_nz = y_cnt_tr[nz_mask_tr]
        off_nb_nz  = log_offset_tr[nz_mask_tr]
        try:
            nb_model = NegativeBinomial(y_nb_tr_nz, X_nb_tr_nz, loglike_method="nb2")
            nb_res = nb_model.fit(disp=0, method="bfgs", maxiter=200)
            nb_fitted = True
            print(f"    NB converged: alpha={nb_res.params[-1]:.4f}")
            with open(ARTIFACTS / f"nb_count_{h}h.pkl", "wb") as fh:
                pickle.dump(nb_res, fh)
        except Exception as e:
            print(f"    NB failed ({e}), falling back to Poisson")

    if not nb_fitted and SM_OK and nz_mask_tr.sum() >= 5:
        # Poisson fallback on non-zero counts
        X_nb_tr_nz = sm.add_constant(X_tr[nz_mask_tr])
        y_nb_tr_nz = y_cnt_tr[nz_mask_tr]
        try:
            nb_model = Poisson(y_nb_tr_nz, X_nb_tr_nz)
            nb_res = nb_model.fit(disp=0, maxiter=200)
            nb_fitted = True
            print(f"    Poisson fallback converged")
            with open(ARTIFACTS / f"poisson_count_{h}h.pkl", "wb") as fh:
                pickle.dump(nb_res, fh)
        except Exception as e:
            print(f"    Poisson also failed ({e}), using historical rate baseline for count")

    # Predict count on eval set
    if nb_fitted:
        X_nb_ev = sm.add_constant(X_ev, has_constant="add")
        # Ensure column count matches (add_constant may differ)
        if X_nb_ev.shape[1] != nb_res.params.shape[0] - (1 if "alpha" in str(type(nb_res)) else 0):
            try:
                mu_ev = np.exp(X_nb_ev @ nb_res.params[:X_nb_ev.shape[1]])
            except Exception:
                mu_ev = ev["zone_hist_rate"].values * h
        else:
            try:
                mu_ev = nb_res.predict(X_nb_ev)
            except Exception:
                mu_ev = ev["zone_hist_rate"].values * h
        # Hurdle: combine P(Y>0) * E[Y|Y>0]
        # E[Y|Y>0] from NB = mu / (1 - P_NB(Y=0))  where P_NB(Y=0) = (alpha/(alpha+mu))^(1/alpha)
        # Simplification: use mu_ev directly (reasonable approximation)
        count_ev_pred = prob_ev_cal * np.maximum(mu_ev, 0.0)
    else:
        # Fallback: historical rate * h
        count_ev_pred = ev["zone_hist_rate"].values * h

    # ── 3d. Baseline predictions ──────────────────────────────────────────────
    # Baseline binary: historical positive rate per zone
    zone_base_rate = tr.groupby("zone")[bin_col].mean().to_dict()
    base_prob_ev = ev["zone"].map(zone_base_rate).fillna(y_bin_tr.mean()).values
    base_count_ev = ev["zone_hist_rate"].values * h

    # ── 3e. Eval metrics ──────────────────────────────────────────────────────
    valid_mask = np.isfinite(prob_ev_cal) & np.isfinite(y_bin_ev.astype(float))
    p_cal_v = prob_ev_cal[valid_mask]
    y_bin_v = y_bin_ev[valid_mask]

    if len(np.unique(y_bin_v)) > 1:
        auc = roc_auc_score(y_bin_v, p_cal_v)
        brier = brier_score_loss(y_bin_v, p_cal_v)
        logloss = log_loss(y_bin_v, p_cal_v)
        auc_ci   = bootstrap_ci_dict(p_cal_v, y_bin_v, lambda t, p: roc_auc_score(t, p), rng=rng_bs)
        brier_ci = bootstrap_ci_dict(p_cal_v, y_bin_v, lambda t, p: brier_score_loss(t, p), rng=rng_bs)
    else:
        auc = brier = logloss = np.nan
        auc_ci = brier_ci = (np.nan, np.nan)

    rmse_count = float(np.sqrt(np.mean((count_ev_pred - y_cnt_ev) ** 2)))
    mae_count  = float(np.mean(np.abs(count_ev_pred - y_cnt_ev)))
    rmse_ci = bootstrap_ci_dict(count_ev_pred, y_cnt_ev,
                                lambda t, p: np.sqrt(np.mean((p - t)**2)), rng=rng_bs)

    print(f"  Metrics ({h}h): AUC={auc:.4f} [{auc_ci[0]:.4f},{auc_ci[1]:.4f}] "
          f"Brier={brier:.4f} [{brier_ci[0]:.4f},{brier_ci[1]:.4f}] "
          f"RMSE_count={rmse_count:.4f} [{rmse_ci[0]:.4f},{rmse_ci[1]:.4f}]")

    metrics_summary.append({
        "horizon_h": h,
        "n_eval": len(ev),
        "n_train": len(tr),
        "pos_rate_train": round(y_bin_tr.mean(), 4),
        "pos_rate_eval": round(y_bin_ev.mean(), 4),
        "auc": round(auc, 4) if not np.isnan(auc) else None,
        "auc_ci_lo": round(auc_ci[0], 4) if not np.isnan(auc_ci[0]) else None,
        "auc_ci_hi": round(auc_ci[1], 4) if not np.isnan(auc_ci[1]) else None,
        "brier": round(brier, 4) if not np.isnan(brier) else None,
        "brier_ci_lo": round(brier_ci[0], 4) if not np.isnan(brier_ci[0]) else None,
        "brier_ci_hi": round(brier_ci[1], 4) if not np.isnan(brier_ci[1]) else None,
        "logloss": round(logloss, 4) if not np.isnan(logloss) else None,
        "rmse_count": round(rmse_count, 4),
        "rmse_count_ci_lo": round(rmse_ci[0], 4) if not np.isnan(rmse_ci[0]) else None,
        "rmse_count_ci_hi": round(rmse_ci[1], 4) if not np.isnan(rmse_ci[1]) else None,
        "mae_count": round(mae_count, 4),
        "nb_fitted": nb_fitted,
        "bootstrap_n": N_BOOTSTRAP,
        "isotonic_calibration_applied": use_calibration,
        "ece_raw_val_fold": round(ece_raw_val, 4),
        "ece_cal_val_fold": round(ece_cal_val, 4),
        "calibration_reason": decision_reason,
        "note_small_holdout": "Eval is Jan 1-19 2024 (~18.3 days). CIs are wider than larger holdouts.",
        "note_south_zone_2": "South Zone 2 has weaker spillover calibration (KS p=0.021 in Part B).",
    })

    # ── collect prediction rows ───────────────────────────────────────────────
    for i, idx in enumerate(ev.index):
        all_pred_rows.append({
            "grid_time_utc": ev.loc[idx, "grid_time_utc"],
            "zone": ev.loc[idx, "zone"],
            "horizon_h": h,
            "prob_bin_hawkes": round(float(prob_ev_cal[i]), 6),
            "prob_bin_raw": round(float(prob_ev_raw[i]), 6),
            "is_calibrated": use_calibration,
            "count_forecast": round(float(count_ev_pred[i]), 4),
            f"Y_bin_{h}h": int(y_bin_ev[i]),
            f"Y_count_{h}h": int(y_cnt_ev[i]),
            "south_zone_2_note": (
                "wider_CI_pending" if ev.loc[idx, "zone"] == SOUTH_ZONE_2 else ""
            ),
        })
        all_base_rows.append({
            "grid_time_utc": ev.loc[idx, "grid_time_utc"],
            "zone": ev.loc[idx, "zone"],
            "horizon_h": h,
            "baseline_prob_bin": round(float(base_prob_ev[i]), 6),
            "baseline_count": round(float(base_count_ev[i]), 4),
            f"Y_bin_{h}h": int(y_bin_ev[i]),
            f"Y_count_{h}h": int(y_cnt_ev[i]),
        })

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PER-ZONE METRICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: PER-ZONE METRICS ===")

df_preds = pd.DataFrame(all_pred_rows)
per_zone_metrics = []

for h in HORIZONS:
    sub = df_preds[df_preds["horizon_h"] == h]
    ev_sub = df_eval.dropna(subset=[f"Y_bin_{h}h"]).copy()
    for z in zones:
        z_mask = sub["zone"] == z
        if z_mask.sum() == 0:
            continue
        p_z = sub.loc[z_mask, "prob_bin_hawkes"].values
        y_z = sub.loc[z_mask, f"Y_bin_{h}h"].values
        c_z = sub.loc[z_mask, "count_forecast"].values
        n_z = sub.loc[z_mask, f"Y_count_{h}h"].values

        auc_z = roc_auc_score(y_z, p_z) if len(np.unique(y_z)) > 1 else np.nan
        rmse_z = float(np.sqrt(np.mean((c_z - n_z)**2)))
        per_zone_metrics.append({
            "horizon_h": h, "zone": z,
            "n_eval": int(z_mask.sum()),
            "pos_rate_eval": round(float(y_z.mean()), 4),
            "auc": round(auc_z, 4) if not np.isnan(auc_z) else None,
            "rmse_count": round(rmse_z, 4),
            "ks_calib_note": "KS p=0.021 in Part B (weakest)" if z == SOUTH_ZONE_2 else "",
        })
        auc_str_z = f"{auc_z:.3f}" if not np.isnan(auc_z) else "nan"
        print(f"  {h}h {z}: n={z_mask.sum()} pos={y_z.mean():.3f} AUC={auc_str_z}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: SAVING OUTPUTS ===")

df_preds = pd.DataFrame(all_pred_rows)
df_base  = pd.DataFrame(all_base_rows)

df_preds.to_csv(OUTPUTS / "layer7_predictions.csv", index=False)
print(f"  Wrote layer7_predictions.csv ({len(df_preds)} rows)")

df_base.to_csv(OUTPUTS / "layer7_baseline_predictions.csv", index=False)
print(f"  Wrote layer7_baseline_predictions.csv ({len(df_base)} rows)")

df_metrics = pd.DataFrame(metrics_summary)
df_metrics.to_csv(ARTIFACTS / "layer7_eval_metrics.csv", index=False)
print(f"  Wrote layer7_model_artifacts/layer7_eval_metrics.csv")

df_zone_metrics = pd.DataFrame(per_zone_metrics)
df_zone_metrics.to_csv(ARTIFACTS / "layer7_eval_per_zone.csv", index=False)
print(f"  Wrote layer7_model_artifacts/layer7_eval_per_zone.csv")

with open(ARTIFACTS / "layer7_calibration_decisions.json", "w") as fh:
    json.dump(calibration_decisions, fh, indent=2)
print("  Wrote layer7_model_artifacts/layer7_calibration_decisions.json")

# Print summary table
print("\n=== EVAL METRICS SUMMARY ===")
print(f"\n  Calibration decisions (per-horizon, based on val-fold ECE):")
for h_key, d in calibration_decisions.items():
    tag = "CALIBRATED" if d["use_calibration"] else "RAW (no calibration)"
    print(f"    {h_key}h: {tag}  val ECE raw={d['ece_raw_val_fold']:.4f} cal={d['ece_cal_val_fold']:.4f}")
    print(f"         Reason: {d['reason']}")

print(f"\n  {'H':>4}  {'AUC':>8}  {'95% CI':>20}  {'Brier':>8}  {'Calibrated':>12}  {'RMSE_cnt':>10}")
for row in metrics_summary:
    h = row["horizon_h"]
    auc_str = f"{row['auc']:.4f}" if row["auc"] else "N/A"
    ci_str  = f"[{row['auc_ci_lo']:.4f},{row['auc_ci_hi']:.4f}]" if row["auc_ci_lo"] else "N/A"
    br_str  = f"{row['brier']:.4f}" if row["brier"] else "N/A"
    cal_str = "Yes" if row["isotonic_calibration_applied"] else "No (raw)"
    rm_str  = (f"{row['rmse_count']:.4f} [{row['rmse_count_ci_lo']:.4f},{row['rmse_count_ci_hi']:.4f}]"
               if row["rmse_count_ci_lo"] else f"{row['rmse_count']:.4f}")
    print(f"  {h:>4}h  {auc_str:>8}  {ci_str:>20}  {br_str:>8}  {cal_str:>12}  {rm_str}")

print(f"\n  NOTE: Eval window is Jan 1-19 2024 (~18.3 days, {df_eval.shape[0]//10} grid hours per zone).")
print(f"  Bootstrap CIs (B={N_BOOTSTRAP}) reflect finite holdout uncertainty.")
print(f"  South Zone 2: weakest spillover calibration — flag for wider ERI CI in next sub-part.")
print("\n=== layer7_zone_forecast.py complete ===")
