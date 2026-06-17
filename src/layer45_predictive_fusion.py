"""
Layer 4.5 — Predictive Decision Intelligence Engine (ASTraM).

Fuses Layer 1–4 intelligence via leak-free as-of surrogate features.
Additive only — does not modify Layers 1–4.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from layer45_asof_features import (
    FittedParams,
    build_asof_feature_matrix,
)
from layer45_feature_registry import export_feature_registry
from layer45_time_split import build_time_split, split_summary

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "events_clean.parquet"
OUT = ROOT / "outputs"
ARTIFACTS = OUT / "layer45_model_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

CAT_PARAMS = dict(
    iterations=500,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=5,
    random_strength=2,
    bagging_temperature=1,
    early_stopping_rounds=50,
    verbose=0,
    random_seed=42,
)

NUM_FEATURES = [
    "asof_p50_duration", "asof_p80_duration", "asof_p95_duration",
    "asof_surv_prob_60", "asof_surv_prob_120", "asof_surv_prob_180",
    "asof_rmst_180", "asof_corridor_burden", "asof_junction_burden",
    "asof_obi_proxy", "asof_corridor_event_rate", "asof_nbr_mean_burden",
    "asof_nbr_mean_duration", "asof_fragility_proxy", "asof_hawkes_intensity",
    "asof_burstiness", "asof_branching_ratio_proxy", "asof_retrieval_confidence",
    "asof_retrieval_n_eff", "asof_retrieval_mean_sim", "asof_retrieval_max_sim",
    "asof_planned_support", "asof_ims_proxy",
    "obi_x_fragility", "retrieval_x_risk", "fragility_x_duration",
    "trust_x_confidence", "hotspot_x_duration", "hawkes_x_obi",
    "nbr_mean_obi", "nbr_mean_fragility", "nbr_mean_duration", "nbr_mean_severity",
    "hour_local", "dow_local", "month", "is_weekend", "trust_score",
    "geo_valid", "duration_anomaly", "iso_flagged", "mnar_predicted_prob",
    "requires_road_closure",
]

CAT_FEATURES = [
    "event_cause", "corridor", "zone", "junction", "priority", "event_type",
    "asof_quantile_fallback_level", "asof_retrieval_fallback",
]

JOSV_NORM_COLS = [
    "duration_pred", "duration_p50", "duration_p80", "duration_p95",
    "duration_ci_lower", "duration_ci_upper",
    "high_impact_prob", "high_impact_prob_calibrated",
    "retrieval_confidence", "novelty_score", "drift_score",
    "trust_score", "fragility_signal", "obi_signal", "ims_proxy",
]


def _log_duration(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


def _exp_duration(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y_log, dtype=float))


def _load_events() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_parquet(DATA)
    return pd.read_csv(ROOT / "data" / "events_clean.csv")


def _training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Resolved events with positive duration for supervised training."""
    m = df["duration_min"].notna() & (df["duration_min"] > 0)
    if "is_censored" in df.columns:
        m &= ~df["is_censored"].astype(bool)
    return df[m].copy()


def _prepare_xy(
    feat_df: pd.DataFrame,
    cause_tau: dict[str, float],
    global_tau: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[int]]:
    X = feat_df.copy()
    for c in NUM_FEATURES:
        if c not in X.columns:
            X[c] = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in CAT_FEATURES:
        if c not in X.columns:
            X[c] = "missing"
        X[c] = X[c].astype(str).fillna("missing")

    feature_cols = NUM_FEATURES + CAT_FEATURES
    y_dur = X["duration_min"].astype(float)
    causes = X["event_cause"].astype(str)
    tau_vec = causes.map(lambda c: cause_tau.get(c, global_tau)).astype(float)
    y_hi = (y_dur > tau_vec).astype(int)
    X_model = X[feature_cols]
    cat_idx = [i for i, c in enumerate(feature_cols) if c in CAT_FEATURES]
    return X_model, y_dur, y_hi, feature_cols, cat_idx


def train_duration_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_idx: list[int],
    loss: str = "RMSE",
):
    if HAS_CATBOOST:
        model = CatBoostRegressor(loss_function=loss, **CAT_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_idx,
        )
        return model
    m = GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42)
    m.fit(X_train, y_train)
    return m


def train_high_impact_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_idx: list[int],
):
    if HAS_CATBOOST:
        model = CatBoostClassifier(loss_function="Logloss", **CAT_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_idx,
        )
        return model
    m = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    m.fit(X_train, y_train)
    return m


def calibrate_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[IsotonicRegression, np.ndarray]:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    calibrated = iso.predict(y_prob)
    return iso, calibrated


def compute_conformal_intervals(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    residuals = np.abs(y_train - pred_train)
    q = float(np.quantile(residuals, 1 - alpha))
    lower = pred_val - q
    upper = pred_val + q
    return lower, upper, q


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        rows.append({
            "bin_lo": bins[i],
            "bin_hi": bins[i + 1],
            "mean_pred": float(y_prob[mask].mean()) if mask.sum() else np.nan,
            "mean_true": float(y_true[mask].mean()) if mask.sum() else np.nan,
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def compute_novelty_drift(
    X_train: pd.DataFrame,
    X_all: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    num_cols = [c for c in X_train.columns if c not in CAT_FEATURES]
    Xn = X_train[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    Xa = X_all[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(Xn)
    scores = -iso.decision_function(Xa)
    flags = iso.predict(Xa) == -1

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Xn)
    Z_all = scaler.transform(Xa)
    cov = np.cov(Z_train.T) + np.eye(Z_train.shape[1]) * 1e-6
    inv = np.linalg.inv(cov)
    mean = Z_train.mean(axis=0)
    drift = np.array([float((z - mean) @ inv @ (z - mean)) for z in Z_all])
    drift_thresh = float(np.quantile(drift[: len(X_train)], 0.95))

    return pd.DataFrame({
        "novelty_score": scores,
        "novelty_flag": flags,
        "drift_score": drift,
        "drift_flag": drift > drift_thresh,
    })


def compute_shap_values(
    model,
    X: pd.DataFrame,
    cat_idx: list[int],
    sample_n: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_SHAP or not HAS_CATBOOST:
        empty = pd.DataFrame({"note": ["SHAP skipped — catboost/shap unavailable"]})
        return empty, empty

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    Xs = X.iloc[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    mean_abs = np.abs(sv).mean(axis=0)
    summary = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    local_rows = []
    for i, row_idx in enumerate(idx):
        top_j = int(np.argmax(np.abs(sv[i])))
        local_rows.append({
            "row_index": int(row_idx),
            "top_feature": X.columns[top_j],
            "top_shap": float(sv[i][top_j]),
            "prediction_driver_rank": 1,
        })
    local = pd.DataFrame(local_rows)
    return summary, local


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(np.median(np.abs(y_true - y_pred)))
    log_true = np.log1p(np.maximum(y_true, 0))
    log_pred = np.log1p(np.maximum(y_pred, 0))
    rmsle = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return {
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "rmsle": rmsle,
        "mape": mape,
    }


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def _fallback_summary(fallback_log: list[dict]) -> pd.DataFrame:
    fb = pd.DataFrame(fallback_log)
    rows = []
    rows.append({"slice": "overall", "metric": "global_fallback_rate",
                 "value": float((fb["quantile_fallback"] == "global").mean())})
    rows.append({"slice": "overall", "metric": "cause_only_fallback_rate",
                 "value": float((fb["quantile_fallback"] == "cause_only").mean())})
    rows.append({"slice": "overall", "metric": "exact_support_rate",
                 "value": float((fb["quantile_fallback"] == "cause_corridor").mean())})
    rows.append({"slice": "overall", "metric": "low_retrieval_support_rate",
                 "value": float((fb["retrieval_fallback"] == "low_support").mean())})
    for cause, g in fb.groupby("event_cause"):
        rows.append({"slice": f"cause:{cause}", "metric": "global_fallback_rate",
                     "value": float((g["quantile_fallback"] == "global").mean())})
    for month, g in fb.groupby(fb["snapshot_date"].str[:7]):
        rows.append({"slice": f"month:{month}", "metric": "global_fallback_rate",
                     "value": float((g["quantile_fallback"] == "global").mean())})
    return pd.DataFrame(rows)


def _build_josv(
    feat_df: pd.DataFrame,
    dur_pred: np.ndarray,
    dur_p50: np.ndarray,
    dur_p80: np.ndarray,
    dur_p95: np.ndarray,
    hi_prob: np.ndarray,
    hi_prob_cal: np.ndarray,
    novelty_df: pd.DataFrame,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    cause_tau: dict[str, float],
    global_tau: float,
) -> pd.DataFrame:
    causes = feat_df["event_cause"].astype(str)
    tau_applied = causes.map(lambda c: cause_tau.get(c, global_tau)).astype(float)
    return pd.DataFrame({
        "event_id": feat_df["event_id"].values,
        "start_local": feat_df["start_local"].values,
        "event_cause": causes.values,
        "high_impact_tau_c": tau_applied.values,
        "duration_pred": dur_pred,
        "duration_p50": dur_p50,
        "duration_p80": dur_p80,
        "duration_p95": dur_p95,
        "duration_ci_lower": ci_lower,
        "duration_ci_upper": ci_upper,
        "high_impact_prob": hi_prob,
        "high_impact_prob_calibrated": hi_prob_cal,
        "retrieval_confidence": feat_df["asof_retrieval_confidence"].values,
        "novelty_score": novelty_df["novelty_score"].values,
        "novelty_flag": novelty_df["novelty_flag"].values,
        "drift_score": novelty_df["drift_score"].values,
        "drift_flag": novelty_df["drift_flag"].values,
        "trust_score": feat_df.get("trust_score", 0.5),
        "fragility_signal": feat_df["asof_fragility_proxy"].values,
        "obi_signal": feat_df["asof_obi_proxy"].values,
        "ims_proxy": feat_df["asof_ims_proxy"].values,
    })


def fit_josv_scaler(josv_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Robust location/scale (median/MAD) fit on training JOSV rows only."""
    stats: dict[str, dict[str, float]] = {}
    for col in JOSV_NORM_COLS:
        if col not in josv_train.columns:
            continue
        x = pd.to_numeric(josv_train[col], errors="coerce").dropna()
        if x.empty:
            continue
        med = float(x.median())
        mad = float((x - med).abs().median())
        if mad < 1e-6:
            mad = float(x.std()) or 1.0
        stats[col] = {"median": med, "mad": mad}
    return stats


def normalize_josv(josv: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Return JOSV with robust z-scores alongside raw values."""
    out = josv.copy()
    for col, st in stats.items():
        if col in out.columns:
            out[f"{col}_z"] = (out[col].astype(float) - st["median"]) / st["mad"]
    return out


def export_outputs(
    feat_df: pd.DataFrame,
    josv: pd.DataFrame,
    josv_normalized: pd.DataFrame,
    metrics: pd.DataFrame,
    shap_summary: pd.DataFrame,
    shap_local: pd.DataFrame,
    novelty_df: pd.DataFrame,
    fallback_summary: pd.DataFrame,
    calibration_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    summary_text: str,
    cause_tau_df: pd.DataFrame | None = None,
    out_dir: Path = OUT,
) -> None:
    out_dir.mkdir(exist_ok=True)
    feat_df.to_csv(out_dir / "layer45_asof_feature_matrix.csv", index=False)
    josv[["event_id", "duration_pred", "duration_p50", "duration_p80", "duration_p95"]].to_csv(
        out_dir / "layer45_duration_predictions.csv", index=False)
    josv[["event_id", "high_impact_prob", "high_impact_prob_calibrated"]].to_csv(
        out_dir / "layer45_high_impact_probabilities.csv", index=False)
    josv.to_csv(out_dir / "layer45_operational_state_vector.csv", index=False)
    josv_normalized.to_csv(out_dir / "layer45_operational_state_vector_normalized.csv", index=False)
    if cause_tau_df is not None:
        cause_tau_df.to_csv(out_dir / "layer45_cause_tau_thresholds.csv", index=False)
    conformal_df.to_csv(out_dir / "layer45_conformal_intervals.csv", index=False)
    calibration_df.to_csv(out_dir / "layer45_calibration.csv", index=False)
    metrics.to_csv(out_dir / "layer45_metrics.csv", index=False)
    shap_summary.to_csv(out_dir / "layer45_shap_summary.csv", index=False)
    shap_local.to_csv(out_dir / "layer45_shap_local.csv", index=False)
    novelty_df.to_csv(out_dir / "layer45_novelty_drift.csv", index=False)
    importance_df.to_csv(out_dir / "layer45_feature_importance.csv", index=False)
    fallback_summary.to_csv(out_dir / "layer45_fallback_summary.csv", index=False)
    coverage_df.to_csv(out_dir / "layer45_coverage_summary.csv", index=False)
    with open(out_dir / "layer45_fusion_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    export_feature_registry(out_dir)


def run_backtest() -> dict:
    """Backtest mode: train Nov 2023–Feb 2024, validate Mar–Apr 2024."""
    logger.info("Layer 4.5 backtest mode")
    raw = _load_events()
    split = build_time_split(raw)
    logger.info("Split: %s", split_summary(raw, split))

    feat_df, params, fallback_log = build_asof_feature_matrix(raw, split.train_mask)
    t = pd.to_datetime(feat_df["start_local"], errors="coerce")
    train_mask = t <= split.train_end
    val_mask = t >= split.val_start

    train_df = _training_rows(feat_df[train_mask])
    val_df = _training_rows(feat_df[val_mask])
    global_tau = params.high_impact_tau
    cause_tau = params.cause_tau

    X_train, y_train_dur, y_train_hi, cols, cat_idx = _prepare_xy(train_df, cause_tau, global_tau)
    X_val, y_val_dur, y_val_hi, _, _ = _prepare_xy(val_df, cause_tau, global_tau)

    y_train_log = pd.Series(_log_duration(y_train_dur), index=y_train_dur.index)
    y_val_log = pd.Series(_log_duration(y_val_dur), index=y_val_dur.index)

    # Duration models — train on log1p(duration), inverse-transform at predict time
    dur_model = train_duration_model(X_train, y_train_log, X_val, y_val_log, cat_idx, "Huber:delta=1")
    q50_model = train_duration_model(X_train, y_train_log, X_val, y_val_log, cat_idx, "Quantile:alpha=0.5")
    q80_model = train_duration_model(X_train, y_train_log, X_val, y_val_log, cat_idx, "Quantile:alpha=0.8")
    q95_model = train_duration_model(X_train, y_train_log, X_val, y_val_log, cat_idx, "Quantile:alpha=0.95")

    hi_model = train_high_impact_model(X_train, y_train_hi, X_val, y_val_hi, cat_idx)

    resolved_feat = _training_rows(feat_df)
    X_all, y_all_dur, _, _, _ = _prepare_xy(resolved_feat, cause_tau, global_tau)
    t_res = pd.to_datetime(resolved_feat["start_local"], errors="coerce")
    resolved_val_mask = (t_res >= split.val_start).values
    resolved_train_mask = (t_res <= split.train_end).values

    pred_train_log = dur_model.predict(X_train)
    pred_val_log = dur_model.predict(X_val)
    pred_train = _exp_duration(pred_train_log)
    pred_val = _exp_duration(pred_val_log)

    ci_lo_log, ci_hi_log, q_band_log = compute_conformal_intervals(
        y_train_log.values, pred_train_log, pred_val_log, alpha=0.1
    )
    ci_lo = _exp_duration(ci_lo_log)
    ci_hi = _exp_duration(ci_hi_log)

    hi_prob_val = hi_model.predict_proba(X_val)[:, 1] if hasattr(hi_model, "predict_proba") else hi_model.predict(X_val)
    iso, hi_cal_val = calibrate_classifier(y_val_hi.values, hi_prob_val)

    novelty_all = compute_novelty_drift(X_train, X_all)
    novelty_all.insert(0, "event_id", resolved_feat["event_id"].values)
    novelty_val = novelty_all.loc[resolved_val_mask].reset_index(drop=True)

    shap_sum, shap_loc = compute_shap_values(dur_model, X_train, cat_idx)

    if HAS_CATBOOST and hasattr(dur_model, "get_feature_importance"):
        imp = pd.DataFrame({
            "feature": cols,
            "importance": dur_model.get_feature_importance(),
        }).sort_values("importance", ascending=False)
    else:
        imp = pd.DataFrame({"feature": cols, "importance": 0})

    # Metrics (original-minute scale + RMSLE)
    reg_m = _regression_metrics(y_val_dur.values, pred_val)
    cls_m = _classification_metrics(y_val_hi.values, hi_cal_val)
    cls_m["ece"] = _ece(y_val_hi.values, hi_cal_val)
    coverage = float(np.mean((y_val_dur.values >= ci_lo) & (y_val_dur.values <= ci_hi)))
    interval_width = float(np.mean(ci_hi - ci_lo))

    metrics_rows = []
    for k, v in reg_m.items():
        metrics_rows.append({"subset": "holdout", "task": "duration", "metric": k, "value": v})
    for k, v in cls_m.items():
        metrics_rows.append({"subset": "holdout", "task": "high_impact", "metric": k, "value": v})
    metrics_rows.append({"subset": "holdout", "task": "conformal", "metric": "coverage_90", "value": coverage})
    metrics_rows.append({"subset": "holdout", "task": "conformal", "metric": "mean_interval_width", "value": interval_width})
    metrics_rows.append({"subset": "holdout", "task": "novelty", "metric": "novelty_rate", "value": float(novelty_val["novelty_flag"].mean())})
    metrics_rows.append({"subset": "holdout", "task": "novelty", "metric": "drift_rate", "value": float(novelty_val["drift_flag"].mean())})

    # Typical-incident slice (duration <= 500 min)
    typical = y_val_dur.values <= 500
    if typical.sum() > 0:
        for k, v in _regression_metrics(y_val_dur.values[typical], pred_val[typical]).items():
            metrics_rows.append({"subset": "holdout_typical_le500", "task": "duration", "metric": k, "value": v})

    if "is_true_planned_event" in val_df.columns:
        pe = val_df["is_true_planned_event"].astype(bool).values
        if pe.sum() > 0:
            for k, v in _regression_metrics(y_val_dur.values[pe], pred_val[pe]).items():
                metrics_rows.append({"subset": "holdout_planned", "task": "duration", "metric": k, "value": v})
            for k, v in _classification_metrics(y_val_hi.values[pe], hi_cal_val[pe]).items():
                metrics_rows.append({"subset": "holdout_planned", "task": "high_impact", "metric": k, "value": v})

    metrics_df = pd.DataFrame(metrics_rows)
    cal_df = _reliability_curve(y_val_hi.values, hi_cal_val)
    cal_df["ece"] = cls_m["ece"]
    cal_df["brier"] = cls_m["brier"]

    conformal_df = pd.DataFrame({
        "event_id": val_df["event_id"].values,
        "duration_pred": pred_val,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "conformal_q_log": q_band_log,
    })

    hi_prob_all = hi_model.predict_proba(X_all)[:, 1] if hasattr(hi_model, "predict_proba") else hi_model.predict(X_all)
    hi_cal_all = iso.predict(hi_prob_all)

    pred_all_log = dur_model.predict(X_all)
    pred_all = _exp_duration(pred_all_log)
    p50_all = _exp_duration(q50_model.predict(X_all))
    p80_all = _exp_duration(q80_model.predict(X_all))
    p95_all = _exp_duration(q95_model.predict(X_all))

    resid_q_log = float(np.quantile(np.abs(y_train_log.values - pred_train_log), 0.9))
    ci_lo_all = _exp_duration(pred_all_log - resid_q_log)
    ci_hi_all = _exp_duration(pred_all_log + resid_q_log)

    josv = _build_josv(
        resolved_feat, pred_all, p50_all, p80_all, p95_all,
        hi_prob_all, hi_cal_all, novelty_all, ci_lo_all, ci_hi_all,
        cause_tau, global_tau,
    )
    josv_train = josv[resolved_train_mask].copy()
    josv_scaler = fit_josv_scaler(josv_train)
    josv_normalized = normalize_josv(josv, josv_scaler)

    cause_tau_df = pd.DataFrame([
        {"event_cause": c, "tau_p75_min": t, "global_fallback_tau": global_tau}
        for c, t in sorted(cause_tau.items())
    ])

    fb_summary = _fallback_summary(fallback_log)
    coverage_df = pd.DataFrame(fallback_log)

    summary = f"""Layer 4.5 Predictive Fusion — Backtest Summary
================================================
Train end: {split.train_end}
Holdout start: {split.val_start}
High-impact: cause-specific tau_c = P75(duration | cause) on train; global fallback = {global_tau:.1f} min
Duration regression: trained on log1p(duration_min), predictions expm1-transformed
Holdout RMSE (minutes): {reg_m['rmse']:.2f}
Holdout MAE (minutes): {reg_m['mae']:.2f}
Holdout Median AE: {reg_m['median_ae']:.2f}
Holdout RMSLE: {reg_m['rmsle']:.4f}
Holdout ROC-AUC (high-impact): {cls_m.get('roc_auc', float('nan'))}
Holdout ECE: {cls_m['ece']:.4f}
Conformal 90% coverage: {coverage:.3f}
Novelty rate (holdout): {float(novelty_val['novelty_flag'].mean()):.3f}

JOSV exports: layer45_operational_state_vector.csv (raw) + _normalized.csv (robust z)
As-of features built from raw history only (no full-dataset L1–L4 joins).
"""

    export_outputs(
        feat_df, josv, josv_normalized, metrics_df, shap_sum, shap_loc, novelty_all,
        fb_summary, cal_df, conformal_df, imp, coverage_df, summary, cause_tau_df,
    )

    joblib.dump(params, ARTIFACTS / "fitted_params.joblib")
    joblib.dump(iso, ARTIFACTS / "calibrator.joblib")
    joblib.dump(josv_scaler, ARTIFACTS / "josv_scaler.joblib")
    joblib.dump(cause_tau, ARTIFACTS / "cause_tau.joblib")
    if HAS_CATBOOST:
        dur_model.save_model(str(ARTIFACTS / "duration_model.cbm"))
        q50_model.save_model(str(ARTIFACTS / "quantile_p50.cbm"))
        q80_model.save_model(str(ARTIFACTS / "quantile_p80.cbm"))
        q95_model.save_model(str(ARTIFACTS / "quantile_p95.cbm"))
        hi_model.save_model(str(ARTIFACTS / "high_impact_model.cbm"))
    else:
        joblib.dump(dur_model, ARTIFACTS / "duration_model.joblib")

    run_deployment_inference(raw, params, retrain=True, josv_scaler=josv_scaler)

    return {"metrics": metrics_df, "tau": global_tau, "n_train": len(train_df), "n_val": len(val_df)}


def run_deployment_inference(
    df: pd.DataFrame | None = None,
    params: FittedParams | None = None,
    cutoff: str | None = None,
    retrain: bool = False,
    josv_scaler: dict | None = None,
) -> pd.DataFrame:
    """
    Deployment mode: fit on all history up to cutoff, score all rows.
    """
    logger.info("Layer 4.5 deployment mode")
    raw = df if df is not None else _load_events()
    if cutoff:
        raw = raw[pd.to_datetime(raw["start_local"]) <= pd.Timestamp(cutoff)]

    all_train_mask = pd.Series(True, index=raw.index)
    feat_df, params, _ = build_asof_feature_matrix(raw, all_train_mask)

    resolved = _training_rows(feat_df)
    global_tau = params.high_impact_tau
    cause_tau = params.cause_tau
    X, y_dur, y_hi, cols, cat_idx = _prepare_xy(resolved, cause_tau, global_tau)
    y_log = pd.Series(_log_duration(y_dur), index=y_dur.index)
    n = len(X)
    split_i = max(int(n * 0.85), 1)

    if retrain or not (ARTIFACTS / "deployment_duration.cbm").exists():
        dur_model = train_duration_model(
            X.iloc[:split_i], y_log.iloc[:split_i], X.iloc[split_i:], y_log.iloc[split_i:], cat_idx, "Huber:delta=1"
        )
        q50 = train_duration_model(
            X.iloc[:split_i], y_log.iloc[:split_i], X.iloc[split_i:], y_log.iloc[split_i:], cat_idx, "Quantile:alpha=0.5"
        )
        hi_model = train_high_impact_model(
            X.iloc[:split_i], y_hi.iloc[:split_i], X.iloc[split_i:], y_hi.iloc[split_i:], cat_idx
        )
        if HAS_CATBOOST:
            dur_model.save_model(str(ARTIFACTS / "deployment_duration.cbm"))
            q50.save_model(str(ARTIFACTS / "deployment_quantile_p50.cbm"))
            hi_model.save_model(str(ARTIFACTS / "deployment_high_impact.cbm"))
    elif HAS_CATBOOST:
        dur_model = CatBoostRegressor()
        dur_model.load_model(str(ARTIFACTS / "deployment_duration.cbm"))
        q50 = CatBoostRegressor()
        q50.load_model(str(ARTIFACTS / "deployment_quantile_p50.cbm"))
        hi_model = CatBoostClassifier()
        hi_model.load_model(str(ARTIFACTS / "deployment_high_impact.cbm"))
    else:
        dur_model = joblib.load(ARTIFACTS / "duration_model.joblib")
        q50 = dur_model
        hi_model = joblib.load(ARTIFACTS / "high_impact_model.joblib")

    pred = _exp_duration(dur_model.predict(X))
    p50 = _exp_duration(q50.predict(X))
    hi_prob = hi_model.predict_proba(X)[:, 1] if hasattr(hi_model, "predict_proba") else hi_model.predict(X)

    iso_path = ARTIFACTS / "calibrator.joblib"
    if iso_path.exists():
        iso = joblib.load(iso_path)
        hi_cal = iso.predict(hi_prob)
    else:
        hi_cal = hi_prob

    novelty = compute_novelty_drift(X.iloc[:split_i], X)

    deploy_raw = pd.DataFrame({
        "event_id": resolved["event_id"],
        "event_cause": resolved["event_cause"].astype(str),
        "duration_pred": pred,
        "duration_p50": p50,
        "high_impact_prob": hi_prob,
        "high_impact_prob_calibrated": hi_cal,
        "novelty_score": novelty["novelty_score"].values,
        "fragility_signal": resolved["asof_fragility_proxy"].values,
        "obi_signal": resolved["asof_obi_proxy"].values,
        "retrieval_confidence": resolved["asof_retrieval_confidence"].values,
        "trust_score": resolved["trust_score"].values if "trust_score" in resolved.columns else 0.5,
        "ims_proxy": resolved["asof_ims_proxy"].values,
        "drift_score": novelty["drift_score"].values,
    })

    scaler = josv_scaler
    if scaler is None and (ARTIFACTS / "josv_scaler.joblib").exists():
        scaler = joblib.load(ARTIFACTS / "josv_scaler.joblib")
    if scaler is None:
        scaler = fit_josv_scaler(deploy_raw.iloc[:split_i])

    deploy_norm = normalize_josv(deploy_raw, scaler)

    OUT.mkdir(exist_ok=True)
    resolved.to_csv(OUT / "layer45_deployment_features.csv", index=False)
    deploy_raw.to_csv(OUT / "layer45_deployment_predictions.csv", index=False)
    deploy_raw.to_csv(OUT / "layer45_deployment_state_vector.csv", index=False)
    deploy_norm.to_csv(OUT / "layer45_deployment_state_vector_normalized.csv", index=False)
    joblib.dump(params, ARTIFACTS / "deployment_params.joblib")
    return deploy_raw


def main() -> None:
    result = run_backtest()
    logger.info("Layer 4.5 complete. Holdout rows: %d", result["n_val"])


if __name__ == "__main__":
    main()
