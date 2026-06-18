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
    fbeta_score,
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
from layer45_duration_guard import (
    apply_fallback_chain,
    blend_with_fallback,
    build_duration_sanity_flags,
    build_scenario_ready_duration_bundle,
    compute_duration_reliability,
    sanitize_quantiles,
)
from layer45_feature_registry import export_feature_registry
from layer45_tail_models import (
    blend_tail_quantiles,
    build_tail_proxy_quantiles,
    calibrate_tail_classifier,
    compute_tail_labels,
    predict_tail_risk,
    train_tail_classifier,
)
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
    "raw_duration_p50", "raw_duration_p80", "raw_duration_p95",
    "safe_duration_p50", "safe_duration_p80", "safe_duration_p95",
    "duration_reliability", "tail_risk_prob",
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


def tune_high_impact_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    causes: np.ndarray,
    beta: float = 2.0,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """
    Per-cause F-beta (beta=2) optimal decision threshold on the validation set.

    Calibrated probabilities and labels must not be reused for model training —
    this is purely a post-hoc operating-point selection step.

    Returns {cause_str: threshold} with a "global" key as fallback.
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)

    def _best(yt: np.ndarray, yp: np.ndarray) -> float:
        best_t, best_f = 0.5, -1.0
        for t in thresholds:
            yh = (yp >= t).astype(int)
            f = float(fbeta_score(yt, yh, beta=beta, zero_division=0))
            if f > best_f:
                best_f, best_t = f, float(t)
        return best_t

    global_t = _best(y_true, y_prob)
    result: dict[str, float] = {"global": global_t}

    for cause in np.unique(causes):
        mask = causes == cause
        if mask.sum() < 20 or int(y_true[mask].sum()) < 3:
            result[str(cause)] = global_t
            continue
        result[str(cause)] = _best(y_true[mask], y_prob[mask])

    return result


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
    scenario_bundle: pd.DataFrame | None = None,
    dur_raw_df: pd.DataFrame | None = None,
    dur_sanitized_df: pd.DataFrame | None = None,
    dur_quality_df: pd.DataFrame | None = None,
    tail_risk_df: pd.DataFrame | None = None,
    hi_threshold_df: pd.DataFrame | None = None,
    leakage_audit_df: pd.DataFrame | None = None,
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
    # Duration quality gate outputs
    if dur_raw_df is not None:
        dur_raw_df.to_csv(out_dir / "layer45_duration_raw_predictions.csv", index=False)
    if dur_sanitized_df is not None:
        dur_sanitized_df.to_csv(out_dir / "layer45_duration_sanitized_predictions.csv", index=False)
    if dur_quality_df is not None:
        dur_quality_df.to_csv(out_dir / "layer45_duration_quality.csv", index=False)
    if tail_risk_df is not None:
        tail_risk_df.to_csv(out_dir / "layer45_tail_risk_probabilities.csv", index=False)
    if hi_threshold_df is not None:
        hi_threshold_df.to_csv(out_dir / "layer45_high_impact_thresholds_by_cause.csv", index=False)
    if scenario_bundle is not None:
        scenario_bundle.to_csv(out_dir / "layer45_scenario_ready_duration.csv", index=False)
    if leakage_audit_df is not None:
        leakage_audit_df.to_csv(out_dir / "layer45_leakage_audit.csv", index=False)
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

    # metrics_df is finalised after the quality gate (below) so guard metrics can be included
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

    # ── DURATION QUALITY GATE ──────────────────────────────────────────────
    logger.info("Duration quality gate: tail classifier, quantile sanitization, fallback blending")

    # Tail labels computed from training-window thresholds (no leakage)
    y_train_tail = compute_tail_labels(y_train_dur, train_df["event_cause"].astype(str), cause_tau, global_tau)
    y_val_tail = compute_tail_labels(y_val_dur, val_df["event_cause"].astype(str), cause_tau, global_tau)

    # Conservative tail-quantile proxies from training-window tail events only
    tail_proxies = build_tail_proxy_quantiles(y_train_dur, y_train_tail, train_df["event_cause"].astype(str))

    # Tail classifier (CatBoost; None → base-rate if sparse)
    _tail_base_rate = float(np.mean(y_train_tail))
    tail_model = train_tail_classifier(X_train, y_train_tail, X_val, y_val_tail, cat_idx)

    # Calibrate tail probabilities on val set
    tail_prob_val_raw = predict_tail_risk(tail_model, X_val, base_rate=_tail_base_rate)
    if tail_model is not None and len(np.unique(y_val_tail)) > 1:
        tail_cal_iso, tail_prob_val_cal = calibrate_tail_classifier(y_val_tail, tail_prob_val_raw)
    else:
        tail_cal_iso = None
        tail_prob_val_cal = tail_prob_val_raw

    # Tail probabilities for all resolved events
    tail_prob_all_raw = predict_tail_risk(tail_model, X_all, base_rate=_tail_base_rate)
    tail_prob_all_cal = (
        tail_cal_iso.predict(tail_prob_all_raw) if tail_cal_iso is not None else tail_prob_all_raw
    )

    # Tail-aware mixture of quantiles
    mix_p50_all, mix_p80_all, mix_p95_all = blend_tail_quantiles(
        p50_all, p80_all, p95_all,
        tail_prob_all_cal,
        resolved_feat["event_cause"].astype(str),
        tail_proxies,
    )

    # Monotone sanitization + safety clamp
    mono_p50_all, mono_p80_all, safe_p95_all, clamp_flags_all, crossing_flags_all = sanitize_quantiles(
        mix_p50_all, mix_p80_all, mix_p95_all
    )

    # Normalise reliability components using training-window stats (no leakage)
    _neff_all = pd.to_numeric(resolved_feat["asof_retrieval_n_eff"], errors="coerce").fillna(0).values
    _neff_train_max = max(float(_neff_all[resolved_train_mask].max()), 1.0)
    support_norm_all = np.clip(_neff_all / _neff_train_max, 0.0, 1.0)

    _nov_all = novelty_all["novelty_score"].values
    _nov_train_finite = _nov_all[resolved_train_mask]
    _nov_train_finite = _nov_train_finite[np.isfinite(_nov_train_finite)]
    _nov_p95 = float(np.percentile(_nov_train_finite, 95)) if len(_nov_train_finite) else 1.0
    nov_norm_all = np.clip(_nov_all / max(_nov_p95, 1e-6), 0.0, 1.0)

    _drift_all = novelty_all["drift_score"].values
    _drift_train_finite = _drift_all[resolved_train_mask]
    _drift_train_finite = _drift_train_finite[np.isfinite(_drift_train_finite)]
    _drift_p95 = float(np.percentile(_drift_train_finite, 95)) if len(_drift_train_finite) else 1.0
    drift_norm_all = np.clip(_drift_all / max(_drift_p95, 1e-6), 0.0, 1.0)

    retr_conf_all_arr = np.clip(
        pd.to_numeric(resolved_feat["asof_retrieval_confidence"], errors="coerce").fillna(0).values,
        0.0, 1.0,
    )

    width_all = safe_p95_all - mono_p50_all
    _width_train = width_all[resolved_train_mask]
    _pos_train_width = _width_train[_width_train > 0]
    _width_p95 = float(np.percentile(_pos_train_width, 95)) if len(_pos_train_width) else 1.0
    width_norm_all = np.clip(width_all / max(_width_p95, 1.0), 0.0, 1.0)

    # Calibration quality: 1 - ECE (global, derived from val; applied uniformly)
    cal_quality_scalar = max(0.0, 1.0 - cls_m["ece"])
    cal_quality_arr = np.full(len(resolved_feat), cal_quality_scalar)

    # Duration reliability score R ∈ [0, 1]
    reliability_all = compute_duration_reliability(
        cal_quality_arr, retr_conf_all_arr, nov_norm_all, drift_norm_all, support_norm_all, width_norm_all,
    )

    # Fallback quantiles from as-of feature chain (cause × corridor → cause → corridor → global)
    _global_fb = {
        "p50": float(y_train_dur.quantile(0.50)),
        "p80": float(y_train_dur.quantile(0.80)),
        "p95": float(y_train_dur.quantile(0.95)),
    }
    fb_p50_all, fb_p80_all, fb_p95_all = apply_fallback_chain(resolved_feat, _global_fb)

    # Reliability-weighted blend: Q_final = R * Q_safe + (1-R) * Q_fallback
    final_p50_all, final_p80_all, final_p95_all, fallback_flags_all = blend_with_fallback(
        mono_p50_all, mono_p80_all, safe_p95_all,
        fb_p50_all, fb_p80_all, fb_p95_all,
        reliability_all,
    )

    # Per-row sanity flags and reason codes
    sanity_flags_all, tail_risk_flags_all, guard_reasons_all = build_duration_sanity_flags(
        crossing_flags_all, clamp_flags_all, fallback_flags_all, tail_prob_all_cal,
    )

    # Scenario-ready bundle (the canonical Layer 5 input)
    scenario_bundle = build_scenario_ready_duration_bundle(
        final_p50_all, final_p80_all, final_p95_all,
        reliability_all, tail_prob_all_cal,
        sanity_flags_all, guard_reasons_all,
        resolved_feat["event_id"].values,
    )

    # ── HIGH-IMPACT THRESHOLD TUNING (F-beta=2 per cause) ─────────────────
    cause_thresholds = tune_high_impact_thresholds(
        y_val_hi.values, hi_cal_val, val_df["event_cause"].astype(str).values, beta=2.0,
    )
    hi_decision_all = np.array([
        int(hi_cal_all[i] >= cause_thresholds.get(str(c), cause_thresholds["global"]))
        for i, c in enumerate(resolved_feat["event_cause"].astype(str))
    ])
    hi_threshold_all = np.array([
        cause_thresholds.get(str(c), cause_thresholds["global"])
        for c in resolved_feat["event_cause"].astype(str)
    ])
    # ── END DURATION QUALITY GATE ──────────────────────────────────────────

    # ── Quality gate metrics (appended now that all guard vars are available) ─
    _val_res = resolved_val_mask
    y_dur_val_res = y_all_dur.values[_val_res]
    final_p50_val = final_p50_all[_val_res]
    raw_p50_val = p50_all[_val_res]

    for k, v in _regression_metrics(y_dur_val_res, final_p50_val).items():
        metrics_rows.append({"subset": "holdout_sanitized", "task": "duration", "metric": k, "value": v})
    for k, v in _regression_metrics(y_dur_val_res, raw_p50_val).items():
        metrics_rows.append({"subset": "holdout_raw_p50", "task": "duration", "metric": k, "value": v})

    _tail_gt500 = y_dur_val_res > 500
    if _tail_gt500.sum() > 0:
        for k, v in _regression_metrics(y_dur_val_res[_tail_gt500], final_p50_val[_tail_gt500]).items():
            metrics_rows.append({"subset": "holdout_tail_gt500_sanitized", "task": "duration", "metric": k, "value": v})
        for k, v in _regression_metrics(y_dur_val_res[_tail_gt500], raw_p50_val[_tail_gt500]).items():
            metrics_rows.append({"subset": "holdout_tail_gt500_raw", "task": "duration", "metric": k, "value": v})

    for _mn, _arr in [
        ("quantile_crossing_rate", crossing_flags_all[_val_res]),
        ("clamp_rate", clamp_flags_all[_val_res]),
        ("fallback_blend_rate", fallback_flags_all[_val_res]),
        ("tail_risk_flag_rate", tail_risk_flags_all[_val_res]),
        ("sanity_pass_rate", sanity_flags_all[_val_res]),
    ]:
        metrics_rows.append({"subset": "holdout", "task": "duration_guard", "metric": _mn, "value": float(_arr.mean())})

    if len(np.unique(y_val_tail)) > 1:
        for k, v in _classification_metrics(y_val_tail, tail_prob_val_cal).items():
            metrics_rows.append({"subset": "holdout", "task": "tail_risk", "metric": k, "value": v})

    hi_decision_val = np.array([
        int(hi_cal_val[i] >= cause_thresholds.get(str(c), cause_thresholds["global"]))
        for i, c in enumerate(val_df["event_cause"].astype(str))
    ])
    hi_tuned_m = {
        "precision_f2_tuned": float(precision_score(y_val_hi.values, hi_decision_val, zero_division=0)),
        "recall_f2_tuned": float(recall_score(y_val_hi.values, hi_decision_val, zero_division=0)),
        "f2_tuned": float(fbeta_score(y_val_hi.values, hi_decision_val, beta=2.0, zero_division=0)),
    }
    for k, v in hi_tuned_m.items():
        metrics_rows.append({"subset": "holdout", "task": "high_impact_tuned", "metric": k, "value": v})

    metrics_df = pd.DataFrame(metrics_rows)

    josv = _build_josv(
        resolved_feat, pred_all, p50_all, p80_all, p95_all,
        hi_prob_all, hi_cal_all, novelty_all, ci_lo_all, ci_hi_all,
        cause_tau, global_tau,
    )
    # Expand JOSV: raw duration columns (explicit diagnostics) + safe columns (Layer 5 optimisation)
    josv["raw_duration_p50"] = p50_all
    josv["raw_duration_p80"] = p80_all
    josv["raw_duration_p95"] = p95_all
    josv["safe_duration_p50"] = final_p50_all
    josv["safe_duration_p80"] = final_p80_all
    josv["safe_duration_p95"] = final_p95_all
    josv["duration_reliability"] = reliability_all
    josv["tail_risk_prob"] = tail_prob_all_cal
    josv["quantile_crossing_flag"] = crossing_flags_all
    josv["clamp_flag"] = clamp_flags_all
    josv["fallback_blend_flag"] = fallback_flags_all
    josv["tail_risk_flag"] = tail_risk_flags_all
    josv["duration_sanity_flag"] = sanity_flags_all
    josv["duration_guard_reason"] = guard_reasons_all
    josv["high_impact_decision"] = hi_decision_all
    josv["high_impact_decision_threshold"] = hi_threshold_all

    josv_train = josv[resolved_train_mask].copy()
    josv_scaler = fit_josv_scaler(josv_train)
    josv_normalized = normalize_josv(josv, josv_scaler)

    cause_tau_df = pd.DataFrame([
        {"event_cause": c, "tau_p75_min": t, "global_fallback_tau": global_tau}
        for c, t in sorted(cause_tau.items())
    ])

    # Raw duration predictions (diagnostics only — Layer 5 must NOT use this file)
    dur_raw_df = pd.DataFrame({
        "event_id": resolved_feat["event_id"].values,
        "raw_duration_p50": p50_all,
        "raw_duration_p80": p80_all,
        "raw_duration_p95": p95_all,
    })

    # Sanitised predictions after full quality gate
    dur_sanitized_df = pd.DataFrame({
        "event_id": resolved_feat["event_id"].values,
        "safe_duration_p50": final_p50_all,
        "safe_duration_p80": final_p80_all,
        "safe_duration_p95": final_p95_all,
        "duration_reliability": reliability_all,
    })

    # Per-row quality flags for auditing
    dur_quality_df = pd.DataFrame({
        "event_id": resolved_feat["event_id"].values,
        "quantile_crossing_flag": crossing_flags_all,
        "clamp_flag": clamp_flags_all,
        "fallback_blend_flag": fallback_flags_all,
        "tail_risk_flag": tail_risk_flags_all,
        "duration_sanity_flag": sanity_flags_all,
        "duration_guard_reason": guard_reasons_all,
        "duration_reliability": reliability_all,
    })

    # Tail-risk probabilities
    tail_risk_df = pd.DataFrame({
        "event_id": resolved_feat["event_id"].values,
        "tail_risk_prob_raw": tail_prob_all_raw,
        "tail_risk_prob": tail_prob_all_cal,
    })

    # Per-cause high-impact decision thresholds (F-beta=2 tuned)
    hi_threshold_df = pd.DataFrame([
        {
            "event_cause": cause,
            "fbeta2_threshold": thr,
            "global_threshold": cause_thresholds["global"],
            "n_support": int((val_df["event_cause"].astype(str) == cause).sum()),
            "n_positive": int(y_val_hi.values[(val_df["event_cause"].astype(str) == cause).values].sum()),
        }
        for cause, thr in cause_thresholds.items()
        if cause != "global"
    ])

    # Leakage audit
    leakage_audit_df = pd.DataFrame([
        {"component": "cause_tau", "fitted_on": "train", "applied_to": "all",
         "note": "P75 duration per cause — training window only"},
        {"component": "asof_features", "fitted_on": "as-of", "applied_to": "each event",
         "note": "History strictly before each event start date — no future leakage"},
        {"component": "josv_scaler", "fitted_on": "train", "applied_to": "all",
         "note": "Robust z-score (median/MAD) fitted on training JOSV rows"},
        {"component": "hi_calibrator", "fitted_on": "val", "applied_to": "all",
         "note": "Isotonic calibration on val predictions — no train labels used"},
        {"component": "tail_calibrator", "fitted_on": "val", "applied_to": "all",
         "note": "Isotonic calibration of tail-risk probs on val set"},
        {"component": "hi_thresholds", "fitted_on": "val", "applied_to": "all",
         "note": "F2-optimal decision thresholds per cause selected on val"},
        {"component": "tail_proxies", "fitted_on": "train", "applied_to": "all",
         "note": "Conservative tail quantiles from training-window tail events only"},
        {"component": "global_fallback_quantiles", "fitted_on": "train", "applied_to": "fallback only",
         "note": "Global P50/P80/P95 from training resolved events"},
    ])

    fb_summary = _fallback_summary(fallback_log)
    coverage_df = pd.DataFrame(fallback_log)

    _dg_cross = float(crossing_flags_all[resolved_val_mask].mean())
    _dg_clamp = float(clamp_flags_all[resolved_val_mask].mean())
    _dg_fb = float(fallback_flags_all[resolved_val_mask].mean())
    _dg_sanity = float(sanity_flags_all[resolved_val_mask].mean())
    _hi_f2 = hi_tuned_m["f2_tuned"]
    summary = f"""Layer 4.5 Predictive Fusion — Backtest Summary
================================================
Train end: {split.train_end}
Holdout start: {split.val_start}
High-impact: cause-specific tau_c = P75(duration | cause) on train; global fallback = {global_tau:.1f} min
Duration regression: trained on log1p(duration_min), predictions expm1-transformed

Raw duration model
  Holdout RMSE (minutes):  {reg_m['rmse']:.2f}
  Holdout MAE (minutes):   {reg_m['mae']:.2f}
  Holdout Median AE:       {reg_m['median_ae']:.2f}
  Holdout RMSLE:           {reg_m['rmsle']:.4f}

Duration quality gate (additive guardrail — Layer 5 consumes sanitized outputs only)
  Quantile crossing rate:  {_dg_cross:.3f}
  Clamp rate (Q95):        {_dg_clamp:.3f}
  Fallback blend rate:     {_dg_fb:.3f}
  Sanity pass rate:        {_dg_sanity:.3f}

High-impact classifier
  Holdout ROC-AUC:         {cls_m.get('roc_auc', float('nan'))}
  Holdout ECE:             {cls_m['ece']:.4f}
  F2-tuned (per-cause):    {_hi_f2:.4f}

Conformal 90% coverage:   {coverage:.3f}
Novelty rate (holdout):   {float(novelty_val['novelty_flag'].mean()):.3f}

Layer 5 must consume:
  outputs/layer45_scenario_ready_duration.csv
  outputs/layer45_operational_state_vector_normalized.csv
NOT: layer45_duration_raw_predictions.csv

JOSV exports: layer45_operational_state_vector.csv (expanded raw+safe) + _normalized.csv (robust z)
As-of features built from raw history only (no full-dataset L1–L4 joins).
"""

    export_outputs(
        feat_df, josv, josv_normalized, metrics_df, shap_sum, shap_loc, novelty_all,
        fb_summary, cal_df, conformal_df, imp, coverage_df, summary, cause_tau_df,
        scenario_bundle=scenario_bundle,
        dur_raw_df=dur_raw_df,
        dur_sanitized_df=dur_sanitized_df,
        dur_quality_df=dur_quality_df,
        tail_risk_df=tail_risk_df,
        hi_threshold_df=hi_threshold_df,
        leakage_audit_df=leakage_audit_df,
    )

    joblib.dump(params, ARTIFACTS / "fitted_params.joblib")
    joblib.dump(iso, ARTIFACTS / "calibrator.joblib")
    joblib.dump(josv_scaler, ARTIFACTS / "josv_scaler.joblib")
    joblib.dump(cause_tau, ARTIFACTS / "cause_tau.joblib")
    joblib.dump(tail_proxies, ARTIFACTS / "tail_proxies.joblib")
    joblib.dump(cause_thresholds, ARTIFACTS / "hi_cause_thresholds.joblib")
    if tail_cal_iso is not None:
        joblib.dump(tail_cal_iso, ARTIFACTS / "tail_calibrator.joblib")
    if HAS_CATBOOST:
        dur_model.save_model(str(ARTIFACTS / "duration_model.cbm"))
        q50_model.save_model(str(ARTIFACTS / "quantile_p50.cbm"))
        q80_model.save_model(str(ARTIFACTS / "quantile_p80.cbm"))
        q95_model.save_model(str(ARTIFACTS / "quantile_p95.cbm"))
        hi_model.save_model(str(ARTIFACTS / "high_impact_model.cbm"))
        if tail_model is not None:
            tail_model.save_model(str(ARTIFACTS / "tail_risk_model.cbm"))
    else:
        joblib.dump(dur_model, ARTIFACTS / "duration_model.joblib")
        if tail_model is not None:
            joblib.dump(tail_model, ARTIFACTS / "tail_risk_model.joblib")

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
