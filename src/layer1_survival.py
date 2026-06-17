"""
Layer 1 — Duration prediction (baseline + advanced survival models)
=====================================================================
Baseline: Kaplan-Meier (cause×corridor), Cox PH, lookup_expected_duration
Advanced: gamma frailty Cox, LogNormal AFT, RSF (tuned + calibrated),
          RMST, UMAP+HDBSCAN archetypes

Run: python src/layer1_survival.py
Outputs → outputs/layer1_*.csv and layer1_cox_summary.txt
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.utils import concordance_index
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "events_clean.parquet"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

MIN_GROUP_SIZE = 15
QUANTILES = (0.50, 0.80, 0.95)
QUANTILE_LABELS = ("p50", "p80", "p95")
RMST_HORIZONS = (60, 180, 360, 720)
MIN_CORRIDOR_FRAILTY_N = 20
MIN_ARCHETYPE_ROWS = 100
FRAILTY_BOOTSTRAP = 100
RSF_N_JOBS = 2
RSF_SHAP_SAMPLES = 100
RSF_IMPORTANCE_SAMPLES = 1500
RSF_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, 15],
    "min_samples_leaf": [10, 20],
}
RSF_CV_MAX_SAMPLES = 4000
CALIBRATION_TIMES = (60, 180, 360)
AFT_SURVIVAL_THRESHOLDS = (60, 120, 240)

try:
    import hdbscan
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import (
        concordance_index_censored,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv

    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def build_survival_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interval-censored survival table.

    Exact events: T in [L, U] with L = U = duration_min, E = 1.
    Right-censored (no end timestamp): T in [L, inf) with
    L = study_end - start, E = 0.
    """
    study_end = df["start_datetime"].max()
    if "modified_datetime" in df.columns and df["modified_datetime"].notna().any():
        study_end = max(study_end, df["modified_datetime"].max())

    work = df[df["start_datetime"].notna()].copy()
    exact = (
        (~work["is_censored"])
        & work["duration_min"].notna()
        & (work["duration_min"] >= 0)
    )
    censored = work["is_censored"].fillna(False)

    work["L"] = np.where(exact, work["duration_min"], np.nan)
    work["U"] = np.where(exact, work["duration_min"], np.inf)
    censor_lower = (study_end - work.loc[censored, "start_datetime"]).dt.total_seconds() / 60.0
    work.loc[censored, "L"] = censor_lower
    work.loc[censored, "U"] = np.inf

    work["T"] = work["L"]
    work["E"] = exact.astype(int)
    work["interval_type"] = np.where(exact, "exact", "right_censored")
    work["weights"] = work["trust_score"].clip(0.01, 1.0)
    work = work[work["T"].notna() & (work["T"] >= 0)].copy()
    return work


def _prepare_covariates(surv: pd.DataFrame) -> pd.DataFrame:
    work = surv.copy()
    work["priority_high"] = (work["priority"] == "High").astype(int)
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)
    work = work.drop(columns=[c for c in work.columns if c.startswith("cause_")], errors="ignore")
    cause_d = pd.get_dummies(work["event_cause"], prefix="cause", drop_first=True)
    return pd.concat([work, cause_d], axis=1)


def _frailty_covariate_cols(work: pd.DataFrame) -> list[str]:
    return sorted(c for c in work.columns if c.startswith("cause_")) + [
        "priority_high",
        "closure_int",
        "hour_sin",
        "hour_cos",
        "is_weekend",
    ]


def _frailty_design_matrix(work: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    mat = work[x_cols].copy()
    for col in mat.columns:
        if mat[col].dtype == bool:
            mat[col] = mat[col].astype(int)
        elif not np.issubdtype(mat[col].dtype, np.number):
            mat[col] = pd.to_numeric(mat[col], errors="coerce")
    return mat.astype(float)


def _turnbull_quantile(grid: np.ndarray, mass: np.ndarray, q: float) -> float | None:
    if len(grid) == 0 or mass.sum() <= 0:
        return None
    cdf = np.cumsum(mass)
    target = q
    idx = np.searchsorted(cdf, target, side="left")
    if idx >= len(grid):
        return float(grid[-1])
    return float(grid[idx])


def fit_turnbull_strata(
    surv: pd.DataFrame, group_cols: list[str], min_size: int = MIN_GROUP_SIZE
) -> pd.DataFrame:
    """
    Interval-censored quantiles via KM on (L, E).

    For exact + right-censored data this matches the Turnbull NPMLE and avoids
    building an O(n^2) mass grid over thousands of unique censoring bounds.
    """
    rows: list[dict[str, Any]] = []
    for keys, grp in surv.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) < min_size:
            continue
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(grp["L"], grp["E"], weights=grp["weights"], label=str(keys))
        except Exception:
            continue
        record = {col: val for col, val in zip(group_cols, keys)}
        record["n"] = len(grp)
        record["n_events"] = int(grp["E"].sum())
        record["n_interval_censored"] = int((grp["interval_type"] == "right_censored").sum())
        record["weighted_n"] = float(grp["weights"].sum())
        for q, label in zip(QUANTILES, QUANTILE_LABELS):
            record[f"{label}_min"] = _weighted_quantile_from_km(kmf, q)
        rows.append(record)
    out = pd.DataFrame(rows)
    print(
        f"Turnbull/KM interval-censored: {len(out)} strata, "
        f"{surv['interval_type'].eq('right_censored').sum():,} right-censored rows used"
    )
    return out


# --- Baseline KM / Cox -------------------------------------------------------

def _weighted_quantile_from_km(kmf: KaplanMeierFitter, q: float) -> float | None:
    sf = kmf.survival_function_
    if sf.empty:
        return None
    times = sf.index.values.astype(float)
    probs = sf.iloc[:, 0].values
    target = 1.0 - q
    below = np.where(probs <= target)[0]
    if len(below) == 0:
        return float(times[-1]) if len(times) else None
    return float(times[below[0]])


def fit_km_strata(
    surv: pd.DataFrame, group_cols: list[str], min_size: int = MIN_GROUP_SIZE
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, grp in surv.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) < min_size:
            continue
        kmf = KaplanMeierFitter()
        try:
            kmf.fit(grp["T"], grp["E"], weights=grp["weights"], label=str(keys))
        except Exception as exc:
            print(f"WARNING: KM failed for {keys}: {exc}")
            continue
        record = {col: val for col, val in zip(group_cols, keys)}
        record["n"] = len(grp)
        record["n_events"] = int(grp["E"].sum())
        record["weighted_n"] = float(grp["weights"].sum())
        for q, label in zip(QUANTILES, QUANTILE_LABELS):
            record[f"{label}_min"] = _weighted_quantile_from_km(kmf, q)
        rows.append(record)
    return pd.DataFrame(rows)


def fit_cox_model(surv: pd.DataFrame) -> tuple[CoxPHFitter | None, str]:
    work = surv.copy()
    work["priority_high"] = (work["priority"] == "High").astype(int)
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)
    top = work["corridor"].value_counts().head(8).index
    work["corridor_top"] = work["corridor"].where(work["corridor"].isin(top), "Other")
    dummies = pd.get_dummies(work["corridor_top"], prefix="corridor", drop_first=True)
    cox_df = pd.concat([
        work[["T", "E", "weights", "priority_high", "closure_int",
              "hour_sin", "hour_cos", "is_weekend"]],
        dummies,
    ], axis=1).dropna()
    if len(cox_df) < 50:
        msg = "WARNING: too few rows for Cox PH; skipping."
        print(msg)
        return None, msg
    cph = CoxPHFitter()
    try:
        cph.fit(cox_df, duration_col="T", event_col="E", weights_col="weights", show_progress=False)
    except Exception as exc:
        return None, f"WARNING: Cox PH fit failed: {exc}"
    lines = [
        "=== Cox Proportional Hazards Summary ===",
        f"Concordance index: {cph.concordance_index_:.3f}",
        "", "Hazard ratios (exp(coef)):",
    ]
    for idx, row in cph.summary.iterrows():
        lines.append(f"  {idx}: HR={np.exp(row['coef']):.3f}, p={row['p']:.4f}")
    report = "\n".join(lines)
    print(report)
    return cph, report


def lookup_expected_duration(
    cause: str, corridor: str, km_table: pd.DataFrame,
    km_fallback: pd.DataFrame, quantile: str = "p50",
) -> dict[str, Any] | None:
    col = f"{quantile}_min"
    match = km_table[(km_table["event_cause"] == cause) & (km_table["corridor"] == corridor)]
    if not match.empty:
        row = match.iloc[0]
        return {"duration_min": row[col], "source": "cause_corridor",
                "n": int(row["n"]), "confidence": "high" if row["n"] >= 30 else "moderate"}
    fb = km_fallback[km_fallback["event_cause"] == cause]
    if not fb.empty:
        row = fb.iloc[0]
        return {"duration_min": row[col], "source": "cause_only_fallback",
                "n": int(row["n"]), "confidence": "moderate" if row["n"] >= 30 else "low"}
    return None


# --- Shared gamma frailty Cox (EM + cluster bootstrap) -----------------------

def _breslow_cumhaz(times: np.ndarray, events: np.ndarray, lps: np.ndarray) -> np.ndarray:
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order].astype(bool)
    lp_sorted = lps[order]
    risk = np.exp(lp_sorted)
    cumhaz = np.zeros(len(times))
    baseline = 0.0
    uniq = np.unique(t_sorted[e_sorted])
    for ut in uniq:
        at_risk = t_sorted >= ut
        d = np.sum(e_sorted & (t_sorted == ut))
        denom = np.sum(risk[at_risk])
        if denom > 0 and d > 0:
            baseline += d / denom
        cumhaz[t_sorted >= ut] = baseline
    return cumhaz


def _cluster_scores(
    fit_df: pd.DataFrame,
    x_cols: list[str],
    beta: pd.Series,
    theta: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    x_mat = _frailty_design_matrix(fit_df, x_cols)
    lps = x_mat.values @ beta.values
    cumhaz = _breslow_cumhaz(fit_df["T"].values, fit_df["E"].values, lps)
    scores: dict[str, float] = {}
    events: dict[str, int] = {}
    frailty: dict[str, float] = {}
    inv_theta = 1.0 / max(theta, 1e-6)
    for corridor, grp in fit_df.groupby("corridor"):
        pos = fit_df.index.get_indexer(grp.index)
        s_j = float(np.sum(cumhaz[pos] * np.exp(lps[pos])))
        d_j = int(grp["E"].sum())
        scores[corridor] = max(s_j, 1e-8)
        events[corridor] = d_j
        frailty[corridor] = (d_j + inv_theta) / (s_j + inv_theta)
    return frailty, scores, events


def _fit_cox_beta(fit_df: pd.DataFrame, x_cols: list[str]) -> pd.Series:
    cox_df = pd.concat([
        fit_df[["T", "E"]],
        _frailty_design_matrix(fit_df, x_cols),
    ], axis=1)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="T", event_col="E", show_progress=False)
    return cph.params_


def _marginal_frailty_ll(
    theta: float,
    scores: dict[str, float],
    events: dict[str, int],
) -> float:
    if theta <= 0:
        return -np.inf
    inv_theta = 1.0 / theta
    ll = 0.0
    for corridor, s_j in scores.items():
        d_j = events[corridor]
        ll += (
            gammaln(d_j + inv_theta)
            - gammaln(inv_theta)
            - (d_j + inv_theta) * np.log(1.0 + theta * s_j)
            + d_j * np.log(theta * s_j + 1e-12)
        )
    return ll


def fit_gamma_frailty_cox(surv: pd.DataFrame) -> pd.DataFrame:
    work = _prepare_covariates(surv)
    x_cols = _frailty_covariate_cols(work)
    eligible = work["corridor"].value_counts()
    eligible = set(eligible[eligible >= MIN_CORRIDOR_FRAILTY_N].index)
    fit_df = work[work["corridor"].isin(eligible)][
        ["T", "E", "corridor"] + x_cols
    ].dropna().copy()
    if len(fit_df) < 100 or fit_df["corridor"].nunique() < 3:
        print("WARNING: insufficient data for gamma frailty Cox.")
        return pd.DataFrame()

    beta = _fit_cox_beta(fit_df, x_cols)
    theta = 1.0
    frailty: dict[str, float] = {}
    for _ in range(12):
        frailty, scores, events = _cluster_scores(fit_df, x_cols, beta, theta)
        opt = minimize_scalar(
            lambda t: -_marginal_frailty_ll(t, scores, events),
            bounds=(1e-3, 50.0),
            method="bounded",
        )
        theta = float(opt.x) if opt.success else theta
    frailty, _, _ = _cluster_scores(fit_df, x_cols, beta, theta)
    corridors = sorted(frailty.keys())
    boot = {c: [] for c in corridors}
    rng = np.random.default_rng(42)
    cluster_ids = fit_df["corridor"].unique()
    cluster_index = {
        c: fit_df.index[fit_df["corridor"] == c].to_numpy()
        for c in cluster_ids
    }
    for b in range(FRAILTY_BOOTSTRAP):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        idx = np.concatenate([cluster_index[c] for c in sampled])
        bdf = fit_df.loc[idx].reset_index(drop=True)
        b_theta = theta
        for _ in range(4):
            _, b_scores, b_events = _cluster_scores(bdf, x_cols, beta, b_theta)
            opt = minimize_scalar(
                lambda t: -_marginal_frailty_ll(t, b_scores, b_events),
                bounds=(1e-3, 50.0),
                method="bounded",
            )
            b_theta = float(opt.x) if opt.success else b_theta
        b_frailty, _, _ = _cluster_scores(bdf, x_cols, beta, b_theta)
        for corridor in corridors:
            if corridor in b_frailty:
                boot[corridor].append(float(b_frailty[corridor]))
        del bdf, b_frailty
        if b % 25 == 0:
            gc.collect()

    rows = []
    for corridor in corridors:
        u = frailty[corridor]
        samples = boot[corridor]
        if len(samples) >= 10:
            lo, hi = np.percentile(samples, [2.5, 97.5])
        else:
            lo, hi = u * 0.85, u * 1.15
        rows.append({
            "corridor": corridor,
            "frailty_effect": u,
            "ci_lower": lo,
            "ci_upper": hi,
            "theta_variance": theta,
            "interpretation": (
                "faster_clearance" if u > 1.05
                else "slower_clearance" if u < 0.95 else "baseline"
            ),
        })
    out = pd.DataFrame(rows).sort_values("frailty_effect", ascending=False)
    print(f"Gamma frailty Cox: {len(out)} corridors, theta={theta:.3f}")
    return out


# --- AFT models + survival probabilities -------------------------------------

def fit_aft_models(surv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = _prepare_covariates(surv)
    work = work[work["E"] == 1].copy()
    x_cols = _frailty_covariate_cols(work)
    aft_df = work[["T", "E", "weights"] + x_cols].dropna()
    aft_df = aft_df[(aft_df["T"] > 0) & (aft_df["T"] < 10080)].select_dtypes(include=[np.number])
    candidates = []
    for name, cls in [("Weibull", WeibullAFTFitter), ("LogNormal", LogNormalAFTFitter)]:
        try:
            aft = cls(penalizer=0.01)
            aft.fit(aft_df, duration_col="T", event_col="E", weights_col="weights", show_progress=False)
            aic = -2 * aft.log_likelihood_ + 2 * len(aft.params_)
            candidates.append((name, aft, aic))
            print(f"{name} AFT: C-index={aft.concordance_index_:.3f}, AIC={aic:.1f}")
        except Exception as exc:
            print(f"WARNING: {name} AFT failed: {exc}")
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    best_name, best_aft, _ = min(candidates, key=lambda x: x[2])
    ln_aft = next((a for n, a, _ in candidates if n == "LogNormal"), best_aft)
    medians = best_aft.predict_median(aft_df).reindex(aft_df.index)
    p90 = best_aft.predict_percentile(aft_df, p=0.9).reindex(aft_df.index)
    pi_lo = ln_aft.predict_percentile(aft_df, p=0.025).reindex(aft_df.index)
    pi_hi = ln_aft.predict_percentile(aft_df, p=0.975).reindex(aft_df.index)
    pred = pd.DataFrame({
        "event_id": work.loc[aft_df.index, "event_id"] if "event_id" in work.columns else aft_df.index,
        "event_cause": work.loc[aft_df.index, "event_cause"].values,
        "corridor": work.loc[aft_df.index, "corridor"].values,
        "predicted_median_min": medians.values,
        "predicted_p90_min": p90.values,
        "pi_lower_95": np.minimum(pi_lo.values, pi_hi.values),
        "pi_upper_95": np.maximum(pi_lo.values, pi_hi.values),
        "model_selected": best_name,
    })
    cmp = pd.DataFrame([{"model": n, "AIC": a, "selected": n == best_name} for n, _, a in candidates])

    surv_probs = pd.DataFrame({
        "event_id": work.loc[aft_df.index, "event_id"] if "event_id" in work.columns else aft_df.index,
    })
    for thr in AFT_SURVIVAL_THRESHOLDS:
        sf = ln_aft.predict_survival_function(aft_df, times=[thr])
        col = f"p_gt_{thr}"
        surv_probs[col] = sf.iloc[0].values if hasattr(sf, "iloc") else sf[thr]
    return pred, cmp, surv_probs


# --- Random Survival Forest (grid search, OOB, calibration) ------------------

def _rsf_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = build_survival_table(df)
    hs_path = OUT_DIR / "layer2_hotspots.csv"
    if hs_path.exists():
        hs = pd.read_csv(hs_path)[["junction", "weighted_intensity", "z_score"]]
        hs = hs.rename(columns={"weighted_intensity": "hotspot_intensity", "z_score": "hotspot_z"})
        work = work.merge(hs, on="junction", how="left")
        work["hotspot_intensity"] = work["hotspot_intensity"].fillna(0)
        work["hotspot_z"] = work["hotspot_z"].fillna(0)
    else:
        work["hotspot_intensity"] = 0.0
        work["hotspot_z"] = 0.0
        print("NOTE: layer2_hotspots.csv not found — RSF hotspot features set to 0.")

    le = LabelEncoder()
    work["cause_enc"] = le.fit_transform(work["event_cause"].fillna("unknown"))
    work["corridor_enc"] = le.fit_transform(work["corridor"].fillna("unknown"))
    work["junction_enc"] = le.fit_transform(work["junction"].fillna("unknown"))
    work["priority_enc"] = work["priority"].map({"High": 2, "Low": 1}).fillna(1)
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)
    feat = [
        "cause_enc", "corridor_enc", "junction_enc", "priority_enc", "closure_int",
        "hour_sin", "hour_cos", "is_weekend", "hotspot_intensity", "hotspot_z",
    ]
    return work, feat


def _rsf_cindex(y, X, **params) -> float:
    rsf = RandomSurvivalForest(random_state=42, n_jobs=RSF_N_JOBS, bootstrap=True, **params)
    rsf.fit(X, y)
    score = float(rsf.score(X, y))
    del rsf
    return score


def _permutation_importance(rsf, X, y, feat: list[str], max_rows: int = RSF_IMPORTANCE_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y
    base = float(rsf.score(Xs, ys))
    imp = []
    for k in range(Xs.shape[1]):
        col = Xs[:, k].copy()
        Xs[:, k] = rng.permutation(col)
        imp.append(max(0.0, base - float(rsf.score(Xs, ys))))
        Xs[:, k] = col
    s = sum(imp) or 1.0
    return pd.DataFrame({"feature": feat, "importance": [v / s for v in imp]})


def _compute_rsf_shap(
    rsf, X: np.ndarray, feat: list[str], work: pd.DataFrame, max_samples: int = RSF_SHAP_SAMPLES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_SHAP:
        print("NOTE: shap not installed; skipping RSF SHAP values.")
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(42)
    n = min(max_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    Xs = np.ascontiguousarray(X[idx])
    shap_values = None
    try:
        explainer = shap.TreeExplainer(rsf, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(Xs, check_additivity=False)
    except Exception:
        bg_idx = rng.choice(len(X), size=min(30, len(X)), replace=False)
        explainer = shap.KernelExplainer(
            rsf.predict, np.ascontiguousarray(X[bg_idx]), link="identity",
        )
        shap_values = explainer.shap_values(Xs, nsamples=50, silent=True)
    finally:
        if "explainer" in locals():
            del explainer
        gc.collect()
    if shap_values is None:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    summary = pd.DataFrame({
        "feature": feat,
        "mean_abs_shap": mean_abs,
        "mean_abs_shap_norm": mean_abs / (mean_abs.sum() or 1.0),
    }).sort_values("mean_abs_shap", ascending=False)

    event_ids = work.iloc[idx]["event_id"].values if "event_id" in work.columns else idx
    top_rows = []
    for row_i in range(len(Xs)):
        order = np.argsort(-np.abs(shap_values[row_i]))[:3]
        for rank, feat_i in enumerate(order, start=1):
            top_rows.append({
                "event_id": event_ids[row_i],
                "rank": rank,
                "feature": feat[feat_i],
                "shap_value": float(shap_values[row_i, feat_i]),
            })
    del shap_values, Xs
    return summary, pd.DataFrame(top_rows)


def _compute_per_cause_calibration(
    rsf, X: np.ndarray, y, work: pd.DataFrame, te: np.ndarray,
) -> pd.DataFrame:
    rows = []
    y_te = y[te]
    train_mask = np.ones(len(y), dtype=bool)
    train_mask[te] = False
    y_train = y[train_mask]
    times = np.array(CALIBRATION_TIMES, dtype=float)
    causes = work.iloc[te]["event_cause"].values if "event_cause" in work.columns else None
    if causes is None or len(te) == 0:
        return pd.DataFrame()
    estimate = np.row_stack([fn(times) for fn in rsf.predict_survival_function(X[te])])
    for cause in sorted(set(causes)):
        mask = causes == cause
        if mask.sum() < 30:
            continue
        y_te_sub = y_te[mask]
        try:
            ibs = float(integrated_brier_score(y_train, y_te_sub, estimate[mask], times))
        except Exception:
            ibs = np.nan
        try:
            auc, _ = cumulative_dynamic_auc(y_train, y_te_sub, rsf.predict(X[te][mask]), times)
            auc_60 = float(auc[0]) if len(auc) else np.nan
        except Exception:
            auc_60 = np.nan
        rows.append({"event_cause": cause, "n_test": int(mask.sum()), "ibs": ibs, "auc_60": auc_60})
    del estimate
    return pd.DataFrame(rows).sort_values("ibs")


def fit_random_survival_forest(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    work, feat = _rsf_feature_matrix(df)
    X = np.ascontiguousarray(work[feat].astype(float).values)
    T, E = work["T"].values, work["E"].astype(bool).values

    metrics_rows = [
        {"metric": "oob_cindex", "value": np.nan},
        {"metric": "cv_cindex_mean", "value": np.nan},
        {"metric": "cv_cindex_std", "value": np.nan},
    ]
    calib_rows = [
        {"metric": "ibs", "value": np.nan},
        {"metric": "auc_60", "value": np.nan},
        {"metric": "auc_180", "value": np.nan},
        {"metric": "auc_360", "value": np.nan},
    ]

    if not HAS_SKSURV:
        print("WARNING: scikit-survival not installed; RSF uses duration ratio proxy.")
        gm = np.median(T[E])
        work["survival_risk_score"] = np.clip(work["T"] / (gm + 1e-6), 0, 5) / 5
        imp_df = pd.DataFrame({"feature": feat, "importance": [1 / len(feat)] * len(feat)})
        cols = [c for c in ["event_id", "event_cause", "corridor", "junction", "T", "survival_risk_score"]
                if c in work.columns]
        return work[cols], imp_df, pd.DataFrame(metrics_rows), pd.DataFrame(calib_rows), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    y = Surv.from_arrays(event=E, time=T)
    rng = np.random.default_rng(42)
    all_idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = all_idx[:split], all_idx[split:]
    best_score = -np.inf
    best_params = {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 20}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if len(X) > RSF_CV_MAX_SAMPLES:
        cv_idx = rng.choice(len(X), size=RSF_CV_MAX_SAMPLES, replace=False)
        X_cv, y_cv = X[cv_idx], y[cv_idx]
        print(f"RSF grid search on {RSF_CV_MAX_SAMPLES:,}/{len(X):,} subsample; final fit uses all rows")
    else:
        X_cv, y_cv = X, y
    print("RSF hyperparameter grid search (5-fold CV)...")
    for n_est in RSF_PARAM_GRID["n_estimators"]:
        for depth in RSF_PARAM_GRID["max_depth"]:
            for leaf in RSF_PARAM_GRID["min_samples_leaf"]:
                params = {"n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf}
                scores = []
                for fold_tr, fold_te in kf.split(X_cv):
                    rsf_fold = RandomSurvivalForest(
                        random_state=42, n_jobs=RSF_N_JOBS, bootstrap=True, **params
                    ).fit(X_cv[fold_tr], y_cv[fold_tr])
                    event, time = y_cv[fold_te]["event"], y_cv[fold_te]["time"]
                    scores.append(concordance_index_censored(event, time, rsf_fold.predict(X_cv[fold_te]))[0])
                    del rsf_fold
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                gc.collect()
    cv_scores = []
    for fold_tr, fold_te in kf.split(X_cv):
        rsf_fold = RandomSurvivalForest(
            random_state=42, n_jobs=RSF_N_JOBS, bootstrap=True, **best_params
        ).fit(X_cv[fold_tr], y_cv[fold_tr])
        event, time = y_cv[fold_te]["event"], y_cv[fold_te]["time"]
        cv_scores.append(concordance_index_censored(event, time, rsf_fold.predict(X_cv[fold_te]))[0])
        del rsf_fold
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    del X_cv, y_cv
    gc.collect()

    rsf = RandomSurvivalForest(
        random_state=42, n_jobs=RSF_N_JOBS, bootstrap=True, oob_score=True, **best_params
    )
    rsf.fit(X, y)
    train_c = float(rsf.score(X, y))
    oob_c = float(getattr(rsf, "oob_score_", np.nan))
    print(
        f"RSF best={best_params} train C={train_c:.3f}, OOB C={oob_c:.3f}, CV C={cv_mean:.3f}±{cv_std:.3f}"
    )

    work["survival_risk_score"] = rsf.predict(X)
    imp_df = _permutation_importance(rsf, X, y, feat)

    metrics_rows = [
        {"metric": "oob_cindex", "value": oob_c},
        {"metric": "cv_cindex_mean", "value": cv_mean},
        {"metric": "cv_cindex_std", "value": cv_std},
        {"metric": "train_cindex", "value": train_c},
        {"metric": "best_n_estimators", "value": best_params["n_estimators"]},
        {"metric": "best_max_depth", "value": best_params["max_depth"]},
        {"metric": "best_min_samples_leaf", "value": best_params["min_samples_leaf"]},
    ]

    rsf_cal = RandomSurvivalForest(
        random_state=42, n_jobs=RSF_N_JOBS, bootstrap=True, **best_params
    ).fit(X[tr], y[tr])
    y_tr, y_te = y[tr], y[te]
    times = np.array(CALIBRATION_TIMES, dtype=float)
    estimate = np.row_stack([fn(times) for fn in rsf_cal.predict_survival_function(X[te])])
    try:
        calib_rows[0]["value"] = float(integrated_brier_score(y_tr, y_te, estimate, times))
    except Exception as exc:
        print(f"WARNING: IBS computation failed: {exc}")
    try:
        auc, _ = cumulative_dynamic_auc(y_tr, y_te, rsf_cal.predict(X[te]), times)
        for i, t in enumerate(CALIBRATION_TIMES):
            calib_rows[i + 1]["value"] = float(auc[i])
    except Exception as exc:
        print(f"WARNING: td-AUC computation failed: {exc}")

    cause_cal = _compute_per_cause_calibration(rsf_cal, X, y, work, te)
    del estimate, rsf_cal
    gc.collect()

    shap_summary, shap_top = _compute_rsf_shap(rsf, X, feat, work)
    if not cause_cal.empty:
        print(f"Per-cause calibration: {len(cause_cal)} causes")
    if not shap_summary.empty:
        print(f"RSF SHAP: top driver {shap_summary.iloc[0]['feature']}")

    cols = [c for c in ["event_id", "event_cause", "corridor", "junction", "T", "survival_risk_score"]
            if c in work.columns]
    return work[cols], imp_df, pd.DataFrame(metrics_rows), pd.DataFrame(calib_rows), shap_summary, shap_top, cause_cal


def compute_rmst(surv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cause, corridor), grp in surv.groupby(["event_cause", "corridor"]):
        if len(grp) < MIN_GROUP_SIZE:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(grp["T"], grp["E"], weights=grp["weights"])
        times = kmf.survival_function_.index.values.astype(float)
        s_vals = kmf.survival_function_.iloc[:, 0].values
        rec = {"event_cause": cause, "corridor": corridor, "n": len(grp)}
        for tau in RMST_HORIZONS:
            grid = np.concatenate([[0.0], times[times <= tau], [tau]])
            s_g = []
            for t in grid:
                i = np.searchsorted(times, t, side="right") - 1
                s_g.append(1.0 if i < 0 else float(s_vals[min(i, len(s_vals) - 1)]))
            rec[f"rmst_{tau}min"] = float(np.trapezoid(s_g, grid))
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(f"RMST: {len(out)} strata")
    return out


def fit_survival_archetypes(surv: pd.DataFrame) -> pd.DataFrame:
    work = surv[surv["E"] == 1].copy()
    work["priority_enc"] = work["priority"].map({"High": 2, "Low": 1}).fillna(1)
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)
    feat_cols = ["T", "priority_enc", "closure_int", "hour_sin", "hour_cos", "trust_score"]
    X_raw = work[feat_cols].astype(float).fillna(work[feat_cols].median())
    if len(X_raw) < MIN_ARCHETYPE_ROWS:
        return pd.DataFrame()

    if not HAS_UMAP:
        print("WARNING: umap-learn/hdbscan not installed; skipping archetypes.")
        return pd.DataFrame()

    X = StandardScaler().fit_transform(X_raw)
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10, prediction_data=False)
    labels = clusterer.fit_predict(embedding)
    probs = clusterer.probabilities_

    label_map: dict[int, str] = {}
    valid = [c for c in set(labels) if c >= 0]
    names = ["rapid_clear", "moderate", "severe", "chronic", "mega_disruption"]
    if valid:
        cluster_means = {c: float(work.loc[labels == c, "T"].mean()) for c in valid}
        ordered = sorted(valid, key=lambda c: cluster_means[c])
        for i, c in enumerate(ordered):
            tier = min(int(i * len(names) / len(ordered)), len(names) - 1)
            label_map[c] = names[tier]
    label_map[-1] = "outlier"

    out = pd.DataFrame({
        "event_id": work["event_id"] if "event_id" in work.columns else work.index,
        "archetype": [label_map.get(int(l), "outlier") for l in labels],
        "probability": probs,
        "cluster_id": labels,
    })
    print(f"UMAP+HDBSCAN archetypes: {len(valid)} clusters, {int((labels == -1).sum())} outliers")
    return out


def run_layer1() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Layer 1: Survival Analysis (baseline + advanced) ===\n")
    df = load_data()
    surv = build_survival_table(df)
    n_exact = int(surv["E"].sum())
    n_cens = int((1 - surv["E"]).sum())
    print(f"Survival table: {len(surv):,} rows ({n_exact:,} exact, {n_cens:,} interval-censored)\n")

    print("--- Baseline: Turnbull + Kaplan-Meier + Cox PH ---")
    turnbull_primary = fit_turnbull_strata(surv, ["event_cause", "corridor"])
    turnbull_fallback = fit_turnbull_strata(surv, ["event_cause"])
    turnbull_primary.to_csv(OUT_DIR / "layer1_turnbull_quantiles.csv", index=False)
    turnbull_fallback.to_csv(OUT_DIR / "layer1_turnbull_fallback.csv", index=False)
    km_primary = fit_km_strata(surv, ["event_cause", "corridor"])
    km_fallback = fit_km_strata(surv, ["event_cause"])
    if not km_primary.empty:
        print(km_primary.sort_values("n", ascending=False).head(5)[
            ["event_cause", "corridor", "n", "p50_min", "p80_min"]
        ].to_string(index=False))
    _, cox_report = fit_cox_model(surv)
    (OUT_DIR / "layer1_cox_summary.txt").write_text(cox_report or "")
    km_primary.to_csv(OUT_DIR / "layer1_survival_quantiles.csv", index=False)
    km_fallback.to_csv(OUT_DIR / "layer1_survival_fallback.csv", index=False)

    print("\n--- Advanced: gamma frailty, AFT, RSF, RMST, archetypes, calibration ---")
    fit_gamma_frailty_cox(surv).to_csv(OUT_DIR / "layer1_frailty_scores.csv", index=False)
    aft_pred, aft_cmp, aft_probs = fit_aft_models(surv)
    if not aft_pred.empty:
        aft_pred.to_csv(OUT_DIR / "layer1_duration_predictions.csv", index=False)
        aft_cmp.to_csv(OUT_DIR / "layer1_aft_model_comparison.csv", index=False)
    if not aft_probs.empty:
        aft_probs.to_csv(OUT_DIR / "layer1_aft_survival_probs.csv", index=False)
    rsf_out, rsf_imp, rsf_metrics, rsf_cal, rsf_shap, rsf_shap_top, rsf_cause_cal = fit_random_survival_forest(df)
    rsf_out.to_csv(OUT_DIR / "layer1_survival_risk_scores.csv", index=False)
    rsf_imp.to_csv(OUT_DIR / "layer1_rsf_feature_importance.csv", index=False)
    rsf_metrics.to_csv(OUT_DIR / "layer1_rsf_metrics.csv", index=False)
    rsf_cal.to_csv(OUT_DIR / "layer1_rsf_calibration.csv", index=False)
    if not rsf_shap.empty:
        rsf_shap.to_csv(OUT_DIR / "layer1_rsf_shap_summary.csv", index=False)
    if not rsf_shap_top.empty:
        rsf_shap_top.to_csv(OUT_DIR / "layer1_rsf_shap_top_drivers.csv", index=False)
    if not rsf_cause_cal.empty:
        rsf_cause_cal.to_csv(OUT_DIR / "layer1_rsf_calibration_by_cause.csv", index=False)
    compute_rmst(surv).to_csv(OUT_DIR / "layer1_rmst_summary.csv", index=False)
    fit_survival_archetypes(surv).to_csv(OUT_DIR / "layer1_incident_archetypes.csv", index=False)

    probe = lookup_expected_duration("protest", "Non-corridor", km_primary, km_fallback)
    print(f"\nLookup probe (protest): {probe}")
    print(f"\nAll Layer 1 outputs → {OUT_DIR}/layer1_*")
    return km_primary, km_fallback


if __name__ == "__main__":
    run_layer1()
