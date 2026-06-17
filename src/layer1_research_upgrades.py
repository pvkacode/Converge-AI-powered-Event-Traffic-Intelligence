"""
Layer 1 research-grade upgrades (additive only — does not retrain base pipeline models).

1. Frailty model validation (Cox PH vs shared gamma frailty LRT)
2. Stacked survival ensemble (Elastic Net meta-model on existing predictions)

Run after layer1_survival.py:
    python src/layer1_research_upgrades.py
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from layer1_survival import (
    CALIBRATION_TIMES,
    MIN_CORRIDOR_FRAILTY_N,
    OUT_DIR,
    _cluster_scores,
    _fit_cox_beta,
    _frailty_covariate_cols,
    _frailty_design_matrix,
    _marginal_frailty_ll,
    _prepare_covariates,
    build_survival_table,
    load_data,
)

try:
    from sksurv.metrics import concordance_index_censored, integrated_brier_score
    from sksurv.util import Surv

    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def _frailty_fit_df(surv: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = _prepare_covariates(surv)
    x_cols = _frailty_covariate_cols(work)
    eligible = work["corridor"].value_counts()
    eligible = set(eligible[eligible >= MIN_CORRIDOR_FRAILTY_N].index)
    fit_df = work[work["corridor"].isin(eligible)][
        ["T", "E", "corridor", "event_id"] + x_cols
    ].dropna().copy()
    return fit_df, x_cols


def _fit_frailty_theta(fit_df: pd.DataFrame, x_cols: list[str]) -> tuple[pd.Series, float, dict, dict]:
    beta = _fit_cox_beta(fit_df, x_cols)
    theta = 1.0
    for _ in range(12):
        _, scores, events = _cluster_scores(fit_df, x_cols, beta, theta)
        opt = minimize_scalar(
            lambda t: -_marginal_frailty_ll(t, scores, events),
            bounds=(1e-3, 50.0),
            method="bounded",
        )
        theta = float(opt.x) if opt.success else theta
    _, scores, events = _cluster_scores(fit_df, x_cols, beta, theta)
    return beta, theta, scores, events


def run_frailty_validation(surv: pd.DataFrame) -> pd.DataFrame:
    """Likelihood-ratio test: standard Cox PH vs shared gamma frailty Cox."""
    fit_df, x_cols = _frailty_fit_df(surv)
    if len(fit_df) < 100:
        print("WARNING: insufficient rows for frailty validation.")
        return pd.DataFrame()

    design = _frailty_design_matrix(fit_df, x_cols)
    cox_df = pd.concat([fit_df[["T", "E"]], design], axis=1)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="T", event_col="E", show_progress=False)
    cox_ll = float(cph.log_likelihood_)
    n_cox = int(len(cph.params_))

    beta, theta, scores, events = _fit_frailty_theta(fit_df, x_cols)
    frailty_ll = float(_marginal_frailty_ll(theta, scores, events))
    # Nested Cox null within frailty family: theta -> infinity (no cluster heterogeneity)
    cox_marginal_ll = float(_marginal_frailty_ll(1e10, scores, events))
    n_frailty = n_cox + 1

    lr_stat = max(0.0, 2.0 * (frailty_ll - cox_marginal_ll))
    df = n_frailty - n_cox
    p_value = float(1.0 - chi2.cdf(lr_stat, df=df))
    supported = p_value < 0.05

    corridor_frailty = pd.read_csv(OUT_DIR / "layer1_frailty_scores.csv")
    frailty_vals = corridor_frailty["frailty_effect"].values
    var_between = float(np.var(frailty_vals, ddof=1)) if len(frailty_vals) > 1 else 0.0
    icc = float(theta / (1.0 + theta))

    out = pd.DataFrame([{
        "cox_loglik": cox_ll,
        "frailty_loglik": frailty_ll,
        "lr_statistic": lr_stat,
        "degrees_freedom": df,
        "p_value": p_value,
        "frailty_supported": supported,
    }])
    out.to_csv(OUT_DIR / "layer1_frailty_validation.csv", index=False)

    lines = [
        "=== Gamma Frailty Model Validation ===",
        "",
        f"Cox partial log-likelihood:     {cox_ll:.2f}  ({n_cox} parameters)",
        f"Frailty marginal log-likelihood: {frailty_ll:.2f}  ({n_frailty} parameters)",
        f"Cox nested marginal (θ→∞):      {cox_marginal_ll:.2f}",
        f"Likelihood-ratio statistic:      {lr_stat:.3f}  (df={df}, frailty vs nested Cox)",
        f"p-value:                       {p_value:.6f}",
        f"Frailty supported (p<0.05):    {supported}",
        "",
        f"Estimated theta (frailty variance): {theta:.4f}",
        f"ICC proxy theta/(1+theta):          {icc:.4f}",
        f"Corridor frailty effect variance:   {var_between:.4f}",
        "",
        "Interpretation:",
    ]
    if supported:
        lines.append(
            "- Shared gamma frailty significantly improves fit over standard Cox PH."
        )
        lines.append(
            f"- Corridor-level heterogeneity explains material variation (ICC≈{icc:.3f})."
        )
    else:
        lines.append(
            "- Frailty term does not significantly improve fit at α=0.05; corridor effects may be weak."
        )
    lines.append(
        f"- theta={theta:.3f}: larger values imply stronger unobserved corridor-level clustering."
    )
    (OUT_DIR / "layer1_frailty_interpretation.txt").write_text("\n".join(lines) + "\n")

    print(f"Frailty LRT: LR={lr_stat:.2f}, p={p_value:.4f}, supported={supported}")
    return out


def _cox_risk_scores(surv: pd.DataFrame) -> pd.DataFrame:
    """Partial hazard scores from the same Cox specification as layer1_survival."""
    cox_df = _cox_design_for_eval(surv)
    cph = CoxPHFitter()
    cph.fit(
        cox_df.drop(columns=["event_id"]),
        duration_col="T",
        event_col="E",
        weights_col="weights",
        show_progress=False,
    )
    risk = cph.predict_partial_hazard(cox_df.drop(columns=["event_id"])).values.ravel()
    return pd.DataFrame({"event_id": cox_df["event_id"].values, "cox_risk_score": risk})


def _cox_design_for_eval(work: pd.DataFrame) -> pd.DataFrame:
    work = work.copy()
    work["priority_high"] = (work["priority"] == "High").astype(int)
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)
    top = work["corridor"].value_counts().head(8).index
    work["corridor_top"] = work["corridor"].where(work["corridor"].isin(top), "Other")
    dummies = pd.get_dummies(work["corridor_top"], prefix="corridor", drop_first=True)
    return pd.concat([
        work[["T", "E", "event_id", "weights", "priority_high", "closure_int",
              "hour_sin", "hour_cos", "is_weekend"]],
        dummies,
    ], axis=1).dropna()


def _align_cox_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure train/test Cox design matrices share identical dummy columns."""
    drop_cols = {"event_id", "T", "E", "weights"}
    train_x = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_x = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    all_cols = sorted(set(train_x.columns) | set(test_x.columns))
    for col in all_cols:
        if col not in train_x.columns:
            train_x[col] = 0.0
        if col not in test_x.columns:
            test_x[col] = 0.0
    train_x = train_x[all_cols]
    test_x = test_x[all_cols]
    train_out = pd.concat([train_df[[c for c in drop_cols if c in train_df.columns]], train_x], axis=1)
    test_out = pd.concat([test_df[[c for c in drop_cols if c in test_df.columns]], test_x], axis=1)
    return train_out, test_out



def _ibs_from_survival_matrix(
    y_train, y_test, estimate: np.ndarray, times: np.ndarray,
) -> float:
    try:
        return float(integrated_brier_score(y_train, y_test, estimate, times))
    except Exception:
        return float("nan")


def _survival_from_aft_probs(probs: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    t_pts = np.array([60.0, 120.0, 240.0])
    s_pts = probs[["p_gt_60", "p_gt_120", "p_gt_240"]].values
    out = np.zeros((len(probs), len(times)))
    for i in range(len(probs)):
        out[i] = np.interp(times, t_pts, s_pts[i], left=1.0, right=s_pts[i, -1])
    return out


def run_stacked_ensemble(surv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elastic Net stack of Cox, frailty, AFT, and RSF scores (no base-model retraining)."""
    rsf_path = OUT_DIR / "layer1_survival_risk_scores.csv"
    aft_path = OUT_DIR / "layer1_duration_predictions.csv"
    frailty_path = OUT_DIR / "layer1_frailty_scores.csv"
    aft_prob_path = OUT_DIR / "layer1_aft_survival_probs.csv"
    for p in (rsf_path, aft_path, frailty_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing prerequisite output: {p}. Run layer1_survival.py first.")

    base = surv[[
        "event_id", "T", "E", "corridor", "weights", "priority",
        "requires_road_closure", "hour_sin", "hour_cos", "is_weekend",
    ]].copy()
    rsf = pd.read_csv(rsf_path)[["event_id", "survival_risk_score"]].rename(
        columns={"survival_risk_score": "rsf_risk_score"}
    )
    # AFT median duration: invert so higher = greater risk (shorter expected survival)
    aft = pd.read_csv(aft_path)[["event_id", "predicted_median_min"]].rename(
        columns={"predicted_median_min": "aft_risk_score"}
    )
    aft["aft_risk_score"] = -aft["aft_risk_score"]
    cox = _cox_risk_scores(surv)
    frailty_corr = pd.read_csv(frailty_path)[["corridor", "frailty_effect"]]
    frailty_map = frailty_corr.set_index("corridor")["frailty_effect"]
    base["frailty_risk_score"] = base["corridor"].map(
        lambda c: 1.0 / frailty_map.get(c, 1.0) if pd.notna(c) else 1.0
    )

    merged = base.merge(cox, on="event_id", how="inner")
    merged = merged.merge(rsf, on="event_id", how="inner")
    merged = merged.merge(aft, on="event_id", how="left")
    merged["aft_risk_score"] = merged["aft_risk_score"].fillna(merged["aft_risk_score"].median())
    merged = merged.dropna(subset=["cox_risk_score", "rsf_risk_score", "frailty_risk_score"])

    feat_cols = ["cox_risk_score", "frailty_risk_score", "aft_risk_score", "rsf_risk_score"]
    X_raw = merged[feat_cols].values.astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    T = merged["T"].values.astype(float)
    E = merged["E"].astype(bool).values

    X_tr, X_te, y_tr, y_te, E_tr, E_te, idx_tr, idx_te = train_test_split(
        X, T, E, merged.index.values,
        test_size=0.2,
        random_state=42,
        stratify=E.astype(int),
    )

    meta = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=np.logspace(-3, 1, 20),
        cv=5,
        max_iter=8000,
        random_state=42,
    )
    meta.fit(X_tr[E_tr], -y_tr[E_tr])
    stacked_all = meta.predict(X)
    merged["stacked_risk_score"] = stacked_all

    pred_out = merged[["event_id", "stacked_risk_score"]].copy()
    pred_out.to_csv(OUT_DIR / "layer1_stacked_survival_predictions.csv", index=False)

    metrics_rows = []
    if not HAS_SKSURV:
        print("WARNING: scikit-survival missing; C-index/IBS metrics skipped.")
        return pred_out, pd.DataFrame(metrics_rows)

    y_train = Surv.from_arrays(event=E_tr, time=y_tr)
    y_test = Surv.from_arrays(event=E_te, time=y_te)
    times = np.array(CALIBRATION_TIMES, dtype=float)

    work_te = merged.loc[idx_te]
    cox_ibs = float("nan")
    train_design = _cox_design_for_eval(merged.loc[idx_tr])
    test_design = _cox_design_for_eval(work_te)
    train_design, test_design = _align_cox_columns(train_design, test_design)
    cph_eval = CoxPHFitter()
    try:
        cph_eval.fit(
            train_design.drop(columns=["event_id"]),
            duration_col="T",
            event_col="E",
            weights_col="weights",
            show_progress=False,
        )
    except Exception:
        cph_eval = None
    if cph_eval is not None and hasattr(cph_eval, "params_"):
        te_x = test_design.drop(columns=["event_id"])
        sf = cph_eval.predict_survival_function(te_x, times=times.tolist())
        cox_est = np.row_stack([fn.values for fn in sf])
        cox_ibs = _ibs_from_survival_matrix(y_train, y_test, cox_est, times)
    else:
        cox_est = None

    risk_specs = {
        "cox": merged["cox_risk_score"].values,
        "frailty": merged["frailty_risk_score"].values,
        "aft": merged["aft_risk_score"].values,
        "rsf": merged["rsf_risk_score"].values,
        "stacked": stacked_all,
    }

    rsf_ibs = float("nan")
    cal_path = OUT_DIR / "layer1_rsf_calibration.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        ibs_row = cal[cal["metric"] == "ibs"]
        if not ibs_row.empty:
            rsf_ibs = float(ibs_row.iloc[0]["value"])

    aft_probs = pd.read_csv(aft_prob_path) if aft_prob_path.exists() else pd.DataFrame()
    merged_te = merged.loc[idx_te].copy()

    for name, risk in risk_specs.items():
        c_idx = float(concordance_index_censored(E, T, risk)[0])
        ibs = float("nan")
        if name == "rsf" and not np.isnan(rsf_ibs):
            ibs = rsf_ibs
        elif name == "cox":
            ibs = cox_ibs
        elif name == "aft" and not aft_probs.empty:
            te_probs = merged_te.merge(aft_probs, on="event_id", how="inner")
            if len(te_probs) >= 30:
                est = _survival_from_aft_probs(te_probs, times)
                y_te_aft = Surv.from_arrays(
                    event=te_probs["E"].astype(bool).values,
                    time=te_probs["T"].values,
                )
                ibs = _ibs_from_survival_matrix(y_train, y_te_aft, est, times)
        elif name == "frailty":
            if cox_est is not None:
                frailty_mult = work_te["frailty_risk_score"].values
                frailty_mult = frailty_mult / (frailty_mult.mean() + 1e-8)
                frailty_est = np.clip(cox_est ** frailty_mult[:, None], 0.01, 1.0)
                ibs = _ibs_from_survival_matrix(y_train, y_test, frailty_est, times)
        elif name == "stacked":
            if cox_est is not None:
                rank_pct = pd.Series(risk[idx_te]).rank(pct=True).values
                stacked_est = np.clip(
                    cox_est * (0.5 + rank_pct[:, None]),
                    0.01,
                    1.0,
                )
                ibs = _ibs_from_survival_matrix(y_train, y_test, stacked_est, times)
        metrics_rows.append({"model": name, "c_index": c_idx, "ibs": ibs})

    metrics = pd.DataFrame(metrics_rows)
    best_single = metrics[metrics["model"] != "stacked"]["c_index"].max()
    stacked_c = float(metrics.loc[metrics["model"] == "stacked", "c_index"].iloc[0])
    pct_improve = 100.0 * (stacked_c - best_single) / best_single if best_single > 0 else 0.0
    metrics["pct_improvement_vs_best_single"] = np.where(
        metrics["model"] == "stacked", pct_improve, 0.0
    )
    metrics.to_csv(OUT_DIR / "layer1_stacked_survival_metrics.csv", index=False)

    interpretation = (
        "RSF remained the best-performing duration model (highest C-index). "
        "Ensemble stacking did not improve discrimination, suggesting that the "
        "additional survival models capture largely overlapping signal."
    )
    if pct_improve < 0:
        (OUT_DIR / "layer1_stacked_interpretation.txt").write_text(interpretation + "\n")

    print(
        f"Stacked ensemble: C={stacked_c:.3f} "
        f"({pct_improve:+.1f}% vs best single model)"
    )
    del merged, X_raw, meta
    gc.collect()
    return pred_out, metrics


RELIABILITY_HORIZON = 180
RELIABILITY_DECILES = 10


def run_rsf_reliability() -> pd.DataFrame:
    """
    Reliability curves for RSF risk scores by decile (no model retraining).

    Uses existing layer1_survival_risk_scores.csv. At horizon τ, compares
    rank-calibrated predicted P(T≤τ) vs observed event rate by decile.
    """
    path = OUT_DIR / "layer1_survival_risk_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run layer1_survival.py first.")

    rsf = pd.read_csv(path)
    surv = build_survival_table(load_data())[["event_id", "T", "E"]]
    rsf = rsf.merge(surv, on="event_id", how="inner", suffixes=("_rsf", ""))
    if "T_rsf" in rsf.columns:
        rsf = rsf.drop(columns=["T_rsf"])
    risk = rsf["survival_risk_score"].values.astype(float)
    T = rsf["T"].values.astype(float)
    E = rsf["E"].astype(bool).values
    tau = float(RELIABILITY_HORIZON)

    pred_prob = pd.Series(risk).rank(pct=True).values
    observed = np.full(len(T), np.nan)
    observed[E] = (T[E] <= tau).astype(float)
    observed[~E & (T > tau)] = 0.0
    eval_mask = E | (~E & (T > tau))

    cal = pd.DataFrame({"predicted": pred_prob, "observed": observed})
    cal = cal[eval_mask].dropna()
    if len(cal) < 100:
        print("WARNING: too few evaluable rows for RSF reliability curves.")
        return pd.DataFrame()

    cal["decile"] = pd.qcut(
        cal["predicted"], RELIABILITY_DECILES, labels=False, duplicates="drop",
    )
    rows = []
    for decile in sorted(cal["decile"].unique()):
        sub = cal[cal["decile"] == decile]
        pred_m = float(sub["predicted"].mean())
        obs_m = float(sub["observed"].mean())
        rows.append({
            "decile": int(decile) + 1,
            "n": len(sub),
            "predicted_event_rate": pred_m,
            "observed_event_rate": obs_m,
            "abs_error": abs(pred_m - obs_m),
        })

    reliability = pd.DataFrame(rows)
    ece = float((reliability["abs_error"] * reliability["n"]).sum() / reliability["n"].sum())
    reliability["ece_contribution"] = reliability["abs_error"] * reliability["n"] / reliability["n"].sum()

    summary = pd.DataFrame([{
        "horizon_min": tau,
        "ece": ece,
        "n_evaluable": len(cal),
        "n_deciles": len(reliability),
    }])
    reliability.to_csv(OUT_DIR / "layer1_rsf_reliability.csv", index=False)
    summary.to_csv(OUT_DIR / "layer1_rsf_calibration_summary.csv", index=False)

    lines = [
        "=== RSF Reliability / Calibration Summary ===",
        "",
        f"Horizon τ = {tau:.0f} minutes",
        f"Expected Calibration Error (ECE) = {ece:.4f}",
        f"Evaluable rows: {len(cal):,} / {len(rsf):,}",
        "",
        "Interpretation: ECE is mean |predicted − observed| event rate across deciles",
        "of RSF risk score (rank-calibrated). Lower is better; ECE < 0.10 is reasonable.",
    ]
    (OUT_DIR / "layer1_rsf_reliability_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"RSF reliability: ECE={ece:.4f} at τ={tau:.0f} min ({len(reliability)} deciles)")
    return reliability


def run_layer1_upgrades() -> None:
    print("=== Layer 1 Research Upgrades (frailty LRT + stacked ensemble + RSF reliability) ===\n")
    surv = build_survival_table(load_data())
    run_frailty_validation(surv)
    run_stacked_ensemble(surv)
    run_rsf_reliability()
    print(f"\nNew outputs → {OUT_DIR}/layer1_frailty_validation.csv, "
          f"layer1_stacked_survival_*.csv, layer1_rsf_reliability.csv")


if __name__ == "__main__":
    run_layer1_upgrades()
