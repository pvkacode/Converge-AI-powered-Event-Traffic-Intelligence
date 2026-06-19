"""
Layer 6 – Drift Detection

Runs three complementary tests on log-duration between the Nov–Feb baseline
and the Mar–Apr feedback batch:

1. Page-Hinkley (PH) test — sequential change-point detection on the ordered
   feedback time series.  Alert when PH_t − min_PH > lambda_ph.

2. Population Stability Index (PSI) — distributional shift between baseline
   and feedback.  Thresholds: PSI < 0.10 stable, 0.10–0.25 moderate, > 0.25 critical.

3. Operational Drift Score (ODS) — computed per calendar week in the feedback
   window:  ODS_t = |mu_t − mu_{t−1}| / sigma_{t−1}

All findings are emitted as recommendation records only.
No upstream files are ever modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PH_DELTA      = 0.02    # sensitivity: expected minimal shift magnitude
PH_LAMBDA     = 5.0     # detection threshold for PH statistic
PSI_MODERATE  = 0.10
PSI_CRITICAL  = 0.25
N_PSI_BINS    = 10
ODS_ALERT     = 1.5     # z-score threshold for week-over-week ODS alert


# ---------------------------------------------------------------------------
# 1. Page-Hinkley test
# ---------------------------------------------------------------------------

def page_hinkley(
    series: np.ndarray,
    mu_baseline: float,
    delta: float = PH_DELTA,
    lambda_: float = PH_LAMBDA,
) -> dict:
    """
    One-sided upward Page-Hinkley test.

    Returns a dict with:
        - ph_series     : list of PH_t values
        - min_series    : running minimum
        - alert_indices : indices where PH_t − min_PH > lambda_
        - first_alert   : index of first alert (or None)
        - max_ph        : peak PH value
    """
    n = len(series)
    ph   = np.zeros(n)
    mn   = np.zeros(n)
    alerts = []

    cumsum = 0.0
    current_min = 0.0

    for t, x in enumerate(series):
        cumsum     += (x - mu_baseline) - delta
        ph[t]       = cumsum
        current_min = min(current_min, ph[t])
        mn[t]       = current_min
        if ph[t] - current_min > lambda_:
            alerts.append(t)

    return {
        "ph_series":     ph.tolist(),
        "min_series":    mn.tolist(),
        "alert_indices": alerts,
        "first_alert":   int(alerts[0]) if alerts else None,
        "max_ph":        float(ph.max()),
        "ph_lambda":     lambda_,
        "ph_delta":      delta,
    }


# ---------------------------------------------------------------------------
# 2. Population Stability Index
# ---------------------------------------------------------------------------

def compute_psi(
    baseline: np.ndarray,
    feedback: np.ndarray,
    n_bins: int = N_PSI_BINS,
) -> tuple[float, pd.DataFrame]:
    """
    Compute PSI between baseline and feedback distributions.

    Returns (psi_total, bin_df) where bin_df has per-bin breakdown.
    """
    # Define bin edges on baseline distribution (percentiles)
    edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    base_counts = np.histogram(baseline, bins=edges)[0]
    feed_counts = np.histogram(feedback, bins=edges)[0]

    base_pct = base_counts / max(len(baseline), 1)
    feed_pct = feed_counts / max(len(feedback), 1)

    # Avoid log(0): replace zeros with small epsilon
    eps = 1e-6
    base_pct = np.clip(base_pct, eps, None)
    feed_pct = np.clip(feed_pct, eps, None)

    psi_per_bin = (feed_pct - base_pct) * np.log(feed_pct / base_pct)
    psi_total   = float(psi_per_bin.sum())

    bin_df = pd.DataFrame({
        "bin":             range(n_bins),
        "edge_lo":         edges[:-1],
        "edge_hi":         edges[1:],
        "baseline_pct":    base_pct,
        "feedback_pct":    feed_pct,
        "psi_contribution": psi_per_bin,
    })

    return psi_total, bin_df


# ---------------------------------------------------------------------------
# 3. Operational Drift Score (weekly ODS)
# ---------------------------------------------------------------------------

def weekly_ods(fb_df: pd.DataFrame, log_col: str = "log_duration") -> pd.DataFrame:
    """
    Compute ODS_t = |mu_t − mu_{t−1}| / sigma_{t−1} for each calendar week
    in the feedback batch.
    """
    df = fb_df[[log_col, "start_local"]].dropna().copy()
    df["week"] = df["start_local"].dt.isocalendar().week.astype(int)
    df["year"] = df["start_local"].dt.isocalendar().year.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    weekly = (
        df.groupby("year_week")[log_col]
        .agg(mu="mean", sigma="std", n="count")
        .reset_index()
        .sort_values("year_week")
    )
    weekly["sigma"] = weekly["sigma"].fillna(1e-6).clip(lower=1e-6)

    ods_values = [None]
    for i in range(1, len(weekly)):
        mu_prev    = weekly.iloc[i - 1]["mu"]
        sig_prev   = weekly.iloc[i - 1]["sigma"]
        mu_curr    = weekly.iloc[i]["mu"]
        ods_values.append(abs(mu_curr - mu_prev) / sig_prev)

    weekly["ods"] = ods_values
    weekly["ods_alert"] = weekly["ods"].apply(
        lambda v: bool(v is not None and v > ODS_ALERT)
    )
    return weekly


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def run_drift_detection(
    prior_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run the full drift detection suite.

    Parameters
    ----------
    prior_df    : Nov–Feb uncensored observations (must have log_duration, start_local)
    feedback_df : Mar–Apr uncensored observations (same)
    features    : additional numeric feature columns to run PSI on

    Returns
    -------
    drift_df : tidy DataFrame with one row per test × variable, containing
               drift scores, flags, severity, and recommendation text.
    """
    records: list[dict] = []

    # ---- restrict to uncensored with valid duration ----
    pr  = prior_df[prior_df["log_duration"].notna()].copy()
    fb  = feedback_df[feedback_df["log_duration"].notna()].copy()
    fb  = fb.sort_values("start_local").reset_index(drop=True)

    mu_baseline    = float(pr["log_duration"].mean())
    sigma_baseline = float(pr["log_duration"].std())

    # ---- 1. Page-Hinkley on log-duration sequence ----
    if len(fb) > 0:
        ph_result = page_hinkley(fb["log_duration"].values, mu_baseline)
        n_alerts  = len(ph_result["alert_indices"])
        severity  = "none"
        if n_alerts > 0:
            severity = "critical" if ph_result["max_ph"] > 2 * PH_LAMBDA else "moderate"
        records.append({
            "test":        "page_hinkley",
            "variable":    "log_duration",
            "score":       round(ph_result["max_ph"], 4),
            "threshold":   PH_LAMBDA,
            "alert":       n_alerts > 0,
            "severity":    severity,
            "n_alerts":    n_alerts,
            "first_alert_idx": ph_result["first_alert"],
            "detail":      f"max_PH={ph_result['max_ph']:.3f}, "
                           f"first_alert_at_index={ph_result['first_alert']}",
            "recommendation": (
                "RETRAIN_TRIGGER: Page-Hinkley detects shift in log-duration sequence. "
                "Recommend updating Layer 4.5 duration priors."
                if n_alerts > 0 else
                "No sequential drift detected in log-duration."
            ),
        })
    else:
        records.append({
            "test": "page_hinkley", "variable": "log_duration",
            "score": 0.0, "threshold": PH_LAMBDA, "alert": False,
            "severity": "none", "n_alerts": 0, "first_alert_idx": None,
            "detail": "no uncensored feedback observations",
            "recommendation": "Insufficient feedback data for PH test.",
        })

    # ---- 2. PSI on log-duration ----
    if len(pr) > 0 and len(fb) > 0:
        psi_total, _ = compute_psi(pr["log_duration"].values, fb["log_duration"].values)
        psi_sev = (
            "critical" if psi_total >= PSI_CRITICAL else
            "moderate" if psi_total >= PSI_MODERATE else "none"
        )
        records.append({
            "test":        "psi",
            "variable":    "log_duration",
            "score":       round(psi_total, 4),
            "threshold":   PSI_CRITICAL,
            "alert":       psi_total >= PSI_MODERATE,
            "severity":    psi_sev,
            "n_alerts":    int(psi_total >= PSI_MODERATE),
            "first_alert_idx": None,
            "detail":      f"PSI={psi_total:.4f} (baseline n={len(pr)}, feedback n={len(fb)})",
            "recommendation": (
                f"RETRAIN_TRIGGER: PSI={psi_total:.4f} indicates "
                f"{'critical' if psi_total >= PSI_CRITICAL else 'moderate'} "
                "distributional shift in log-duration. "
                "Recommend Layer 4.5 recalibration."
                if psi_total >= PSI_MODERATE else
                f"PSI={psi_total:.4f} — distribution stable."
            ),
        })

    # ---- 3. PSI on additional features ----
    if features:
        for feat in features:
            pr_feat = pr[feat].dropna().values if feat in pr.columns else np.array([])
            fb_feat = fb[feat].dropna().values if feat in fb.columns else np.array([])
            if len(pr_feat) < 5 or len(fb_feat) < 5:
                continue
            psi_f, _ = compute_psi(pr_feat, fb_feat)
            psi_sev  = (
                "critical" if psi_f >= PSI_CRITICAL else
                "moderate" if psi_f >= PSI_MODERATE else "none"
            )
            records.append({
                "test":         "psi",
                "variable":     feat,
                "score":        round(psi_f, 4),
                "threshold":    PSI_CRITICAL,
                "alert":        psi_f >= PSI_MODERATE,
                "severity":     psi_sev,
                "n_alerts":     int(psi_f >= PSI_MODERATE),
                "first_alert_idx": None,
                "detail":       f"PSI={psi_f:.4f}",
                "recommendation": (
                    f"RETRAIN_TRIGGER: PSI={psi_f:.4f} on '{feat}' indicates "
                    f"{psi_sev} distributional shift."
                    if psi_f >= PSI_MODERATE else
                    f"PSI={psi_f:.4f} on '{feat}' — stable."
                ),
            })

    # ---- 4. ODS (weekly) ----
    if len(fb) > 0:
        weekly = weekly_ods(fb)
        ods_alerts = weekly[weekly["ods_alert"] == True]  # noqa: E712
        max_ods    = float(weekly["ods"].dropna().max()) if len(weekly) > 1 else 0.0
        ods_sev    = "critical" if max_ods > 2 * ODS_ALERT else ("moderate" if max_ods > ODS_ALERT else "none")
        records.append({
            "test":        "ods_weekly",
            "variable":    "log_duration",
            "score":       round(max_ods, 4),
            "threshold":   ODS_ALERT,
            "alert":       len(ods_alerts) > 0,
            "severity":    ods_sev,
            "n_alerts":    len(ods_alerts),
            "first_alert_idx": None,
            "detail":      (
                f"max_ODS={max_ods:.3f} across {len(weekly)} weeks; "
                f"alert weeks: {ods_alerts['year_week'].tolist()}"
            ),
            "recommendation": (
                f"RETRAIN_TRIGGER: ODS={max_ods:.3f} — week-over-week mean shift "
                "exceeds threshold. Recommend reviewing duration priors."
                if len(ods_alerts) > 0 else
                f"ODS={max_ods:.3f} — no week-over-week drift detected."
            ),
        })

    # ---- 5. Mean-shift z-test ----
    if len(pr) > 0 and len(fb) > 0:
        mu_fb   = float(fb["log_duration"].mean())
        se      = sigma_baseline / np.sqrt(len(fb))
        z_score = abs(mu_fb - mu_baseline) / max(se, 1e-9)
        alert   = z_score > 2.0
        records.append({
            "test":        "mean_shift_ztest",
            "variable":    "log_duration",
            "score":       round(z_score, 4),
            "threshold":   2.0,
            "alert":       alert,
            "severity":    "critical" if z_score > 3.0 else ("moderate" if alert else "none"),
            "n_alerts":    int(alert),
            "first_alert_idx": None,
            "detail":      (
                f"baseline_mu={mu_baseline:.3f}, feedback_mu={mu_fb:.3f}, "
                f"z={z_score:.3f}"
            ),
            "recommendation": (
                f"RETRAIN_TRIGGER: Mean log-duration shifted by z={z_score:.2f} "
                "standard errors. Layer 4.5 global prior should be updated."
                if alert else
                f"Mean shift z={z_score:.2f} — within acceptable range."
            ),
        })

    drift_df = pd.DataFrame(records)
    return drift_df
