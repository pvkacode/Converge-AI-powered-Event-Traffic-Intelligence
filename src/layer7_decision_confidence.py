"""
Layer 7 — M5 Part A: Decision Confidence Service.

Deterministic. NO model, NO retraining. Combines existing uncertainty,
robustness, and reliability signals into a per-active-site Decision Confidence
Score (DCS):

    DCS = (1 - normalized_uncertainty) * robustness_score * duration_reliability

All three factors are in [0,1], so DCS in [0,1]. Tiers (High/Moderate/Low) are
assigned by terciles of the DCS distribution (rank-based, consistent with the M1
ORS / M3 active-tier convention), guaranteeing populated tiers.

NOTE on uncertainty: layer6_posterior_uncertainty.csv is keyed by
stratum_cause x stratum_corridor, NOT event_id. Uncertainty is therefore mapped
to each event via its event_cause (mean posterior CI width across that cause's
strata), normalized against the GLOBAL stratum CI-width range (fixed anchor, not
the active batch). Causes absent from the posterior table fall back to the global
mean width.

ADDITIVE ONLY. Writes only outputs/layer7_decision_confidence.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def _uncertainty_by_cause() -> tuple[dict, float, float, float]:
    pu = pd.read_csv(OUT / "layer6_posterior_uncertainty.csv")
    w = pd.to_numeric(pu["ci95_width_log"], errors="coerce")
    gmin, gmax = float(w.min()), float(w.max())
    gmean = float(w.mean())
    by_cause = pu.groupby("stratum_cause")["ci95_width_log"].mean().to_dict()
    return by_cause, gmin, gmax, gmean


def build_decision_confidence() -> pd.DataFrame:
    astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)
    dq = pd.read_csv(OUT / "layer45_duration_quality.csv")
    dq["event_id"] = dq["event_id"].astype(str)

    by_cause, gmin, gmax, gmean = _uncertainty_by_cause()
    span = max(gmax - gmin, 1e-9)

    df = astate[["event_id", "event_cause", "robustness_score"]].copy()
    # reliability per event (prefer L4.5 duration_quality; fallback to L5 allocation)
    df = df.merge(dq[["event_id", "duration_reliability"]], on="event_id", how="left")
    if df["duration_reliability"].isna().any():
        rel5 = alloc[["event_id", "duration_reliability"]].rename(
            columns={"duration_reliability": "rel5"})
        df = df.merge(rel5, on="event_id", how="left")
        df["duration_reliability"] = df["duration_reliability"].fillna(df["rel5"])
        df = df.drop(columns=["rel5"])

    # uncertainty per event via cause
    cause_width = df["event_cause"].map(by_cause).fillna(gmean)
    norm_unc = ((cause_width - gmin) / span).clip(0, 1)
    df["uncertainty_component"] = (1.0 - norm_unc).clip(0, 1)
    df["robustness_component"] = pd.to_numeric(
        df["robustness_score"], errors="coerce").fillna(0.0).clip(0, 1)
    df["reliability_component"] = pd.to_numeric(
        df["duration_reliability"], errors="coerce").fillna(0.0).clip(0, 1)

    df["decision_confidence_score"] = (
        df["uncertainty_component"]
        * df["robustness_component"]
        * df["reliability_component"]
    ).clip(0, 1)

    # tercile tiers (rank-based) -> always populated
    pct = df["decision_confidence_score"].rank(pct=True, method="average")
    df["decision_confidence_tier"] = np.where(
        pct >= 2 / 3, "High", np.where(pct >= 1 / 3, "Moderate", "Low"))
    df["generated_at"] = _NOW_ISO

    cols = [
        "event_id", "decision_confidence_score", "decision_confidence_tier",
        "uncertainty_component", "robustness_component", "reliability_component",
        "generated_at",
    ]
    return df[cols].sort_values(
        "decision_confidence_score", ascending=False).reset_index(drop=True)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_decision_confidence()
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / "layer7_decision_confidence.csv", index=False)
    return df, _validate(df)


def _validate(df: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []

    def chk(cid, passed, detail, severity="critical"):
        checks.append({"check_id": cid, "phase": "decision_confidence",
                       "passed": bool(passed), "detail": detail,
                       "severity": "info" if passed else severity})

    s = df["decision_confidence_score"]
    chk("m5_dcs_range", bool(((s >= 0) & (s <= 1)).all()),
        f"DCS range [{s.min():.4f}, {s.max():.4f}]")
    chk("m5_dcs_no_nan", int(df.isna().sum().sum()) == 0,
        f"{int(df.isna().sum().sum())} NaN in DCS output")
    tiers = set(df["decision_confidence_tier"].unique())
    chk("m5_dcs_tiers_present", len(tiers) >= 1 and df["decision_confidence_tier"].notna().all(),
        f"tier distribution: {df['decision_confidence_tier'].value_counts().to_dict()}")
    return checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print("=== Layer 7 M5 Decision Confidence ===")
    print(df.head().to_string(index=False))
    print("tiers:", df["decision_confidence_tier"].value_counts().to_dict())
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
