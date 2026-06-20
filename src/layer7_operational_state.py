"""
Layer 7 — Phase 2: Operational State Engine.

Deterministic. NO machine learning. NO retraining. Consumes only existing
L4.5 (normalized JOSV), L5 (resource_allocation, chance_constraint_violations)
and L6 outputs.

Operational Risk Score (ORS):
    ORS_base = sigmoid( 0.30*tail_risk_prob_z + 0.20*fragility_signal_z
                      + 0.15*obi_signal_z     + 0.15*drift_score_z
                      + 0.10*novelty_score_z  + 0.10*critical_alert_indicator )
    ORS = 100 * ORS_base
              * (1 - 0.50*robustness_score)            # L5 discount (where available)
              * (1 - 0.30*(1 - duration_reliability))  # L4.5 discount

Tiers (Normal / Elevated / Critical / Emergency) by percentile ranking.

Outputs:
    outputs/layer7_operational_state.csv
    outputs/layer7_site_risk_ranking.csv
    outputs/layer7_state_summary.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import (
    JOSV_NORMALIZED,
    ORS_PCT_SCALE,
    ORS_RELIABILITY_DISCOUNT,
    ORS_ROBUSTNESS_DISCOUNT,
    ORS_TIER_CUTS,
    ORS_WEIGHTS,
    OUT,
)
from layer7_loader import Store

_NOW = datetime.now(timezone.utc).isoformat()

_Z_COLS = [
    "tail_risk_prob_z",
    "fragility_signal_z",
    "obi_signal_z",
    "drift_score_z",
    "novelty_score_z",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _assign_tier(percentile: np.ndarray) -> np.ndarray:
    """percentile in [0,1] -> tier label using ORS_TIER_CUTS (descending cuts)."""
    out = np.empty(len(percentile), dtype=object)
    out[:] = "Normal"
    # apply from lowest to highest so the highest cut wins
    for label, cut in sorted(ORS_TIER_CUTS, key=lambda t: t[1]):
        out[percentile >= cut] = label
    return out


def build_operational_state(store: Store) -> pd.DataFrame:
    josv = store.get(JOSV_NORMALIZED)
    if josv is None or len(josv) == 0:
        raise RuntimeError("Operational State Engine: JOSV normalized input unavailable")

    df = josv.copy()

    # --- z-score inputs (defensive: fill non-finite with 0 = neutral, flag coverage)
    coverage_missing = np.zeros(len(df), dtype=int)
    for c in _Z_COLS:
        if c not in df.columns:
            df[c] = 0.0
            coverage_missing += 1
        else:
            col = pd.to_numeric(df[c], errors="coerce")
            coverage_missing += col.isna().to_numpy().astype(int)
            df[c] = col.fillna(0.0)

    # --- critical_alert_indicator: event-keyed critical signal.
    # Among M1 alert sources, only L5 chance-constraint violations are event-keyed;
    # system-level critical alerts (drift/global) are handled by the Alert engine and
    # would not differentiate sites, so they are intentionally excluded here.
    ccv = store.get("layer5_chance_constraint_violations.csv")
    critical_events: set = set()
    if ccv is not None and len(ccv) > 0 and {"event_id", "violation_flag"} <= set(ccv.columns):
        flag = (
            ccv["violation_flag"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        )
        critical_events = set(ccv.loc[flag, "event_id"].astype(str))
    df["critical_alert_indicator"] = (
        df["event_id"].astype(str).isin(critical_events).astype(float)
    )

    # --- PATCH F-001: percentile-rank normalization before weighting.
    # Each z feature -> within-population percentile in (0,1], centred to [-0.5,0.5].
    # This removes the variance-driven dominance of tail_risk_prob_z (was ~94%): every
    # feature now enters on a uniform scale so the published weights actually govern
    # influence. The raw z columns are still emitted unchanged (schema-preserving).
    centred = {}
    for c in _Z_COLS:
        pctc = df[c].rank(pct=True, method="average") - 0.5
        centred[c] = pctc.to_numpy(dtype=float)
    logit = ORS_PCT_SCALE * (
        ORS_WEIGHTS["tail_risk_prob_z"] * centred["tail_risk_prob_z"]
        + ORS_WEIGHTS["fragility_signal_z"] * centred["fragility_signal_z"]
        + ORS_WEIGHTS["obi_signal_z"] * centred["obi_signal_z"]
        + ORS_WEIGHTS["drift_score_z"] * centred["drift_score_z"]
        + ORS_WEIGHTS["novelty_score_z"] * centred["novelty_score_z"]
    ) + ORS_WEIGHTS["critical_alert_indicator"] * df["critical_alert_indicator"].to_numpy(dtype=float)
    ors_base = _sigmoid(logit)

    # --- L5 robustness discount (left-join; only the active sites carry it)
    alloc = store.get("layer5_resource_allocation.csv")
    if alloc is not None and "event_id" in alloc.columns:
        keep = [c for c in ["event_id", "robustness_score", "service_tier"] if c in alloc.columns]
        a = alloc[keep].drop_duplicates("event_id").copy()
        a["event_id"] = a["event_id"].astype(str)
        df["event_id"] = df["event_id"].astype(str)
        df = df.merge(a, on="event_id", how="left", suffixes=("", "_l5"))
    if "robustness_score" not in df.columns:
        df["robustness_score"] = np.nan
    if "service_tier" not in df.columns:
        df["service_tier"] = np.nan

    df["in_layer5_flag"] = df["robustness_score"].notna()
    robustness = pd.to_numeric(df["robustness_score"], errors="coerce").fillna(0.0).clip(0, 1)
    reliability = pd.to_numeric(df.get("duration_reliability"), errors="coerce").fillna(1.0).clip(0, 1)

    robustness_factor = 1.0 - ORS_ROBUSTNESS_DISCOUNT * robustness
    reliability_factor = 1.0 - ORS_RELIABILITY_DISCOUNT * (1.0 - reliability)

    # --- PATCH F-014: Layer 6 integrity gate -> bounded quality down-weight on ORS.
    qpath = OUT / "layer7_quality_gate.csv"
    if qpath.exists():
        qg = pd.read_csv(qpath)[["event_id", "quality_weight", "data_quality_flag"]].copy()
        qg["event_id"] = qg["event_id"].astype(str)
        df = df.merge(qg, on="event_id", how="left")
    if "quality_weight" not in df.columns:
        df["quality_weight"] = 1.0
        df["data_quality_flag"] = "unknown"
    df["quality_weight"] = pd.to_numeric(df["quality_weight"], errors="coerce").fillna(1.0).clip(0, 1)
    df["data_quality_flag"] = df["data_quality_flag"].fillna("unknown")
    quality_factor = df["quality_weight"].to_numpy(dtype=float)

    df["operational_risk_score"] = (
        100.0 * ors_base * robustness_factor * reliability_factor * quality_factor
    )
    df["ors_base"] = ors_base
    df["robustness_factor"] = robustness_factor
    df["reliability_factor"] = reliability_factor
    df["quality_factor"] = quality_factor
    df["coverage_missing_signals"] = coverage_missing
    df["coverage_flag"] = coverage_missing > 0

    # --- percentile rank + tier
    pct = df["operational_risk_score"].rank(pct=True, method="average").to_numpy()
    df["ors_percentile"] = pct
    df["operational_tier"] = _assign_tier(pct)

    # --- component contributions for explainability (PATCH F-001: percentile-centred,
    # logit-space). Shares = |contrib_i| / sum|contrib_i| (scale cancels in the ratio).
    for c in _Z_COLS:
        df[f"contrib_{c}"] = ORS_WEIGHTS[c] * centred[c]
    df["contrib_critical_alert_indicator"] = (
        ORS_WEIGHTS["critical_alert_indicator"] * df["critical_alert_indicator"]
    )
    df["generated_at"] = _NOW

    out_cols = [
        "event_id", "event_cause", "operational_risk_score", "ors_percentile",
        "operational_tier", "ors_base", "robustness_factor", "reliability_factor",
        "tail_risk_prob_z", "fragility_signal_z", "obi_signal_z", "drift_score_z",
        "novelty_score_z", "critical_alert_indicator", "duration_reliability",
        "robustness_score", "service_tier", "in_layer5_flag",
        "quality_factor", "data_quality_flag",
        "coverage_missing_signals", "coverage_flag",
        "contrib_tail_risk_prob_z", "contrib_fragility_signal_z", "contrib_obi_signal_z",
        "contrib_drift_score_z", "contrib_novelty_score_z",
        "contrib_critical_alert_indicator", "generated_at",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].copy()


def build_ranking(state: pd.DataFrame) -> pd.DataFrame:
    rank = state.sort_values(
        ["operational_risk_score", "event_id"], ascending=[False, True]
    ).reset_index(drop=True)
    rank.insert(0, "risk_rank", np.arange(1, len(rank) + 1))
    cols = [
        "risk_rank", "event_id", "event_cause", "operational_risk_score",
        "ors_percentile", "operational_tier", "service_tier", "in_layer5_flag",
        "critical_alert_indicator", "generated_at",
    ]
    return rank[[c for c in cols if c in rank.columns]].copy()


def build_summary(state: pd.DataFrame) -> pd.DataFrame:
    s = state["operational_risk_score"]
    rows = [
        {"metric": "n_sites", "value": int(len(state))},
        {"metric": "n_in_layer5", "value": int(state["in_layer5_flag"].sum())},
        {"metric": "ors_mean", "value": round(float(s.mean()), 6)},
        {"metric": "ors_std", "value": round(float(s.std(ddof=0)), 6)},
        {"metric": "ors_variance", "value": round(float(s.var(ddof=0)), 6)},
        {"metric": "ors_min", "value": round(float(s.min()), 6)},
        {"metric": "ors_max", "value": round(float(s.max()), 6)},
        {"metric": "n_nan_ors", "value": int(s.isna().sum())},
        {"metric": "n_coverage_flagged", "value": int(state["coverage_flag"].sum())},
    ]
    for label, _ in ORS_TIER_CUTS:
        rows.append({
            "metric": f"tier_{label.lower()}_count",
            "value": int((state["operational_tier"] == label).sum()),
        })
    rows.append({"metric": "generated_at", "value": _NOW})
    return pd.DataFrame(rows)


def run(store: Store, write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    state = build_operational_state(store)
    ranking = build_ranking(state)
    summary = build_summary(state)

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        state.to_csv(OUT / "layer7_operational_state.csv", index=False)
        ranking.to_csv(OUT / "layer7_site_risk_ranking.csv", index=False)
        summary.to_csv(OUT / "layer7_state_summary.csv", index=False)

    checks = _validate(state)
    return {"state": state, "ranking": ranking, "summary": summary}, checks


def _validate(state: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []

    def chk(cid: str, passed: bool, detail: str, severity: str = "critical") -> None:
        checks.append({
            "check_id": cid, "phase": "operational_state", "passed": bool(passed),
            "detail": detail, "severity": "info" if passed else severity,
        })

    tiers_present = set(state["operational_tier"].unique())
    expected = {label for label, _ in ORS_TIER_CUTS}
    chk("state_no_missing_tiers", expected <= tiers_present,
        f"tiers present: {sorted(tiers_present)} (expected {sorted(expected)})")

    n_nan = int(state["operational_risk_score"].isna().sum())
    chk("state_no_nan_risk_scores", n_nan == 0, f"{n_nan} NaN ORS values")

    var = float(state["operational_risk_score"].var(ddof=0))
    chk("state_nonzero_variance", var > 1e-9, f"ORS variance = {var:.6g}")

    smin = float(state["operational_risk_score"].min())
    smax = float(state["operational_risk_score"].max())
    chk("state_score_in_range", smin >= 0 and smax <= 100,
        f"ORS range [{smin:.3f}, {smax:.3f}]")

    return checks


if __name__ == "__main__":
    from layer7_loader import audit_inputs

    _store, _ = audit_inputs(write=False)
    _tables, _checks = run(_store, write=True)
    print("=== Layer 7 Operational State Engine ===")
    print(_tables["summary"].to_string(index=False))
    for c in _checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
