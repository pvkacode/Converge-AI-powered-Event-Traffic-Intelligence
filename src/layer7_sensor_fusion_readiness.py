"""
Layer 7 — M7 Part B: Sensor Fusion Readiness Engine.

Estimates how much VALUE future sensor inputs would add per active site (where
would new sensors most reduce operational uncertainty). Read-only; no model, no
ingestion.

    SVI = 0.40*uncertainty + 0.30*operational_risk + 0.20*alert_density + 0.10*drift_pressure
Higher SVI => greater value from future sensors. SVI in [0,1].

ADDITIVE ONLY. Writes only outputs/layer7_sensor_fusion_readiness.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_W = {"uncertainty": 0.40, "operational_risk": 0.30, "alert_density": 0.20, "drift_pressure": 0.10}


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _drift_pressure() -> float:
    dr = _read("layer6_drift_report.csv")
    if not len(dr):
        return 0.0
    n_alert = int(dr["alert"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    frac = n_alert / max(1, len(dr))
    urg = float(np.clip(pd.to_numeric(dr.get("retrain_urgency"), errors="coerce").max(), 0, 1))
    return float(np.clip(frac * urg, 0, 1))


def build_svi() -> pd.DataFrame:
    astate = _read("layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    dcs = _read("layer7_decision_confidence.csv")
    dcs["event_id"] = dcs["event_id"].astype(str)
    alerts = _read("layer7_prioritized_alerts.csv")

    # uncertainty = 1 - uncertainty_component (component = 1 - normalized_uncertainty)
    unc_map = (dict(zip(dcs["event_id"],
                        (1.0 - pd.to_numeric(dcs["uncertainty_component"], errors="coerce").fillna(0)).clip(0, 1)))
               if len(dcs) else {})
    # alert density per site
    if len(alerts):
        alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
        ncount = alerts.groupby("affected_event_id").size()
        amax = max(1, int(ncount.max()))
    else:
        ncount, amax = pd.Series(dtype=int), 1
    drift_pressure = _drift_pressure()

    df = astate[["event_id", "operational_risk_score"]].copy()
    df["uncertainty"] = df["event_id"].map(unc_map).fillna(0.0).clip(0, 1)
    df["operational_risk"] = (pd.to_numeric(df["operational_risk_score"], errors="coerce")
                              .fillna(0) / 100.0).clip(0, 1)
    df["alert_density"] = df["event_id"].map(
        lambda e: min(1.0, int(ncount.get(e, 0)) / amax)).fillna(0.0)
    df["drift_pressure"] = drift_pressure

    df["sensor_value_index"] = (
        _W["uncertainty"] * df["uncertainty"]
        + _W["operational_risk"] * df["operational_risk"]
        + _W["alert_density"] * df["alert_density"]
        + _W["drift_pressure"] * df["drift_pressure"]
    ).clip(0, 1)

    pct = df["sensor_value_index"].rank(pct=True, method="average")
    df["svi_tier"] = np.where(pct >= 2 / 3, "High", np.where(pct >= 1 / 3, "Moderate", "Low"))
    df["generated_at"] = _NOW_ISO
    cols = ["event_id", "sensor_value_index", "svi_tier", "uncertainty",
            "operational_risk", "alert_density", "drift_pressure", "generated_at"]
    return df[cols].sort_values("sensor_value_index", ascending=False).reset_index(drop=True)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_svi()
    if write:
        df.to_csv(OUT / "layer7_sensor_fusion_readiness.csv", index=False)
    s = df["sensor_value_index"]
    checks = [{
        "check_id": "m7_svi_bounded", "phase": "sensor_fusion_readiness",
        "passed": bool(((s >= 0) & (s <= 1)).all()) and int(df.isna().sum().sum()) == 0,
        "detail": f"SVI range [{s.min():.4f}, {s.max():.4f}]; tiers={df['svi_tier'].value_counts().to_dict()}; "
                  f"{int(df.isna().sum().sum())} NaN",
        "severity": "critical" if not ((s >= 0) & (s <= 1)).all() else "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
