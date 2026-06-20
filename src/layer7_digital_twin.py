"""
Layer 7 — M6 Part A: Digital Twin Core.

A virtual operational copy of the current traffic-control state, assembled (read-only)
from existing Layer 5 / Layer 6 / Layer 7 outputs. NO optimization, NO retraining,
NO model. Simulation layer only.

DigitalTwinState (one per active site) snapshots: operational risk, active tier,
decision confidence, allocated resources, diversion status, alert state, override
state, robustness.

ADDITIVE ONLY. Writes only outputs/layer7_digital_twin_state.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


class DigitalTwinState(NamedTuple):
    event_id: str
    operational_risk_score: float
    active_operational_tier: str
    decision_confidence_score: float
    allocated_resources: str
    diversion_status: str
    alert_state: str
    override_state: str
    robustness_score: float
    generated_at: str


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_twin_state() -> pd.DataFrame:
    astate = _read("layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    tiers = _read("layer7_active_site_tiers.csv")
    tier_map = (dict(zip(tiers["event_id"].astype(str), tiers["active_operational_tier"]))
                if len(tiers) else {})
    dcs = _read("layer7_decision_confidence.csv")
    dcs_map = (dict(zip(dcs["event_id"].astype(str), dcs["decision_confidence_score"]))
               if len(dcs) else {})
    alloc = _read("layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)

    # alert state per event
    alerts = _read("layer7_prioritized_alerts.csv")
    alert_state: dict[str, str] = {}
    if len(alerts):
        alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
        for eid, grp in alerts.groupby("affected_event_id"):
            if eid and eid.lower() != "nan":
                top = grp.sort_values("alert_severity_score", ascending=False).iloc[0]
                alert_state[eid] = f"{top['priority']} ({len(grp)})"

    # override state per event (from append-only audit log)
    audit = _read("layer7_override_audit_log.csv")
    ov_state: dict[str, str] = {}
    if len(audit):
        for _, r in audit.iterrows():
            eid = str(r.get("event_id", ""))
            ov_state.setdefault(eid, f"{r.get('override_id', '')}:{r.get('approval_status', '')}")

    states: list[DigitalTwinState] = []
    for _, r in astate.iterrows():
        eid = str(r["event_id"])
        arow = alloc[alloc["event_id"] == eid]
        if len(arow):
            a = arow.iloc[0]
            res = (f"officers={int(a.get('officers_allocated', 0) or 0)};"
                   f"barricades={int(a.get('barricades_allocated', 0) or 0)};"
                   f"tow={int(a.get('tow_trucks_allocated', 0) or 0)};"
                   f"qru={int(a.get('qru_allocated', 0) or 0)}")
            div = ("enabled" if str(a.get("diversion_activated", "")).strip().lower()
                   in ("1", "true", "yes") else "disabled")
        else:
            res, div = "officers=0;barricades=0;tow=0;qru=0", "disabled"
        states.append(DigitalTwinState(
            event_id=eid,
            operational_risk_score=round(float(r.get("operational_risk_score", 0.0)), 6),
            active_operational_tier=tier_map.get(eid, "Normal"),
            decision_confidence_score=round(float(dcs_map.get(eid, 0.0)), 6),
            allocated_resources=res,
            diversion_status=div,
            alert_state=alert_state.get(eid, "none"),
            override_state=ov_state.get(eid, "none"),
            robustness_score=round(float(r.get("robustness_score", 0.0)), 6),
            generated_at=_NOW_ISO,
        ))
    return pd.DataFrame([s._asdict() for s in states])


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_twin_state()
    if write:
        df.to_csv(OUT / "layer7_digital_twin_state.csv", index=False)
    checks = [{
        "check_id": "m6_twin_state_built", "phase": "digital_twin",
        "passed": len(df) > 0 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} twin states; {int(df.isna().sum().sum())} NaN",
        "severity": "info" if (len(df) > 0 and int(df.isna().sum().sum()) == 0) else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
