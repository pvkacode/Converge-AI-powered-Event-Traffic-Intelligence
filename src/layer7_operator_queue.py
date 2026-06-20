"""
Layer 7 — M5 Part F: Operator Action Queue.

Priority-ordered queue of operator actions, assembled (no new model) from existing
Layer 7 outputs. Priority bands:
  1. Emergency sites
  2. Low-confidence recommendations
  3. Critical alerts
  4. High-impact overrides
  5. Drift-triggered investigations

Output: outputs/layer7_operator_action_queue.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_queue() -> pd.DataFrame:
    tiers = _read("layer7_active_site_tiers.csv")
    dcs = _read("layer7_decision_confidence.csv")
    alerts = _read("layer7_prioritized_alerts.csv")
    ext_ois = _read("layer7_override_impact_extended.csv")
    audit = _read("layer7_override_audit_log.csv")
    drift = _read("layer6_drift_report.csv")

    dcs_map = (
        dict(zip(dcs["event_id"].astype(str), dcs["decision_confidence_score"]))
        if len(dcs) else {})
    rows: list[dict] = []

    # 1. Emergency sites
    if len(tiers):
        for _, r in tiers[tiers["active_operational_tier"] == "Emergency"].iterrows():
            eid = str(r["event_id"])
            rows.append({"priority": 1, "event_id": eid,
                         "reason": "Emergency operational tier",
                         "decision_confidence_score": dcs_map.get(eid, ""),
                         "recommended_action": "Immediate deployment review and on-site command"})

    # 2. Low-confidence recommendations
    if len(dcs):
        for _, r in dcs[dcs["decision_confidence_tier"] == "Low"].iterrows():
            eid = str(r["event_id"])
            rows.append({"priority": 2, "event_id": eid,
                         "reason": "Low decision confidence",
                         "decision_confidence_score": r["decision_confidence_score"],
                         "recommended_action": "Validate recommendation; collect more evidence"})

    # 3. Critical alerts (P1)
    if len(alerts):
        for _, r in alerts[alerts["priority"].astype(str) == "P1"].iterrows():
            eid = str(r.get("affected_event_id", "")).strip()
            eid = eid if eid and eid.lower() != "nan" else "SYSTEM"
            rows.append({"priority": 3, "event_id": eid,
                         "reason": f"Critical alert ({r.get('topic_key', '')})",
                         "decision_confidence_score": dcs_map.get(eid, ""),
                         "recommended_action": "Investigate alert root cause"})

    # 4. High-impact overrides
    if len(ext_ois):
        # PATCH F-004: quantile-based impact tiers (absolute_ois>=0.50 was unreachable)
        hi = ext_ois[ext_ois["impact_level"].isin(["High", "Critical"])]
        ov_event = (dict(zip(audit["override_id"].astype(str), audit["event_id"].astype(str)))
                    if len(audit) else {})
        for _, r in hi.iterrows():
            ovid = str(r["override_id"])
            eid = ov_event.get(ovid, "")
            rows.append({"priority": 4, "event_id": eid,
                         "reason": f"High-impact override {ovid} (OIS={r['absolute_ois']:.2f})",
                         "decision_confidence_score": dcs_map.get(eid, ""),
                         "recommended_action": "Review override justification and impact"})

    # 5. Drift-triggered investigations
    if len(drift):
        da = drift[drift["alert"].astype(str).str.lower().isin(["true", "1", "yes"])]
        for _, r in da.iterrows():
            rows.append({"priority": 5, "event_id": "GLOBAL",
                         "reason": f"Drift trigger: {r.get('test', '')} on {r.get('variable', '')}",
                         "decision_confidence_score": "",
                         "recommended_action": "Schedule retrain/recalibration investigation"})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=["priority", "event_id", "reason",
                                   "decision_confidence_score", "recommended_action"])
    df = df.sort_values(["priority", "event_id"]).reset_index(drop=True)
    df.insert(0, "queue_rank", range(1, len(df) + 1))
    df["generated_at"] = _NOW_ISO
    return df[["queue_rank", "event_id", "reason", "priority",
               "decision_confidence_score", "recommended_action", "generated_at"]]


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_queue()
    if write:
        df.to_csv(OUT / "layer7_operator_action_queue.csv", index=False)
    checks = [{
        "check_id": "m5_operator_queue_built", "phase": "operator_queue",
        "passed": True,
        "detail": f"{len(df)} queue items; priority bands: "
                  f"{df['priority'].value_counts().sort_index().to_dict() if len(df) else {}}",
        "severity": "info",
    }, {
        "check_id": "m5_operator_queue_no_nan", "phase": "operator_queue",
        "passed": int(df.isna().sum().sum()) == 0,
        "detail": f"{int(df.isna().sum().sum())} NaN in queue",
        "severity": "info" if int(df.isna().sum().sum()) == 0 else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head(15).to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
