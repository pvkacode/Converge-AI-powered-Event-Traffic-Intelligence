"""
Layer 7 — M5 Part H: Governance Diagnostics.

Read-only governance KPIs derived from existing Layer 7 / Layer 6 outputs:
  - override approval rate
  - high-impact override rate
  - alert acknowledgement rate
  - low-confidence recommendation rate
  - drift-trigger frequency

Output: outputs/layer7_governance_summary.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_governance() -> pd.DataFrame:
    audit = _read("layer7_override_audit_log.csv")
    ext = _read("layer7_override_impact_extended.csv")
    alerts = _read("layer7_prioritized_alerts.csv")
    dcs = _read("layer7_decision_confidence.csv")
    drift = _read("layer6_drift_report.csv")

    rows: list[dict] = []

    def add(metric, value, numerator, denominator, detail=""):
        rows.append({"metric": metric, "value": round(float(value), 6),
                     "numerator": numerator, "denominator": denominator,
                     "detail": detail, "generated_at": _NOW_ISO})

    n_ov = len(audit)
    if n_ov:
        status = audit["approval_status"].astype(str).str.lower()
        n_appr = int(status.isin(["approved", "executed"]).sum())
        add("override_approval_rate", n_appr / n_ov, n_appr, n_ov,
            "approved+executed / total overrides")
        n_ack = int((audit["override_type"].astype(str).str.lower()
                     .isin(["acknowledge_alert", "suppress_alert"])
                     & status.isin(["approved", "executed"])).sum())
    else:
        add("override_approval_rate", 0.0, 0, 0, "no overrides")
        n_ack = 0

    if len(ext):
        # PATCH F-004: use quantile-based impact tiers (absolute_ois>=0.50 was unreachable)
        n_hi = int(ext["impact_level"].isin(["High", "Critical"]).sum())
        add("high_impact_override_rate", n_hi / max(1, len(ext)), n_hi, len(ext),
            "impact_level in {High,Critical} (quantile tiers)")
    else:
        add("high_impact_override_rate", 0.0, 0, 0, "no override impacts")

    n_alerts = len(alerts)
    add("alert_acknowledgement_rate", (n_ack / n_alerts) if n_alerts else 0.0,
        n_ack, n_alerts, "acknowledged/suppressed (approved) per active alert")

    if len(dcs):
        n_low = int((dcs["decision_confidence_tier"] == "Low").sum())
        add("low_confidence_recommendation_rate", n_low / len(dcs), n_low, len(dcs),
            "DCS tier == Low / active sites")
    else:
        add("low_confidence_recommendation_rate", 0.0, 0, 0, "no DCS")

    if len(drift):
        n_da = int(drift["alert"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        add("drift_trigger_frequency", n_da / len(drift), n_da, len(drift),
            "drift tests with alert==True / total tests")
    else:
        add("drift_trigger_frequency", 0.0, 0, 0, "no drift report")

    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_governance()
    if write:
        df.to_csv(OUT / "layer7_governance_summary.csv", index=False)
    checks = [{
        "check_id": "m5_governance_generated", "phase": "governance",
        "passed": len(df) == 5 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} governance metrics; "
                  f"{int(df.isna().sum().sum())} NaN",
        "severity": "info" if (len(df) == 5 and int(df.isna().sum().sum()) == 0) else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df[["metric", "value", "numerator", "denominator"]].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
