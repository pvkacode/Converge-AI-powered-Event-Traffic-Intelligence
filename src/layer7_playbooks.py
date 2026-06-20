"""
Layer 7 — M7 Part E: Operational Playbook Engine.

Converts current operational conditions into concrete operator actions. Read-only;
each playbook is matched against existing Layer 6/7 outputs to count current triggers.

Playbooks: High Risk + Low Confidence, Critical Drift, Repeated Override,
Resource Saturation, Emergency Site Escalation.

ADDITIVE ONLY. Writes only outputs/layer7_operational_playbooks.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_playbooks() -> pd.DataFrame:
    tiers = _read("layer7_active_site_tiers.csv")
    dcs = _read("layer7_decision_confidence.csv")
    drift = _read("layer6_drift_report.csv")
    audit = _read("layer7_override_audit_log.csv")
    opt = _read("layer5_optimization_metrics.csv")

    # High Risk + Low Confidence
    hr_lc = []
    if len(tiers) and len(dcs):
        low = set(dcs[dcs["decision_confidence_tier"] == "Low"]["event_id"].astype(str))
        hi = set(tiers[tiers["active_operational_tier"].isin(["Emergency", "Critical"])]["event_id"].astype(str))
        hr_lc = sorted(hi & low)

    # Critical Drift
    n_crit_drift = 0
    if len(drift):
        n_crit_drift = int(((drift["alert"].astype(str).str.lower().isin(["true", "1", "yes"]))
                            & (drift["severity"].astype(str).str.lower() == "critical")).sum())

    # Repeated Override (event_id with >1 override in audit log)
    rep_override = []
    if len(audit):
        vc = audit["event_id"].astype(str).value_counts()
        rep_override = sorted(vc[vc > 1].index.tolist())

    # Resource Saturation (any budget fully used)
    saturated = []
    if len(opt):
        m = opt.set_index("metric")["value"].to_dict()
        caps = {"total_officers_deployed": 120, "total_barricades_deployed": 100,
                "total_tow_trucks_deployed": 15, "total_qru_deployed": 10}
        saturated = [k.replace("total_", "").replace("_deployed", "")
                     for k, cap in caps.items() if float(m.get(k, 0)) >= cap]

    # Emergency Site Escalation
    emerg = []
    if len(tiers):
        emerg = sorted(tiers[tiers["active_operational_tier"] == "Emergency"]["event_id"].astype(str))

    defs = [
        ("PB-001", "High Risk + Low Confidence",
         "Site in Emergency/Critical tier AND decision confidence tier Low",
         "Dispatch senior officer; collect ground truth; defer automated reallocation",
         "high", len(hr_lc), ";".join(hr_lc[:10])),
        ("PB-002", "Critical Drift",
         "One or more critical drift alerts in Layer 6 drift report",
         "Freeze automated duration priors; schedule Layer 4.5 retrain review",
         "critical" if n_crit_drift else "info", n_crit_drift,
         "critical drift tests present" if n_crit_drift else ""),
        ("PB-003", "Repeated Override",
         "Same event overridden more than once in the audit log",
         "Escalate to supervisor; review whether the recommendation model is mis-serving the site",
         "moderate", len(rep_override), ";".join(rep_override[:10])),
        ("PB-004", "Resource Saturation",
         "One or more city-wide resource budgets fully deployed",
         "Trigger mutual-aid request; reprioritize via DIS/ORS; consider diversion-first plans",
         "high" if saturated else "info", len(saturated), ";".join(saturated)),
        ("PB-005", "Emergency Site Escalation",
         "Active site classified Emergency operational tier",
         "Activate incident command; pre-position tow + QRU; open diversion corridors",
         "critical" if emerg else "info", len(emerg), ";".join(emerg[:10])),
    ]
    rows = [{
        "playbook_id": pid, "playbook_name": name, "trigger_condition": cond,
        "recommended_actions": act, "severity": sev, "n_matched": n,
        "matched_examples": ex, "generated_at": _NOW_ISO,
    } for (pid, name, cond, act, sev, n, ex) in defs]
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_playbooks()
    if write:
        df.to_csv(OUT / "layer7_operational_playbooks.csv", index=False)
    checks = [{
        "check_id": "m7_playbooks_generated", "phase": "playbooks",
        "passed": len(df) == 5 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} playbooks; total matches={int(df['n_matched'].sum())}",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df[["playbook_id", "playbook_name", "severity", "n_matched"]].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
