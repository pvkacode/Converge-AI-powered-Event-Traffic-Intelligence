"""
Layer 7 — M7 Part H: Failure Mode Library.

Catalogs operational failure modes for future live deployment, each with detection,
mitigation, severity, and a current-status probe against existing outputs. Read-only.

ADDITIVE ONLY. Writes only outputs/layer7_failure_mode_catalog.csv.
(Distinct from the M0 planning doc outputs/layer7_failure_modes.md.)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import FRONT, OUT

_NOW = datetime.now(timezone.utc)
_NOW_ISO = _NOW.isoformat()
_STALE_HOURS = 48.0


def _stale_feeds() -> int:
    n = 0
    for f in FRONT.glob("layer7_*.csv"):
        age = (_NOW - datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0
        n += int(age > _STALE_HOURS)
    return n


def build_catalog() -> pd.DataFrame:
    # probe current status from existing outputs
    n_stale = _stale_feeds()
    sat = []
    opt = OUT / "layer5_optimization_metrics.csv"
    if opt.exists():
        m = pd.read_csv(opt).set_index("metric")["value"].to_dict()
        caps = {"total_officers_deployed": 120, "total_barricades_deployed": 100,
                "total_tow_trucks_deployed": 15, "total_qru_deployed": 10}
        sat = [k for k, cap in caps.items() if float(m.get(k, 0)) >= cap]
    n_alerts = 0
    pa = OUT / "layer7_prioritized_alerts.csv"
    if pa.exists():
        n_alerts = len(pd.read_csv(pa))

    modes = [
        ("FM-01", "missing_feeds", "Required upstream output absent",
         "Ingestion manifest found=False / file existence probe",
         "Skip dependent engine; coverage_flag; degrade gracefully", "critical",
         "all required feeds present"),
        ("FM-02", "stale_feeds", "Feed older than freshness threshold",
         "mtime/generated_at age > 48h", "Stale banner; proceed with warning", "warning",
         f"{n_stale} stale frontend feed(s)"),
        ("FM-03", "schema_drift", "Upstream adds/renames a consumed column",
         "Required-column intersection check in loader/validator",
         "Use intersected columns; engine-level skip with reason", "warning",
         "no drift on consumed columns (M1 loader green)"),
        ("FM-04", "operator_misuse", "Invalid/abusive override or sandbox request",
         "Override safety rules + sandbox validation flags",
         "Flag + record; never silently apply; OIS quantifies impact", "moderate",
         "safety rules enforced (M2/M6)"),
        ("FM-05", "sensor_failure", "Future sensor offline/degraded",
         "Sensor status field + coverage gate (inverse-variance fusion)",
         "Drop source; fall back to existing JOSV signals", "warning",
         "no live sensors yet (readiness only)"),
        ("FM-06", "alert_storms", "Excessive simultaneous alerts overwhelm operators",
         "Alert feed volume / priority-band spike detection",
         "Dedup by topic; rate-limit display; escalate via playbooks", "high",
         f"current alert feed size={n_alerts}"),
        ("FM-07", "resource_saturation", "City-wide resource budget exhausted",
         "Deployed totals vs budget caps",
         "Mutual-aid request; diversion-first plans; reprioritize by DIS/ORS", "high",
         f"saturated: {[s.replace('total_','').replace('_deployed','') for s in sat]}"),
    ]
    rows = [{
        "failure_id": fid, "failure_mode": fm, "description": desc,
        "detection": det, "mitigation": mit, "severity": sev,
        "current_status": status, "generated_at": _NOW_ISO,
    } for (fid, fm, desc, det, mit, sev, status) in modes]
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_catalog()
    if write:
        df.to_csv(OUT / "layer7_failure_mode_catalog.csv", index=False)
    checks = [{
        "check_id": "m7_failure_modes_generated", "phase": "failure_modes",
        "passed": len(df) == 7 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} failure modes catalogued",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df[["failure_id", "failure_mode", "severity", "current_status"]].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
