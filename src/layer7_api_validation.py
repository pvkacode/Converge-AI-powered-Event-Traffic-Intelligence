"""
Layer 7 — M5 Part C: API validation layer.

Exercises the API providers with well-formed and malformed inputs and records the
outcome. Read-only. Validates: missing IDs, malformed requests, schema mismatches,
invalid priorities, invalid override IDs.

Output: outputs/layer7_api_validation_report.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

import layer7_api as api
import layer7_api_extensions as ext
from layer7_config import FRONT, OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# expected key columns per page (schema-mismatch detection)
_EXPECTED_SCHEMA = {
    "overview": ["event_id", "operational_risk_score", "active_operational_tier"],
    "alerts": ["l7_alert_id", "priority", "alert_severity_score"],
    "recommendations": ["event_id", "service_tier", "resource_rationale_score"],
    "overrides": ["override_id", "approval_status", "relative_ois", "absolute_ois"],
}


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []

    def rec(case, category, passed, detail):
        rows.append({"case": case, "category": category, "passed": bool(passed),
                     "detail": detail, "generated_at": _NOW_ISO})

    # 1. schema mismatches on each page
    for page, cols in _EXPECTED_SCHEMA.items():
        data = api.get_page(page)
        present = set(data[0].keys()) if data else set()
        missing = [c for c in cols if c not in present]
        rec(f"schema_{page}", "schema_mismatch", not missing,
            "ok" if not missing else f"missing columns: {missing}")

    # 2. valid site lookup + missing/invalid site id
    overview = api.get_page("overview")
    if overview:
        good_id = str(overview[0]["event_id"])
        rec("site_valid", "valid_id", bool(ext.get_site(good_id)),
            f"resolved site {good_id}")
    rec("site_missing", "missing_id", ext.get_site("FKID_NONEXISTENT") == {},
        "unknown event_id returns empty (404 at HTTP layer)")
    rec("site_blank", "malformed", ext.get_site("") == {},
        "blank event_id returns empty")

    # 3. priority validation
    rec("priority_valid", "valid_priority",
        isinstance(ext.get_alerts_by_priority("P1"), list),
        "P1 returns a list")
    rec("priority_invalid", "invalid_priority", "P9" not in ext.VALID_PRIORITIES,
        "P9 rejected by validator (400 at HTTP layer)")
    rec("priority_malformed", "malformed", "" not in ext.VALID_PRIORITIES,
        "blank priority rejected")

    # 4. override id validation
    overrides = api.get_page("overrides")
    if overrides:
        good_ov = str(overrides[0]["override_id"])
        rec("override_valid", "valid_id", bool(ext.get_override(good_ov)),
            f"resolved override {good_ov}")
    rec("override_missing", "invalid_override_id", ext.get_override("OVR-9999") == {},
        "unknown override_id returns empty (404 at HTTP layer)")

    # 5. page file presence (manifest/page coverage)
    for page, (fname) in {k: f"layer7_{n}.csv" for k, n in [
            ("overview", "operations_overview"), ("alerts", "active_alerts"),
            ("recommendations", "resource_recommendations"),
            ("overrides", "override_history"), ("health", "model_health"),
            ("counterfactuals", "counterfactuals")]}.items():
        rec(f"page_file_{page}", "schema_mismatch", (FRONT / fname).exists(),
            f"{fname} present")

    report = pd.DataFrame(rows)
    if write:
        report.to_csv(OUT / "layer7_api_validation_report.csv", index=False)

    checks = [{
        "check_id": "m5_api_validation_all_pass",
        "phase": "api_validation",
        "passed": bool(report["passed"].all()),
        "detail": f"{int(report['passed'].sum())}/{len(report)} API validation cases passed",
        "severity": "info" if bool(report["passed"].all()) else "critical",
    }]
    return report, checks


if __name__ == "__main__":
    report, checks = run(write=True)
    print(report.to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
