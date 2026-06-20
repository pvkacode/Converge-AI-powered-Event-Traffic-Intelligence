"""
Layer 7 — M7 Part G: Observability Layer.

Read-only health rollup across the Layer 7 service surface: pipeline, data freshness,
API, dashboard, digital twin, governance. Produces a normalized score per component
(reused by the Deployment Readiness Score).

ADDITIVE ONLY. Writes only outputs/layer7_observability_report.csv.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from layer7_config import FRONT, OUT

_NOW = datetime.now(timezone.utc)
_NOW_ISO = _NOW.isoformat()
_STALE_HOURS = 48.0

_PIPELINE_KEY_OUTPUTS = [
    "layer7_operational_state.csv", "layer7_prioritized_alerts.csv",
    "layer7_active_site_state.csv", "layer7_override_audit_log.csv",
    "layer7_site_explanations.csv", "layer7_decision_confidence.csv",
    "layer7_digital_twin_state.csv", "layer7_twin_scenarios.csv",
]
_FRONTEND_PAGES = [
    "layer7_operations_overview.csv", "layer7_active_alerts.csv",
    "layer7_resource_recommendations.csv", "layer7_override_history.csv",
    "layer7_model_health.csv", "layer7_counterfactuals.csv",
]


def build_observability() -> pd.DataFrame:
    rows = []

    def add(component, score, status, detail):
        rows.append({"component": component, "score": round(float(score), 4),
                     "status": status, "detail": detail, "generated_at": _NOW_ISO})

    # pipeline health: key outputs present
    present = sum((OUT / f).exists() for f in _PIPELINE_KEY_OUTPUTS)
    score = present / len(_PIPELINE_KEY_OUTPUTS)
    add("pipeline_health", score, "healthy" if score == 1 else "warning",
        f"{present}/{len(_PIPELINE_KEY_OUTPUTS)} key outputs present")

    # data freshness: frontend feeds
    fresh, total = 0, 0
    for f in _FRONTEND_PAGES:
        p = FRONT / f
        if p.exists():
            total += 1
            age = (_NOW - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0
            fresh += int(age <= _STALE_HOURS)
    score = fresh / max(1, total)
    add("data_freshness", score, "healthy" if score == 1 else "warning",
        f"{fresh}/{total} feeds within {_STALE_HOURS}h")

    # API health: from service health report
    svc = OUT / "layer7_service_health.csv"
    if svc.exists():
        s = pd.read_csv(svc)
        api = s[s["component"] == "api"]
        n_ok = int((api["status"] == "healthy").sum())
        score = n_ok / max(1, len(api))
        add("api_health", score, "healthy" if score == 1 else "warning",
            f"{n_ok}/{len(api)} api checks healthy")
    else:
        add("api_health", 0.0, "critical", "service_health report missing")

    # dashboard health: manifest + pages
    man = FRONT / "layer7_dashboard_manifest.json"
    pages_ok = sum((FRONT / f).exists() for f in _FRONTEND_PAGES)
    man_ok = man.exists()
    score = (pages_ok / len(_FRONTEND_PAGES)) * (1.0 if man_ok else 0.5)
    add("dashboard_health", score, "healthy" if (man_ok and pages_ok == len(_FRONTEND_PAGES)) else "warning",
        f"manifest={man_ok}; pages={pages_ok}/{len(_FRONTEND_PAGES)}")

    # digital twin health
    tw = OUT / "layer7_digital_twin_health.csv"
    if tw.exists():
        t = pd.read_csv(tw)
        n_ok = int((t["status"] == "healthy").sum())
        score = n_ok / max(1, len(t))
        add("digital_twin_health", score, "healthy" if score == 1 else "warning",
            f"{n_ok}/{len(t)} twin components healthy")
    else:
        add("digital_twin_health", 0.0, "critical", "twin health report missing")

    # governance health: metrics present, no critical
    gov = OUT / "layer7_governance_summary.csv"
    if gov.exists():
        g = pd.read_csv(gov)
        add("governance_health", 1.0 if len(g) == 5 else 0.5,
            "healthy" if len(g) == 5 else "warning", f"{len(g)} governance metrics")
    else:
        add("governance_health", 0.0, "critical", "governance summary missing")

    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_observability()
    if write:
        df.to_csv(OUT / "layer7_observability_report.csv", index=False)
    n_crit = int((df["status"] == "critical").sum())
    checks = [{
        "check_id": "m7_observability_built", "phase": "observability",
        "passed": n_crit == 0 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} components; mean_score={df['score'].mean():.3f}; "
                  f"status={df['status'].value_counts().to_dict()}",
        "severity": "info" if n_crit == 0 else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
