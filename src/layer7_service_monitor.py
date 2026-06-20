"""
Layer 7 — M5 Part G: Service Health Monitor.

Tracks operational health of the Layer 7 service surface (read-only):
  - API health      (app constructs; expected routes present; or degraded-if-absent)
  - feed freshness   (age of each dashboard export vs now)
  - manifest consistency (manifest page row-counts vs actual files)
  - endpoint coverage (catalog endpoints vs app routes)

Output: outputs/layer7_service_health.csv
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import layer7_api as api
import layer7_api_extensions as ext
from layer7_config import FRONT, OUT

_NOW = datetime.now(timezone.utc)
_NOW_ISO = _NOW.isoformat()
_STALE_HOURS = 48.0

_PAGE_FILES = [
    "layer7_operations_overview.csv", "layer7_active_alerts.csv",
    "layer7_resource_recommendations.csv", "layer7_override_history.csv",
    "layer7_model_health.csv", "layer7_counterfactuals.csv",
]


def build_service_health() -> pd.DataFrame:
    rows: list[dict] = []

    def add(component, metric, value, status, detail=""):
        rows.append({"component": component, "metric": metric, "value": value,
                     "status": status, "detail": detail, "generated_at": _NOW_ISO})

    # API health
    fastapi_ok = ext._HAS_FASTAPI
    routes_ok, n_routes, missing = True, 0, []
    if fastapi_ok:
        try:
            app = ext.create_app()
            routes = {getattr(r, "path", None) for r in app.routes}
            missing = sorted(ext.expected_routes() - routes)
            n_routes = len([r for r in routes if r])
            routes_ok = not missing
        except Exception as exc:  # pragma: no cover
            routes_ok = False
            missing = [str(exc)]
    add("api", "fastapi_installed", fastapi_ok, "healthy" if fastapi_ok else "degraded",
        "API serves when installed; file backend works regardless")
    add("api", "endpoint_coverage", n_routes,
        "healthy" if routes_ok else "warning",
        f"missing routes: {missing}" if missing else "all expected routes present")
    add("api", "uptime_state", "constructible" if (fastapi_ok and routes_ok) else "file_only",
        "healthy" if (fastapi_ok and routes_ok) else "degraded",
        "in-process construct check (no persistent server required)")

    # feed freshness
    n_stale = 0
    for f in _PAGE_FILES:
        p = FRONT / f
        if p.exists():
            age = (_NOW - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0
            stale = age > _STALE_HOURS
            n_stale += int(stale)
            add("feed", f"freshness_{f}", round(age, 4),
                "warning" if stale else "healthy",
                f"age_hours={age:.3f}")
        else:
            n_stale += 1
            add("feed", f"freshness_{f}", "", "critical", "missing export")
    add("feed", "n_stale_feeds", n_stale, "healthy" if n_stale == 0 else "warning", "")

    # manifest consistency
    man_path = FRONT / "layer7_dashboard_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        mism = []
        for key, meta in man.get("pages", {}).items():
            fname = meta["file"].split("/")[-1]
            fp = FRONT / fname
            actual = (len(pd.read_csv(fp)) if fp.exists() else -1)
            if actual != meta.get("rows"):
                mism.append(f"{key}:{meta.get('rows')}!={actual}")
        add("manifest", "row_count_consistency", "ok" if not mism else "mismatch",
            "healthy" if not mism else "critical",
            f"mismatches: {mism}" if mism else "manifest matches files")
    else:
        add("manifest", "row_count_consistency", "missing", "critical", "manifest absent")

    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_service_health()
    if write:
        df.to_csv(OUT / "layer7_service_health.csv", index=False)
    n_critical = int((df["status"] == "critical").sum())
    checks = [{
        "check_id": "m5_service_health_ok", "phase": "service_monitor",
        "passed": n_critical == 0,
        "detail": f"{len(df)} health rows; critical={n_critical}; "
                  f"status counts={df['status'].value_counts().to_dict()}",
        "severity": "info" if n_critical == 0 else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
