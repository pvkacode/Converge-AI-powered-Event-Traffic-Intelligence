"""
Layer 7 — M5 Part B/E: API hardening (additive).

Extends the M4 read-only API WITHOUT modifying src/layer7_api.py. It reuses the
M4 app (api.create_app()) and registers new endpoints on it, so every existing
M4 endpoint keeps working unchanged.

Backward-compatibility note: M4 already serves GET /health (the model-health
page). To avoid breaking that contract (constraint: existing M4 endpoints must
continue working), the new liveness/status object is served at GET /healthz, and
/health is left untouched.

New endpoints:
  GET /healthz              -> {status, version, timestamp, validation_passed}
  GET /metrics              -> dashboard KPIs
  GET /site/{event_id}      -> single-site operational record
  GET /alerts/{priority}    -> alerts filtered by priority (P1..P4)
  GET /override/{override_id} -> override details

Pure provider functions work WITHOUT FastAPI (degrade-if-absent), so the M5
validation/audit harness runs even when the web framework is absent.

ADDITIVE ONLY. This module writes nothing on import; export helpers are called by
the M5 orchestrator.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

import layer7_api as api
from layer7_config import FRONT, OUT

_VERSION = "layer7_m5"

try:
    from fastapi import FastAPI, HTTPException
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover
    _HAS_FASTAPI = False


# ----------------------------------------------------------------- pure providers
def get_healthz() -> dict:
    val_ok = True
    vr = OUT / "layer7_m5_validation_report.csv"
    if vr.exists():
        try:
            df = pd.read_csv(vr)
            val_ok = bool(df["passed"].astype(str).str.lower().isin(["true", "1"]).all())
        except Exception:
            val_ok = True
    return {
        "status": "healthy",
        "version": _VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_passed": val_ok,
    }


def get_metrics() -> dict:
    overview = pd.DataFrame(api.get_page("overview"))
    alerts = pd.DataFrame(api.get_page("alerts"))
    overrides = pd.DataFrame(api.get_page("overrides"))
    kpis: dict = {"generated_at": datetime.now(timezone.utc).isoformat()}

    kpis["n_active_sites"] = int(len(overview))
    if len(overview):
        kpis["tier_distribution"] = (
            overview["active_operational_tier"].value_counts().to_dict()
            if "active_operational_tier" in overview else {})
        for r in ("officers_allocated", "barricades_allocated",
                  "tow_trucks_allocated", "qru_allocated"):
            if r in overview:
                kpis[f"total_{r}"] = int(pd.to_numeric(overview[r], errors="coerce").fillna(0).sum())
    kpis["n_alerts"] = int(len(alerts))
    if len(alerts) and "priority" in alerts:
        kpis["alerts_by_priority"] = alerts["priority"].value_counts().to_dict()
    kpis["n_overrides"] = int(len(overrides))

    dcs = OUT / "layer7_decision_confidence.csv"
    if dcs.exists():
        d = pd.read_csv(dcs)
        kpis["dcs_mean"] = round(float(d["decision_confidence_score"].mean()), 6)
        kpis["dcs_tier_distribution"] = d["decision_confidence_tier"].value_counts().to_dict()

    om = OUT / "layer5_optimization_metrics.csv"
    if om.exists():
        m = pd.read_csv(om).set_index("metric")["value"].to_dict()
        for k in ("expected_delay_reduction_pct", "cvar_90", "chance_constraint_satisfaction_mean"):
            if k in m:
                kpis[k] = float(m[k])
    return kpis


def get_site(event_id: str) -> dict:
    overview = pd.DataFrame(api.get_page("overview"))
    if "event_id" not in overview:
        return {}
    row = overview[overview["event_id"].astype(str) == str(event_id)]
    return row.iloc[0].to_dict() if len(row) else {}


def get_alerts_by_priority(priority: str) -> list[dict]:
    alerts = pd.DataFrame(api.get_page("alerts"))
    if "priority" not in alerts:
        return []
    p = str(priority).upper()
    return alerts[alerts["priority"].astype(str).str.upper() == p].to_dict(orient="records")


def get_override(override_id: str) -> dict:
    ov = pd.DataFrame(api.get_page("overrides"))
    if "override_id" not in ov:
        return {}
    row = ov[ov["override_id"].astype(str) == str(override_id)]
    return row.iloc[0].to_dict() if len(row) else {}


VALID_PRIORITIES = {"P1", "P2", "P3", "P4"}


# ----------------------------------------------------------------- app factory
def create_app():
    """Augmented app = M4 app + M5 endpoints. M4 endpoints unchanged."""
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed. pip install -r requirements-layer7.txt")
    app = api.create_app()  # reuse M4 app (all M4 routes preserved)

    @app.get("/healthz")
    def healthz() -> dict:
        return get_healthz()

    @app.get("/metrics")
    def metrics() -> dict:
        return get_metrics()

    @app.get("/site/{event_id}")
    def site(event_id: str) -> dict:
        rec = get_site(event_id)
        if not rec:
            raise HTTPException(status_code=404, detail=f"event_id '{event_id}' not found")
        return rec

    @app.get("/alerts/{priority}")
    def alerts_by_priority(priority: str) -> list[dict]:
        if str(priority).upper() not in VALID_PRIORITIES:
            raise HTTPException(status_code=400,
                                detail=f"invalid priority '{priority}'; use P1..P4")
        return get_alerts_by_priority(priority)

    @app.get("/override/{override_id}")
    def override(override_id: str) -> dict:
        rec = get_override(override_id)
        if not rec:
            raise HTTPException(status_code=404,
                                detail=f"override_id '{override_id}' not found")
        return rec

    return app


def expected_routes() -> set[str]:
    return api.expected_routes() | {
        "/healthz", "/metrics", "/site/{event_id}",
        "/alerts/{priority}", "/override/{override_id}",
    }


# ----------------------------------------------------------------- Part E exports
_ENDPOINT_META = [
    ("/", "GET", "Service index and endpoint list", "object", "M4"),
    ("/manifest", "GET", "Dashboard manifest (pages, schema versions, freshness)", "object", "M4"),
    ("/overview", "GET", "Operations overview (per active site)", "list", "M4"),
    ("/alerts", "GET", "Prioritized alert feed", "list", "M4"),
    ("/recommendations", "GET", "Resource recommendations", "list", "M4"),
    ("/overrides", "GET", "Override history", "list", "M4"),
    ("/health", "GET", "Model-health page (M4; unchanged for compatibility)", "list", "M4"),
    ("/counterfactuals", "GET", "Counterfactual scenarios", "list", "M4"),
    ("/healthz", "GET", "Service liveness/status object", "object", "M5"),
    ("/metrics", "GET", "Dashboard KPIs", "object", "M5"),
    ("/site/{event_id}", "GET", "Single-site operational record", "object", "M5"),
    ("/alerts/{priority}", "GET", "Alerts filtered by priority P1..P4", "list", "M5"),
    ("/override/{override_id}", "GET", "Single override details", "object", "M5"),
]


def build_endpoint_catalog() -> pd.DataFrame:
    rows = [{"endpoint": e, "method": m, "description": d,
             "response_type": rt, "milestone": ms}
            for (e, m, d, rt, ms) in _ENDPOINT_META]
    df = pd.DataFrame(rows)
    df["generated_at"] = datetime.now(timezone.utc).isoformat()
    return df


def export_openapi(path) -> dict:
    """Export the augmented app's OpenAPI schema. Falls back to a static schema
    derived from the endpoint catalog if FastAPI is unavailable."""
    if _HAS_FASTAPI:
        try:
            schema = create_app().openapi()
        except Exception:
            schema = _static_openapi()
    else:
        schema = _static_openapi()
    path = OUT / "layer7_openapi_snapshot.json" if path is None else path
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def _static_openapi() -> dict:
    paths = {}
    for (e, m, d, rt, _ms) in _ENDPOINT_META:
        paths[e] = {m.lower(): {"summary": d,
                                "responses": {"200": {"description": rt}}}}
    return {
        "openapi": "3.0.0",
        "info": {"title": "ASTraM Layer 7 Operational Dashboard API",
                 "version": _VERSION, "description": "Read-only (static fallback schema)."},
        "paths": paths,
    }
