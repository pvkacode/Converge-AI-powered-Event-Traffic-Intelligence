"""
Layer 7 — M4: Operational Dashboard Backend (file-based).

Pure assembly. NO model, NO retraining, NO score recomputation. Joins existing
M1 / M1.1 / M2 / M3 + Layer 5 / Layer 6 outputs into denormalized, read-optimized
per-page payloads for the operations dashboard, plus a manifest.

ADDITIVE ONLY. Writes only outputs/frontend/layer7_*.csv|json. The seven canonical
L1-L4 frontend files are never touched — M4 adds new layer7_* files beside them.

Page outputs (outputs/frontend/):
  layer7_operations_overview.csv
  layer7_active_alerts.csv
  layer7_resource_recommendations.csv
  layer7_override_history.csv
  layer7_model_health.csv
  layer7_counterfactuals.csv
  layer7_dashboard_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import FRONT, OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# Page file names + primary keys + the key columns that must be non-null.
PAGES = {
    "overview": ("layer7_operations_overview.csv", "event_id",
                 ["event_id", "operational_risk_score", "active_operational_tier"]),
    "alerts": ("layer7_active_alerts.csv", "l7_alert_id",
               ["l7_alert_id", "priority", "alert_severity_score"]),
    "recommendations": ("layer7_resource_recommendations.csv", "event_id",
                        ["event_id", "service_tier", "resource_rationale_score"]),
    "overrides": ("layer7_override_history.csv", "override_id",
                  ["override_id", "approval_status", "relative_ois", "absolute_ois"]),
    "model_health": ("layer7_model_health.csv", None,
                     ["panel", "metric", "status_normalized"]),
    "counterfactuals": ("layer7_counterfactuals.csv", None,
                        ["event_id", "scenario_type", "counterfactual_score"]),
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def _schema_version(cols) -> str:
    return hashlib.md5(",".join(sorted(map(str, cols))).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- pages
def build_overview() -> pd.DataFrame:
    astate = _read("layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    tiers = _read("layer7_active_site_tiers.csv")[["event_id", "active_operational_tier"]]
    tiers["event_id"] = tiers["event_id"].astype(str)
    alloc = _read("layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)
    site = _read("layer7_site_explanations.csv")[
        ["event_id", "top_factor_1", "top_factor_1_share"]]
    site["event_id"] = site["event_id"].astype(str)

    # per-site alert count
    al = _read("layer7_prioritized_alerts.csv")
    al["affected_event_id"] = al["affected_event_id"].astype(str)
    ncount = al.groupby("affected_event_id").size().rename("n_alerts")

    df = astate[["event_id", "event_cause", "operational_risk_score",
                 "operational_tier", "service_tier", "robustness_score"]].copy()
    df = df.merge(tiers, on="event_id", how="left")
    df = df.merge(site, on="event_id", how="left")
    res_cols = ["event_id", "officers_allocated", "barricades_allocated",
                "tow_trucks_allocated", "qru_allocated", "diversion_activated"]
    df = df.merge(alloc[[c for c in res_cols if c in alloc.columns]], on="event_id", how="left")
    df["n_alerts"] = df["event_id"].map(ncount).fillna(0).astype(int)
    df = df.rename(columns={"top_factor_1": "top_risk_factor",
                            "top_factor_1_share": "top_risk_share"})
    df["generated_at"] = _NOW_ISO
    return df.sort_values(["operational_risk_score", "event_id"], ascending=[False, True])


def build_alerts() -> pd.DataFrame:
    al = _read("layer7_prioritized_alerts.csv")
    expl = _read("layer7_alert_explanations.csv")[
        ["alert_id", "root_cause", "corroboration_sources"]].rename(
        columns={"alert_id": "l7_alert_id"})
    df = al.merge(expl, on="l7_alert_id", how="left")
    cols = ["priority_rank", "l7_alert_id", "priority", "severity_raw",
            "alert_severity_score", "affected_layer", "affected_event_id",
            "root_cause", "corroboration_sources", "description", "generated_at"]
    df = df.rename(columns={"severity_raw": "severity"})
    cols = [c if c != "severity_raw" else "severity" for c in cols]
    df["generated_at"] = _NOW_ISO
    return df[[c for c in cols if c in df.columns]].sort_values("priority_rank")


def build_recommendations() -> pd.DataFrame:
    alloc = _read("layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)
    res = _read("layer7_resource_explanations.csv")
    res["event_id"] = res["event_id"].astype(str)
    cols = ["event_id", "event_cause", "service_tier", "officers_allocated",
            "barricades_allocated", "tow_trucks_allocated", "qru_allocated",
            "diversion_activated", "diversion_reason", "expected_delay_reduction_min",
            "effectiveness", "robustness_score"]
    df = alloc[[c for c in cols if c in alloc.columns]].merge(
        res[["event_id", "resource_rationale_score", "resource_explanation"]],
        on="event_id", how="left")
    df["generated_at"] = _NOW_ISO
    return df.sort_values("resource_rationale_score", ascending=False)


def build_overrides() -> pd.DataFrame:
    audit = _read("layer7_override_audit_log.csv").astype(str)
    ext = _read("layer7_override_impact_extended.csv")
    ext["override_id"] = ext["override_id"].astype(str)
    safety = _read("layer7_override_safety_report.csv")[
        ["override_id", "safety_status", "flags"]].drop_duplicates("override_id")
    safety["override_id"] = safety["override_id"].astype(str)
    df = audit.merge(ext[["override_id", "relative_ois", "absolute_ois", "impact_level"]],
                     on="override_id", how="left")
    df = df.merge(safety, on="override_id", how="left")
    cols = ["override_id", "event_id", "override_type", "approval_status",
            "operator_id", "justification", "override_source", "timestamp",
            "relative_ois", "absolute_ois", "impact_level", "safety_status", "flags"]
    df["generated_at"] = _NOW_ISO
    return df[[c for c in cols if c in df.columns] + ["generated_at"]]


def build_model_health() -> pd.DataFrame:
    rows = []
    mh = _read("layer6_model_health_summary.csv")
    for _, r in mh.iterrows():
        rows.append({
            "panel": "health", "metric_group": r.get("metric_group", ""),
            "metric": r.get("metric", ""),
            "holdout_value": r.get("holdout_value", ""),
            "feedback_value": r.get("feedback_value", ""),
            "status_normalized": str(r.get("status", "")).lower(),
            "detail": r.get("note", ""),
            "overall_health": r.get("overall_health", ""),
        })
    dr = _read("layer6_drift_report.csv")
    for _, r in dr.iterrows():
        rows.append({
            "panel": "drift", "metric_group": "drift",
            "metric": f"{r.get('test', '')}:{r.get('variable', '')}",
            "holdout_value": "", "feedback_value": r.get("score", ""),
            "status_normalized": str(r.get("severity", "none")).lower(),
            "detail": r.get("detail", ""), "overall_health": "",
        })
    md = _read("layer6_monitoring_diagnostics.csv")
    key_terms = ("retrain", "urgency", "knowledge", "retention")
    for _, r in md.iterrows():
        metric = str(r.get("metric", ""))
        if any(k in metric.lower() for k in key_terms):
            rows.append({
                "panel": "monitoring", "metric_group": r.get("diagnostic_group", ""),
                "metric": metric, "holdout_value": "", "feedback_value": r.get("value", ""),
                "status_normalized": str(r.get("flag", "")).lower(),
                "detail": r.get("notes", ""), "overall_health": "",
            })
    df = pd.DataFrame(rows)
    df["generated_at"] = _NOW_ISO
    return df


def build_counterfactuals() -> pd.DataFrame:
    cf = _read("layer7_counterfactual_analysis.csv")
    cf["event_id"] = cf["event_id"].astype(str)
    tiers = _read("layer7_active_site_tiers.csv")[["event_id", "active_operational_tier"]]
    tiers["event_id"] = tiers["event_id"].astype(str)
    df = cf.merge(tiers, on="event_id", how="left")
    cols = ["event_id", "active_operational_tier", "scenario_type",
            "expected_delay_delta", "expected_risk_delta", "expected_alert_delta",
            "counterfactual_score", "generated_at"]
    if "generated_at" not in df.columns:
        df["generated_at"] = _NOW_ISO
    return df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------------- manifest
def build_manifest(frames: dict[str, pd.DataFrame]) -> dict:
    pages = {}
    for key, (fname, pk, _keycols) in PAGES.items():
        df = frames[key]
        src = FRONT / fname
        pages[key] = {
            "file": f"outputs/frontend/{fname}",
            "rows": int(len(df)),
            "primary_key": pk,
            "schema_version": _schema_version(df.columns),
            "columns": list(map(str, df.columns)),
            "mtime": (datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc).isoformat()
                      if src.exists() else None),
        }
    return {
        "layer7_milestone": "M4",
        "title": "ASTraM Layer 7 Operational Dashboard Backend",
        "generated_at": _NOW_ISO,
        "read_only": True,
        "pages": pages,
        "api_endpoints": ["/", "/manifest", "/overview", "/alerts",
                          "/recommendations", "/overrides", "/health", "/counterfactuals"],
        "note": "Additive-only. Layers 1-6 frozen. Canonical L1-L4 frontend files untouched.",
    }


# --------------------------------------------------------------------------- run
def build_all() -> dict[str, pd.DataFrame]:
    return {
        "overview": build_overview(),
        "alerts": build_alerts(),
        "recommendations": build_recommendations(),
        "overrides": build_overrides(),
        "model_health": build_model_health(),
        "counterfactuals": build_counterfactuals(),
    }


def run(write: bool = True) -> tuple[dict[str, pd.DataFrame], dict, list[dict]]:
    frames = build_all()
    # display fill: render upstream-missing cells as blank (no NaN in artifacts)
    frames = {k: v.fillna("") for k, v in frames.items()}

    if write:
        FRONT.mkdir(parents=True, exist_ok=True)
        for key, (fname, _pk, _kc) in PAGES.items():
            frames[key].to_csv(FRONT / fname, index=False)

    manifest = build_manifest(frames)
    if write:
        (FRONT / "layer7_dashboard_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    checks = _validate(frames, manifest)
    return frames, manifest, checks


def _validate(frames: dict[str, pd.DataFrame], manifest: dict) -> list[dict]:
    checks: list[dict] = []

    def chk(cid, passed, detail, severity="critical"):
        checks.append({"check_id": cid, "phase": "dashboard_backend",
                       "passed": bool(passed), "detail": detail,
                       "severity": "info" if passed else severity})

    for key, (fname, pk, keycols) in PAGES.items():
        df = frames[key]
        chk(f"m4_{key}_populated", len(df) > 0, f"{fname}: {len(df)} rows")
        present = [c for c in keycols if c in df.columns]
        missing = [c for c in keycols if c not in df.columns]
        # key columns non-blank
        blank = 0
        for c in present:
            blank += int((df[c].astype(str).str.strip() == "").sum())
        chk(f"m4_{key}_keycols_nonnull", not missing and blank == 0,
            f"missing={missing}, blank_keycells={blank}")
        if pk:
            n_dup = int(df[pk].astype(str).duplicated().sum())
            chk(f"m4_{key}_pk_unique", n_dup == 0, f"{n_dup} duplicate {pk}")

    # manifest references every page with matching row counts
    ok = all(
        manifest["pages"][k]["rows"] == len(frames[k]) for k in PAGES
    )
    chk("m4_manifest_consistent", ok, "manifest row counts match page frames")

    return checks


if __name__ == "__main__":
    frames, manifest, checks = run(write=True)
    print("=== Layer 7 M4 Dashboard Backend ===")
    for k in PAGES:
        print(f"  {k}: {len(frames[k])} rows")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
