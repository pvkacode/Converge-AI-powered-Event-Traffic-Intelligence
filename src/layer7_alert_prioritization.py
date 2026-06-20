"""
Layer 7 — Phase 3: Alert Prioritization Engine.

Deterministic. Consumes L6 active_alerts, retrain_triggers, drift_report and
L5 chance_constraint_violations into one normalized, deduplicated, ranked feed.

Alert Severity Score (ASS):
    ASS = base_severity * corroboration_factor * recency_factor
        base_severity:        info=0.25, warning=0.50, moderate=0.65, critical=0.90
        corroboration_factor: 1 + 0.15 * (# independent sources on the same topic)
        recency_factor:       exp(-age_hours / 24)

Priority levels P1..P4 by ASS thresholds.

Outputs:
    outputs/layer7_prioritized_alerts.csv
    outputs/layer7_alert_summary.csv
    outputs/layer7_alert_diagnostics.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import (
    ASS_MAX_SOURCES,
    ASS_PRIORITY_CUTS,
    ASS_PRIORITY_PCT_CUTS,
    BASE_SEVERITY,
    CORROBORATION_PER_SOURCE,
    OUT,
    RECENCY_HALFLIFE_HOURS,
    SEVERITY_FALLBACK,
)
from layer7_loader import Store

# PATCH F-002: theoretical maximum of the multiplicative ASS, used to rescale it into
# [0,1] (monotone, ordering preserved). max base severity x max corroboration x max recency.
_ASS_MAX = max(BASE_SEVERITY.values()) * (1.0 + CORROBORATION_PER_SOURCE * ASS_MAX_SOURCES) * 1.0

_NOW = datetime.now(timezone.utc)
_NOW_ISO = _NOW.isoformat()

_DRIFT_VAR_TOKENS = ("log_duration", "hour_local", "trust_score", "iso_anomaly_score")


def _file_mtime_iso(name: str) -> str:
    p = OUT / name
    if p.exists():
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    return _NOW_ISO


def _topic_key(original_id: str, affected_layer: str, variable: str) -> str:
    """Deterministic topic for corroboration grouping across feeds."""
    oid = str(original_id or "")
    if oid.startswith("DUR_SHIFT"):
        return "dur_shift::" + oid.replace("DUR_SHIFT_", "").lower()
    var = str(variable or "").strip().lower()
    if var and var not in ("nan", "general", "none", ""):
        return f"{str(affected_layer).lower()}::{var}"
    low = oid.lower()
    for v in _DRIFT_VAR_TOKENS:
        if v in low:
            return f"layer4.5::{v}"
    return f"{str(affected_layer).lower()}::general"


def _collect(store: Store) -> pd.DataFrame:
    """Normalize all four feeds into one long alert frame."""
    recs: list[dict] = []

    # L6 active alerts ------------------------------------------------------
    aa = store.get("layer6_active_alerts.csv")
    if aa is not None and len(aa) > 0:
        for _, r in aa.iterrows():
            recs.append({
                "source_feed": "active_alerts",
                "original_id": str(r.get("alert_id", "")),
                "severity_raw": str(r.get("severity", "")).strip().lower(),
                "affected_layer": str(r.get("affected_layer", "")),
                "variable": "",
                "affected_event_id": "",
                "description": str(r.get("description", "")),
                "timestamp": str(r.get("generated_at", "")) or _NOW_ISO,
            })

    # L6 retrain triggers ---------------------------------------------------
    rt = store.get("layer6_retrain_triggers.csv")
    if rt is not None and len(rt) > 0:
        for _, r in rt.iterrows():
            recs.append({
                "source_feed": "retrain_triggers",
                "original_id": str(r.get("trigger_id", "")),
                "severity_raw": str(r.get("severity", "")).strip().lower(),
                "affected_layer": str(r.get("affected_layer", "")),
                "variable": str(r.get("variable", "")),
                "affected_event_id": "",
                "description": str(r.get("recommendation", "")),
                "timestamp": str(r.get("generated_at", "")) or _NOW_ISO,
            })

    # L6 drift report (alert==True only) ------------------------------------
    dr = store.get("layer6_drift_report.csv")
    if dr is not None and len(dr) > 0:
        mtime = _file_mtime_iso("layer6_drift_report.csv")
        alert_mask = dr["alert"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        for _, r in dr[alert_mask].iterrows():
            recs.append({
                "source_feed": "drift_report",
                "original_id": f"DRIFT_{r.get('test', '')}_{r.get('variable', '')}".upper(),
                "severity_raw": str(r.get("severity", "")).strip().lower(),
                "affected_layer": "Layer4.5",
                "variable": str(r.get("variable", "")),
                "affected_event_id": "",
                "description": str(r.get("recommendation", "")),
                "timestamp": mtime,
            })

    # L5 chance-constraint violations (violation_flag true) -----------------
    cc = store.get("layer5_chance_constraint_violations.csv")
    if cc is not None and len(cc) > 0 and "violation_flag" in cc.columns:
        mtime = _file_mtime_iso("layer5_chance_constraint_violations.csv")
        flag = cc["violation_flag"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        for _, r in cc[flag].iterrows():
            eid = str(r.get("event_id", ""))
            recs.append({
                "source_feed": "cc_violations",
                "original_id": f"CCV_{eid}",
                # a hard chance-constraint breach is treated as critical
                "severity_raw": "critical",
                "affected_layer": "Layer5",
                "variable": f"event:{eid}",
                "affected_event_id": eid,
                "description": (
                    f"Chance-constraint violation: tier={r.get('service_tier', '')} "
                    f"realized={r.get('realized_satisfaction_rate', '')} "
                    f"required={r.get('required_prob', '')} margin={r.get('violation_margin', '')}"
                ),
                "timestamp": mtime,
            })

    return pd.DataFrame(recs)


def prioritize(store: Store) -> pd.DataFrame:
    df = _collect(store)
    if len(df) == 0:
        # valid all-healthy batch
        return pd.DataFrame(columns=[
            "l7_alert_id", "source_feed", "original_id", "topic_key", "severity_raw",
            "base_severity", "corroboration_sources", "corroboration_factor",
            "age_hours", "recency_factor", "alert_severity_score", "priority",
            "affected_layer", "affected_event_id", "description", "timestamp",
            "merged_count", "generated_at",
        ])

    # base severity
    df["base_severity"] = df["severity_raw"].map(BASE_SEVERITY)
    df["severity_unknown"] = df["base_severity"].isna()
    df["base_severity"] = df["base_severity"].fillna(SEVERITY_FALLBACK)

    # topic + corroboration (distinct feeds per topic)
    df["topic_key"] = [
        _topic_key(o, a, v)
        for o, a, v in zip(df["original_id"], df["affected_layer"], df["variable"])
    ]
    topic_sources = df.groupby("topic_key")["source_feed"].nunique()
    df["corroboration_sources"] = df["topic_key"].map(topic_sources).astype(int)
    df["corroboration_factor"] = 1.0 + CORROBORATION_PER_SOURCE * df["corroboration_sources"]

    # recency (timestamps mix '...Z' and '+00:00' ISO forms -> parse as mixed)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    age_hours = (_NOW - ts).dt.total_seconds() / 3600.0
    df["age_hours"] = age_hours.clip(lower=0).fillna(0.0)
    df["recency_factor"] = np.exp(-df["age_hours"] / RECENCY_HALFLIFE_HOURS)

    # ASS (PATCH F-002: keep multiplicative form, bound to [0,1] by theoretical max)
    df["alert_severity_raw"] = (
        df["base_severity"] * df["corroboration_factor"] * df["recency_factor"]
    )
    df["alert_severity_score"] = (df["alert_severity_raw"] / _ASS_MAX).clip(0.0, 1.0)
    # priority is assigned AFTER dedup, by quantile of the bounded ASS (PATCH F-003)

    # unique l7 alert id (+ dedupe identical source/original pairs, keep max ASS)
    df["l7_alert_id"] = df["source_feed"] + "::" + df["original_id"]
    df = df.sort_values("alert_severity_score", ascending=False)
    merged = df.groupby("l7_alert_id").size().rename("merged_count")
    df = df.drop_duplicates("l7_alert_id", keep="first").merge(
        merged, on="l7_alert_id", how="left"
    )
    df["generated_at"] = _NOW_ISO

    # deterministic ordering
    df = df.sort_values(["alert_severity_score", "l7_alert_id"], ascending=[False, True])
    df = df.reset_index(drop=True)
    df.insert(0, "priority_rank", np.arange(1, len(df) + 1))

    # PATCH F-003: quantile-based priority on the bounded ASS distribution, so P1 is the
    # genuine top band instead of every corroborated critical clearing a fixed 0.85 cut.
    ass_pct = df["alert_severity_score"].rank(pct=True, method="average")

    def _priority_pct(p: float) -> str:
        for label, cut in ASS_PRIORITY_PCT_CUTS:
            if p >= cut:
                return label
        return ASS_PRIORITY_PCT_CUTS[-1][0]

    df["priority"] = ass_pct.apply(_priority_pct)

    cols = [
        "priority_rank", "l7_alert_id", "source_feed", "original_id", "topic_key",
        "severity_raw", "severity_unknown", "base_severity", "corroboration_sources",
        "corroboration_factor", "age_hours", "recency_factor",
        "alert_severity_raw", "alert_severity_score",
        "priority", "affected_layer", "affected_event_id", "description", "timestamp",
        "merged_count", "generated_at",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def build_summary(alerts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for label, _ in ASS_PRIORITY_CUTS:
        rows.append({
            "dimension": "priority", "key": label,
            "n_alerts": int((alerts["priority"] == label).sum()) if len(alerts) else 0,
        })
    for sev in ("critical", "moderate", "warning", "info"):
        n = int((alerts["severity_raw"] == sev).sum()) if len(alerts) else 0
        rows.append({"dimension": "severity_raw", "key": sev, "n_alerts": n})
    if len(alerts):
        for feed, n in alerts["source_feed"].value_counts().items():
            rows.append({"dimension": "source_feed", "key": feed, "n_alerts": int(n)})
    rows.append({"dimension": "total", "key": "all", "n_alerts": int(len(alerts))})
    rows.append({"dimension": "meta", "key": "generated_at", "n_alerts": _NOW_ISO})
    return pd.DataFrame(rows)


def build_diagnostics(alerts: pd.DataFrame) -> pd.DataFrame:
    if len(alerts) == 0:
        return pd.DataFrame(columns=[
            "topic_key", "n_alerts", "n_sources", "sources", "max_ass",
            "top_priority", "generated_at",
        ])
    g = alerts.groupby("topic_key")
    diag = g.agg(
        n_alerts=("l7_alert_id", "count"),
        n_sources=("source_feed", "nunique"),
        sources=("source_feed", lambda s: ";".join(sorted(set(s)))),
        max_ass=("alert_severity_score", "max"),
        top_priority=("priority", "min"),  # P1 < P2 lexically
    ).reset_index()
    diag["generated_at"] = _NOW_ISO
    return diag.sort_values("max_ass", ascending=False).reset_index(drop=True)


def run(store: Store, write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    alerts = prioritize(store)
    summary = build_summary(alerts)
    diagnostics = build_diagnostics(alerts)

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        alerts.to_csv(OUT / "layer7_prioritized_alerts.csv", index=False)
        summary.to_csv(OUT / "layer7_alert_summary.csv", index=False)
        diagnostics.to_csv(OUT / "layer7_alert_diagnostics.csv", index=False)

    checks = _validate(alerts)
    return {"alerts": alerts, "summary": summary, "diagnostics": diagnostics}, checks


def _validate(alerts: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []

    def chk(cid: str, passed: bool, detail: str, severity: str = "critical") -> None:
        checks.append({
            "check_id": cid, "phase": "alert_prioritization", "passed": bool(passed),
            "detail": detail, "severity": "info" if passed else severity,
        })

    n_dup = int(alerts["l7_alert_id"].duplicated().sum()) if len(alerts) else 0
    chk("alerts_no_duplicate_ids", n_dup == 0, f"{n_dup} duplicate l7_alert_id")

    if len(alerts) > 1:
        ass = alerts["alert_severity_score"].to_numpy()
        monotone = bool(np.all(ass[:-1] >= ass[1:] - 1e-12))
        chk("alerts_ordering_reproducible", monotone,
            "feed sorted by ASS desc (deterministic tie-break on id)")
    else:
        chk("alerts_ordering_reproducible", True, f"{len(alerts)} alert(s); ordering trivial")

    if len(alerts):
        dist = alerts["priority"].value_counts().to_dict()
        n_nan = int(alerts["alert_severity_score"].isna().sum())
        chk("alerts_no_nan_ass", n_nan == 0, f"{n_nan} NaN ASS values")
        chk("alerts_severity_distribution_reported", True, f"priority distribution: {dist}")
    else:
        chk("alerts_no_nan_ass", True, "empty feed (valid all-healthy batch)")
        chk("alerts_severity_distribution_reported", True, "priority distribution: {} (empty feed)")

    return checks


if __name__ == "__main__":
    from layer7_loader import audit_inputs

    _store, _ = audit_inputs(write=False)
    _tables, _checks = run(_store, write=True)
    print("=== Layer 7 Alert Prioritization Engine ===")
    print(_tables["summary"].to_string(index=False))
    for c in _checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
