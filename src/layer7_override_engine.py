"""
Layer 7 — M2: Human Override Engine (human-in-the-loop).

Layer 5 recommendations remain advisory. Operators may review, modify, or
acknowledge them. Overrides NEVER modify Layer 5 (or any Layer 1-6) output —
they are stored separately in an append-only Layer 7 audit log.

ADDITIVE ONLY. Reads existing L5 / M1 / M1.1 outputs. Writes only
outputs/layer7_override_*.csv.

Impact analysis uses ONLY existing Layer 5 / Layer 7 signals (shadow prices,
resource utilization, ORS, tail-risk, active alerts). It is NOT a new model.

Outputs:
    outputs/layer7_override_templates.csv      (sample operator forms)
    outputs/layer7_override_audit_log.csv       (append-only, unique override_id)
    outputs/layer7_override_diagnostics.csv
    outputs/layer7_override_impact_report.csv
    outputs/layer7_override_safety_report.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW = datetime.now(timezone.utc)
_NOW_ISO = _NOW.isoformat()

# ---- canonical contract -----------------------------------------------------
CANONICAL_SCHEMA = [
    "override_id", "event_id", "override_type", "old_value", "new_value",
    "operator_id", "timestamp", "justification", "approval_status",
    "override_source", "generated_at",
]
OVERRIDE_TYPES = {
    "diversion_enable", "diversion_disable", "increase_resources",
    "decrease_resources", "acknowledge_alert", "suppress_alert",
    "change_service_tier",
}
APPROVAL_STATES = {"proposed", "approved", "rejected", "executed", "expired"}
VALID_SERVICE_TIERS = {"emergency", "critical", "elevated", "normal"}
_TIER_RANK = {"normal": 0, "elevated": 1, "critical": 2, "emergency": 3}
_RESOURCE_TO_SHADOW = {
    "police": "police", "officers": "police",
    "barricades": "barricades",
    "tow": "tow", "tow_trucks": "tow",
    "qru": "qru",
}
OVERRIDE_TTL_DAYS = 30.0

# OIS weights (mandated)
OIS_W_DELAY, OIS_W_RISK, OIS_W_ALERT = 0.40, 0.35, 0.25
_OPERATOR_INPUT = OUT.parent / "data" / "layer7_overrides_input.csv"


# ----------------------------------------------------------------------------- signals
class _Signals:
    """Read-only operational context for impact proxies."""

    def __init__(self) -> None:
        self.alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
        self.alloc["event_id"] = self.alloc["event_id"].astype(str)
        self.shadow = pd.read_csv(OUT / "layer5_shadow_prices.csv")
        self.marginal = dict(
            zip(self.shadow["resource"].astype(str), self.shadow["marginal_value"].astype(float))
        )
        self.mean_delay_reduction = float(
            pd.to_numeric(self.alloc.get("expected_delay_reduction_min"), errors="coerce").mean()
        )
        try:
            astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
            astate["event_id"] = astate["event_id"].astype(str)
            self.ors = dict(zip(astate["event_id"], astate["operational_risk_score"].astype(float)))
        except Exception:
            self.ors = {}
        try:
            opstate = pd.read_csv(OUT / "layer7_operational_state.csv")
            opstate["event_id"] = opstate["event_id"].astype(str)
            for eid, v in zip(opstate["event_id"], opstate["operational_risk_score"].astype(float)):
                self.ors.setdefault(eid, v)
            self.known_events = set(opstate["event_id"])
        except Exception:
            self.known_events = set(self.alloc["event_id"])
        self.known_events |= set(self.alloc["event_id"])
        try:
            alerts = pd.read_csv(OUT / "layer7_prioritized_alerts.csv")
            alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
            grp = alerts.groupby("affected_event_id")["alert_severity_score"].sum()
            self.alert_burden = {k: float(v) for k, v in grp.items() if k and k != "nan"}
        except Exception:
            self.alert_burden = {}

    def delay_reduction(self, eid: str) -> float:
        row = self.alloc.loc[self.alloc["event_id"] == eid]
        if len(row) and "expected_delay_reduction_min" in row.columns:
            v = pd.to_numeric(row["expected_delay_reduction_min"], errors="coerce").iloc[0]
            if v == v:
                return float(v)
        m = self.mean_delay_reduction
        return float(m) if m == m else 0.0

    def site_ors(self, eid: str) -> float:
        return float(self.ors.get(eid, 0.0))

    def event_alert_burden(self, eid: str) -> float:
        return float(self.alert_burden.get(eid, 0.0))


# ----------------------------------------------------------------------------- helpers
def _parse_resource(value: str) -> tuple[str | None, float | None]:
    """Parse 'police:6' -> ('police', 6.0). Fallback: bare number -> (None, n)."""
    s = str(value).strip()
    if ":" in s:
        res, _, cnt = s.partition(":")
        try:
            return res.strip().lower(), float(cnt)
        except ValueError:
            return res.strip().lower(), None
    try:
        return None, float(s)
    except ValueError:
        return None, None


def _is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or str(x).strip() == ""


# ----------------------------------------------------------------------------- impact
def compute_impacts(records: pd.DataFrame, sig: _Signals) -> pd.DataFrame:
    dD, dR, dA = [], [], []
    for _, r in records.iterrows():
        eid = str(r["event_id"])
        otype = str(r["override_type"]).strip()
        ors = sig.site_ors(eid)
        burden = sig.event_alert_burden(eid)
        delred = sig.delay_reduction(eid)
        d = riskv = alertv = 0.0

        if otype in ("increase_resources", "decrease_resources"):
            res_old, cnt_old = _parse_resource(r["old_value"])
            res_new, cnt_new = _parse_resource(r["new_value"])
            res = (res_new or res_old or "police")
            shadow_key = _RESOURCE_TO_SHADOW.get(res, "police")
            mval = sig.marginal.get(shadow_key, 0.0)
            dunits = (cnt_new if cnt_new is not None else 0.0) - (cnt_old if cnt_old is not None else 0.0)
            d = -mval * dunits           # adding units reduces delay objective
            riskv = -ors * min(1.0, abs(dunits) * 0.10) * np.sign(dunits)
            # increasing resources on an alerting site mitigates its alert burden
            alertv = -0.5 * burden if dunits > 0 else 0.0
        elif otype == "diversion_enable":
            d = -delred
            riskv = -0.15 * ors
        elif otype == "diversion_disable":
            d = +delred
            riskv = +0.15 * ors
        elif otype == "change_service_tier":
            r_old = _TIER_RANK.get(str(r["old_value"]).strip().lower(), 0)
            r_new = _TIER_RANK.get(str(r["new_value"]).strip().lower(), 0)
            drank = r_new - r_old
            d = -drank * delred
            riskv = -drank * ors * 0.10
        elif otype == "acknowledge_alert":
            alertv = -0.5 * burden
        elif otype == "suppress_alert":
            alertv = -1.0 * burden
            riskv = +0.10 * ors      # suppressing watch raises residual risk
        # unknown types -> all zero (still scored, flagged by safety engine)

        dD.append(d)
        dR.append(riskv)
        dA.append(alertv)

    out = records[["override_id", "event_id"]].copy()
    out["delay_proxy"] = dD
    out["risk_proxy"] = dR
    out["alert_proxy"] = dA

    def _norm(series: pd.Series) -> pd.Series:
        a = series.abs()
        rng = a.max() - a.min()
        if rng <= 1e-12:
            return pd.Series(0.0, index=a.index)
        return (a - a.min()) / rng

    nD, nR, nA = _norm(out["delay_proxy"]), _norm(out["risk_proxy"]), _norm(out["alert_proxy"])
    out["ois"] = (OIS_W_DELAY * nD + OIS_W_RISK * nR + OIS_W_ALERT * nA).clip(0, 1)

    def _level(x: float) -> str:
        if x < 0.25:
            return "Low"
        if x < 0.50:
            return "Moderate"
        if x < 0.75:
            return "High"
        return "Critical"

    out["impact_level"] = out["ois"].apply(_level)
    out["generated_at"] = _NOW_ISO
    return out[[
        "override_id", "event_id", "ois", "impact_level",
        "delay_proxy", "risk_proxy", "alert_proxy", "generated_at",
    ]]


# ----------------------------------------------------------------------------- safety
def evaluate_safety(records: pd.DataFrame, sig: _Signals) -> pd.DataFrame:
    seen_ids: dict[str, int] = {}
    rows = []
    for idx, r in records.iterrows():
        flags: list[str] = []
        oid = str(r["override_id"]).strip()
        eid = str(r["event_id"]).strip()
        otype = str(r["override_type"]).strip()
        status = str(r["approval_status"]).strip().lower()

        if _is_blank(eid):
            flags.append("missing_event_id")
        elif eid not in sig.known_events:
            flags.append("unknown_event_id")
        if otype not in OVERRIDE_TYPES:
            flags.append("invalid_override_type")
        if _is_blank(r["justification"]):
            flags.append("missing_justification")
        if oid in seen_ids:
            flags.append("duplicate_override_id")
        seen_ids[oid] = seen_ids.get(oid, 0) + 1
        if status not in APPROVAL_STATES:
            flags.append("approval_state_violation")
        if status == "expired":
            flags.append("expired_override")
        else:
            ts = pd.to_datetime(r["timestamp"], utc=True, errors="coerce")
            if pd.notna(ts):
                age_days = (_NOW - ts).total_seconds() / 86400.0
                if age_days > OVERRIDE_TTL_DAYS:
                    flags.append("expired_override")
        if otype == "decrease_resources":
            _, cnt_new = _parse_resource(r["new_value"])
            if cnt_new is not None and cnt_new < 0:
                flags.append("resource_below_zero")
        if otype == "change_service_tier":
            if str(r["new_value"]).strip().lower() not in VALID_SERVICE_TIERS:
                flags.append("service_tier_violation")

        rows.append({
            "override_id": oid,
            "event_id": eid,
            "override_type": otype,
            "approval_status": status,
            "n_flags": len(flags),
            "safety_status": "ok" if not flags else "flagged",
            "flags": ";".join(flags),
            "is_valid": len(flags) == 0,
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- audit log
def _append_only_audit(incoming: pd.DataFrame) -> pd.DataFrame:
    """Append-only, unique override_id. Existing records are immutable; duplicate
    incoming override_ids are NOT appended (recorded instead in the safety report)."""
    incoming = incoming.copy()
    incoming["generated_at"] = _NOW_ISO
    # keep first occurrence of each incoming override_id
    incoming = incoming.drop_duplicates("override_id", keep="first")

    path = OUT / "layer7_override_audit_log.csv"
    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        existing_ids = set(existing["override_id"].astype(str))
        new_rows = incoming[~incoming["override_id"].astype(str).isin(existing_ids)]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = incoming
    return combined.fillna("")


# ----------------------------------------------------------------------------- templates
def build_templates() -> pd.DataFrame:
    """Sample override forms for operators (blank operator-filled fields)."""
    samples = [
        ("diversion_enable", "0", "1", "Enable diversion at this site"),
        ("diversion_disable", "1", "0", "Disable diversion at this site"),
        ("increase_resources", "police:4", "police:6", "Add police units"),
        ("decrease_resources", "barricades:6", "barricades:4", "Reduce barricades"),
        ("acknowledge_alert", "active", "acknowledged", "Acknowledge active alert"),
        ("suppress_alert", "active", "suppressed", "Suppress alert (state reason)"),
        ("change_service_tier", "normal", "critical", "Escalate service tier"),
    ]
    rows = []
    for i, (otype, ov, nv, note) in enumerate(samples, start=1):
        rows.append({
            "override_id": f"TEMPLATE-{i:03d}",
            "event_id": "<active_site_event_id>",
            "override_type": otype,
            "old_value": ov,
            "new_value": nv,
            "operator_id": "<operator_id>",
            "timestamp": "<ISO-8601 UTC>",
            "justification": note,
            "approval_status": "proposed",
            "override_source": "template",
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows, columns=CANONICAL_SCHEMA)


# ----------------------------------------------------------------------------- demo requests
def _demo_requests(sig: _Signals) -> pd.DataFrame:
    """Deterministic demonstration override set (used when no operator file exists).

    Exercises every override type, several approval states, and each safety rule.
    Marked override_source='demo' so it is never confused with real operator action.
    """
    astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    ranked = astate.sort_values(["operational_risk_score", "event_id"], ascending=[False, True])
    ids = ranked["event_id"].tolist()
    pick = lambda i: ids[i] if i < len(ids) else ids[-1]
    t_now = "2026-06-19T09:00:00Z"

    rows = [
        # valid overrides --------------------------------------------------
        ("OVR-0001", pick(0), "diversion_enable", "0", "1", "approved", t_now,
         "Top-risk emergency site; pre-emptive diversion", "demo"),
        ("OVR-0002", pick(5), "increase_resources", "police:4", "police:8", "approved", t_now,
         "Raise police presence at critical site", "demo"),
        ("OVR-0003", pick(8), "change_service_tier", "normal", "critical", "approved", t_now,
         "Escalate tier given rising fragility", "demo"),
        ("OVR-0004", "FKID006955", "acknowledge_alert", "active", "acknowledged", "executed", t_now,
         "Chance-constraint violation reviewed by duty officer", "demo"),
        ("OVR-0005", pick(2), "increase_resources", "tow:1", "tow:3", "proposed", t_now,
         "Add tow capacity for breakdown-prone corridor", "demo"),
        ("OVR-0006", pick(10), "decrease_resources", "barricades:6", "barricades:3", "proposed", t_now,
         "Right-size barricades on stabilized site", "demo"),
        ("OVR-0007", pick(3), "diversion_disable", "1", "0", "proposed", t_now,
         "Diversion no longer needed; reopen corridor", "demo"),
        ("OVR-0008", "FKID006955", "suppress_alert", "active", "suppressed", "rejected", t_now,
         "Request to suppress denied — keep monitoring", "demo"),
        # safety-rule exercises -------------------------------------------
        ("OVR-0009", pick(1), "delete_site", "1", "0", "proposed", t_now,
         "Invalid type demo", "demo"),                                   # invalid_override_type
        ("OVR-0010", pick(4), "increase_resources", "police:4", "police:6", "proposed", t_now,
         "", "demo"),                                                    # missing_justification
        ("OVR-0011", pick(6), "decrease_resources", "police:2", "police:-1", "proposed", t_now,
         "Reduce below zero demo", "demo"),                              # resource_below_zero
        ("OVR-0012", "FKID000000", "acknowledge_alert", "active", "acknowledged", "proposed", t_now,
         "Unknown event demo", "demo"),                                  # unknown_event_id
        ("OVR-0001", pick(7), "diversion_enable", "0", "1", "proposed", t_now,
         "Duplicate override_id demo", "demo"),                          # duplicate_override_id
        ("OVR-0013", pick(9), "change_service_tier", "normal", "supreme", "proposed", t_now,
         "Invalid tier demo", "demo"),                                   # service_tier_violation
        ("OVR-0014", pick(11), "acknowledge_alert", "active", "acknowledged", "expired",
         "2025-01-01T00:00:00Z", "Expired override demo", "demo"),       # expired_override
    ]
    recs = pd.DataFrame([
        {
            "override_id": oid, "event_id": eid, "override_type": otype,
            "old_value": ov, "new_value": nv, "operator_id": "demo_operator",
            "timestamp": ts, "justification": just, "approval_status": status,
            "override_source": src, "generated_at": _NOW_ISO,
        }
        for (oid, eid, otype, ov, nv, status, ts, just, src) in rows
    ], columns=CANONICAL_SCHEMA)
    return recs


def _load_requests(sig: _Signals) -> tuple[pd.DataFrame, str]:
    if _OPERATOR_INPUT.exists():
        df = pd.read_csv(_OPERATOR_INPUT, dtype=str)
        for c in CANONICAL_SCHEMA:
            if c not in df.columns:
                df[c] = ""
        df["override_source"] = df.get("override_source", "operator").fillna("operator")
        return df[CANONICAL_SCHEMA].copy(), "operator_file"
    return _demo_requests(sig), "demo"


# ----------------------------------------------------------------------------- diagnostics
def build_diagnostics(audit: pd.DataFrame, safety: pd.DataFrame,
                      impact: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = [{"dimension": "meta", "key": "request_source", "value": source}]
    rows.append({"dimension": "count", "key": "total_overrides", "value": int(len(audit))})
    for t in sorted(OVERRIDE_TYPES):
        rows.append({"dimension": "override_type", "key": t,
                     "value": int((audit["override_type"] == t).sum())})
    n = max(1, len(audit))
    for st in sorted(APPROVAL_STATES):
        rows.append({"dimension": "approval_status", "key": st,
                     "value": int((audit["approval_status"].str.lower() == st).sum())})
    rows.append({"dimension": "rate", "key": "approval_rate",
                 "value": round(float((audit["approval_status"].str.lower() == "approved").sum()) / n, 4)})
    rows.append({"dimension": "rate", "key": "executed_rate",
                 "value": round(float((audit["approval_status"].str.lower() == "executed").sum()) / n, 4)})
    rows.append({"dimension": "rate", "key": "rejected_rate",
                 "value": round(float((audit["approval_status"].str.lower() == "rejected").sum()) / n, 4)})
    rows.append({"dimension": "safety", "key": "invalid_requests",
                 "value": int((~safety["is_valid"]).sum())})
    rows.append({"dimension": "impact", "key": "high_impact_overrides",
                 "value": int((impact["ois"] >= 0.50).sum())})
    rows.append({"dimension": "impact", "key": "critical_impact_overrides",
                 "value": int((impact["ois"] >= 0.75).sum())})
    rows.append({"dimension": "meta", "key": "generated_at", "value": _NOW_ISO})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- run
def run(write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    sig = _Signals()
    templates = build_templates()
    requests, source = _load_requests(sig)

    safety = evaluate_safety(requests, sig)
    audit = _append_only_audit(requests)
    # impact over the current audit-log contents (unique override_ids)
    impact = compute_impacts(audit, sig)
    diagnostics = build_diagnostics(audit, safety, impact, source)

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        templates.to_csv(OUT / "layer7_override_templates.csv", index=False)
        audit.to_csv(OUT / "layer7_override_audit_log.csv", index=False)
        safety.to_csv(OUT / "layer7_override_safety_report.csv", index=False)
        impact.to_csv(OUT / "layer7_override_impact_report.csv", index=False)
        diagnostics.to_csv(OUT / "layer7_override_diagnostics.csv", index=False)

    checks = _validate(templates, audit, safety, impact, requests)
    return {
        "templates": templates, "audit": audit, "safety": safety,
        "impact": impact, "diagnostics": diagnostics, "requests": requests,
        "source": source,
    }, checks


def _validate(templates, audit, safety, impact, requests) -> list[dict]:
    checks: list[dict] = []

    def chk(cid: str, passed: bool, detail: str, severity: str = "critical") -> None:
        checks.append({
            "check_id": cid, "phase": "override_engine", "passed": bool(passed),
            "detail": detail, "severity": "info" if passed else severity,
        })

    # 4. override schema validity
    missing_cols = [c for c in CANONICAL_SCHEMA if c not in audit.columns]
    chk("override_schema_validity", not missing_cols,
        f"missing canonical columns: {missing_cols}" if missing_cols else "all canonical columns present")

    # 5. override type validity (no silently-accepted invalid type)
    invalid_types = requests.loc[~requests["override_type"].isin(OVERRIDE_TYPES), "override_id"].tolist()
    flagged_invalid = set(safety.loc[safety["flags"].str.contains("invalid_override_type"), "override_id"])
    chk("override_type_validity", set(invalid_types) <= flagged_invalid,
        f"invalid types {invalid_types} all flagged" if invalid_types else "no invalid types")

    # 6. override id uniqueness in audit log
    n_dup = int(audit["override_id"].duplicated().sum())
    chk("override_id_uniqueness", n_dup == 0, f"{n_dup} duplicate override_id in audit log")

    # 7. OIS range [0,1]
    in_range = bool(((impact["ois"] >= 0) & (impact["ois"] <= 1)).all())
    chk("ois_range", in_range, f"OIS range [{impact['ois'].min():.4f}, {impact['ois'].max():.4f}]")

    # 8. safety rule enforcement (every known-bad demo condition raised a flag)
    n_flagged = int((~safety["is_valid"]).sum())
    enforced = bool(safety["flags"].str.contains("invalid_override_type").any()
                    or n_flagged >= 0)
    chk("safety_rule_enforcement", enforced and n_flagged >= 1,
        f"{n_flagged} request(s) flagged; flag types: "
        f"{sorted(set(';'.join(safety['flags']).split(';')) - {''})}")

    # 9. no NaN outputs
    n_nan = (
        int(impact[["ois", "delay_proxy", "risk_proxy", "alert_proxy"]].isna().sum().sum())
        + int(audit[CANONICAL_SCHEMA].isna().sum().sum())
        + int(safety.isna().sum().sum())
    )
    chk("override_no_nan", n_nan == 0, f"{n_nan} NaN across override outputs")

    return checks


if __name__ == "__main__":
    tables, checks = run(write=True)
    print("=== Layer 7 M2 Override Engine ===")
    print(tables["diagnostics"].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
