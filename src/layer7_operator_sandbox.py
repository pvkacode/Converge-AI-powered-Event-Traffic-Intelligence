"""
Layer 7 — M6 Part G: Operator Sandbox.

Lets operators test hypothetical overrides (resource increases/reductions, diversion
changes) against the Digital Twin WITHOUT touching any production output. Read-only:
it consumes existing signals (shadow prices, counterfactual deltas, site context) and
estimates impact; it never writes to production override/audit logs.

Input (optional, operator-provided): data/layer7_sandbox_override.csv
  columns: sandbox_id, event_id, change_type, resource, delta_units, note
  change_type in {resource_increase, resource_reduction, diversion_enable, diversion_disable}
If absent, a deterministic demo set is generated (sandbox_source='demo').

Outputs:
  outputs/layer7_sandbox_results.csv
  outputs/layer7_sandbox_validation.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT
from layer7_explanation_engine import compute_absolute_anchors
from layer7_scenario_simulator import build_site_context

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_INPUT = OUT.parent / "data" / "layer7_sandbox_override.csv"

_CHANGE_TYPES = {"resource_increase", "resource_reduction",
                 "diversion_enable", "diversion_disable"}
_RES_TO_SHADOW = {"police": "police", "officers": "police", "barricades": "barricades",
                  "tow": "tow", "tow_trucks": "tow", "qru": "qru"}


def _shadow() -> dict:
    sp = pd.read_csv(OUT / "layer5_shadow_prices.csv")
    return dict(zip(sp["resource"].astype(str), sp["marginal_value"].astype(float)))


def _demo_requests(ctx: pd.DataFrame) -> pd.DataFrame:
    ids = ctx.sort_values("operational_risk_score", ascending=False)["event_id"].tolist()
    pick = lambda i: ids[i] if i < len(ids) else ids[-1]
    rows = [
        ("SBX-001", pick(0), "resource_increase", "police", 3, "Test +3 police at top site"),
        ("SBX-002", pick(1), "resource_increase", "tow", 2, "Test +2 tow"),
        ("SBX-003", pick(2), "resource_reduction", "barricades", 2, "Test -2 barricades"),
        ("SBX-004", pick(3), "diversion_enable", "", 0, "Test enabling diversion"),
        ("SBX-005", pick(4), "diversion_disable", "", 0, "Test disabling diversion"),
        # safety exercises
        ("SBX-006", "FKID000000", "resource_increase", "police", 2, "Unknown event"),
        ("SBX-007", pick(5), "teleport_units", "police", 1, "Invalid change_type"),
        ("SBX-008", pick(6), "resource_reduction", "qru", 99, "Reduction below zero"),
    ]
    return pd.DataFrame([{
        "sandbox_id": s, "event_id": e, "change_type": c, "resource": r,
        "delta_units": u, "note": n, "sandbox_source": "demo",
    } for (s, e, c, r, u, n) in rows])


def _load_requests(ctx: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if _INPUT.exists():
        df = pd.read_csv(_INPUT, dtype=str)
        for col in ["sandbox_id", "event_id", "change_type", "resource", "delta_units", "note"]:
            if col not in df.columns:
                df[col] = ""
        df["sandbox_source"] = df.get("sandbox_source", "operator").fillna("operator")
        return df, "operator_file"
    return _demo_requests(ctx), "demo"


def _validate_request(r, known_events: set, ctx_map: dict) -> tuple[list[str], float | None]:
    flags = []
    eid = str(r["event_id"]).strip()
    ctype = str(r["change_type"]).strip()
    if not eid:
        flags.append("missing_event_id")
    elif eid not in known_events:
        flags.append("unknown_event_id")
    if ctype not in _CHANGE_TYPES:
        flags.append("invalid_change_type")
    units = None
    if ctype in ("resource_increase", "resource_reduction"):
        try:
            units = float(r["delta_units"])
        except (ValueError, TypeError):
            flags.append("invalid_delta_units")
        if ctype == "resource_reduction" and units is not None and eid in ctx_map:
            # current allocation isn't tracked per-resource here; flag absurd reductions
            if units > 12:
                flags.append("reduction_exceeds_capacity")
    return flags, units


def run(write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    ctx = build_site_context()
    ctx_map = {str(r["event_id"]): r for _, r in ctx.iterrows()}
    known = set(ctx["event_id"].astype(str))
    anchors = compute_absolute_anchors()
    marg = _shadow()

    requests, source = _load_requests(ctx)
    res_rows, val_rows = [], []
    for _, r in requests.iterrows():
        flags, units = _validate_request(r, known, ctx_map)
        eid = str(r["event_id"]).strip()
        ctype = str(r["change_type"]).strip()
        c = ctx_map.get(eid)
        dD = dR = dA = 0.0
        if not flags and c is not None:
            ors = float(c["operational_risk_score"]); D_r = float(c["expected_delay_reduction_min"])
            burden = float(c["burden"])
            if ctype == "resource_increase":
                mval = marg.get(_RES_TO_SHADOW.get(str(r["resource"]).lower(), "police"), 0.0)
                dD = -mval * (units or 0.0)
                dR = -ors * min(1.0, (units or 0.0) * 0.10)
                dA = -0.25 * burden
            elif ctype == "resource_reduction":
                mval = marg.get(_RES_TO_SHADOW.get(str(r["resource"]).lower(), "police"), 0.0)
                dD = +mval * (units or 0.0)
                dR = +ors * min(1.0, (units or 0.0) * 0.10)
                dA = +0.25 * burden
            elif ctype == "diversion_enable":
                dD = -0.30 * D_r; dR = -0.15 * ors
            elif ctype == "diversion_disable":
                dD = +0.30 * D_r; dR = +0.15 * ors

        def comp(delta, anchor):
            return float(np.clip(0.5 + 0.5 * delta / max(anchor, 1e-9), 0, 1))

        ss = (0.5 * comp(dD, anchors["delay"]) + 0.4 * comp(dR, anchors["risk"])
              + 0.1 * comp(dA, anchors["alert"]))
        res_rows.append({
            "sandbox_id": r["sandbox_id"], "event_id": eid, "change_type": ctype,
            "resource": r.get("resource", ""), "delta_units": r.get("delta_units", ""),
            "expected_delay_change": round(dD, 4), "expected_risk_change": round(dR, 4),
            "expected_alert_change": round(dA, 4),
            "sandbox_simulation_score": round(float(np.clip(ss, 0, 1)), 6),
            "valid": len(flags) == 0, "flags": ";".join(flags),
            "sandbox_source": source, "generated_at": _NOW_ISO,
        })
        val_rows.append({
            "sandbox_id": r["sandbox_id"], "event_id": eid, "change_type": ctype,
            "n_flags": len(flags), "valid": len(flags) == 0, "flags": ";".join(flags),
            "generated_at": _NOW_ISO,
        })

    results = pd.DataFrame(res_rows)
    validation = pd.DataFrame(val_rows)
    if write:
        results.to_csv(OUT / "layer7_sandbox_results.csv", index=False)
        validation.to_csv(OUT / "layer7_sandbox_validation.csv", index=False)

    n_valid = int(validation["valid"].sum())
    n_flag = int((~validation["valid"]).sum())
    checks = [{
        "check_id": "m6_sandbox_works", "phase": "operator_sandbox",
        "passed": len(results) > 0 and int(results.isna().sum().sum()) == 0,
        "detail": f"source={source}; {len(results)} requests; valid={n_valid}; flagged={n_flag}; "
                  f"flag_types={sorted(set(';'.join(validation['flags']).split(';')) - {''})}",
        "severity": "info" if (len(results) > 0 and int(results.isna().sum().sum()) == 0) else "critical",
    }, {
        "check_id": "m6_sandbox_safety_enforced", "phase": "operator_sandbox",
        "passed": n_flag >= 1,
        "detail": f"{n_flag} request(s) flagged by sandbox safety checks",
        "severity": "info",
    }]
    return {"results": results, "validation": validation, "source": source}, checks


if __name__ == "__main__":
    tables, checks = run(write=True)
    print(tables["results"][["sandbox_id", "event_id", "change_type",
                             "sandbox_simulation_score", "valid", "flags"]].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
