"""
Layer 7 — M3 (Part B): Counterfactual Analysis Engine.

Answers the operator question "what if we did NOT follow the recommendation?"
via SURROGATE analysis only — NO retraining, NO optimization reruns, NO new
model. Deltas are closed-form restatements of published Layer 5 quantities
(shadow prices, expected_delay_reduction_min, effectiveness, robustness) and
Layer 7 ORS / alert burden.

Scenarios per active site:
  A Recommended action (baseline reference, deltas = 0)
  B No action          (remove resources)
  C Reduced resources  (x0.5)
  D Increased resources (+2 officers, +1 tow)
  E Diversion disabled

ADDITIVE ONLY. Writes only outputs/layer7_counterfactual_analysis.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT
from layer7_explanation_engine import (
    OIS_W_ALERT,
    OIS_W_DELAY,
    OIS_W_RISK,
    compute_absolute_anchors,
)

_NOW_ISO = datetime.now(timezone.utc).isoformat()

_SCENARIOS = ["A_recommended", "B_no_action", "C_reduced_resources",
              "D_increased_resources", "E_diversion_disabled"]


def _cf_score(dD, dR, dA, anchors) -> float:
    nd = min(1.0, abs(dD) / anchors["delay"])
    nr = min(1.0, abs(dR) / anchors["risk"])
    na = min(1.0, abs(dA) / anchors["alert"])
    return float(np.clip(OIS_W_DELAY * nd + OIS_W_RISK * nr + OIS_W_ALERT * na, 0, 1))


def build_counterfactuals() -> pd.DataFrame:
    anchors = compute_absolute_anchors()
    alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)
    shadow = pd.read_csv(OUT / "layer5_shadow_prices.csv")
    marg = dict(zip(shadow["resource"].astype(str), shadow["marginal_value"].astype(float)))
    mval_police = marg.get("police", 0.0)
    mval_tow = marg.get("tow", 0.0)

    try:
        astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
        astate["event_id"] = astate["event_id"].astype(str)
        ors_map = dict(zip(astate["event_id"], astate["operational_risk_score"].astype(float)))
    except Exception:
        ors_map = {}
    try:
        alerts = pd.read_csv(OUT / "layer7_prioritized_alerts.csv")
        alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
        burden_map = alerts.groupby("affected_event_id")["alert_severity_score"].sum().to_dict()
    except Exception:
        burden_map = {}

    rows = []
    for _, r in alloc.iterrows():
        eid = str(r["event_id"])
        D_r = float(pd.to_numeric(r.get("expected_delay_reduction_min"), errors="coerce") or 0.0)
        e_r = float(pd.to_numeric(r.get("effectiveness"), errors="coerce") or 0.0)
        ors = float(ors_map.get(eid, 0.0))
        burden = float(burden_map.get(eid, 0.0))
        div_active = str(r.get("diversion_activated", "")).strip().lower() in ("1", "true", "yes")

        # Each scenario's (Δdelay, Δrisk, Δalert) relative to the recommended plan.
        scen = {
            "A_recommended": (0.0, 0.0, 0.0),
            # remove all resources -> lose the full recommended delay reduction
            "B_no_action": (+D_r, +ors * e_r, +burden),
            # halve resources -> lose ~half
            "C_reduced_resources": (+0.5 * D_r, +0.5 * ors * e_r, +0.5 * burden),
            # add 2 officers + 1 tow -> extra marginal delay reduction (negative delta)
            "D_increased_resources": (
                -(2.0 * mval_police + 1.0 * mval_tow),
                -ors * min(1.0, 0.20),
                -0.25 * burden,
            ),
            # disable diversion -> lose its share of the reduction if it was active
            "E_diversion_disabled": (
                (+0.30 * D_r) if div_active else 0.0,
                (+0.15 * ors) if div_active else 0.0,
                0.0,
            ),
        }
        for stype in _SCENARIOS:
            dD, dR, dA = scen[stype]
            rows.append({
                "event_id": eid,
                "scenario_type": stype,
                "expected_delay_delta": round(dD, 4),
                "expected_risk_delta": round(dR, 4),
                "expected_alert_delta": round(dA, 4),
                "counterfactual_score": round(_cf_score(dD, dR, dA, anchors), 6),
                "generated_at": _NOW_ISO,
            })
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    cf = build_counterfactuals()
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        cf.to_csv(OUT / "layer7_counterfactual_analysis.csv", index=False)
    checks = _validate(cf)
    return cf, checks


def _validate(cf: pd.DataFrame) -> list[dict]:
    checks: list[dict] = []

    def chk(cid, passed, detail, severity="critical"):
        checks.append({"check_id": cid, "phase": "counterfactual_engine",
                       "passed": bool(passed), "detail": detail,
                       "severity": "info" if passed else severity})

    n_sites = cf["event_id"].nunique()
    expected = n_sites * len(_SCENARIOS)
    chk("m3_counterfactuals_generated", len(cf) == expected,
        f"{len(cf)} rows = {n_sites} sites x {len(_SCENARIOS)} scenarios (expected {expected})")

    n_nan = int(cf.isna().sum().sum())
    chk("m3_counterfactuals_no_nan", n_nan == 0, f"{n_nan} NaN in counterfactual output")

    in_range = bool(((cf["counterfactual_score"] >= 0) & (cf["counterfactual_score"] <= 1)).all())
    chk("m3_counterfactual_score_range", in_range,
        f"score range [{cf['counterfactual_score'].min():.4f}, {cf['counterfactual_score'].max():.4f}]")

    # baseline scenario A must be the zero-reference
    a = cf[cf["scenario_type"] == "A_recommended"]
    base_zero = bool((a[["expected_delay_delta", "expected_risk_delta",
                         "expected_alert_delta"]].abs().to_numpy() < 1e-9).all())
    chk("m3_counterfactual_baseline_zero", base_zero,
        "Scenario A (recommended) deltas are zero reference")

    return checks


if __name__ == "__main__":
    cf, checks = run(write=True)
    print("=== Layer 7 M3 Counterfactual Engine ===")
    print(cf.groupby("scenario_type")["counterfactual_score"].mean().round(4).to_dict())
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
