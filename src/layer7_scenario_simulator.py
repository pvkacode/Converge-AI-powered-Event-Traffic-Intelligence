"""
Layer 7 — M6 Part C/D/F: Scenario Simulation Engine.

For every active site, simulates all 8 library scenarios and estimates delay / risk /
alert / override / confidence impact using existing signals (counterfactual outputs,
shadow prices, robustness, DCS). NO new model, NO optimization.

Simulation Score (badness, lower is better; 0.5 = baseline reference):
    SS = 0.35*delay + 0.30*risk + 0.15*alert + 0.10*override + 0.10*confidence
each component in [0,1], so SS in [0,1].

Component mapping (0.5-centred so improvements score < 0.5, worsening > 0.5):
    score = clip(0.5 + 0.5 * delta / anchor, 0, 1)

ADDITIVE ONLY. Writes outputs/layer7_twin_scenarios.csv and
outputs/layer7_scenario_ranking.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT
from layer7_explanation_engine import compute_absolute_anchors
from layer7_scenario_library import SCENARIOS

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# SS component weights (mandated)
_W = {"delay": 0.35, "risk": 0.30, "alert": 0.15, "override": 0.10, "confidence": 0.10}


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_site_context() -> pd.DataFrame:
    """One row per active site with all signals the simulator/sandbox need."""
    alloc = _read("layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)
    dcs = _read("layer7_decision_confidence.csv")
    dcs["event_id"] = dcs["event_id"].astype(str)
    astate = _read("layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    alerts = _read("layer7_prioritized_alerts.csv")
    burden = {}
    if len(alerts):
        alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
        burden = alerts.groupby("affected_event_id")["alert_severity_score"].sum().to_dict()
    # override impact joined via audit event_id
    audit = _read("layer7_override_audit_log.csv")
    imp = _read("layer7_override_impact_report.csv")
    ext = _read("layer7_override_impact_extended.csv")
    ov_by_event = {}
    if len(audit) and len(imp):
        m = imp.merge(ext[["override_id", "absolute_ois"]], on="override_id", how="left") \
            if len(ext) else imp.assign(absolute_ois=0.0)
        a2 = audit[["override_id", "event_id"]].astype(str)
        m = m.astype({"override_id": str}).merge(a2, on="override_id", how="left",
                                                 suffixes=("", "_audit"))
        for _, r in m.iterrows():
            eid = str(r.get("event_id_audit", r.get("event_id", "")))
            ov_by_event.setdefault(eid, {
                "delay": float(r.get("delay_proxy", 0.0) or 0.0),
                "risk": float(r.get("risk_proxy", 0.0) or 0.0),
                "alert": float(r.get("alert_proxy", 0.0) or 0.0),
                "abs_ois": float(r.get("absolute_ois", 0.0) or 0.0),
            })

    ctx = astate[["event_id", "operational_risk_score", "robustness_score"]].copy()
    ctx = ctx.merge(alloc[["event_id", "expected_delay_reduction_min", "effectiveness"]],
                    on="event_id", how="left")
    ctx = ctx.merge(dcs[["event_id", "decision_confidence_score"]], on="event_id", how="left")
    ctx["burden"] = ctx["event_id"].map(burden).fillna(0.0)
    ctx["ov_delay"] = ctx["event_id"].map(lambda e: ov_by_event.get(e, {}).get("delay", 0.0))
    ctx["ov_risk"] = ctx["event_id"].map(lambda e: ov_by_event.get(e, {}).get("risk", 0.0))
    ctx["ov_alert"] = ctx["event_id"].map(lambda e: ov_by_event.get(e, {}).get("alert", 0.0))
    ctx["ov_abs_ois"] = ctx["event_id"].map(lambda e: ov_by_event.get(e, {}).get("abs_ois", 0.0))
    for c in ["expected_delay_reduction_min", "effectiveness", "decision_confidence_score"]:
        ctx[c] = pd.to_numeric(ctx[c], errors="coerce").fillna(0.0)
    return ctx


def _drift_factor() -> float:
    dr = _read("layer6_drift_report.csv")
    if not len(dr):
        return 0.5
    da = dr[dr["alert"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if not len(da):
        return 0.0
    return float(np.clip(pd.to_numeric(da["retrain_urgency"], errors="coerce").max(), 0, 1))


def _counterfactual_deltas() -> dict:
    cf = _read("layer7_counterfactual_analysis.csv")
    out: dict[tuple, tuple] = {}
    if len(cf):
        for _, r in cf.iterrows():
            out[(str(r["event_id"]), str(r["scenario_type"]))] = (
                float(r["expected_delay_delta"]), float(r["expected_risk_delta"]),
                float(r["expected_alert_delta"]))
    return out


def simulate() -> pd.DataFrame:
    ctx = build_site_context()
    anchors = compute_absolute_anchors()
    cf = _counterfactual_deltas()
    drift = _drift_factor()

    def comp(delta: float, anchor: float) -> float:
        return float(np.clip(0.5 + 0.5 * delta / max(anchor, 1e-9), 0, 1))

    rows = []
    for _, r in ctx.iterrows():
        eid = str(r["event_id"])
        D_r = float(r["expected_delay_reduction_min"])
        ors = float(r["operational_risk_score"])
        burden = float(r["burden"])
        e_r = float(r["effectiveness"])
        dcs = float(r["decision_confidence_score"])
        for (sid, name, _desc, cf_map, deriv, conf_delta) in SCENARIOS:
            if deriv == "baseline":
                dD = dR = dA = 0.0
            elif deriv == "counterfactual":
                dD, dR, dA = cf.get((eid, cf_map), (0.0, 0.0, 0.0))
            elif deriv == "override":
                dD, dR, dA = float(r["ov_delay"]), float(r["ov_risk"]), float(r["ov_alert"])
            elif deriv == "drift_worst":
                dD = +0.5 * D_r * drift
                dR = +ors * drift
                dA = +burden * drift
            elif deriv == "stabilized":
                dD = -0.5 * D_r
                dR = -ors * e_r
                dA = -burden
            else:  # pragma: no cover
                dD = dR = dA = 0.0

            delay_score = comp(dD, anchors["delay"])
            risk_score = comp(dR, anchors["risk"])
            alert_score = comp(dA, anchors["alert"])
            # override component: applying an override is a deviation cost
            ov_dev = float(r["ov_abs_ois"]) if deriv == "override" else 0.0
            override_score = float(np.clip(0.5 + 0.5 * ov_dev, 0, 1))
            # confidence component: improvement (positive conf_delta) lowers badness
            confidence_score = float(np.clip(0.5 - 0.5 * conf_delta, 0, 1))

            ss = (_W["delay"] * delay_score + _W["risk"] * risk_score
                  + _W["alert"] * alert_score + _W["override"] * override_score
                  + _W["confidence"] * confidence_score)
            rows.append({
                "event_id": eid, "scenario_id": sid, "scenario_name": name,
                "delay_delta": round(dD, 4), "risk_delta": round(dR, 4),
                "alert_delta": round(dA, 4), "confidence_delta": round(conf_delta, 4),
                "delay_score": round(delay_score, 6), "risk_score": round(risk_score, 6),
                "alert_score": round(alert_score, 6), "override_score": round(override_score, 6),
                "confidence_score": round(confidence_score, 6),
                "simulation_score": round(float(np.clip(ss, 0, 1)), 6),
                "generated_at": _NOW_ISO,
            })
    return pd.DataFrame(rows)


def build_ranking(sims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eid, grp in sims.groupby("event_id"):
        g = grp.sort_values(["simulation_score", "scenario_id"])
        best = g.iloc[0]
        worst = g.iloc[-1]
        base = grp[grp["scenario_id"] == "SCENARIO_0"].iloc[0]
        rows.append({
            "event_id": eid,
            "best_scenario_id": best["scenario_id"], "best_scenario_score": best["simulation_score"],
            "worst_scenario_id": worst["scenario_id"], "worst_scenario_score": worst["simulation_score"],
            "baseline_scenario_id": "SCENARIO_0", "baseline_score": base["simulation_score"],
            "best_vs_baseline": round(float(base["simulation_score"] - best["simulation_score"]), 6),
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    sims = simulate()
    ranking = build_ranking(sims)
    twin = sims[["event_id", "scenario_id", "scenario_name", "delay_delta",
                 "risk_delta", "alert_delta", "confidence_delta", "simulation_score"]].copy()
    if write:
        twin.to_csv(OUT / "layer7_twin_scenarios.csv", index=False)
        ranking.to_csv(OUT / "layer7_scenario_ranking.csv", index=False)
        sims.to_csv(OUT / "layer7_twin_scenarios_full.csv", index=False)

    n_sites = sims["event_id"].nunique()
    n_scen = sims["scenario_id"].nunique()
    checks = [
        {"check_id": "m6_all_scenarios_simulated", "phase": "simulator",
         "passed": len(sims) == n_sites * 8 and n_scen == 8,
         "detail": f"{len(sims)} rows = {n_sites} sites x {n_scen} scenarios",
         "severity": "info" if (len(sims) == n_sites * 8 and n_scen == 8) else "critical"},
        {"check_id": "m6_simulation_score_bounded", "phase": "simulator",
         "passed": bool(((sims["simulation_score"] >= 0) & (sims["simulation_score"] <= 1)).all()),
         "detail": f"SS range [{sims['simulation_score'].min():.4f}, {sims['simulation_score'].max():.4f}]",
         "severity": "critical"},
        {"check_id": "m6_simulation_no_nan", "phase": "simulator",
         "passed": int(sims.isna().sum().sum()) == 0,
         "detail": f"{int(sims.isna().sum().sum())} NaN in simulations", "severity": "critical"},
        {"check_id": "m6_ranking_complete", "phase": "simulator",
         "passed": len(ranking) == n_sites and int(ranking.isna().sum().sum()) == 0,
         "detail": f"ranking rows={len(ranking)} (sites={n_sites}), no NaN",
         "severity": "critical"},
    ]
    return {"sims": sims, "twin": twin, "ranking": ranking}, checks


if __name__ == "__main__":
    tables, checks = run(write=True)
    print("mean SS by scenario:")
    print(tables["sims"].groupby("scenario_id")["simulation_score"].mean().round(4).to_string())
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
