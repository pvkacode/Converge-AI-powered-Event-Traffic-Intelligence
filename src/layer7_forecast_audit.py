"""
Layer 7 — M7B.1 Part A: Forecast Confidence Realism audit.

Re-runs the per-site Kalman snapshot and evaluates BOTH horizon uncertainty-growth
models (Approach 1 = h*Q, Approach 2 = factor*ΣA^iQA^iᵀ), compares them against the
original "decayed" model, selects the better (monotone-decreasing confidence with the
clearest, still-finite spread), and writes the audit + uncertainty-summary outputs.

ADDITIVE ONLY. Writes only outputs/layer7_forecast_confidence_audit.csv and
outputs/layer7_forecast_uncertainty_summary.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT
from layer7_kalman import KalmanFilter
from layer7_state_forecasting import HORIZON_MIN, _SCALE, forecast
from layer7_state_space import (
    A,
    H,
    N_FILTER_STEPS,
    P0_DIAG,
    Q,
    build_exogenous,
    build_target,
    effective_R,
    observation_vector,
)

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_MODES = ["decayed", "linear_h", "factor"]


def _site_states():
    fused = pd.read_csv(OUT / "layer7_sensor_fusion.csv")
    fused["event_id"] = fused["event_id"].astype(str)
    exo = build_exogenous().set_index("event_id")
    states = []
    for _, o in fused.iterrows():
        eid = str(o["event_id"])
        obs = {k: float(o[k]) for k in ["traffic_speed", "queue_length", "travel_time",
                                        "incident_probability", "lane_availability"]}
        conf = float(o["confidence"]); conflict = float(o["conflict_score"])
        fallback = str(o["fallback_mode"]).lower() in ("true", "1", "yes")
        exo_row = exo.loc[eid].to_dict() if eid in exo.index else {}
        y = observation_vector(obs); tstar = build_target(obs, exo_row)
        R_eff, _ = effective_R(conf, conflict, fallback)
        kf = KalmanFilter(A, H, Q, R_eff, x0=y.copy(), P0=np.diag(P0_DIAG))
        for _ in range(N_FILTER_STEPS):
            kf.predict(forcing=(np.eye(7) - A) @ tstar)
            kf.update(y, R=R_eff)
        states.append((eid, kf.x, kf.P, tstar))
    return states


def _evaluate(states, mode):
    recs = []
    for eid, x, P, tstar in states:
        for hmin, q, mean, var, conf in forecast(A, x, P, Q, tstar, mode=mode):
            recs.append({"horizon": hmin, "quantity": q, "variance": var, "confidence": conf})
    return pd.DataFrame(recs)


def run(write: bool = True) -> tuple[dict, list[dict], str]:
    states = _site_states()
    per_mode = {m: _evaluate(states, m) for m in _MODES}

    # build audit table (per mode x horizon)
    audit_rows = []
    for m in _MODES:
        df = per_mode[m]
        for h in HORIZON_MIN:
            sub = df[df["horizon"] == h]
            audit_rows.append({
                "model": m, "horizon": h,
                "mean_variance": round(float(sub["variance"].mean()), 6),
                "median_variance": round(float(sub["variance"].median()), 6),
                "mean_confidence": round(float(sub["confidence"].mean()), 6),
                "median_confidence": round(float(sub["confidence"].median()), 6),
            })
    audit = pd.DataFrame(audit_rows)
    # variance growth rate per model: mean_var(30)/mean_var(5)
    for m in _MODES:
        v5 = audit[(audit.model == m) & (audit.horizon == 5)]["mean_variance"].iloc[0]
        v30 = audit[(audit.model == m) & (audit.horizon == 30)]["mean_variance"].iloc[0]
        audit.loc[audit.model == m, "variance_growth_rate"] = round(v30 / max(v5, 1e-9), 4)

    # selection: prefer monotone-decreasing mean_confidence with the largest spread
    def score(m):
        a = audit[audit.model == m].sort_values("horizon")
        c = a["mean_confidence"].to_numpy()
        monotone = bool(c[0] >= c[1] >= c[2])
        spread = float(c[0] - c[2])
        return monotone, spread
    ranked = sorted([(m,) + score(m) for m in ("linear_h", "factor")],
                    key=lambda t: (t[1], t[2]), reverse=True)
    selected = ranked[0][0]
    audit["selected"] = audit["model"] == selected
    audit["generated_at"] = _NOW_ISO

    # uncertainty summary for the SELECTED model
    sel = per_mode[selected]
    summ_rows = []
    for h in HORIZON_MIN:
        sub = sel[sel["horizon"] == h]
        std = np.sqrt(sub["variance"].clip(0))
        summ_rows.append({
            "horizon": h,
            "mean_std": round(float(std.mean()), 6),
            "median_std": round(float(std.median()), 6),
            "confidence_mean": round(float(sub["confidence"].mean()), 6),
            "confidence_std": round(float(sub["confidence"].std(ddof=0)), 6),
            "generated_at": _NOW_ISO,
        })
    summary = pd.DataFrame(summ_rows)

    if write:
        audit.to_csv(OUT / "layer7_forecast_confidence_audit.csv", index=False)
        summary.to_csv(OUT / "layer7_forecast_uncertainty_summary.csv", index=False)

    sel_conf = audit[audit.model == selected].sort_values("horizon")["mean_confidence"].tolist()
    checks = [{
        "check_id": "m7b1_forecast_confidence_monotone", "phase": "forecast_audit",
        "passed": bool(sel_conf[0] >= sel_conf[1] >= sel_conf[2]),
        "detail": f"selected={selected}; mean_confidence 5/15/30 = {sel_conf}",
        "severity": "info",
    }, {
        "check_id": "m7b1_forecast_finite", "phase": "forecast_audit",
        "passed": bool(np.isfinite(per_mode[selected]["variance"]).all()
                       and (per_mode[selected]["variance"] >= 0).all()),
        "detail": "selected-model forecast variances finite and non-negative",
        "severity": "critical",
    }]
    return {"audit": audit, "summary": summary, "per_mode": per_mode}, checks, selected


if __name__ == "__main__":
    tables, checks, selected = run(write=True)
    print(f"SELECTED uncertainty-growth model: {selected}")
    print(tables["audit"][["model", "horizon", "mean_variance", "mean_confidence",
                           "variance_growth_rate", "selected"]].to_string(index=False))
    print()
    print(tables["summary"].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
