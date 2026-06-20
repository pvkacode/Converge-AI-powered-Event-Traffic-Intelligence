"""
Layer 7 — M7B: Traffic State Estimation Layer (orchestrator).

Estimates the hidden traffic state that generated the M7A fused observations, with a
linear Kalman filter, confidence-aware observation noise, state uncertainty tiers,
short-horizon forecasts, capacity utilization, and incident escalation risk.

Additive, new files only. NO Layer 1-6 / existing Layer 7 output modified. NO scoring
logic changed. Kalman filtering only (NOT MPC / control / GNN).

Outputs:
  layer7_state_estimates.csv      layer7_state_forecasts.csv
  layer7_state_uncertainty.csv    layer7_capacity_utilization.csv
  layer7_escalation_risk.csv      layer7_state_diagnostics.csv
  layer7_state_summary.txt
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT
from layer7_kalman import KalmanFilter
from layer7_state_diagnostics import build_diagnostics
from layer7_state_forecasting import forecast
from layer7_state_space import (
    A,
    H,
    IDX,
    N_FILTER_STEPS,
    P0_DIAG,
    Q,
    STATE_NAMES,
    build_exogenous,
    build_target,
    derive_capacity_util,
    effective_R,
    observation_vector,
)

_NOW_ISO = datetime.now(timezone.utc).isoformat()
# reference (acceptable) std per state for the confidence metric: confidence falls as the
# posterior std approaches this reference, so uncertainty tiers are meaningful (not all-LOW).
_VAR_REF = {"speed": 8.0, "density": 0.15, "queue_length": 45.0, "travel_time": 22.0,
            "incident_intensity": 0.15, "lane_availability": 0.15, "capacity_utilization": 0.15}


def _uncertainty_tier(conf: float) -> str:
    u = 1.0 - conf
    return "LOW" if u < 0.33 else ("MEDIUM" if u < 0.66 else "HIGH")


def _cap_tier(u: float) -> str:
    return "NORMAL" if u < 0.60 else ("ELEVATED" if u < 0.85 else "SATURATED")


def _esc_tier(x: float) -> str:
    return ("CRITICAL" if x >= 0.75 else "HIGH" if x >= 0.50
            else "MEDIUM" if x >= 0.25 else "LOW")


def run(write: bool = True) -> tuple[dict, list[dict]]:
    fused = pd.read_csv(OUT / "layer7_sensor_fusion.csv")
    fused["event_id"] = fused["event_id"].astype(str)
    exo = build_exogenous().set_index("event_id")

    est_rows, unc_rows, fc_rows, cap_rows, esc_rows, stab = [], [], [], [], [], []

    for _, o in fused.iterrows():
        eid = str(o["event_id"])
        obs = {k: float(o[k]) for k in ["traffic_speed", "queue_length", "travel_time",
                                        "incident_probability", "lane_availability"]}
        conf = float(o["confidence"]); conflict = float(o["conflict_score"])
        fallback = str(o["fallback_mode"]).lower() in ("true", "1", "yes")
        src_count = int(o.get("source_count", 0))
        exo_row = exo.loc[eid].to_dict() if eid in exo.index else {}

        y = observation_vector(obs)
        tstar = build_target(obs, exo_row)
        R_eff, R_mult = effective_R(conf, conflict, fallback)
        forcing = (np.eye(7) - A) @ tstar

        kf = KalmanFilter(A, H, Q, R_eff, x0=y.copy(), P0=np.diag(P0_DIAG))
        for _ in range(N_FILTER_STEPS):
            kf.predict(forcing=forcing)
            kf.update(y, R=R_eff)

        xp = kf.x
        var = kf.variances
        stab.append(kf.stability_report())

        # Step 1/5/10: per-state estimates + uncertainty + provenance
        for i, name in enumerate(STATE_NAMES):
            val = float(xp[i])
            if name in ("incident_intensity", "lane_availability", "capacity_utilization", "density"):
                val = float(np.clip(val, 0.0, 1.0))
            elif name in ("speed", "queue_length", "travel_time"):
                val = float(max(0.0, val))
            v = float(var[i])
            sconf = float(1.0 / (1.0 + v / (_VAR_REF[name] ** 2)))
            est_rows.append({
                "event_id": eid, "state_dimension": i, "state_name": name,
                "state_value": round(val, 4), "posterior_mean": round(val, 4),
                "posterior_variance": round(v, 6), "kalman_update_count": kf.update_count,
                "source_observations": src_count, "obs_confidence": round(conf, 4),
                "conflict_score": round(conflict, 4), "fallback_mode": fallback,
                "generated_at": _NOW_ISO,
            })
            unc_rows.append({
                "event_id": eid, "state_name": name,
                "posterior_mean": round(val, 4), "posterior_variance": round(v, 6),
                "confidence": round(sconf, 4), "uncertainty_tier": _uncertainty_tier(sconf),
                "effective_observation_variance": round(float(R_eff[i, i]), 6),
                "obs_noise_multiplier": round(R_mult, 4), "generated_at": _NOW_ISO,
            })

        # Step 6: forecasts
        fc = forecast(A, xp, kf.P, Q, tstar)
        for hmin, q, mean, fvar, fconf in fc:
            fc_rows.append({"event_id": eid, "horizon_min": hmin, "quantity": q,
                            "forecast_mean": mean, "forecast_variance": fvar,
                            "forecast_confidence": fconf, "generated_at": _NOW_ISO})
        # queue growth (30-min) for escalation
        q_now = float(max(0.0, xp[IDX["queue_length"]]))
        q_30 = next(m for (h, qq, m, _v, _c) in fc if h == 30 and qq == "queue_length")
        queue_growth = float(np.clip((q_30 - q_now) / 600.0, 0.0, 1.0))

        # Step 7: capacity utilization (from filtered observed channels)
        util, load, eff_cap = derive_capacity_util(
            float(max(0.0, xp[IDX["speed"]])), q_now,
            float(np.clip(xp[IDX["incident_intensity"]], 0, 1)),
            float(np.clip(xp[IDX["lane_availability"]], 0, 1)))
        cap_rows.append({"event_id": eid, "capacity_utilization": round(util, 4),
                         "capacity_tier": _cap_tier(util), "traffic_load": round(load, 4),
                         "effective_capacity": round(eff_cap, 4), "generated_at": _NOW_ISO})

        # Step 8: incident escalation risk
        incident = float(np.clip(xp[IDX["incident_intensity"]], 0, 1))
        alert_d = float(exo_row.get("alert_density", 0.0))
        esc = float(np.clip(0.35 * incident + 0.25 * queue_growth + 0.25 * util + 0.15 * alert_d, 0, 1))
        esc_rows.append({"event_id": eid, "escalation_risk": round(esc, 4),
                         "escalation_tier": _esc_tier(esc), "incident_intensity": round(incident, 4),
                         "queue_growth_30min": round(queue_growth, 4),
                         "capacity_utilization": round(util, 4), "alert_density": round(alert_d, 4),
                         "generated_at": _NOW_ISO})

    est = pd.DataFrame(est_rows); unc = pd.DataFrame(unc_rows)
    fc_df = pd.DataFrame(fc_rows); cap = pd.DataFrame(cap_rows); esc = pd.DataFrame(esc_rows)
    diag = build_diagnostics(est, cap, stab)

    if write:
        est.to_csv(OUT / "layer7_state_estimates.csv", index=False)
        unc.to_csv(OUT / "layer7_state_uncertainty.csv", index=False)
        fc_df.to_csv(OUT / "layer7_state_forecasts.csv", index=False)
        cap.to_csv(OUT / "layer7_capacity_utilization.csv", index=False)
        esc.to_csv(OUT / "layer7_escalation_risk.csv", index=False)
        diag.to_csv(OUT / "layer7_state_diagnostics.csv", index=False)

    checks = _validate(est, unc, fc_df, cap, esc, stab)
    if write:
        _summary(est, unc, fc_df, cap, esc, stab, checks)
    return {"estimates": est, "uncertainty": unc, "forecasts": fc_df, "capacity": cap,
            "escalation": esc, "diagnostics": diag, "stability": pd.DataFrame(stab)}, checks


def _validate(est, unc, fc, cap, esc, stab) -> list[dict]:
    checks = []

    def chk(cid, passed, detail, sev="critical"):
        checks.append({"check_id": cid, "phase": "state_estimation", "passed": bool(passed),
                       "detail": detail, "severity": "info" if passed else sev})

    sr = pd.DataFrame(stab)
    chk("m7b_state_values_finite", bool(np.isfinite(est["state_value"]).all()),
        f"{int((~np.isfinite(est['state_value'])).sum())} non-finite state values")
    chk("m7b_variances_nonneg", bool((est["posterior_variance"] >= 0).all()),
        "all posterior variances >= 0")
    chk("m7b_capacity_bounded", bool(((cap["capacity_utilization"] >= 0)
        & (cap["capacity_utilization"] <= 1)).all()), "capacity in [0,1]")
    chk("m7b_forecasts_finite", bool(np.isfinite(fc["forecast_mean"]).all()
        and np.isfinite(fc["forecast_variance"]).all()), "all forecast values finite")
    sp = est[est["state_name"] == "speed"]["state_value"]
    ql = est[est["state_name"] == "queue_length"]["state_value"]
    chk("m7b_no_negative_speed", bool((sp >= 0).all()), f"min speed={sp.min():.3f}")
    chk("m7b_no_negative_queue", bool((ql >= 0).all()), f"min queue={ql.min():.3f}")
    chk("m7b_escalation_bounded", bool(((esc["escalation_risk"] >= 0)
        & (esc["escalation_risk"] <= 1)).all()), "escalation_risk in [0,1]")
    chk("m7b_kalman_stable", bool(sr["stable"].all())
        and float(sr["spectral_radius_A"].max()) < 1.0,
        f"all sites stable; max spectral_radius_A={sr['spectral_radius_A'].max():.4f}; "
        f"max_post_var={sr['max_posterior_variance'].max():.2f}")
    return checks


def _summary(est, unc, fc, cap, esc, stab, checks) -> None:
    sr = pd.DataFrame(stab)
    n_pass = sum(1 for c in checks if c["passed"]); n_fail = sum(1 for c in checks if not c["passed"])
    fb = int(est.groupby("event_id")["fallback_mode"].first().sum())
    lines = [
        "LAYER 7 — M7B TRAFFIC STATE ESTIMATION SUMMARY",
        "=" * 52,
        f"generated_at: {_NOW_ISO}",
        "method: linear Kalman filter (7-state) on M7A fused observations",
        "",
        f"A. state vectors estimated: {est['event_id'].nunique()} sites x {len(STATE_NAMES)} states "
        f"= {len(est)} estimates",
        f"B. forecast horizon coverage: {sorted(fc['horizon_min'].unique())} min x "
        f"{sorted(fc['quantity'].unique())} ({len(fc)} forecasts)",
        f"C. state confidence: mean={unc['confidence'].mean():.3f}; "
        f"uncertainty tiers={unc['uncertainty_tier'].value_counts().to_dict()}",
        f"D. escalation-risk tiers: {esc['escalation_tier'].value_counts().to_dict()} "
        f"(risk range [{esc['escalation_risk'].min():.3f},{esc['escalation_risk'].max():.3f}])",
        f"E. capacity-utilization tiers: {cap['capacity_tier'].value_counts().to_dict()} "
        f"(util range [{cap['capacity_utilization'].min():.3f},{cap['capacity_utilization'].max():.3f}])",
        f"F. Kalman stability: all_stable={bool(sr['stable'].all())}; "
        f"spectral_radius(A)={sr['spectral_radius_A'].max():.4f}<1; "
        f"max_post_var={sr['max_posterior_variance'].max():.2f}; "
        f"min_P_eig={sr['min_P_eigenvalue'].min():.2e}; updates/site={int(sr['update_count'].max())}",
        f"G. fallback behaviour: {fb} sites carry M7A fallback flag; observation noise inflated "
        f"(R_eff *= 1/conf * (1+conflict) * 1.5) -> wider, still-finite estimates; never crashes",
        "H. layer compatibility: reads M7A + L5/L6 only; no upstream or existing-L7 output modified.",
        "",
        f"VALIDATION: {n_pass} passed / {n_fail} failed",
    ]
    for c in checks:
        lines.append(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['check_id']}: {c['detail']}")
    (OUT / "layer7_state_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("LAYER 7 — M7B TRAFFIC STATE ESTIMATION")
    print("=" * 60)
    tables, checks = run(write=True)
    print(f"sites: {tables['estimates']['event_id'].nunique()}  "
          f"estimates: {len(tables['estimates'])}  forecasts: {len(tables['forecasts'])}")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
    n_fail = sum(1 for c in checks if not c["passed"])
    print(f"\nValidation: {len(checks)-n_fail} passed / {n_fail} failed")


if __name__ == "__main__":
    main()
