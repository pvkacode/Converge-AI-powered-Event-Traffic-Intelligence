"""
Layer 7 — M7B Steps 1/2/4: Traffic state-space model.

State vector x_t (7-dim) per active site:
    [speed, density, queue_length, travel_time, incident_intensity,
     lane_availability, capacity_utilization]

Observation model y_t = H x_t + v_t, with H = I7: 5 channels come directly from the
M7A fused observation; density and capacity_utilization are DERIVED pseudo-measurements
(fundamental-diagram / load-over-capacity), so all 7 states receive Kalman updates.

Dynamics x_t = A x_(t-1) + B u_t + w_t are implemented as mean-reversion to a target
t* (B u_t = (I-A) t*), so the system is provably stable (diagonal A, spectral radius<1)
and settles on an observation-/exogenous-shaped equilibrium rather than decaying to 0.

ADDITIVE ONLY. This module writes nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from layer7_config import OUT

STATE_NAMES = ["speed", "density", "queue_length", "travel_time",
               "incident_intensity", "lane_availability", "capacity_utilization"]
N = 7
IDX = {n: i for i, n in enumerate(STATE_NAMES)}

# diagonal transition (mean-reversion rates); all < 1 => spectral radius < 1 (stable)
A_DIAG = np.array([0.95, 0.95, 0.97, 0.95, 0.92, 0.98, 0.95])
A = np.diag(A_DIAG)
H = np.eye(N)

# process noise (per-state, in each state's units^2)
Q = np.diag([4.0, 0.002, 100.0, 25.0, 0.002, 0.002, 0.004])
# base observation noise (per channel, units^2) before confidence/conflict scaling
R_BASE = np.array([25.0, 0.01, 900.0, 100.0, 0.01, 0.01, 0.02])
# initial covariance
P0_DIAG = R_BASE * 4.0

FALLBACK_R_PENALTY = 1.5
# Single honest snapshot: M7A provides ONE fused observation per site, so we run one
# predict+update cycle (repeating the same y would fabricate information and collapse P).
N_FILTER_STEPS = 1
STEP_MINUTES = 5            # one filter step ~ 5 min (sets forecast horizons)


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def derive_density(speed: float, queue: float) -> float:
    """Occupancy-like density proxy in [0,1] from fused speed + queue."""
    return float(np.clip(0.6 * (queue / 600.0) + 0.4 * ((70.0 - speed) / 65.0), 0.0, 1.0))


def derive_capacity_util(speed: float, queue: float, incident: float, lane: float) -> tuple:
    """Step 7: capacity_utilization = traffic_load / effective_capacity, bounded [0,1]."""
    load = 0.5 * (queue / 600.0) + 0.3 * float(incident) + 0.2 * (1.0 - speed / 70.0)
    eff_cap = max(0.10, float(lane))
    util = float(np.clip(load / eff_cap, 0.0, 1.0))
    return util, float(load), eff_cap


def build_exogenous() -> pd.DataFrame:
    """u_t exogenous factors from Layer 5 / Layer 6, per active site."""
    st = _read("layer7_operational_state.csv"); st["event_id"] = st["event_id"].astype(str)
    active = _read("layer7_active_site_state.csv"); active["event_id"] = active["event_id"].astype(str)
    df = active[["event_id"]].merge(st[["event_id", "operational_risk_score"]], on="event_id", how="left")
    df["op_risk"] = (pd.to_numeric(df["operational_risk_score"], errors="coerce").fillna(0) / 100.0).clip(0, 1)

    alerts = _read("layer7_prioritized_alerts.csv")
    if len(alerts):
        alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
        nc = alerts.groupby("affected_event_id").size()
        df["alert_density"] = df["event_id"].map(lambda e: min(1.0, int(nc.get(e, 0)) / 3.0))
    else:
        df["alert_density"] = 0.0

    drift = _read("layer6_drift_report.csv")
    drift_sev = 0.0
    if len(drift):
        da = drift[drift["alert"].astype(str).str.lower().isin(["true", "1", "yes"])]
        drift_sev = float(np.clip(pd.to_numeric(da.get("retrain_urgency"), errors="coerce").max()
                                  if len(da) else 0.0, 0, 1))
    df["drift_severity"] = drift_sev

    alloc = _read("layer5_resource_allocation.csv")
    div = {}
    if len(alloc):
        alloc["event_id"] = alloc["event_id"].astype(str)
        div = dict(zip(alloc["event_id"],
                       alloc["diversion_activated"].astype(str).str.lower().isin(["1", "true", "yes"])))
    df["diversion"] = df["event_id"].map(lambda e: 1.0 if div.get(e, False) else 0.0)
    return df[["event_id", "op_risk", "alert_density", "drift_severity", "diversion"]]


def build_target(obs: dict, exo: dict) -> np.ndarray:
    """Mean-reversion target t* = derived state shaped mildly by exogenous factors."""
    speed = float(obs["traffic_speed"]); queue = float(obs["queue_length"])
    travel = float(obs["travel_time"]); incident = float(obs["incident_probability"])
    lane = float(obs["lane_availability"])
    density = derive_density(speed, queue)
    cap, _, _ = derive_capacity_util(speed, queue, incident, lane)
    r = float(exo.get("op_risk", 0.0)); drift = float(exo.get("drift_severity", 0.0))
    alert = float(exo.get("alert_density", 0.0))
    t = np.array([
        speed * (1.0 - 0.10 * r),                       # higher risk -> lower target speed
        float(np.clip(density + 0.10 * r, 0, 1)),
        queue * (1.0 + 0.10 * r),
        travel * (1.0 + 0.10 * r),
        float(np.clip(incident + 0.20 * drift + 0.10 * alert, 0, 1)),
        float(np.clip(lane, 0, 1)),
        float(np.clip(cap + 0.10 * r, 0, 1)),
    ], dtype=float)
    return t


def observation_vector(obs: dict) -> np.ndarray:
    speed = float(obs["traffic_speed"]); queue = float(obs["queue_length"])
    travel = float(obs["travel_time"]); incident = float(obs["incident_probability"])
    lane = float(obs["lane_availability"])
    density = derive_density(speed, queue)
    cap, _, _ = derive_capacity_util(speed, queue, incident, lane)
    return np.array([speed, density, queue, travel, incident, lane, cap], dtype=float)


def effective_R(confidence: float, conflict: float, fallback: bool) -> tuple:
    """Step 4: R_eff = R_base * (1/confidence) * (1+conflict) * fallback_penalty."""
    conf = max(float(confidence), 0.05)
    mult = (1.0 / conf) * (1.0 + float(conflict)) * (FALLBACK_R_PENALTY if fallback else 1.0)
    return np.diag(R_BASE * mult), float(mult)
