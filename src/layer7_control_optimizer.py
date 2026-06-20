"""
Layer 7 — M7C Step 3/4/6: MPC-lite objective, spillover, action scoring.

Receding-horizon objective (per site, evaluated on the M7B forecast state):

    J = w1*queue + w2*travel_time + w3*escalation_risk + w4*spillover_risk + w5*control_cost

Step 4 spillover (first use of topology):
    spillover_i = Σ_j adjacency_weight(i,j) * capacity_utilization(j)

Step 6 scoring (per action):
    benefit_score = norm(Δqueue) + norm(Δtravel) + Δrisk
    cost_score    = action_cost
    control_score = benefit_score - λ*cost_score

The recommended action minimizes J (equivalently maximizes net benefit). NO ML/RL.

ADDITIVE ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# objective weights
W = {"queue": 0.30, "travel": 0.20, "escalation": 0.25, "spillover": 0.15, "cost": 0.10}
LAMBDA = 0.05                 # cost penalty in control_score
# benefit normalizers chosen so queue / travel / risk contribute on comparable scales
Q_SCALE, T_SCALE, RISK_W = 120.0, 90.0, 2.0
# J objective still scales queue/travel by realistic ranges
Q_OBJ_SCALE, T_OBJ_SCALE = 600.0, 60.0


def compute_spillover(sites: pd.DataFrame, topo: pd.DataFrame) -> dict:
    """spillover_i = Σ_j adjacency_weight(i,j) * capacity_utilization(j)."""
    cap = dict(zip(sites["event_id"].astype(str), sites["capacity_utilization"].astype(float)))
    spill = {e: 0.0 for e in cap}
    if len(topo):
        for _, e in topo.iterrows():
            a, b, w = str(e["site_a"]), str(e["site_b"]), float(e["adjacency_weight"])
            if a in spill and b in cap:
                spill[a] += w * cap[b]
            if b in spill and a in cap:
                spill[b] += w * cap[a]
    return spill


def objective_J(queue, travel, escalation, spillover, cost) -> float:
    return float(W["queue"] * (queue / Q_OBJ_SCALE) + W["travel"] * (travel / T_OBJ_SCALE)
                 + W["escalation"] * escalation + W["spillover"] * spillover
                 + W["cost"] * cost)


def score_action(site: dict, action: dict, sim: dict) -> dict:
    benefit = sim["d_queue"] / Q_SCALE + sim["d_travel"] / T_SCALE + RISK_W * sim["d_risk"]
    cost = float(action["action_cost"])
    control = benefit - LAMBDA * cost
    # J for the post-action state (lower is better); spillover is fixed exogenous load
    j_action = objective_J(site["queue_length"] - sim["d_queue"],
                           site["travel_time"] - sim["d_travel"],
                           max(0.0, site["escalation_risk"] - sim["d_risk"]),
                           site["spillover_norm"], cost)
    return {"benefit_score": round(benefit, 6), "cost_score": round(cost, 4),
            "control_score": round(control, 6), "objective_J": round(j_action, 6)}


def baseline_J(site: dict) -> float:
    return objective_J(site["queue_length"], site["travel_time"],
                       site["escalation_risk"], site["spillover_norm"], 0.0)
