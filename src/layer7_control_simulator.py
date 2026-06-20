"""
Layer 7 — M7C Step 5/10: Candidate Action Simulation (surrogate, no ML/RL).

Estimates the effect of each candidate action on a site using deterministic surrogate
relations grounded in the M7B state. Provides per-action deltas and a baseline-vs-action
what-if comparison.

ADDITIVE ONLY. No I/O of its own.
"""

from __future__ import annotations


def simulate_action(site: dict, action: dict) -> dict:
    """Return estimated impact of one action on one site (NO_ACTION -> zero deltas)."""
    q = float(site["queue_length"]); tt = float(site["travel_time"])
    esc = float(site["escalation_risk"]); inc = float(site.get("incident_intensity", 0.0))
    if action is None or action.get("action_id") == "NO_ACTION":
        return {"d_queue": 0.0, "d_travel": 0.0, "d_risk": 0.0,
                "queue_after": q, "travel_after": tt, "risk_after": esc}
    aid = action["action_id"]
    # operationally-grounded condition multipliers (surrogate; NOT learned):
    #  - a tow clears the incident driving the queue (only resource that does);
    #  - QRU/police are most effective when escalation is high;
    #  - a diversion cannot clear an ON-SITE incident, so its effect is damped by incident.
    mq = mt = mr = 1.0
    if aid == "DISPATCH_TOW":
        mq *= 1.0 + 2.0 * inc; mr *= 1.0 + 2.0 * inc
    elif aid in ("DISPATCH_QRU", "DISPATCH_POLICE"):
        mr *= 1.0 + 2.0 * esc
    elif aid == "ACTIVATE_DIVERSION":
        damp = 1.0 - 0.5 * inc
        mq *= damp; mt *= damp
    elif aid in ("SIGNAL_EXTEND_GREEN", "SIGNAL_REDUCE_GREEN", "SIGNAL_OFFSET_ADJUST"):
        # signal timing manages flow but cannot clear an on-site incident
        damp = 1.0 - 0.7 * inc
        mq *= damp; mt *= damp
    d_queue = q * float(action["q_frac"]) * mq
    d_travel = tt * float(action["t_frac"]) * mt
    # risk reduction scales with current escalation; capped so risk stays >= 0
    d_risk = min(esc, float(action["risk_abs"]) * mr * (0.5 + esc) + esc * 0.05)
    return {
        "d_queue": round(d_queue, 4), "d_travel": round(d_travel, 4), "d_risk": round(d_risk, 6),
        "queue_after": round(max(0.0, q - d_queue), 4),
        "travel_after": round(max(0.0, tt - d_travel), 4),
        "risk_after": round(max(0.0, esc - d_risk), 6),
    }


def whatif(site: dict, action: dict) -> dict:
    """Baseline vs recommended-action comparison for the simulation output."""
    sim = simulate_action(site, action)
    return {
        "baseline_queue": round(float(site["queue_length"]), 4),
        "baseline_travel": round(float(site["travel_time"]), 4),
        "baseline_risk": round(float(site["escalation_risk"]), 6),
        "action_queue": sim["queue_after"], "action_travel": sim["travel_after"],
        "action_risk": sim["risk_after"],
        "queue_impact": sim["d_queue"], "travel_impact": sim["d_travel"],
        "risk_impact": sim["d_risk"],
        "queue_reduction_pct": round(100.0 * sim["d_queue"] / max(float(site["queue_length"]), 1e-9), 2),
        "risk_reduction_pct": round(100.0 * sim["d_risk"] / max(float(site["escalation_risk"]), 1e-9), 2),
    }
