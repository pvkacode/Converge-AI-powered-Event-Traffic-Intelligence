"""
Layer 7 — M7C Step 1: Control Action Library.

Candidate traffic-control actions (signal / traffic / resource / information). Each
action carries cost, duration, approval requirement, optional resource demand, and
deterministic surrogate effect coefficients (fractional queue/travel reduction, absolute
risk reduction) used by the lightweight simulator. NO ML, NO RL.

Human-in-the-loop: approval_required = True for everything except INFORMATION_ONLY.

ADDITIVE ONLY. This module defines data only; writes nothing.
"""

from __future__ import annotations

# resource pool consumed by an action: (resource_type, count) or (None, 0)
# effect: (queue_reduction_frac, travel_reduction_frac, risk_reduction_abs)
# action_id, action_type, action_cost, action_duration_min, approval_required,
#   resource_type, resource_count, q_frac, t_frac, risk_abs, info_only
ACTIONS = [
    # signal
    ("SIGNAL_EXTEND_GREEN", "signal", 1.0, 15, True, None, 0, 0.12, 0.10, 0.03, False),
    ("SIGNAL_REDUCE_GREEN", "signal", 1.0, 15, True, None, 0, 0.05, 0.03, 0.02, False),
    ("SIGNAL_OFFSET_ADJUST", "signal", 1.0, 20, True, None, 0, 0.08, 0.08, 0.02, False),
    # traffic
    ("ACTIVATE_DIVERSION", "traffic", 3.0, 60, True, None, 0, 0.18, 0.15, 0.06, False),
    ("DEACTIVATE_DIVERSION", "traffic", 1.0, 10, True, None, 0, 0.04, 0.05, 0.01, False),
    ("QUEUE_RELIEF", "traffic", 2.0, 30, True, None, 0, 0.15, 0.10, 0.05, False),
    # resource
    ("DISPATCH_POLICE", "resource", 4.0, 45, True, "police", 1, 0.10, 0.08, 0.07, False),
    ("DISPATCH_MARSHAL", "resource", 2.0, 45, True, "marshal", 1, 0.07, 0.05, 0.04, False),
    ("DISPATCH_TOW", "resource", 3.0, 30, True, "tow", 1, 0.12, 0.06, 0.06, False),
    ("DISPATCH_QRU", "resource", 5.0, 30, True, "qru", 1, 0.10, 0.07, 0.09, False),
    # information (no approval needed)
    ("VMS_MESSAGE", "information", 0.5, 30, False, None, 0, 0.04, 0.06, 0.02, True),
    ("OPERATOR_ESCALATION", "information", 0.5, 5, False, None, 0, 0.00, 0.00, 0.03, True),
    ("ROADWORK_ALERT", "information", 0.5, 60, False, None, 0, 0.03, 0.02, 0.02, True),
]

ACTION_FIELDS = ["action_id", "action_type", "action_cost", "action_duration",
                 "approval_required", "resource_type", "resource_count",
                 "q_frac", "t_frac", "risk_abs", "info_only"]

INFORMATION_ONLY = {a[0] for a in ACTIONS if a[10]}


def action_dicts() -> list[dict]:
    return [dict(zip(ACTION_FIELDS, a)) for a in ACTIONS]


def is_applicable(a: dict, site: dict) -> bool:
    """Lightweight applicability gate per site context (capacity/escalation/incident/diversion)."""
    cap = site["capacity_utilization"]; esc = site["escalation_risk"]
    inc = site["incident_intensity"]; div = site["diversion_active"]
    aid = a["action_id"]
    if aid == "ACTIVATE_DIVERSION":
        return (cap >= 0.5 or esc >= 0.25) and not div
    if aid == "DEACTIVATE_DIVERSION":
        return div and esc < 0.20
    if aid == "DISPATCH_TOW":
        return inc >= 0.20
    if aid == "DISPATCH_QRU":
        return esc >= 0.30
    if aid == "DISPATCH_POLICE":
        return esc >= 0.20
    if aid in ("SIGNAL_EXTEND_GREEN", "QUEUE_RELIEF"):
        return cap >= 0.30
    if aid == "SIGNAL_REDUCE_GREEN":
        return site.get("spillover_norm", 0.0) >= 0.5
    if aid == "OPERATOR_ESCALATION":
        return esc >= 0.25 or site.get("forecast_uncertainty", 0.0) >= 0.5
    return True  # signal_offset, marshal, info actions broadly applicable
