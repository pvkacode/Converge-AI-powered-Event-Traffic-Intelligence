"""
Layer 7 — M7C Step 9: Control Explanation Engine.

Generates human-readable, metric-grounded explanations for each recommendation. Every
sentence references actual M7B/M7C numbers (queue, escalation before/after, spillover).

ADDITIVE ONLY.
"""

from __future__ import annotations


def explain(site: dict, action_id: str, sim: dict, spillover: float) -> str:
    if action_id == "NO_ACTION":
        return (f"No control action recommended: queue {site['queue_length']:.0f} m, "
                f"escalation {site['escalation_risk']:.2f}, capacity {site['capacity_utilization']:.2f} "
                f"are within operating range; net benefit of all candidate actions is non-positive.")
    qpct = 100.0 * sim["d_queue"] / max(site["queue_length"], 1e-9)
    return (
        f"Forecast queue {site['queue_length']:.0f} m and capacity utilization "
        f"{site['capacity_utilization']:.2f} at this site (spillover load {spillover:.2f}). "
        f"{action_id} is expected to reduce queue by {qpct:.0f}% "
        f"({sim['d_queue']:.0f} m) and travel time by {sim['d_travel']:.1f} min, "
        f"lowering escalation risk from {site['escalation_risk']:.2f} to {sim['risk_after']:.2f}."
    )


def risk_if_ignored(site: dict) -> str:
    esc, cap = site["escalation_risk"], site["capacity_utilization"]
    if esc >= 0.30 or cap >= 0.85:
        return "HIGH — escalation/capacity elevated; queue and spillover likely to grow."
    if esc >= 0.20 or cap >= 0.60:
        return "MODERATE — conditions degrading; monitor and pre-position resources."
    return "LOW — conditions stable; deferral has limited downside."
