"""
Layer 7 — M7C Step 7: Resource Feasibility / Constraints.

Layer 5 already deploys its full optimized budget (police 120 / barricades 100 / tow 15 /
qru 10 — saturated), so M7C control actions draw from a separate operator CONTINGENCY
RESERVE (marshals have no Layer 5 equivalent and get their own reserve). Recommended
resource actions are checked, in operator-priority order, against the reserve; once a
reserve is exhausted, further actions of that type are flagged resource_constrained.

ADDITIVE ONLY. This module computes flags; the engine writes the output.
"""

from __future__ import annotations

import pandas as pd

# operator contingency reserve available to the control layer (beyond L5's base plan)
CONTROL_RESERVE = {"police": 20, "marshal": 30, "tow": 5, "qru": 4}


def apply_resource_feasibility(recs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate recommendations with feasible / resource_constrained flags (priority order)
    and return a per-resource constraints table."""
    remaining = dict(CONTROL_RESERVE)
    demand = {k: 0 for k in CONTROL_RESERVE}
    feasible_flags, constrained_flags = [], []

    order = recs.sort_values("operator_priority_score", ascending=False).index
    decision = {}
    for idx in order:
        r = recs.loc[idx]
        rtype = r.get("resource_type")
        rcount = int(r.get("resource_count", 0) or 0)
        if not rtype or rtype == "None" or pd.isna(rtype) or rcount == 0:
            decision[idx] = (True, False)        # no resource needed -> feasible
            continue
        demand[rtype] = demand.get(rtype, 0) + rcount
        if remaining.get(rtype, 0) >= rcount:
            remaining[rtype] -= rcount
            decision[idx] = (True, False)
        else:
            decision[idx] = (False, True)        # reserve exhausted

    for idx in recs.index:
        f, c = decision.get(idx, (True, False))
        feasible_flags.append(f); constrained_flags.append(c)
    recs = recs.copy()
    recs["feasible"] = feasible_flags
    recs["resource_constrained"] = constrained_flags

    rows = []
    for rtype, cap in CONTROL_RESERVE.items():
        used = cap - remaining[rtype]
        rows.append({"resource_type": rtype, "reserve_available": cap,
                     "demand": demand.get(rtype, 0), "allocated": used,
                     "remaining": remaining[rtype],
                     "bottleneck": demand.get(rtype, 0) > cap})
    constraints = pd.DataFrame(rows)
    return recs, constraints
