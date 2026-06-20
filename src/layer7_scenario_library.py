"""
Layer 7 — M6 Part B: Scenario Library.

Declarative catalog of the 8 simulation scenarios. Each entry carries the
parameters the simulator uses to derive deltas from existing signals (counterfactual
outputs, shadow prices, robustness, DCS) — NO new model, NO optimization.

`counterfactual_map` links a scenario to a precomputed M3 counterfactual delta where
one exists; otherwise the simulator computes the delta directly (see `derive`).
`confidence_delta` is a fixed, documented per-scenario shift in decision confidence.

ADDITIVE ONLY. Writes only outputs/layer7_scenario_catalog.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# scenario_id, name, description, counterfactual_map, derive, confidence_delta
SCENARIOS = [
    ("SCENARIO_0", "Current recommended plan",
     "Layer 5 recommended allocation as-is (baseline reference).",
     "A_recommended", "baseline", 0.00),
    ("SCENARIO_1", "No intervention",
     "Remove all resources; lose the recommended delay reduction.",
     "B_no_action", "counterfactual", -0.30),
    ("SCENARIO_2", "Resource reduction",
     "Halve allocated resources.",
     "C_reduced_resources", "counterfactual", -0.15),
    ("SCENARIO_3", "Resource increase",
     "Add resources (extra officers + tow) via shadow-price marginal value.",
     "D_increased_resources", "counterfactual", 0.10),
    ("SCENARIO_4", "Diversion disabled",
     "Turn off diversion where active; lose its share of the reduction.",
     "E_diversion_disabled", "counterfactual", -0.10),
    ("SCENARIO_5", "Operator override plan",
     "Apply the operator override impact (if any) for the site.",
     "", "override", 0.05),
    ("SCENARIO_6", "Worst-case drift escalation",
     "Amplify risk/delay/alerts by the Layer 6 drift escalation factor.",
     "", "drift_worst", -0.40),
    ("SCENARIO_7", "Best-case stabilized state",
     "Full effectiveness: maximal delay/risk/alert reduction.",
     "", "stabilized", 0.40),
]


def build_catalog() -> pd.DataFrame:
    rows = [{
        "scenario_id": sid, "scenario_name": name, "description": desc,
        "counterfactual_map": cf, "derivation": dv, "confidence_delta": cd,
        "generated_at": _NOW_ISO,
    } for (sid, name, desc, cf, dv, cd) in SCENARIOS]
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_catalog()
    if write:
        df.to_csv(OUT / "layer7_scenario_catalog.csv", index=False)
    checks = [{
        "check_id": "m6_scenario_catalog_complete", "phase": "scenario_library",
        "passed": len(df) == 8,
        "detail": f"{len(df)} scenarios catalogued",
        "severity": "info" if len(df) == 8 else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df[["scenario_id", "scenario_name", "derivation", "confidence_delta"]].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
