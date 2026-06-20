"""
Layer 7 — M6 Part E: Plan Comparison Engine.

Compares the Current Plan (SCENARIO_0 baseline) against an Alternative Plan
(the best-scoring non-baseline scenario per site) using the simulator outputs.
Read-only; no optimization.

Output: outputs/layer7_plan_comparison.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def build_comparison(sims: pd.DataFrame | None = None) -> pd.DataFrame:
    if sims is None:
        sims = pd.read_csv(OUT / "layer7_twin_scenarios_full.csv")

    rows = []
    for eid, grp in sims.groupby("event_id"):
        base = grp[grp["scenario_id"] == "SCENARIO_0"].iloc[0]
        alt_pool = grp[grp["scenario_id"] != "SCENARIO_0"].sort_values(
            ["simulation_score", "scenario_id"])
        alt = alt_pool.iloc[0]
        rows.append({
            "event_id": eid,
            "current_scenario": "SCENARIO_0",
            "alternative_scenario": alt["scenario_id"],
            "alternative_name": alt["scenario_name"],
            # deltas are expressed relative to the recommended baseline (SCENARIO_0 deltas = 0)
            "expected_delay_change": round(float(alt["delay_delta"] - base["delay_delta"]), 4),
            "expected_risk_change": round(float(alt["risk_delta"] - base["risk_delta"]), 4),
            "expected_alert_change": round(float(alt["alert_delta"] - base["alert_delta"]), 4),
            "expected_confidence_change": round(
                float(alt["confidence_delta"] - base["confidence_delta"]), 4),
            "current_simulation_score": round(float(base["simulation_score"]), 6),
            "alternative_simulation_score": round(float(alt["simulation_score"]), 6),
            "recommendation": ("prefer_alternative"
                               if alt["simulation_score"] < base["simulation_score"]
                               else "keep_current"),
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


def run(sims: pd.DataFrame | None = None, write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_comparison(sims)
    if write:
        df.to_csv(OUT / "layer7_plan_comparison.csv", index=False)
    checks = [{
        "check_id": "m6_plan_comparison_built", "phase": "plan_comparison",
        "passed": len(df) > 0 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} site comparisons; "
                  f"prefer_alternative={int((df['recommendation'] == 'prefer_alternative').sum())}; "
                  f"{int(df.isna().sum().sum())} NaN",
        "severity": "info" if (len(df) > 0 and int(df.isna().sum().sum()) == 0) else "critical",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
