"""
Layer 7 — M7 Part F: Deployment Readiness Score (DRS).

Aggregates Layer 7 maturity into a single 0-100 readiness score. Read-only.

    DRS = (0.25*validation + 0.20*integrity + 0.20*observability
           + 0.20*governance + 0.15*simulation) * 100

Component scores (all in [0,1]) are derived from existing Layer 7 outputs:
  validation   = passed / total checks across all M1-M6 validation reports
  integrity    = fraction of reports whose namespace/integrity check passed
  observability= mean observability component score
  governance   = 1 - 0.5*low_confidence_rate - 0.5*high_impact_override_rate
  simulation   = fraction of digital-twin-health components healthy

ADDITIVE ONLY. Writes only outputs/layer7_deployment_readiness.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_W = {"validation": 0.25, "integrity": 0.20, "observability": 0.20,
      "governance": 0.20, "simulation": 0.15}

_VAL_REPORTS = [
    "layer7_validation_report.csv", "layer7_m2_validation_report.csv",
    "layer7_m3_validation_report.csv", "layer7_m4_validation_report.csv",
    "layer7_m5_validation_report.csv", "layer7_m6_validation_report.csv",
]


def _validation_and_integrity() -> tuple[float, float]:
    total, passed = 0, 0
    n_reports, n_integrity_ok = 0, 0
    for f in _VAL_REPORTS:
        p = OUT / f
        if not p.exists():
            continue
        df = pd.read_csv(p)
        flags = df["passed"].astype(str).str.lower().isin(["true", "1"])
        total += len(df); passed += int(flags.sum())
        n_reports += 1
        integ = df[df["check_id"].astype(str).str.contains("no_layer16_modifications")]
        if len(integ):
            n_integrity_ok += int(integ["passed"].astype(str).str.lower().isin(["true", "1"]).all())
        else:
            n_integrity_ok += 1  # report predates the explicit integrity check
    val = passed / max(1, total)
    integ = n_integrity_ok / max(1, n_reports)
    return val, integ


def _observability_score() -> float:
    p = OUT / "layer7_observability_report.csv"
    if not p.exists():
        return 0.0
    return float(pd.read_csv(p)["score"].mean())


def _governance_score() -> float:
    p = OUT / "layer7_governance_summary.csv"
    if not p.exists():
        return 0.5
    m = pd.read_csv(p).set_index("metric")["value"].to_dict()
    low = float(m.get("low_confidence_recommendation_rate", 0.0))
    hi = float(m.get("high_impact_override_rate", 0.0))
    return float(np.clip(1.0 - 0.5 * low - 0.5 * hi, 0, 1))


def _simulation_score() -> float:
    p = OUT / "layer7_digital_twin_health.csv"
    if not p.exists():
        return 0.0
    t = pd.read_csv(p)
    return float((t["status"] == "healthy").mean())


def build_readiness() -> pd.DataFrame:
    val, integ = _validation_and_integrity()
    obs = _observability_score()
    gov = _governance_score()
    sim = _simulation_score()
    comps = {"validation": val, "integrity": integ, "observability": obs,
             "governance": gov, "simulation": sim}
    drs = sum(_W[k] * comps[k] for k in comps) * 100.0

    rows = [{
        "component": k, "weight": _W[k], "score_0_1": round(comps[k], 6),
        "weighted_contribution": round(_W[k] * comps[k] * 100, 4),
        "generated_at": _NOW_ISO,
    } for k in comps]
    rows.append({
        "component": "deployment_readiness_score", "weight": 1.0,
        "score_0_1": round(drs / 100.0, 6), "weighted_contribution": round(drs, 4),
        "generated_at": _NOW_ISO,
    })
    return pd.DataFrame(rows), drs


def run(write: bool = True) -> tuple[pd.DataFrame, float, list[dict]]:
    df, drs = build_readiness()
    if write:
        df.to_csv(OUT / "layer7_deployment_readiness.csv", index=False)
    checks = [{
        "check_id": "m7_drs_bounded", "phase": "deployment_readiness",
        "passed": 0 <= drs <= 100 and int(df.isna().sum().sum()) == 0,
        "detail": f"DRS={drs:.2f}/100; components="
                  f"{ {r['component']: r['score_0_1'] for _, r in df.iterrows() if r['component'] != 'deployment_readiness_score'} }",
        "severity": "critical" if not (0 <= drs <= 100) else "info",
    }]
    return df, drs, checks


if __name__ == "__main__":
    df, drs, checks = run(write=True)
    print(df.to_string(index=False))
    print(f"DRS = {drs:.2f}/100")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
