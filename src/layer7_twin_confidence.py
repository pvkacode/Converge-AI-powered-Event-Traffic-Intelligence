"""
Layer 7 — M6 Part H: Twin Confidence Analysis.

Estimates how trustworthy each site's simulation is. Read-only; no model.

    TCS = 0.4*DCS + 0.3*robustness + 0.3*(1 - uncertainty)

The DCS file already exposes uncertainty_component = (1 - normalized_uncertainty),
so (1 - uncertainty) is taken directly from that component (consistent anchor).

Output: outputs/layer7_twin_confidence.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def _scenario_stability() -> dict:
    """Per-event entropy-based stability from the twin scenario distribution.
    stability = 1 - H/log(K), where p_s = SS_s / sum_s SS_s over the K=8 scenarios."""
    path = OUT / "layer7_twin_scenarios_full.csv"
    if not path.exists():
        return {}
    sims = pd.read_csv(path)
    sims["event_id"] = sims["event_id"].astype(str)
    out: dict[str, float] = {}
    for eid, grp in sims.groupby("event_id"):
        ss = pd.to_numeric(grp["simulation_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
        k = len(ss)
        total = ss.sum()
        if k <= 1 or total <= 1e-12:
            out[eid] = 0.5
            continue
        p = ss / total
        p = p[p > 0]
        h = float(-(p * np.log(p)).sum())
        out[eid] = float(np.clip(1.0 - h / np.log(k), 0.0, 1.0))
    return out


def build_twin_confidence() -> pd.DataFrame:
    dcs = pd.read_csv(OUT / "layer7_decision_confidence.csv")
    df = pd.DataFrame({"event_id": dcs["event_id"].astype(str)})
    df["dcs_component"] = pd.to_numeric(dcs["decision_confidence_score"], errors="coerce").fillna(0).clip(0, 1)
    # robustness/uncertainty kept as DIAGNOSTIC columns only (schema-preserving); they are
    # NO LONGER summed into TCS (PATCH F-005: they already live inside DCS -> double counting).
    df["robustness_component"] = pd.to_numeric(dcs["robustness_component"], errors="coerce").fillna(0).clip(0, 1)
    df["uncertainty_inv_component"] = pd.to_numeric(
        dcs["uncertainty_component"], errors="coerce").fillna(0).clip(0, 1)

    # PATCH F-005: independent second axis = entropy-based scenario stability of the
    # twin's per-site scenario distribution. stability = 1 - H/log(K), K=8 scenarios.
    # High stability = decisive (peaked) scenario differentiation; low = indistinct.
    stab = _scenario_stability()
    df["scenario_stability"] = df["event_id"].map(stab).fillna(0.5).clip(0, 1)

    df["twin_confidence_score"] = (
        0.5 * df["dcs_component"] + 0.5 * df["scenario_stability"]
    ).clip(0, 1)

    pct = df["twin_confidence_score"].rank(pct=True, method="average")
    df["tcs_tier"] = np.where(pct >= 2 / 3, "High", np.where(pct >= 1 / 3, "Moderate", "Low"))
    df["generated_at"] = _NOW_ISO
    return df.sort_values("twin_confidence_score", ascending=False).reset_index(drop=True)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_twin_confidence()
    if write:
        df.to_csv(OUT / "layer7_twin_confidence.csv", index=False)
    s = df["twin_confidence_score"]
    checks = [{
        "check_id": "m6_tcs_bounded", "phase": "twin_confidence",
        "passed": bool(((s >= 0) & (s <= 1)).all()) and int(df.isna().sum().sum()) == 0,
        "detail": f"TCS range [{s.min():.4f}, {s.max():.4f}]; tiers={df['tcs_tier'].value_counts().to_dict()}; "
                  f"{int(df.isna().sum().sum())} NaN",
        "severity": "critical" if not ((s >= 0) & (s <= 1)).all() else "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
