"""
Layer 7 — M1.1: Active-Site Operational View.

M1 ranks the full Layer 4.5 population (~3,498 events). Layer 5 only optimizes a
50-event active subset. This module produces a dedicated, actionable operational
view restricted to the Layer 5 active sites.

ADDITIVE ONLY. Reads existing M1 outputs + layer5_resource_allocation.csv.
Writes only outputs/layer7_active_site_*.csv. Does not modify M1 outputs.

Outputs:
    outputs/layer7_active_site_state.csv
    outputs/layer7_active_site_ranking.csv
    outputs/layer7_active_site_validation.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

_STATE_COLS = [
    "event_id", "event_cause", "operational_risk_score", "operational_tier",
    "service_tier", "robustness_score", "active_site_rank",
    "active_site_percentile", "generated_at",
]


def build_active_site_state() -> pd.DataFrame:
    alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
    state = pd.read_csv(OUT / "layer7_operational_state.csv")

    # Authoritative active-site set = the 50 Layer 5 event_ids (deduplicated).
    active_ids = (
        alloc[["event_id", "event_cause"]]
        .assign(event_id=lambda d: d["event_id"].astype(str))
        .drop_duplicates("event_id")
    )

    s = state.copy()
    s["event_id"] = s["event_id"].astype(str)
    keep = [
        "event_id", "operational_risk_score", "operational_tier",
        "service_tier", "robustness_score",
    ]
    keep = [c for c in keep if c in s.columns]
    merged = active_ids.merge(s[keep], on="event_id", how="left")

    # Coverage guard: any L5 event missing from M1 state (should not happen).
    merged["missing_from_state"] = merged["operational_risk_score"].isna()
    merged["operational_risk_score"] = pd.to_numeric(
        merged["operational_risk_score"], errors="coerce"
    ).fillna(0.0)
    merged["operational_tier"] = merged["operational_tier"].fillna("Normal")
    if "service_tier_x" in merged.columns:  # event_cause came from alloc, tier from state
        merged = merged.rename(columns={"service_tier_y": "service_tier"})

    # Rank within the active subset: ORS desc, tie-break event_id asc.
    merged = merged.sort_values(
        ["operational_risk_score", "event_id"], ascending=[False, True]
    ).reset_index(drop=True)
    merged["active_site_rank"] = np.arange(1, len(merged) + 1)
    merged["active_site_percentile"] = (
        merged["operational_risk_score"].rank(pct=True, method="average")
    )
    merged["generated_at"] = _NOW_ISO

    out_cols = [c for c in _STATE_COLS if c in merged.columns]
    return merged[out_cols + (["missing_from_state"] if merged["missing_from_state"].any() else [])].copy()


def build_active_site_ranking(state: pd.DataFrame) -> pd.DataFrame:
    rank = state.sort_values(
        ["operational_risk_score", "event_id"], ascending=[False, True]
    ).reset_index(drop=True)
    cols = [
        "active_site_rank", "event_id", "event_cause", "operational_risk_score",
        "operational_tier", "service_tier", "robustness_score",
        "active_site_percentile", "generated_at",
    ]
    return rank[[c for c in cols if c in rank.columns]].copy()


def validate(state: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
    l5_ids = set(alloc["event_id"].astype(str).unique())
    state_ids = state["event_id"].astype(str)

    checks: list[dict] = []

    def chk(cid: str, passed: bool, detail: str, severity: str = "critical") -> None:
        checks.append({
            "check_id": cid, "phase": "active_site_view", "passed": bool(passed),
            "detail": detail, "severity": "info" if passed else severity,
        })

    # every L5 event appears exactly once
    counts = state_ids.value_counts()
    each_once = set(counts.index) == l5_ids and bool((counts == 1).all())
    chk("active_site_coverage", each_once,
        f"L5 events={len(l5_ids)}, in view={state_ids.nunique()}, "
        f"max_count={int(counts.max()) if len(counts) else 0}")

    # no duplicate event_id
    n_dup = int(state_ids.duplicated().sum())
    chk("active_site_uniqueness", n_dup == 0, f"{n_dup} duplicate event_id")

    # rankings reproducible (deterministic sort; ranks strictly increasing)
    ranks = state.sort_values(
        ["operational_risk_score", "event_id"], ascending=[False, True]
    )["active_site_rank"].to_numpy()
    reproducible = bool(np.array_equal(ranks, np.sort(ranks)) and len(set(ranks)) == len(ranks))
    chk("active_site_ranking_reproducibility", reproducible,
        "deterministic ORS-desc / event_id tie-break")

    # percentiles in [0,1]
    pct = state["active_site_percentile"]
    in_range = bool(((pct >= 0) & (pct <= 1)).all())
    chk("active_site_percentile_range", in_range,
        f"percentile range [{pct.min():.4f}, {pct.max():.4f}]")

    # tiers populated
    tiers = state["operational_tier"].dropna()
    chk("active_site_tiers_populated", len(tiers) == len(state) and tiers.ne("").all(),
        f"tier distribution: {state['operational_tier'].value_counts().to_dict()}")

    # no NaN in required columns
    req = ["event_id", "operational_risk_score", "operational_tier",
           "active_site_rank", "active_site_percentile"]
    n_nan = int(state[[c for c in req if c in state.columns]].isna().sum().sum())
    chk("active_site_no_nan", n_nan == 0, f"{n_nan} NaN in required columns")

    report = pd.DataFrame(checks)
    report["active_count"] = int(len(state))
    report["tier_distribution"] = str(state["operational_tier"].value_counts().to_dict())
    report["generated_at"] = _NOW_ISO
    return report, checks


def run(write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    state = build_active_site_state()
    ranking = build_active_site_ranking(state)
    report, checks = validate(state)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        state.to_csv(OUT / "layer7_active_site_state.csv", index=False)
        ranking.to_csv(OUT / "layer7_active_site_ranking.csv", index=False)
        report.to_csv(OUT / "layer7_active_site_validation.csv", index=False)
    return {"state": state, "ranking": ranking, "report": report}, checks


if __name__ == "__main__":
    tables, checks = run(write=True)
    print("=== Layer 7 M1.1 Active-Site View ===")
    print(f"active_count: {len(tables['state'])}")
    print(f"tier_distribution: {tables['state']['operational_tier'].value_counts().to_dict()}")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
