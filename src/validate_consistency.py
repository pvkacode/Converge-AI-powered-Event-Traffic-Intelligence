"""
Validate Day 1 pipeline consistency with original codebase design.

Run after data_pipeline.py + layer1_survival.py:
    python src/validate_consistency.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "events_clean.parquet"


def main() -> None:
    df = pd.read_parquet(CLEAN)
    errors: list[str] = []
    notes: list[str] = []

    if "is_true_planned_event" not in df.columns:
        errors.append("Missing column: is_true_planned_event")
    else:
        n_true = int(df["is_true_planned_event"].sum())
        notes.append(f"is_true_planned_event count: {n_true} (expect ~191 on full ASTraM export)")
        if not (180 <= n_true <= 200):
            errors.append(f"is_true_planned_event={n_true} outside expected ~191 range")

    debris_n = (df["event_cause"] == "debris").sum()
    notes.append(f"debris (merged Debris+debris): {debris_n} (expect 13 on full export)")
    if debris_n != 13:
        errors.append(f"debris merge count={debris_n}, expected 13")

    fog_n = (df["event_cause"] == "fog_low_visibility").sum()
    notes.append(f"fog_low_visibility (merged): {fog_n} (expect 2 on full export)")

    if df["requires_road_closure"].dtype == object:
        errors.append("requires_road_closure should be bool, not object")

    if "trust_score" not in df.columns:
        errors.append("Missing trust_score column")

    censored_with_dur = df["is_censored"] & df["duration_min"].notna()
    if censored_with_dur.any():
        errors.append(
            f"{censored_with_dur.sum()} censored rows have non-null duration_min "
            "(should be impossible with end-marker logic)"
        )

    study_end = df["start_datetime"].max()
    exact = (
        (~df["is_censored"])
        & df["duration_min"].notna()
        & (df["duration_min"] >= 0)
        & df["start_datetime"].notna()
    )
    censored = df["is_censored"].fillna(False) & df["start_datetime"].notna()
    notes.append(f"Interval-censored survival rows: {int(exact.sum() + censored.sum()):,}")
    notes.append(f"  exact events (E=1): {int(exact.sum()):,}")
    notes.append(f"  right-censored [L, inf): {int(censored.sum()):,}")
    notes.append(
        "  Censored rows contribute lower bound L = study_end - start "
        f"(study_end={study_end})"
    )

    turnbull = ROOT / "outputs" / "layer1_turnbull_quantiles.csv"
    if turnbull.exists():
        tb = pd.read_csv(turnbull)
        notes.append(f"  Turnbull strata exported: {len(tb)}")

    if "trust_score" in df.columns and df["trust_score"].isna().any():
        errors.append("trust_score has NaNs in clean data")

    print("=== CONSISTENCY VALIDATION ===\n")
    for line in notes:
        print(f"  {line}")

    if errors:
        print("\nFAILURES:")
        for e in errors:
            print(f"  ✗ {e}")
        raise SystemExit(1)

    print("\n✓ All consistency checks passed.")


if __name__ == "__main__":
    main()
