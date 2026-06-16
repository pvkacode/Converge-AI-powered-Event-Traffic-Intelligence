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

    # --- data_pipeline checks ---
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

    # Censored rows must have null duration_min (original design)
    censored_with_dur = df["is_censored"] & df["duration_min"].notna()
    if censored_with_dur.any():
        errors.append(
            f"{censored_with_dur.sum()} censored rows have non-null duration_min "
            "(should be impossible with end-marker logic)"
        )

    # --- layer1 survival design checks (mirrors build_survival_table) ---
    surv = df[df["duration_min"].notna() & (df["duration_min"] >= 0)].copy()
    surv["E"] = (~surv["is_censored"]).astype(int)

    notes.append(f"Survival-modelable rows: {len(surv):,}")
    notes.append(f"  events observed (E=1): {int(surv['E'].sum()):,}")
    notes.append(f"  censored in survival table (E=0): {int((1 - surv['E']).sum()):,}")

    if (1 - surv["E"]).sum() != 0:
        notes.append(
            "  NOTE: E=0 count is 0 by design — censored rows lack duration_min "
            "and are excluded from KM quantiles (same as original v1). Their "
            "impact is via trust_score + missingness test, not KM censoring time."
        )

    # Original v1 used hard 1440-min cap; upgraded pipeline uses MAD + trust instead
    if (surv["duration_min"] > 1440).any():
        notes.append(
            f"  Rows with duration > 1440 min retained: "
            f"{(surv['duration_min'] > 1440).sum()} "
            "(intentional — stratified MAD + trust_score replaces global cutoff)"
        )

    # Weights available for Layer 1
    if "trust_score" in surv.columns and surv["trust_score"].isna().any():
        errors.append("trust_score has NaNs in survival-modelable rows")

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
