"""
Layer 1 — Duration prediction via survival analysis (Kaplan-Meier + Cox PH)
=========================================================================
WHY survival analysis: duration data is right-censored (~4,500 rows lack end
timestamps) and heavily right-skewed. OLS is wrong; KM/Cox are built for this.

trust_score from data_pipeline.py is passed as `weights=` so low-trust rows
contribute proportionally less rather than being hard-dropped.

Run: python src/layer1_survival.py
Outputs:
  - outputs/layer1_survival_quantiles.csv
  - outputs/layer1_survival_fallback.csv
  - outputs/layer1_cox_summary.txt
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "events_clean.parquet"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

MIN_GROUP_SIZE = 15
QUANTILES = (0.50, 0.80, 0.95)
QUANTILE_LABELS = ("p50", "p80", "p95")


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def build_survival_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    T = duration_min, E = 1 if NOT is_censored else 0.

    Rows with null/negative duration_min are hard-excluded (no time information).
    Censored rows without end timestamps therefore leave the KM table — their
    impact is carried via trust_score down-weighting in the cleaned data, and
    the missingness test documents systematic censoring bias.

    WHY not admin-censor at study end: with ~55% missing end timestamps,
    artificial censoring times push KM quantiles to ~160 days, destroying
    operational readability (breakdowns should read ~30-50 min, not study horizon).
    """
    surv = df.copy()
    valid = surv["duration_min"].notna() & (surv["duration_min"] >= 0)
    surv = surv[valid].copy()
    surv["T"] = surv["duration_min"]
    surv["E"] = (~surv["is_censored"]).astype(int)
    surv["weights"] = surv["trust_score"].clip(0.01, 1.0)
    return surv


def _weighted_quantile_from_km(
    kmf: KaplanMeierFitter, q: float
) -> float | None:
    """Read duration quantile from fitted KM survival function S(t)."""
    sf = kmf.survival_function_
    if sf.empty:
        return None
    times = sf.index.values.astype(float)
    probs = sf.iloc[:, 0].values
    target = 1.0 - q
    below = np.where(probs <= target)[0]
    if len(below) == 0:
        return float(times[-1]) if len(times) else None
    return float(times[below[0]])


def fit_km_strata(
    surv: pd.DataFrame,
    group_cols: list[str],
    min_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Kaplan-Meier per stratum: S(t) = prod_{t_i <= t}(1 - d_i/n_i).
    Uses lifelines with trust_score weights.
    """
    rows: list[dict[str, Any]] = []

    for keys, grp in surv.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) < min_size:
            continue

        kmf = KaplanMeierFitter()
        try:
            kmf.fit(
                durations=grp["T"],
                event_observed=grp["E"],
                weights=grp["weights"],
                label=str(keys),
            )
        except Exception as exc:
            print(f"WARNING: KM failed for {keys}: {exc}")
            continue

        record = {col: val for col, val in zip(group_cols, keys)}
        record["n"] = len(grp)
        record["n_events"] = int(grp["E"].sum())
        record["weighted_n"] = float(grp["weights"].sum())

        for q, label in zip(QUANTILES, QUANTILE_LABELS):
            record[f"{label}_min"] = _weighted_quantile_from_km(kmf, q)

        record["median_ci_lower"] = (
            kmf.confidence_interval_survival_function_.index.min()
            if hasattr(kmf, "confidence_interval_survival_function_")
            else np.nan
        )
        rows.append(record)

    return pd.DataFrame(rows)


def fit_cox_model(surv: pd.DataFrame) -> tuple[CoxPHFitter | None, str]:
    """
    Cox PH: h(t|x) = h_0(t) exp(beta^T x).
    Hazard ratios = exp(beta_i) — interpretable effect sizes.
    """
    work = surv.copy()
    work["priority_high"] = (work["priority"] == "High").astype(int)
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)

    top_corridors = work["corridor"].value_counts().head(8).index
    work["corridor_top"] = work["corridor"].where(
        work["corridor"].isin(top_corridors), "Other"
    )
    corridor_dummies = pd.get_dummies(work["corridor_top"], prefix="corridor", drop_first=True)

    cox_df = pd.concat([
        work[["T", "E", "weights", "priority_high", "closure_int",
              "hour_sin", "hour_cos", "is_weekend"]],
        corridor_dummies,
    ], axis=1).dropna()

    if len(cox_df) < 50:
        msg = "WARNING: too few rows for Cox PH; skipping."
        print(msg)
        return None, msg

    cph = CoxPHFitter()
    try:
        cph.fit(
            cox_df,
            duration_col="T",
            event_col="E",
            weights_col="weights",
            show_progress=False,
        )
    except Exception as exc:
        msg = f"WARNING: Cox PH fit failed: {exc}"
        print(msg)
        return None, msg

    lines = [
        "=== Cox Proportional Hazards Summary ===",
        f"Concordance index: {cph.concordance_index_:.3f}",
        "(0.5 = random, 1.0 = perfect; values near 0.5 mean weak covariate signal)",
        "",
        "Hazard ratios (exp(coef)):",
    ]
    summary = cph.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])
    for idx, row in summary.iterrows():
        lines.append(
            f"  {idx}: HR={row['hazard_ratio']:.3f}, p={row['p']:.4f}"
        )
    lines.append("")
    lines.append(
        "INTERPRETATION: corridor and cause (via KM stratification) typically "
        "explain duration better than priority/time-of-day/closure-flag alone."
    )
    report = "\n".join(lines)
    print(report)
    return cph, report


def lookup_expected_duration(
    cause: str,
    corridor: str,
    km_table: pd.DataFrame,
    km_fallback: pd.DataFrame,
    quantile: str = "p50",
) -> dict[str, Any] | None:
    """
    Public lookup: cause+corridor -> cause-only -> None.
    Returns duration estimate with source and confidence metadata.
    """
    col = f"{quantile}_min"
    if col not in km_table.columns:
        raise ValueError(f"Unknown quantile '{quantile}'; use one of {QUANTILE_LABELS}")

    match = km_table[
        (km_table["event_cause"] == cause) & (km_table["corridor"] == corridor)
    ]
    if not match.empty:
        row = match.iloc[0]
        return {
            "duration_min": row[col],
            "source": "cause_corridor",
            "n": int(row["n"]),
            "confidence": "high" if row["n"] >= 30 else "moderate",
        }

    fb = km_fallback[km_fallback["event_cause"] == cause]
    if not fb.empty:
        row = fb.iloc[0]
        return {
            "duration_min": row[col],
            "source": "cause_only_fallback",
            "n": int(row["n"]),
            "confidence": "moderate" if row["n"] >= 30 else "low",
        }

    return None


def run_layer1() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data()
    surv = build_survival_table(df)
    print(f"Survival table: {len(surv):,} rows "
          f"({surv['E'].sum():,} events, {(1-surv['E']).sum():,} censored)")

    km_primary = fit_km_strata(surv, ["event_cause", "corridor"])
    km_fallback = fit_km_strata(surv, ["event_cause"])

    if km_primary.empty:
        print("WARNING: no KM strata met MIN_GROUP_SIZE; check data.")
    else:
        print(f"\nKM strata (cause×corridor): {len(km_primary)} groups")
        preview = km_primary.sort_values("n", ascending=False).head(8)
        show_cols = ["event_cause", "corridor", "n", "p50_min", "p80_min", "p95_min"]
        print(preview[show_cols].to_string(index=False))

    _, cox_report = fit_cox_model(surv)
    (OUT_DIR / "layer1_cox_summary.txt").write_text(cox_report)

    km_primary.to_csv(OUT_DIR / "layer1_survival_quantiles.csv", index=False)
    km_fallback.to_csv(OUT_DIR / "layer1_survival_fallback.csv", index=False)
    print(f"\nSaved quantiles to {OUT_DIR / 'layer1_survival_quantiles.csv'}")
    print(f"Saved fallback to {OUT_DIR / 'layer1_survival_fallback.csv'}")

    # Sanity: sparse cause should fall back
    probe = lookup_expected_duration(
        "protest", "Non-corridor", km_primary, km_fallback, quantile="p50"
    )
    print(f"\nLookup probe (protest, Non-corridor): {probe}")

    return km_primary, km_fallback


if __name__ == "__main__":
    run_layer1()
