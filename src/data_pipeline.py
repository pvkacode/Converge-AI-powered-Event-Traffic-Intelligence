"""
Day 1 data pipeline — ASTraM Bengaluru incident log (~8,170 rows).

WHY this file exists: downstream survival analysis (Layer 1) and spatial
hotspot detection (Layer 2) require a single cleaned table with correct
censoring labels, cyclical time features, and a composite trust_score
that weights rows proportionally instead of binary keep/drop filtering.

The trust_score replaces a blunt global duration cutoff with stratified
MAD outlier detection, a missingness-mechanism test, and Isolation Forest
multivariate anomaly detection — all combined via noisy-OR.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "events_raw.csv"
DATA_CLEAN_PARQUET = ROOT / "data" / "events_clean.parquet"
DATA_CLEAN_CSV = ROOT / "data" / "events_clean.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MISSINGNESS_OUT = OUT_DIR / "missingness_test.txt"

DATETIME_COLS = [
    "start_datetime",
    "end_datetime",
    "modified_datetime",
    "created_date",
    "closed_datetime",
    "resolved_datetime",
]

CATEGORICAL_COLS = [
    "event_cause",
    "corridor",
    "zone",
    "priority",
    "junction",
    "police_station",
    "veh_type",
]

MAD_THRESHOLD = 3.5
MIN_STRATUM_SIZE = 10
MIN_GROUP_SIZE_KM = 15  # documented for Layer 1; used in summary only here

TRUST_WEIGHTS = {
    "duration_anomaly": 0.30,
    "not_geo_valid": 0.40,
    "mnar_censored": 0.30,
    "iso_flagged": 0.30,
}

LOCAL_TZ = "Asia/Kolkata"

# Load-bearing for project framing (Layer 4 case-based retrieval): true discrete
# planned disruptions are a small minority of total rows (~191 / 8,170).
TRUE_PLANNED_CAUSES = frozenset({
    "public_event",
    "procession",
    "vip_movement",
    "protest",
})

# Known typo / casing variants in source ASTraM export
EVENT_CAUSE_ALIASES = {
    "debris": "debris",
    "fog / low visibility": "fog_low_visibility",
    "fog_low_visibility": "fog_low_visibility",
}


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Load raw CSV and normalize column names to the pipeline schema."""
    df = pd.read_csv(path, low_memory=False)
    if "event_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "event_id"})
    return df


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse UTC datetime columns; local columns use Asia/Kolkata (+5:30)."""
    df = df.copy()
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if "start_datetime" in df.columns:
        start_local = df["start_datetime"].dt.tz_convert(LOCAL_TZ)
        df["start_local"] = start_local
        df["hour_local"] = start_local.dt.hour
        df["dow_local"] = start_local.dt.dayofweek  # Monday=0
        df["date_local"] = start_local.dt.date
    return df


def _normalize_event_cause(value) -> str | float:
    """Map known typo/case variants to canonical snake_case cause labels."""
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    if key in {"nan", "none", "null", ""}:
        return np.nan
    return EVENT_CAUSE_ALIASES.get(key, key.replace(" ", "_"))


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string categoricals so stratification keys are stable.

    WHY: raw export has inconsistent casing ("Debris"/"debris", "Fog / Low
    Visibility") that would split KM strata and inflate apparent category counts.
    """
    df = df.copy()
    if "corridor" in df.columns:
        df["corridor"] = df["corridor"].fillna("Non-corridor")

    for col in CATEGORICAL_COLS + ["event_type", "status"]:
        if col not in df.columns:
            continue
        if col == "event_cause":
            df[col] = df[col].apply(_normalize_event_cause)
            continue
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "NULL": np.nan, "": np.nan})
        )

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].str.lower()

    if "requires_road_closure" in df.columns:
        df["requires_road_closure"] = df["requires_road_closure"].apply(
            _parse_closure_bool
        )

    df["is_true_planned_event"] = df["event_cause"].isin(TRUE_PLANNED_CAUSES)
    return df


def _parse_closure_bool(value) -> bool | float:
    """Parse requires_road_closure from bool, string TRUE/FALSE, or missing."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in {"TRUE", "T", "1", "YES"}:
        return True
    if text in {"FALSE", "F", "0", "NO"}:
        return False
    return np.nan


def build_duration_and_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct duration_min and is_censored respecting right-censoring.

    End-marker priority: closed_datetime → end_datetime → resolved_datetime.
    is_censored=True when NO end marker exists (includes active incidents and
    closed-status rows missing all timestamps — a known source data quality issue).
    """
    df = df.copy()
    end_marker = (
        df["closed_datetime"]
        .fillna(df["end_datetime"])
        .fillna(df["resolved_datetime"])
    )
    df["end_marker"] = end_marker
    df["is_censored"] = end_marker.isna()
    df["duration_min"] = (
        (end_marker - df["start_datetime"]).dt.total_seconds() / 60.0
    )
    return df


def _mad_modified_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Iglewicz-Hoaglin modified z-score using median and MAD."""
    x = values.astype(float)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if mad == 0 or not np.isfinite(mad):
        return np.full_like(x, np.nan, dtype=float), median, mad
    modified_z = 0.6745 * (x - median) / mad
    return modified_z, median, mad


def flag_duration_anomalies(
    df: pd.DataFrame,
    threshold: float = MAD_THRESHOLD,
    min_stratum_size: int = MIN_STRATUM_SIZE,
) -> pd.DataFrame:
    """
    Stratified robust outlier detection via Median Absolute Deviation.

    WHY MAD not global cutoff: a 12-hour construction incident on ORR East 2
    is operationally normal; the same duration for vehicle_breakdown elsewhere
    is likely bad data. Modified z within (cause × corridor) captures that.

    Fallback hierarchy: (cause, corridor) → cause-only → global.
    """
    out = df.copy()
    out["modified_z"] = np.nan
    out["duration_anomaly"] = False

    valid = out["duration_min"].notna() & (out["duration_min"] >= 0)
    if valid.sum() == 0:
        print("WARNING: no valid duration_min values for MAD anomaly detection.")
        return out

    assigned = pd.Series(False, index=out.index)

    def apply_mad_to_indices(indices: pd.Index, reference_values: np.ndarray) -> None:
        """Score `indices` using MAD computed on `reference_values`."""
        if len(indices) == 0:
            return
        z, _, mad = _mad_modified_z(reference_values)
        if mad == 0 or not np.isfinite(mad) or np.all(np.isnan(z)):
            return
        if len(indices) == len(reference_values):
            out.loc[indices, "modified_z"] = z
        else:
            ref_median = np.nanmedian(reference_values)
            ref_mad = np.nanmedian(np.abs(reference_values - ref_median))
            if ref_mad == 0:
                return
            scored = 0.6745 * (out.loc[indices, "duration_min"].values - ref_median) / ref_mad
            out.loc[indices, "modified_z"] = scored
            z = scored
        out.loc[indices, "duration_anomaly"] = np.abs(out.loc[indices, "modified_z"]) > threshold
        assigned.loc[indices] = True

    # Primary: cause × corridor
    for _, sub in out[valid].groupby(["event_cause", "corridor"], dropna=False):
        if len(sub) >= min_stratum_size:
            apply_mad_to_indices(sub.index, sub["duration_min"].values)

    # Fallback: cause-only for unassigned valid rows
    still = valid & ~assigned
    for cause, sub in out[still].groupby("event_cause", dropna=False):
        cause_ref = out[valid & (out["event_cause"] == cause)]
        if len(cause_ref) >= min_stratum_size:
            apply_mad_to_indices(sub.index, cause_ref["duration_min"].values)

    # Global fallback for any remaining
    still = valid & ~assigned
    if still.any():
        apply_mad_to_indices(out[still].index, out.loc[valid, "duration_min"].values)

    return out


def add_geo_valid(df: pd.DataFrame) -> pd.DataFrame:
    """geo_valid=False for null coords or placeholder (0,0) values."""
    df = df.copy()
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    df["geo_valid"] = (
        lat.notna()
        & lon.notna()
        & (lat.abs() > 1)
        & (lon.abs() > 1)
    )
    return df


def _top_n_dummies(series: pd.Series, n: int = 8, other_label: str = "Other") -> pd.DataFrame:
    top = series.value_counts().head(n).index.tolist()
    collapsed = series.where(series.isin(top), other_label)
    return pd.get_dummies(collapsed, prefix=series.name, drop_first=True)


def run_missingness_test(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Test whether missing end-timestamps are MCAR or systematic (MAR/MNAR).

    WHY: naive Kaplan-Meier assumes non-informative censoring. If corridor
    or cause predicts missingness, censored rows are not exchangeable with
    observed durations — we must down-weight high-risk censored rows.
    """
    work = df.copy()
    work["corridor_fill"] = work["corridor"].fillna("Non-corridor")
    work["priority_fill"] = work["priority"].fillna("Unknown")
    work["cause_fill"] = work["event_cause"].fillna("Unknown")
    work["hour_local"] = work["hour_local"].fillna(work["hour_local"].median())
    work["closure_int"] = work["requires_road_closure"].fillna(False).astype(int)

    y = work["is_censored"].astype(int)
    X_parts = [
        _top_n_dummies(work["corridor_fill"], n=8, other_label="Other"),
        _top_n_dummies(work["priority_fill"], n=5, other_label="Other"),
        _top_n_dummies(work["cause_fill"], n=12, other_label="Other"),
        work[["hour_local", "closure_int"]],
    ]
    X = pd.concat(X_parts, axis=1).astype(float)
    X = sm.add_constant(X, has_constant="add")

    # Drop zero-variance columns that make the Hessian singular
    nzv = X.columns[X.std() == 0]
    if len(nzv):
        X = X.drop(columns=nzv)

    default_prob = float(y.mean())
    work["mnar_predicted_prob"] = default_prob
    work["mnar_censored_flag"] = work["is_censored"] & (work["mnar_predicted_prob"] > 0.7)

    try:
        null_model = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0, maxiter=200)
        full_model = sm.Logit(y, X).fit(disp=0, maxiter=200, method="bfgs")

        lr_stat = 2 * (full_model.llf - null_model.llf)
        df_diff = X.shape[1] - 1
        lr_p = stats.chi2.sf(lr_stat, df_diff)

        sig_cols = []
        for col in X.columns:
            if col == "const":
                continue
            pval = full_model.pvalues.get(col, np.nan)
            if pval < 0.05:
                sig_cols.append(f"{col} (p={pval:.4f})")

        work["mnar_predicted_prob"] = full_model.predict(X)
        work["mnar_censored_flag"] = work["is_censored"] & (work["mnar_predicted_prob"] > 0.7)

        if lr_p < 0.05:
            verdict = (
                f"VERDICT: Missing end-timestamps are NOT completely at random (MCAR).\n"
                f"Likelihood-ratio test: LR={lr_stat:.2f}, df={df_diff}, p={lr_p:.2e}.\n"
                f"Interpretation: missingness is systematic (MAR/MNAR). Naive survival "
                f"curves may be biased; censored rows with high predicted missingness "
                f"probability are down-weighted via trust_score.\n"
                f"Significant covariates (p<0.05): "
                f"{', '.join(sig_cols) if sig_cols else 'none individually significant'}"
            )
        else:
            verdict = (
                f"VERDICT: No strong evidence against MCAR (LR={lr_stat:.2f}, p={lr_p:.4f}).\n"
                f"Censoring may still be partially informative; trust_score still "
                f"down-weights censored rows with high predicted P(missing)."
            )
    except Exception as exc:
        verdict = (
            f"VERDICT: Logistic missingness model could not be fit ({exc}).\n"
            f"Falling back to baseline P(censored)={default_prob:.3f} for MNAR flagging.\n"
            f"Censored rows still down-weighted via the is_censored component of trust_score."
        )
        print(f"WARNING: missingness test fallback — {exc}")

    MISSINGNESS_OUT.write_text(verdict + "\n")
    print(verdict)

    return work, verdict


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multivariate anomaly detection for joint weirdness no single rule catches.

    O(n log n), no distributional assumptions — appropriate for skewed traffic data.
    """
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_local"].fillna(12) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_local"].fillna(12) / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow_local"].fillna(0) / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow_local"].fillna(0) / 7)
    out["is_weekend"] = out["dow_local"].isin([5, 6]).astype(int)

    dur = out["duration_min"].copy()
    dur = dur.where(dur >= 0)
    dur_imputed = dur.fillna(dur.median())

    le = LabelEncoder()
    priority_enc = le.fit_transform(out["priority"].fillna("Unknown").astype(str))

    lat = pd.to_numeric(out["latitude"], errors="coerce").fillna(out["latitude"].median())
    lon = pd.to_numeric(out["longitude"], errors="coerce").fillna(out["longitude"].median())
    closure_int = out["requires_road_closure"].fillna(False).astype(int)

    feature_matrix = np.column_stack([
        dur_imputed,
        out["hour_sin"],
        out["hour_cos"],
        out["dow_sin"],
        out["dow_cos"],
        lat,
        lon,
        priority_enc,
        closure_int,
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    iso = IsolationForest(contamination="auto", random_state=42)
    iso.fit(X_scaled)
    scores = iso.score_samples(X_scaled)  # more negative = more anomalous

    out["iso_anomaly_score"] = scores
    cutoff = np.quantile(scores, 0.05)
    out["iso_flagged"] = scores <= cutoff
    return out


def compute_trust_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite Data Trust Score via noisy-OR over independent evidence flags:

        trust_i = prod_k (1 - w_k * flag_k,i)

    Rows contribute proportionally to KM/Gi* rather than binary exclusion.
    """
    out = df.copy()
    flags = {
        "duration_anomaly": out["duration_anomaly"].astype(float),
        "not_geo_valid": (~out["geo_valid"]).astype(float),
        "mnar_censored": out.get("mnar_censored_flag", pd.Series(False, index=out.index)).astype(float),
        "iso_flagged": out["iso_flagged"].astype(float),
    }
    trust = np.ones(len(out), dtype=float)
    for key, weight in TRUST_WEIGHTS.items():
        trust *= 1.0 - weight * flags[key].values
    out["trust_score"] = np.clip(trust, 0.0, 1.0)
    return out


def add_category_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Stable integer codes for categoricals; original strings preserved."""
    out = df.copy()
    for col in CATEGORICAL_COLS:
        if col not in out.columns:
            continue
        filled = out[col].fillna("__MISSING__").astype(str)
        codes, uniques = pd.factorize(filled, sort=True)
        out[f"{col}_code"] = codes
    return out


def print_summary(df: pd.DataFrame, verdict: str) -> None:
    """Human-readable pipeline summary — quote these numbers in the concept note."""
    closed_no_ts = (
        (df["status"].astype(str).str.lower() == "closed")
        & df["is_censored"]
    ).sum()
    true_planned = int(df["is_true_planned_event"].sum())
    true_planned_pct = 100.0 * true_planned / len(df)
    planned_rows = int((df["event_type"] == "planned").sum())

    print("\n=== DATA PIPELINE SUMMARY ===")
    print(f"Total rows: {len(df):,}")
    print(f"geo_valid rows: {df['geo_valid'].sum():,}")
    print(f"Censored (no end timestamp): {df['is_censored'].sum():,}")
    print(f"Closed status but NO end timestamp: {closed_no_ts:,}  "
          f"(data-quality finding — status/timestamp inconsistency)")
    print(f"is_true_planned_event: {true_planned} ({true_planned_pct:.1f}% of all rows)")
    print(f"Planned rows (event_type=planned): {planned_rows}")
    print("\nevent_type value_counts:")
    print(df["event_type"].value_counts().to_string())
    print(f"\nDuration anomalies (stratified MAD): {df['duration_anomaly'].sum():,}")
    print(f"Isolation Forest flagged (bottom 5%): {df['iso_flagged'].sum():,}")
    print(f"Mean trust_score: {df['trust_score'].mean():.3f}")
    if "start_local" in df.columns and df["start_local"].notna().any():
        print(f"Date range (start_local): {df['start_local'].min()} → {df['start_local'].max()}")
    print(f"Missingness test saved to: {MISSINGNESS_OUT}")


def run_pipeline() -> pd.DataFrame:
    """Execute full cleaning pipeline and persist outputs."""
    print(f"Loading raw data from {DATA_RAW}")
    df = load_raw()
    df = clean_categoricals(df)
    df = parse_datetimes(df)
    df = build_duration_and_censoring(df)
    df = add_geo_valid(df)
    df = flag_duration_anomalies(df)
    df, verdict = run_missingness_test(df)
    df = run_isolation_forest(df)
    df = compute_trust_score(df)
    df = add_category_codes(df)

    print_summary(df, verdict)

    df.to_parquet(DATA_CLEAN_PARQUET, index=False)
    df.to_csv(DATA_CLEAN_CSV, index=False)
    print(f"\nSaved cleaned data:\n  {DATA_CLEAN_PARQUET}\n  {DATA_CLEAN_CSV}")
    return df


if __name__ == "__main__":
    run_pipeline()
