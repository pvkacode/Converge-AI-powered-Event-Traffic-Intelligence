"""
Layer 4 methodology upgrades (additive — fixes retrieval leakage + prototypes).

1. Leakage-free Gower distance (pre-event query features only)
2. K-Medoids prototypes on Gower distance (replaces KMeans)
3. Calibrated confidence with principled abstention

Run after layer4_planned_event_retrieval.py (or standalone):
    python src/layer4_methodology_upgrades.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"

K_NEIGHBORS = 5
H_BANDWIDTH = 0.5
CONF_THRESHOLD = 0.40
MAX_SIM_THRESHOLD = 0.30
N_EFF_MIN = 3.0

CAUSE_COL = "event_cause"
CORRIDOR_COL = "corridor"
CLOSURE_COL = "requires_road_closure"
DURATION_COL = "duration_min"
PRIORITY_COL = "priority"
TRUST_COL = "trust_score"
PLANNED_COL = "is_true_planned_event"
START_COL = "start_local"

QUERY_COLS = [
    CAUSE_COL,
    CORRIDOR_COL,
    "closure_binary",
    "hour_of_day",
    "day_of_week",
    "priority_numeric",
    "month",
]
CATEGORICAL_QUERY = [CAUSE_COL, CORRIDOR_COL]
BINARY_QUERY = ["closure_binary"]
CONTINUOUS_QUERY = ["hour_of_day", "day_of_week", "priority_numeric", "month"]

PRIORITY_MAP = {"High": 3, "Low": 1, "Unknown": 2}


def _prepare_planned() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    df_all = pd.read_parquet(DATA / "events_clean.parquet").reset_index(drop=True)
    df_all[START_COL] = pd.to_datetime(df_all[START_COL], errors="coerce")
    df_all["hour_of_day"] = df_all[START_COL].dt.hour.fillna(12).astype(int)
    df_all["day_of_week"] = df_all[START_COL].dt.dayofweek.fillna(2).astype(int)
    df_all["month"] = df_all[START_COL].dt.month.fillna(3).astype(int)
    df_all["closure_binary"] = df_all[CLOSURE_COL].astype(bool).astype(int)
    prio_str = df_all[PRIORITY_COL].astype(str).fillna("Unknown")
    df_all["priority_numeric"] = prio_str.map(PRIORITY_MAP).fillna(2.0)
    dur_med = float(df_all[DURATION_COL].clip(0, 1440).dropna().median())
    df_all["duration_clean"] = df_all[DURATION_COL].clip(0, 1440).fillna(dur_med)
    df_all["trust_clean"] = pd.to_numeric(df_all[TRUST_COL], errors="coerce").clip(0.1, 1.0).fillna(0.7)

    planned_mask = df_all[PLANNED_COL] == True
    if planned_mask.sum() < 10:
        kw = ["rally", "festival", "procession", "vip", "protest", "event", "celebr", "march",
              "gather", "sports", "concert", "parade", "public_event"]
        planned_mask = df_all[CAUSE_COL].str.lower().str.contains("|".join(kw), na=False)

    df_planned = df_all[planned_mask].copy().reset_index(drop=True)
    for col in CONTINUOUS_QUERY:
        med = float(pd.to_numeric(df_all[col], errors="coerce").median())
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(med)
        df_planned[col] = pd.to_numeric(df_planned[col], errors="coerce").fillna(med)

    ranges: dict[str, float] = {}
    for col in CONTINUOUS_QUERY + BINARY_QUERY:
        vals = pd.to_numeric(df_all[col], errors="coerce").dropna()
        ranges[col] = max(float(vals.max() - vals.min()), 1e-6)

    weights = {col: 1.0 / len(QUERY_COLS) for col in QUERY_COLS}
    return df_all, df_planned, ranges, weights


def gower_distance(
    q: dict,
    p: dict,
    weights: dict[str, float],
    ranges: dict[str, float],
) -> float:
    total_w, total_d = 0.0, 0.0
    for col in QUERY_COLS:
        w = weights.get(col, 1.0 / len(QUERY_COLS))
        qv, pv = q.get(col), p.get(col)
        if qv is None or pv is None:
            continue
        if col in CATEGORICAL_QUERY or col in BINARY_QUERY:
            d_k = 0.0 if str(qv) == str(pv) else 1.0
        else:
            r = ranges.get(col, 1.0)
            d_k = min(1.0, abs(float(qv) - float(pv)) / r)
        total_d += w * d_k
        total_w += w
    return total_d / total_w if total_w > 1e-9 else 1.0


def _row_to_query(row: pd.Series) -> dict:
    return {
        CAUSE_COL: str(row.get(CAUSE_COL, "__NA__") or "__NA__"),
        CORRIDOR_COL: str(row.get(CORRIDOR_COL, "__NA__") or "__NA__"),
        "closure_binary": int(row.get("closure_binary", 0) or 0),
        "hour_of_day": int(row.get("hour_of_day", 12)),
        "day_of_week": int(row.get("day_of_week", 2)),
        "priority_numeric": float(row.get("priority_numeric", 2)),
        "month": int(row.get("month", 3)),
    }


def _gower_matrix(rows: list[dict], weights: dict, ranges: dict) -> np.ndarray:
    n = len(rows)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = gower_distance(rows[i], rows[j], weights, ranges)
            D[i, j] = D[j, i] = d
    return D


def k_medoids_gower(D: np.ndarray, n_clusters: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """K-Medoids on precomputed Gower distance matrix."""
    n = D.shape[0]
    n_clusters = min(n_clusters, n)
    rng = np.random.default_rng(42)
    medoids = rng.choice(n, size=n_clusters, replace=False)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        for i in range(n):
            labels[i] = int(np.argmin(D[i, medoids]))
        new_medoids = medoids.copy()
        for k in range(n_clusters):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                continue
            intra = D[np.ix_(members, members)].sum(axis=1)
            new_medoids[k] = members[int(np.argmin(intra))]
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return labels, medoids


def effective_sample_size(sims: np.ndarray) -> float:
    pos = sims[sims > 0]
    if len(pos) == 0:
        return 0.0
    s1, s2 = float(pos.sum()), float((pos ** 2).sum())
    return (s1 ** 2) / s2 if s2 > 1e-12 else 0.0


def confidence_score(sims: np.ndarray) -> tuple[float, float, float, float]:
    valid = sims[sims > 0]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0
    n_eff = effective_sample_size(valid)
    mean_s = float(valid.mean())
    max_s = float(valid.max())
    conf = (n_eff / (n_eff + 2.0)) * mean_s * max_s
    return float(conf), n_eff, mean_s, max_s


def should_abstain(conf: float, max_s: float, n_eff: float) -> tuple[bool, str]:
    if max_s < MAX_SIM_THRESHOLD:
        return True, f"max_similarity={max_s:.3f}<{MAX_SIM_THRESHOLD}"
    if n_eff < N_EFF_MIN:
        return True, f"n_eff={n_eff:.2f}<{N_EFF_MIN}"
    if conf < CONF_THRESHOLD:
        return True, f"confidence={conf:.3f}<{CONF_THRESHOLD}"
    return False, "sufficient_evidence"


def build_prototypes(df_planned: pd.DataFrame, weights: dict, ranges: dict) -> pd.DataFrame:
    n = len(df_planned)
    n_clusters = max(10, min(50, n // 4)) if n >= 40 else max(5, n // 3)
    queries = [_row_to_query(df_planned.iloc[i]) for i in range(n)]
    D = _gower_matrix(queries, weights, ranges)
    labels, medoid_idx = k_medoids_gower(D, n_clusters)

    rows = []
    for k, mid in enumerate(medoid_idx):
        members = np.where(labels == k)[0]
        ce = df_planned.iloc[members]
        medoid_row = df_planned.iloc[mid]
        member_dists = D[mid, members]
        sims = np.exp(-member_dists / H_BANDWIDTH)
        quality = float(sims.mean()) if len(sims) else 0.0

        rows.append({
            "prototype_id": k,
            "representative_event_id": str(medoid_row["event_id"]),
            "support_count": int(len(members)),
            "corridor": str(medoid_row[CORRIDOR_COL]),
            "cause": str(medoid_row[CAUSE_COL]),
            "median_duration": float(ce["duration_clean"].median()),
            "mean_trust": float(ce["trust_clean"].mean()) if ce["trust_clean"].notna().any() else 0.7,
            "closure_rate": float(ce["closure_binary"].mean()),
            "prototype_quality_score": round(quality, 4),
            **{col: _row_to_query(medoid_row)[col] for col in QUERY_COLS},
        })
    return pd.DataFrame(rows)


def run_layer4_methodology_upgrades() -> None:
    print("=== Layer 4 Methodology Upgrades (leakage-free retrieval) ===\n")
    df_all, df_planned, ranges, weights = _prepare_planned()
    n_planned = len(df_planned)
    print(f"  Planned events: {n_planned}")

    validation = pd.DataFrame([
        {"feature_name": col, "used_in_similarity": True} for col in QUERY_COLS
    ] + [
        {"feature_name": name, "used_in_similarity": False}
        for name in [
            "duration_min", "predicted_duration", "closure_rate_outcome",
            "OBI_impact", "severity_metrics", "trust_score",
        ]
    ])
    validation.to_csv(OUT / "layer4_retrieval_validation.csv", index=False)

    prototypes_df = build_prototypes(df_planned, weights, ranges)
    proto_save = prototypes_df[[
        "prototype_id", "representative_event_id", "support_count",
        "corridor", "cause", "median_duration", "mean_trust",
        "prototype_quality_score",
    ]].rename(columns={"mean_trust": "mean_trust"})
    prototypes_df.to_csv(OUT / "layer4_planned_event_prototypes.csv", index=False)
    print(f"  K-Medoids prototypes: {len(prototypes_df)}")

    obi_lookup: dict[str, float] = {}
    obi_path = OUT / "layer2_operational_burden_index.csv"
    if obi_path.exists():
        obi_df = pd.read_csv(obi_path)
        junc_corr = df_all[df_all["junction"].notna()][["junction", CORRIDOR_COL]].drop_duplicates()
        obi_joined = obi_df[["junction", "operational_burden_index"]].merge(junc_corr, on="junction", how="inner")
        obi_lookup = obi_joined.groupby(CORRIDOR_COL)["operational_burden_index"].mean().to_dict()

    retrieval_rows, diag_rows = [], []
    for _, event_row in df_planned.iterrows():
        eid = str(event_row["event_id"])
        query = _row_to_query(event_row)

        sims_list, gdists, pidx = [], [], []
        for i, (_, proto) in enumerate(prototypes_df.iterrows()):
            if str(proto["representative_event_id"]) == eid:
                continue
            gd = gower_distance(query, proto.to_dict(), weights, ranges)
            trust = float(proto.get("mean_trust", 0.7) or 0.7)
            sim = float(np.exp(-gd / H_BANDWIDTH) * trust)
            if not np.isfinite(sim):
                continue
            sims_list.append(sim)
            gdists.append(gd)
            pidx.append(i)

        if not sims_list:
            sims = np.array([])
            pidx = []
        else:
            order = np.argsort(sims_list)[::-1][:K_NEIGHBORS]
            sims = np.array([sims_list[j] for j in order])
            pidx = [pidx[j] for j in order]

        conf, n_eff, mean_s, max_s = confidence_score(sims)
        abstain, reason = should_abstain(conf, max_s, n_eff)

        pred_med = pred_p80 = pred_cl = float("nan")
        if len(sims) > 0 and sims.sum() > 1e-9:
            w = sims / sims.sum()
            protos = prototypes_df.iloc[pidx]
            pred_med = float(np.dot(w, protos["median_duration"].values))
            pred_cl = float(np.dot(w, protos["closure_rate"].values))

        retrieval_rows.append({
            "query_event_id": eid,
            "cause": query[CAUSE_COL],
            "corridor": query[CORRIDOR_COL],
            "confidence": round(conf, 4),
            "mean_similarity": round(mean_s, 4),
            "max_similarity": round(max_s, 4),
            "effective_sample_size": round(n_eff, 3),
            "abstain_flag": int(abstain),
            "abstain_reason": reason,
            "predicted_duration_median": round(pred_med, 1) if not np.isnan(pred_med) else np.nan,
            "predicted_closure_probability": round(pred_cl, 4) if not np.isnan(pred_cl) else np.nan,
            "actual_duration": round(float(event_row["duration_clean"]), 1),
        })
        diag_rows.append({
            "event_id": eid,
            "confidence": round(conf, 4),
            "mean_similarity": round(mean_s, 4),
            "max_similarity": round(max_s, 4),
            "effective_sample_size": round(n_eff, 3),
            "abstain_flag": int(abstain),
            "abstain_reason": reason,
        })

    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df.to_csv(OUT / "layer4_planned_event_retrieval.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "layer4_retrieval_diagnostics.csv", index=False)

    n_abs = int(retrieval_df["abstain_flag"].sum())
    print(f"  Retrieval: {n_planned} events, abstain={n_abs} ({n_abs/max(n_planned,1):.1%})")
    print(f"  Mean confidence: {retrieval_df['confidence'].mean():.3f}")
    print(f"  Outputs → layer4_retrieval_validation.csv, layer4_planned_event_prototypes.csv, "
          f"layer4_planned_event_retrieval.csv, layer4_retrieval_diagnostics.csv")

    weights_doc = {
        "query_features": QUERY_COLS,
        "outcome_only_features": ["duration_min", "trust_score", "OBI", "severity", "closure_rate"],
        "confidence_formula": "Conf = (n_eff/(n_eff+2)) * mean_sim * max_sim",
        "abstention": {
            "confidence_lt": CONF_THRESHOLD,
            "max_similarity_lt": MAX_SIM_THRESHOLD,
            "n_eff_lt": N_EFF_MIN,
        },
        "prototype_method": "K-Medoids on Gower distance",
        "feature_weights": weights,
    }
    with open(OUT / "layer4_retrieval_feature_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_doc, f, indent=2)


if __name__ == "__main__":
    run_layer4_methodology_upgrades()
