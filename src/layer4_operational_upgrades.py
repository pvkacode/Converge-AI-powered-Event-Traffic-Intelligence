"""
Layer 4 operational upgrade — three-tier evidence framework (final pre-frontend).

Reuses leakage-free Gower + K-Medoids + confidence from layer4_methodology_upgrades.
Does NOT change similarity, trust weighting, prototypes, or confidence formula.

Run after layer4_methodology_upgrades.py:
    python src/layer4_operational_upgrades.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from layer4_methodology_upgrades import (
    CAUSE_COL,
    CORRIDOR_COL,
    H_BANDWIDTH,
    K_NEIGHBORS,
    confidence_score,
    gower_distance,
    _prepare_planned,
    _row_to_query,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"

HIGH_CONF = 0.70
MEDIUM_CONF = 0.40
OPERATOR_WARNING = (
    "Historical evidence is moderate. Recommendations should be reviewed before deployment."
)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q, cw, v))


def confidence_band(conf: float) -> str:
    if conf >= HIGH_CONF:
        return "HIGH"
    if conf >= MEDIUM_CONF:
        return "MEDIUM"
    return "LOW"


def confidence_reason(band: str, mean_s: float, max_s: float, n_eff: float) -> str:
    if band == "HIGH":
        return "High similarity and strong historical support."
    if band == "MEDIUM":
        return "Moderate similarity; evidence partially supported."
    return (
        f"Low similarity and insufficient comparable events "
        f"(mean_sim={mean_s:.2f}, max_sim={max_s:.2f}, n_eff={n_eff:.1f})."
    )


def recommendation_source(band: str) -> str:
    if band == "HIGH":
        return "RETRIEVAL"
    if band == "MEDIUM":
        return "HYBRID"
    return "LAYER3_FALLBACK"


def _corridor_junction_map(df_all: pd.DataFrame) -> dict[str, str]:
    sub = df_all[df_all["junction"].notna() & df_all[CORRIDOR_COL].notna()]
    return (
        sub.groupby(CORRIDOR_COL)["junction"]
        .agg(lambda s: s.mode().iloc[0] if len(s) else "")
        .to_dict()
    )


def _load_layer3_fallback() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mp = pd.read_csv(OUT / "layer3_manpower_recommendations.csv")
    surv = pd.read_csv(OUT / "layer1_survival_quantiles.csv")
    dis = pd.read_csv(OUT / "layer3_disruption_impact_scores.csv")
    return mp, surv, dis


def _l3_row(corridor: str, corr_junc: dict[str, str], mp: pd.DataFrame,
            surv: pd.DataFrame, dis: pd.DataFrame) -> dict:
    junc = corr_junc.get(corridor, "")
    mp_row = mp[mp["junction"] == junc]
    if mp_row.empty:
        mp_row = mp.nlargest(1, "ods_score")
    r = mp_row.iloc[0]
    sq = surv[surv["corridor"] == corridor]
    if sq.empty:
        sq = surv.groupby("corridor")["p80_min"].mean().reset_index().nlargest(1, "p80_min")
    s = sq.iloc[0]
    p50 = float(s.get("p50_min", s.get("p80_min", 60)) or 60)
    p80 = float(s.get("p80_min", 90) or 90)
    p95 = float(s.get("p95_min", p80 * 1.4) or p80 * 1.4)
    impact = float(dis[dis["junction"] == junc]["dis_score"].iloc[0]) if junc and (dis["junction"] == junc).any() else 50.0
    return {
        "pred_duration_p50": p50,
        "pred_duration_p80": min(p80, 360),
        "pred_duration_p95": min(p95, 480),
        "pred_impact_p50": impact,
        "pred_impact_p80": min(impact * 1.15, 100),
        "pred_impact_p95": min(impact * 1.3, 100),
        "recommended_officers": int(r.get("allocated_officers", r.get("officers", 4))),
        "recommended_barricades": int(r.get("allocated_barricades", r.get("barricades_from_ods", 8))),
        "recommended_tow_units": int(r.get("allocated_tow", r.get("tow_vehicles", 0))),
        "recommended_supervisors": int(r.get("allocated_supervisors", r.get("supervisors", 1))),
        "recommended_qru_units": int(r.get("qru_units", 0)),
        "ods_score": float(r.get("ods_score", 0)),
        "dis_score": float(r.get("dis_score", impact)),
    }


def _retrieve_sims(
    query: dict,
    eid: str,
    prototypes_df: pd.DataFrame,
    weights: dict,
    ranges: dict,
) -> tuple[np.ndarray, list[int]]:
    sims_list, pidx = [], []
    for i, (_, proto) in enumerate(prototypes_df.iterrows()):
        if str(proto["representative_event_id"]) == eid:
            continue
        gd = gower_distance(query, proto.to_dict(), weights, ranges)
        trust = float(proto.get("mean_trust", 0.7) or 0.7)
        sim = float(np.exp(-gd / H_BANDWIDTH) * trust)
        if not np.isfinite(sim):
            continue
        sims_list.append(sim)
        pidx.append(i)
    if not sims_list:
        return np.array([]), []
    order = np.argsort(sims_list)[::-1][:K_NEIGHBORS]
    return np.array([sims_list[j] for j in order]), [pidx[j] for j in order]


def run_layer4_operational_upgrades() -> pd.DataFrame:
    print("=== Layer 4 Operational Upgrade (evidence tiers + uncertainty) ===\n")

    proto_path = OUT / "layer4_planned_event_prototypes.csv"
    if not proto_path.exists():
        raise FileNotFoundError("Run layer4_methodology_upgrades.py first.")

    df_all, df_planned, ranges, weights = _prepare_planned()
    prototypes_df = pd.read_csv(proto_path)
    mp, surv, dis = _load_layer3_fallback()
    corr_junc = _corridor_junction_map(df_all)

    obi_lookup: dict[str, float] = {}
    obi_path = OUT / "layer2_operational_burden_index.csv"
    if obi_path.exists():
        obi_df = pd.read_csv(obi_path)
        junc_corr = df_all[df_all["junction"].notna()][["junction", CORRIDOR_COL]].drop_duplicates()
        obi_joined = obi_df[["junction", "operational_burden_index"]].merge(junc_corr, on="junction", how="inner")
        obi_lookup = obi_joined.groupby(CORRIDOR_COL)["operational_burden_index"].mean().to_dict()

    rows = []
    for _, event_row in df_planned.iterrows():
        eid = str(event_row["event_id"])
        query = _row_to_query(event_row)
        corridor = query[CORRIDOR_COL]

        sims, pidx = _retrieve_sims(query, eid, prototypes_df, weights, ranges)
        conf, n_eff, mean_s, max_s = confidence_score(sims)
        band = confidence_band(conf)
        reason = confidence_reason(band, mean_s, max_s, n_eff)
        source = recommendation_source(band)
        abstain = int(band == "LOW")

        l3 = _l3_row(corridor, corr_junc, mp, surv, dis)

        if band in ("HIGH", "MEDIUM") and len(sims) > 0 and sims.sum() > 1e-9:
            protos = prototypes_df.iloc[pidx]
            w = sims
            durations = protos["median_duration"].astype(float).values
            impacts = np.array([obi_lookup.get(str(c), 0.5) * 100 for c in protos["corridor"]])
            pred = {
                "pred_duration_p50": weighted_quantile(durations, w, 0.50),
                "pred_duration_p80": weighted_quantile(durations, w, 0.80),
                "pred_duration_p95": weighted_quantile(durations, w, 0.95),
                "pred_impact_p50": weighted_quantile(impacts, w, 0.50),
                "pred_impact_p80": weighted_quantile(impacts, w, 0.80),
                "pred_impact_p95": weighted_quantile(impacts, w, 0.95),
            }
            res = {k: l3[k] for k in [
                "recommended_officers", "recommended_barricades", "recommended_tow_units",
                "recommended_supervisors", "recommended_qru_units",
            ]}
        else:
            pred = {k: l3[k] for k in [
                "pred_duration_p50", "pred_duration_p80", "pred_duration_p95",
                "pred_impact_p50", "pred_impact_p80", "pred_impact_p95",
            ]}
            res = {k: l3[k] for k in [
                "recommended_officers", "recommended_barricades", "recommended_tow_units",
                "recommended_supervisors", "recommended_qru_units",
            ]}
            reason = reason + " Using Layer 3 DIS/ODS resource optimization as fallback."

        row = {
            "event_id": eid,
            "cause": query[CAUSE_COL],
            "corridor": corridor,
            "confidence": round(conf, 4),
            "effective_sample_size": round(n_eff, 3),
            "mean_similarity": round(mean_s, 4),
            "max_similarity": round(max_s, 4),
            "confidence_band": band,
            "confidence_reason": reason,
            "recommendation_source": source,
            "operator_warning": OPERATOR_WARNING if band == "MEDIUM" else "",
            "abstain_flag": abstain,
            "abstain_reason": "retrieval_insufficient" if abstain else "",
            **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in pred.items()},
            **res,
            "actual_duration": round(float(event_row["duration_clean"]), 1),
        }
        rows.append(row)

    rec = pd.DataFrame(rows)
    rec.to_csv(OUT / "layer4_planned_event_retrieval.csv", index=False)

    diag = rec[[
        "event_id", "confidence", "mean_similarity", "max_similarity",
        "effective_sample_size", "confidence_band", "confidence_reason",
        "abstain_flag", "abstain_reason", "recommendation_source",
    ]].copy()
    diag.to_csv(OUT / "layer4_retrieval_diagnostics.csv", index=False)

    n = len(rec)
    summary = pd.DataFrame([
        {"metric": "planned_events_total", "value": n},
        {"metric": "high_confidence_count", "value": int((rec["confidence_band"] == "HIGH").sum())},
        {"metric": "medium_confidence_count", "value": int((rec["confidence_band"] == "MEDIUM").sum())},
        {"metric": "low_confidence_count", "value": int((rec["confidence_band"] == "LOW").sum())},
        {"metric": "abstention_count", "value": int(rec["abstain_flag"].sum())},
        {"metric": "mean_confidence", "value": round(float(rec["confidence"].mean()), 4)},
        {"metric": "mean_similarity", "value": round(float(rec["mean_similarity"].mean()), 4)},
        {"metric": "mean_effective_sample_size", "value": round(float(rec["effective_sample_size"].mean()), 3)},
        {"metric": "retrieval_coverage_pct", "value": round(100 * (rec["confidence_band"] != "LOW").mean(), 1)},
    ])
    summary.to_csv(OUT / "layer4_retrieval_quality_summary.csv", index=False)

    bins = [i / 10 for i in range(11)]
    hist, _ = np.histogram(rec["confidence"], bins=bins)
    dist = pd.DataFrame({
        "confidence_bin": [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(10)],
        "count": hist,
    })
    dist.to_csv(OUT / "layer4_confidence_distribution.csv", index=False)

    _update_knowledge_base(rec)

    n_high = int((rec["confidence_band"] == "HIGH").sum())
    n_med = int((rec["confidence_band"] == "MEDIUM").sum())
    n_low = int((rec["confidence_band"] == "LOW").sum())
    print(f"  Events: {n} | HIGH={n_high} MEDIUM={n_med} LOW={n_low}")
    print(f"  Retrieval coverage (non-LOW): {(n_high+n_med)/n:.1%}")
    print(f"  Mean confidence: {rec['confidence'].mean():.3f}")
    return rec


def _update_knowledge_base(rec: pd.DataFrame) -> None:
    kb_path = OUT / "layer4_event_knowledge_base.json"
    if not kb_path.exists():
        print("  [SKIP] knowledge base not found")
        return

    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    rec_lu = rec.set_index("event_id").to_dict(orient="index")
    for entry in kb.get("knowledge_entries", []):
        eid = entry.get("event_id")
        if eid not in rec_lu or not entry.get("is_planned_event"):
            continue
        r = rec_lu[eid]
        entry.update({
            "confidence_band": r["confidence_band"],
            "recommendation_source": r["recommendation_source"],
            "confidence_reason": r["confidence_reason"],
            "pred_duration_p50": r["pred_duration_p50"],
            "pred_duration_p80": r["pred_duration_p80"],
            "pred_duration_p95": r["pred_duration_p95"],
            "pred_impact_p50": r["pred_impact_p50"],
            "pred_impact_p80": r["pred_impact_p80"],
            "pred_impact_p95": r["pred_impact_p95"],
            "operator_warning": r.get("operator_warning", ""),
        })

    kb["operational_upgrade"] = {
        "evidence_bands": {"HIGH": f">={HIGH_CONF}", "MEDIUM": f"{MEDIUM_CONF}–{HIGH_CONF}", "LOW": f"<{MEDIUM_CONF}"},
        "confidence_formula": "Conf = (n_eff/(n_eff+2)) * mean_similarity * max_similarity",
    }
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)
    print(f"  Updated layer4_event_knowledge_base.json ({len(rec)} planned enrichments)")


if __name__ == "__main__":
    run_layer4_operational_upgrades()
    from frontend_exports import run_frontend_exports
    run_frontend_exports()
