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


NON_CORRIDOR = {"", "non-corridor", "Non-corridor"}
PLANNED_QRU_CAUSES = {"procession", "protest", "vip_movement", "public_event"}
RESOURCE_KEYS = [
    "recommended_officers",
    "recommended_barricades",
    "recommended_tow_units",
    "recommended_supervisors",
    "recommended_qru_units",
]
PRED_KEYS = [
    "pred_duration_p50",
    "pred_duration_p80",
    "pred_duration_p95",
    "pred_impact_p50",
    "pred_impact_p80",
    "pred_impact_p95",
]


def _is_non_corridor(corridor: str) -> bool:
    return str(corridor or "").strip() in NON_CORRIDOR


def _median_row(df: pd.DataFrame, col: str = "ods_score") -> pd.Series:
    if df.empty:
        raise ValueError("empty dataframe")
    ranked = df.sort_values(col)
    return ranked.iloc[len(ranked) // 2]


def _load_hotspot_coords() -> pd.DataFrame:
    path = OUT / "layer2_hotspots.csv"
    if not path.exists():
        return pd.DataFrame(columns=["junction", "latitude", "longitude"])
    geo = pd.read_csv(path)
    cols = {"junction", "latitude", "longitude"}
    if not cols.issubset(geo.columns):
        return pd.DataFrame(columns=["junction", "latitude", "longitude"])
    return geo[list(cols)].dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def _nearest_junction(lat: float, lng: float, geo: pd.DataFrame) -> str:
    if geo.empty or not np.isfinite(lat) or not np.isfinite(lng):
        return ""
    dlat = geo["latitude"].astype(float) - lat
    dlng = geo["longitude"].astype(float) - lng
    dist2 = dlat * dlat + dlng * dlng
    idx = int(dist2.argmin())
    return str(geo.iloc[idx]["junction"])


def _event_junction_map(df_all: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in df_all.iterrows():
        eid = str(row.get("event_id", ""))
        junc = str(row.get("junction", "") or "").strip()
        if eid and junc:
            out[eid] = junc
    return out


def _cause_junction_pool(df_planned: pd.DataFrame) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = {}
    for _, row in df_planned.iterrows():
        cause = str(row.get(CAUSE_COL, "")).strip()
        junc = str(row.get("junction", "") or "").strip()
        corridor = str(row.get(CORRIDOR_COL, "") or "").strip()
        if not cause or not junc or _is_non_corridor(corridor):
            continue
        pool.setdefault(cause, set()).add(junc)
    return {k: sorted(v) for k, v in pool.items()}


def _build_junc_obi_lookup() -> dict[str, float]:
    path = OUT / "layer2_operational_burden_index.csv"
    if not path.exists():
        return {}
    obi = pd.read_csv(path)
    if "junction" not in obi.columns or "operational_burden_index" not in obi.columns:
        return {}
    return {
        str(r["junction"]): float(r["operational_burden_index"]) * 100.0
        for _, r in obi.iterrows()
        if pd.notna(r["junction"])
    }


def _build_junc_dis_lookup(dis: pd.DataFrame) -> dict[str, float]:
    if "junction" not in dis.columns or "dis_score" not in dis.columns:
        return {}
    return {
        str(r["junction"]): float(r["dis_score"])
        for _, r in dis.iterrows()
        if pd.notna(r["junction"])
    }


def _resolve_junction(
    corridor: str,
    corr_junc: dict[str, str],
    *,
    cause: str,
    lat: float | None,
    lng: float | None,
    cause_junc_pool: dict[str, list[str]],
    mp: pd.DataFrame,
    geo: pd.DataFrame,
) -> str:
    junc = corr_junc.get(corridor, "").strip()
    if junc and not _is_non_corridor(corridor):
        return junc

    if lat is not None and lng is not None:
        near = _nearest_junction(float(lat), float(lng), geo)
        if near:
            return near

    pool = cause_junc_pool.get(cause, [])
    if pool:
        mp_pool = mp[mp["junction"].isin(pool)]
        if not mp_pool.empty:
            return str(_median_row(mp_pool)["junction"])

    if not mp.empty:
        return str(_median_row(mp)["junction"])
    return ""


def _mp_resources(mp_row: pd.Series) -> dict[str, int]:
    return {
        "recommended_officers": int(mp_row.get("allocated_officers", mp_row.get("officers", 4))),
        "recommended_barricades": int(mp_row.get("allocated_barricades", mp_row.get("barricades_from_ods", 8))),
        "recommended_tow_units": int(mp_row.get("allocated_tow", mp_row.get("tow_vehicles", 0))),
        "recommended_supervisors": int(mp_row.get("allocated_supervisors", mp_row.get("supervisors", 1))),
        "recommended_qru_units": int(mp_row.get("qru_units", 0)),
    }


def _impact_from_junction(
    junc: str,
    junc_dis: dict[str, float],
    junc_obi: dict[str, float],
    fallback: float = 50.0,
) -> float:
    if junc and junc in junc_dis:
        return float(junc_dis[junc])
    if junc and junc in junc_obi:
        return float(junc_obi[junc])
    return fallback


def _impact_triplet(impact: float) -> dict[str, float]:
    impact = float(np.clip(impact, 5.0, 100.0))
    return {
        "pred_impact_p50": impact,
        "pred_impact_p80": min(impact * 1.15, 100.0),
        "pred_impact_p95": min(impact * 1.3, 100.0),
    }


def _duration_triplet_from_survival(
    corridor: str,
    cause: str,
    surv: pd.DataFrame,
    df_planned: pd.DataFrame,
) -> dict[str, float]:
    sq = surv[surv["corridor"] == corridor] if corridor and not _is_non_corridor(corridor) else pd.DataFrame()
    if sq.empty and cause:
        cause_corridors = (
            df_planned.loc[df_planned[CAUSE_COL] == cause, CORRIDOR_COL]
            .astype(str)
            .loc[lambda s: ~s.map(_is_non_corridor)]
            .unique()
        )
        sq = surv[surv["corridor"].isin(cause_corridors)]
    if sq.empty:
        sq = surv.groupby("corridor")["p80_min"].mean().reset_index().nlargest(1, "p80_min")
    s = sq.iloc[0]
    p50 = float(s.get("p50_min", s.get("p80_min", 60)) or 60)
    p80 = float(s.get("p80_min", 90) or 90)
    p95 = float(s.get("p95_min", p80 * 1.4) or p80 * 1.4)
    return {
        "pred_duration_p50": p50,
        "pred_duration_p80": min(p80, 360),
        "pred_duration_p95": min(p95, 480),
    }


def _l3_row(
    corridor: str,
    corr_junc: dict[str, str],
    mp: pd.DataFrame,
    surv: pd.DataFrame,
    dis: pd.DataFrame,
    *,
    cause: str = "",
    lat: float | None = None,
    lng: float | None = None,
    cause_junc_pool: dict[str, list[str]] | None = None,
    geo: pd.DataFrame | None = None,
    junc_dis: dict[str, float] | None = None,
    junc_obi: dict[str, float] | None = None,
    df_planned: pd.DataFrame | None = None,
) -> dict:
    cause_junc_pool = cause_junc_pool or {}
    geo = geo if geo is not None else pd.DataFrame()
    junc_dis = junc_dis or {}
    junc_obi = junc_obi or {}

    junc = _resolve_junction(
        corridor,
        corr_junc,
        cause=cause,
        lat=lat,
        lng=lng,
        cause_junc_pool=cause_junc_pool,
        mp=mp,
        geo=geo,
    )

    mp_row = mp[mp["junction"] == junc] if junc else pd.DataFrame()
    if mp_row.empty and cause:
        pool = cause_junc_pool.get(cause, [])
        mp_pool = mp[mp["junction"].isin(pool)]
        if not mp_pool.empty:
            mp_row = _median_row(mp_pool).to_frame().T
    if mp_row.empty:
        mp_row = _median_row(mp).to_frame().T

    r = mp_row.iloc[0]
    impact = _impact_from_junction(junc, junc_dis, junc_obi)
    durations = _duration_triplet_from_survival(
        corridor,
        cause,
        surv,
        df_planned if df_planned is not None else pd.DataFrame(),
    )
    res = _mp_resources(r)
    return {
        **durations,
        **_impact_triplet(impact),
        **res,
        "ods_score": float(r.get("ods_score", 0)),
        "dis_score": float(r.get("dis_score", impact)),
        "_junction": junc,
    }


def _prototype_impacts(
    protos: pd.DataFrame,
    weights: np.ndarray,
    event_junc: dict[str, str],
    junc_dis: dict[str, float],
    junc_obi: dict[str, float],
) -> np.ndarray:
    impacts = []
    for _, proto in protos.iterrows():
        eid = str(proto.get("representative_event_id", ""))
        junc = event_junc.get(eid, "")
        if junc:
            impacts.append(_impact_from_junction(junc, junc_dis, junc_obi, fallback=np.nan))
        else:
            dur = float(proto.get("median_duration", np.nan))
            impacts.append(min(dur / 1.5, 100.0) if np.isfinite(dur) else np.nan)
    arr = np.array(impacts, dtype=float)
    if np.all(~np.isfinite(arr)):
        return np.full(len(protos), 50.0)
    fill = float(np.nanmedian(arr))
    arr[~np.isfinite(arr)] = fill
    return arr


def _resources_from_retrieval(
    protos: pd.DataFrame,
    weights: np.ndarray,
    event_junc: dict[str, str],
    mp: pd.DataFrame,
    base_res: dict[str, int],
    pred_impact_p50: float,
    cause: str,
) -> dict[str, int]:
    if protos.empty or weights.sum() <= 0:
        return dict(base_res)

    stacks: dict[str, list[float]] = {k: [] for k in RESOURCE_KEYS}
    w_list: list[float] = []
    for (_, proto), w in zip(protos.iterrows(), weights):
        eid = str(proto.get("representative_event_id", ""))
        junc = event_junc.get(eid, "")
        mp_row = mp[mp["junction"] == junc] if junc else pd.DataFrame()
        if mp_row.empty:
            continue
        res = _mp_resources(mp_row.iloc[0])
        for k in RESOURCE_KEYS:
            stacks[k].append(float(res[k]))
        w_list.append(float(w))

    if not w_list:
        return dict(base_res)

    w_arr = np.array(w_list)
    w_arr = w_arr / w_arr.sum()
    blended = {
        k: float(np.dot(w_arr, stacks[k]))
        for k in RESOURCE_KEYS
    }

    base_impact = max(float(base_res.get("dis_score", pred_impact_p50)), 1.0)
    scale = np.clip(np.sqrt(max(pred_impact_p50, 1.0) / base_impact), 0.75, 1.5)
    out = {
        "recommended_officers": int(np.clip(round(blended["recommended_officers"] * scale), 0, 25)),
        "recommended_barricades": int(np.clip(round(blended["recommended_barricades"] * scale), 0, 40)),
        "recommended_tow_units": int(np.clip(round(blended["recommended_tow_units"] * scale), 0, 5)),
        "recommended_supervisors": int(np.clip(round(blended["recommended_supervisors"] * scale), 0, 5)),
        "recommended_qru_units": int(np.clip(round(blended["recommended_qru_units"]), 0, 2)),
    }

    if (
        out["recommended_qru_units"] == 0
        and cause in PLANNED_QRU_CAUSES
        and pred_impact_p50 >= 45.0
    ):
        out["recommended_qru_units"] = 1

    if out["recommended_officers"] == 0 and base_res["recommended_officers"] > 0:
        out["recommended_officers"] = base_res["recommended_officers"]
    return out


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
    geo = _load_hotspot_coords()
    cause_junc_pool = _cause_junction_pool(df_planned)
    event_junc = _event_junction_map(df_all)
    junc_obi = _build_junc_obi_lookup()
    junc_dis = _build_junc_dis_lookup(dis)

    rows = []
    for _, event_row in df_planned.iterrows():
        eid = str(event_row["event_id"])
        query = _row_to_query(event_row)
        corridor = query[CORRIDOR_COL]
        cause = str(query[CAUSE_COL])
        lat = pd.to_numeric(event_row.get("latitude"), errors="coerce")
        lng = pd.to_numeric(event_row.get("longitude"), errors="coerce")
        lat_f = float(lat) if pd.notna(lat) else None
        lng_f = float(lng) if pd.notna(lng) else None

        sims, pidx = _retrieve_sims(query, eid, prototypes_df, weights, ranges)
        conf, n_eff, mean_s, max_s = confidence_score(sims)
        band = confidence_band(conf)
        reason = confidence_reason(band, mean_s, max_s, n_eff)
        source = recommendation_source(band)
        abstain = int(band == "LOW")

        l3 = _l3_row(
            corridor,
            corr_junc,
            mp,
            surv,
            dis,
            cause=cause,
            lat=lat_f,
            lng=lng_f,
            cause_junc_pool=cause_junc_pool,
            geo=geo,
            junc_dis=junc_dis,
            junc_obi=junc_obi,
            df_planned=df_planned,
        )
        l3.pop("_junction", None)
        base_res = {k: l3[k] for k in RESOURCE_KEYS}

        if band in ("HIGH", "MEDIUM") and len(sims) > 0 and sims.sum() > 1e-9:
            protos = prototypes_df.iloc[pidx]
            w = sims
            durations = protos["median_duration"].astype(float).values
            impacts = _prototype_impacts(protos, w, event_junc, junc_dis, junc_obi)
            pred = {
                "pred_duration_p50": weighted_quantile(durations, w, 0.50),
                "pred_duration_p80": weighted_quantile(durations, w, 0.80),
                "pred_duration_p95": weighted_quantile(durations, w, 0.95),
                "pred_impact_p50": weighted_quantile(impacts, w, 0.50),
                "pred_impact_p80": weighted_quantile(impacts, w, 0.80),
                "pred_impact_p95": weighted_quantile(impacts, w, 0.95),
            }
            res = _resources_from_retrieval(
                protos,
                w,
                event_junc,
                mp,
                {**base_res, "dis_score": l3["dis_score"]},
                pred["pred_impact_p50"],
                cause,
            )
        else:
            pred = {k: l3[k] for k in PRED_KEYS}
            res = dict(base_res)
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
    n_combos = rec.groupby(RESOURCE_KEYS).ngroups
    n_qru = int((rec["recommended_qru_units"] > 0).sum())
    print(f"  Events: {n} | HIGH={n_high} MEDIUM={n_med} LOW={n_low}")
    print(f"  Retrieval coverage (non-LOW): {(n_high+n_med)/n:.1%}")
    print(f"  Mean confidence: {rec['confidence'].mean():.3f}")
    print(f"  Unique resource bundles: {n_combos} | rows with QRU>0: {n_qru}")
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
