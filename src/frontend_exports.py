"""
Frontend export layer — single read path for dashboard integration.

Generates clean copies under outputs/frontend/ from canonical pipeline outputs.
The frontend must NOT read research-validation CSVs directly.

Run after all layer scripts:
    python src/frontend_exports.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
FRONT = OUT / "frontend"


def _export(src_name: str, dst_name: str, columns: list[str] | None = None) -> bool:
    src = OUT / src_name
    if not src.exists():
        print(f"  [SKIP] {src_name} not found")
        return False
    df = pd.read_csv(src)
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    df.to_csv(FRONT / dst_name, index=False)
    print(f"  [OK] {dst_name} ({len(df)} rows)")
    return True


def run_frontend_exports() -> None:
    print("=== Frontend Exports ===\n")
    FRONT.mkdir(parents=True, exist_ok=True)

    _export(
        "layer1_survival_quantiles.csv",
        "duration_lookup.csv",
        ["event_cause", "corridor", "n", "p50_min", "p80_min", "p95_min"],
    )
    _export(
        "layer1_survival_risk_scores.csv",
        "risk_scores.csv",
        ["event_id", "event_cause", "corridor", "junction", "T", "survival_risk_score"],
    )
    _export(
        "layer2_multiscale_hotspots.csv",
        "hotspot_rankings.csv",
        ["junction", "sps", "nhi", "gi_star_h1", "gi_star_h2", "gi_star_h3", "gi_star_h5"],
    )
    if not (FRONT / "hotspot_rankings.csv").exists():
        _export(
            "layer2_hotspots.csv",
            "hotspot_rankings.csv",
            ["junction", "weighted_intensity", "z_score", "p_sim", "is_significant"],
        )
    _export(
        "layer2_operational_burden_index.csv",
        "operational_burden.csv",
        ["junction", "operational_burden_index", "severity_norm", "persistence_norm",
         "hawkes_norm", "duration_risk_norm", "frailty_norm"],
    )
    _export(
        "layer2_obi_stable_top25.csv",
        "top25_locations.csv",
        ["junction", "mean_rank", "rank_std", "prob_top10", "prob_top25"],
    )
    if not (FRONT / "top25_locations.csv").exists():
        _export(
            "layer2_operational_burden_top25.csv",
            "top25_locations.csv",
            ["junction", "operational_burden_index"],
        )

    frag_path = OUT / "layer3_corridor_fragility.csv"
    if frag_path.exists():
        frag = pd.read_csv(frag_path)
        cols = ["corridor", "mu", "alpha", "beta", "branching_ratio",
                "current_intensity", "fragility_raw", "fragility_log"]
        cols = [c for c in cols if c in frag.columns]
        if "fragility_log" not in frag.columns and "current_fragility" in frag.columns:
            mu = frag["mu"].astype(float)
            lam = frag["current_intensity"].astype(float)
            frag["fragility_raw"] = frag["current_fragility"]
            frag["fragility_log"] = (lam - mu).clip(lower=0) / (mu + 0.01)
            frag["fragility_log"] = frag["fragility_log"].apply(lambda x: __import__("math").log1p(x))
            cols = [c for c in cols + ["fragility_raw", "fragility_log"] if c in frag.columns]
        frag[cols].to_csv(FRONT / "corridor_fragility.csv", index=False)
        print(f"  [OK] corridor_fragility.csv ({len(frag)} rows)")

    _export(
        "layer4_planned_event_retrieval.csv",
        "planned_event_recommendations.csv",
        [
            "event_id", "cause", "corridor", "confidence", "confidence_band",
            "recommendation_source", "confidence_reason", "operator_warning",
            "effective_sample_size", "mean_similarity", "max_similarity",
            "pred_duration_p50", "pred_duration_p80", "pred_duration_p95",
            "pred_impact_p50", "pred_impact_p80", "pred_impact_p95",
            "recommended_officers", "recommended_barricades", "recommended_tow_units",
            "recommended_supervisors", "recommended_qru_units",
            "abstain_flag",
        ],
    )
    _export(
        "layer4_retrieval_diagnostics.csv",
        "layer4_retrieval_diagnostics.csv",
        [
            "query_event_id", "cause", "corridor", "confidence", "effective_sample_size",
            "mean_similarity", "abstain_flag", "actual_duration", "predicted_duration_median",
            "abs_error", "rel_error",
            "geo_radius_2km_count", "geo_radius_nearest_km", "geo_sigma_km",
        ],
    )
    _export("layer4_geo_radius_matches.csv", "layer4_geo_radius_matches.csv")

    dis_path = OUT / "layer3_disruption_impact_scores.csv"
    if dis_path.exists() and not (FRONT / "top25_locations.csv").exists():
        dis = pd.read_csv(dis_path).nlargest(25, "dis_score")
        dis[["junction", "dis_score", "risk_level"]].to_csv(
            FRONT / "top25_locations.csv", index=False
        )
        print(f"  [OK] top25_locations.csv from DIS ({len(dis)} rows)")

    print(f"\nFrontend exports → {FRONT}/")


if __name__ == "__main__":
    run_frontend_exports()
