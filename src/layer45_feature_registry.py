"""
Layer 4.5 — feature registry and leakage audit metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

# Features produced by as-of builders (all marked fold-safe / as-of safe)
ASOF_FEATURE_META = [
    ("asof_p50_duration", "duration", "asof_safe", "prediction_time"),
    ("asof_p80_duration", "duration", "asof_safe", "prediction_time"),
    ("asof_p95_duration", "duration", "asof_safe", "prediction_time"),
    ("asof_surv_prob_60", "duration", "asof_safe", "prediction_time"),
    ("asof_surv_prob_120", "duration", "asof_safe", "prediction_time"),
    ("asof_surv_prob_180", "duration", "asof_safe", "prediction_time"),
    ("asof_rmst_180", "duration", "asof_safe", "prediction_time"),
    ("asof_quantile_fallback_level", "duration", "asof_safe", "prediction_time"),
    ("asof_corridor_burden", "spatial", "asof_safe", "prediction_time"),
    ("asof_junction_burden", "spatial", "asof_safe", "prediction_time"),
    ("asof_obi_proxy", "spatial", "asof_safe", "prediction_time"),
    ("asof_corridor_event_rate", "spatial", "asof_safe", "prediction_time"),
    ("asof_nbr_mean_burden", "spatial", "asof_safe", "prediction_time"),
    ("asof_nbr_mean_duration", "spatial", "asof_safe", "prediction_time"),
    ("asof_fragility_proxy", "fragility", "asof_safe", "prediction_time"),
    ("asof_hawkes_intensity", "fragility", "asof_safe", "prediction_time"),
    ("asof_burstiness", "fragility", "asof_safe", "prediction_time"),
    ("asof_branching_ratio_proxy", "fragility", "asof_safe", "prediction_time"),
    ("asof_retrieval_confidence", "retrieval", "asof_safe", "prediction_time"),
    ("asof_retrieval_n_eff", "retrieval", "asof_safe", "prediction_time"),
    ("asof_retrieval_mean_sim", "retrieval", "asof_safe", "prediction_time"),
    ("asof_retrieval_max_sim", "retrieval", "asof_safe", "prediction_time"),
    ("asof_planned_support", "retrieval", "asof_safe", "prediction_time"),
    ("asof_ims_proxy", "retrieval", "asof_safe", "prediction_time"),
    ("obi_x_fragility", "interaction", "asof_safe", "prediction_time"),
    ("retrieval_x_risk", "interaction", "asof_safe", "prediction_time"),
    ("fragility_x_duration", "interaction", "asof_safe", "prediction_time"),
    ("trust_x_confidence", "interaction", "asof_safe", "prediction_time"),
    ("hotspot_x_duration", "interaction", "asof_safe", "prediction_time"),
    ("hawkes_x_obi", "interaction", "asof_safe", "prediction_time"),
    ("nbr_mean_obi", "graph", "asof_safe", "prediction_time"),
    ("nbr_mean_fragility", "graph", "asof_safe", "prediction_time"),
    ("nbr_mean_duration", "graph", "asof_safe", "prediction_time"),
    ("nbr_mean_severity", "graph", "asof_safe", "prediction_time"),
]

BASE_FEATURE_META = [
    ("event_cause", "base", "asof_safe", "prediction_time"),
    ("corridor", "base", "asof_safe", "prediction_time"),
    ("zone", "base", "asof_safe", "prediction_time"),
    ("junction", "base", "asof_safe", "prediction_time"),
    ("priority", "base", "asof_safe", "prediction_time"),
    ("requires_road_closure", "base", "asof_safe", "prediction_time"),
    ("event_type", "base", "asof_safe", "prediction_time"),
    ("hour_local", "base", "asof_safe", "prediction_time"),
    ("dow_local", "base", "asof_safe", "prediction_time"),
    ("month", "base", "asof_safe", "prediction_time"),
    ("is_weekend", "base", "asof_safe", "prediction_time"),
    ("trust_score", "base", "asof_safe", "prediction_time"),
    ("geo_valid", "base", "asof_safe", "prediction_time"),
    ("duration_anomaly", "base", "asof_safe", "prediction_time"),
    ("iso_flagged", "base", "asof_safe", "prediction_time"),
    ("mnar_predicted_prob", "base", "asof_safe", "prediction_time"),
    ("is_true_planned_event", "base", "asof_safe", "prediction_time"),
]

BLOCKED_FEATURES = [
    ("layer1_survival_risk_score", "layer1_output", "blocked", "full_dataset_leakage"),
    ("layer2_operational_burden_index", "layer2_output", "blocked", "full_dataset_leakage"),
    ("layer3_dis_score", "layer3_output", "blocked", "full_dataset_leakage"),
    ("layer4_retrieval_confidence_full", "layer4_output", "blocked", "full_dataset_leakage"),
    ("duration_min", "target", "training_only", "label"),
]

TARGET_META = [
    ("duration_min", "target", "training_only", "label"),
    ("high_impact", "target", "training_only", "label"),
]


def build_feature_registry() -> list[dict]:
    rows = []
    for name, group, safety, avail in ASOF_FEATURE_META + BASE_FEATURE_META:
        rows.append({
            "feature_name": name,
            "feature_group": group,
            "leakage_status": safety,
            "availability": avail,
            "used_in_training": True,
            "notes": "Point-in-time as-of snapshot feature",
        })
    for name, group, safety, avail in BLOCKED_FEATURES:
        rows.append({
            "feature_name": name,
            "feature_group": group,
            "leakage_status": safety,
            "availability": avail,
            "used_in_training": False,
            "notes": "Blocked — computed on full Nov–Apr batch",
        })
    for name, group, safety, avail in TARGET_META:
        rows.append({
            "feature_name": name,
            "feature_group": group,
            "leakage_status": safety,
            "availability": avail,
            "used_in_training": False,
            "notes": "Supervised target only",
        })
    return rows


def export_feature_registry(out_dir: Path = OUT) -> pd.DataFrame:
    import json

    rows = build_feature_registry()
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "layer45_feature_registry.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "layer45_leakage_audit.csv", index=False)
    return audit
