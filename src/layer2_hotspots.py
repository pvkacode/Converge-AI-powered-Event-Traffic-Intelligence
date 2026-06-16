"""
Layer 2 — Spatial Hotspot Significance via Getis-Ord Gi* (trust-weighted)
=========================================================================
WHY Gi* not a raw heatmap: heatmaps conflate "busy area" with "anomalously
clustered area." Gi* tests whether local incident density exceeds chance.

x_j = trust-weighted incident intensity (sum of trust_score), not raw count.

SIGNIFICANCE: permutation p_sim is PRIMARY (p < 0.05), not asymptotic z > 1.96.
KNN weights on irregular small samples often break the normal approximation.

Run: python src/layer2_hotspots.py
Outputs:
  - outputs/layer2_hotspots.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from esda.getisord import G_Local
from libpysal.weights import KNN

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "events_clean.parquet"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

K_NEIGHBORS = 6
N_PERMUTATIONS = 999
Z_THRESHOLD = 1.96
P_SIM_THRESHOLD = 0.05


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def build_junction_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per junction: trust-weighted intensity + mean lat/long.
    Drops junctions with no valid geo — Gi* requires real coordinates.
    """
    geo = df[df["geo_valid"]].copy()
    geo = geo[geo["junction"].notna() & (geo["junction"].astype(str).str.strip() != "")]

    if geo.empty:
        raise ValueError("No usable junction rows with valid geo.")

    agg = geo.groupby("junction").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        raw_count=("trust_score", "size"),
        weighted_intensity=("trust_score", "sum"),
        mean_trust=("trust_score", "mean"),
    ).reset_index()

    return agg


def compute_getis_ord(
    junctions: pd.DataFrame,
    k: int = K_NEIGHBORS,
    permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """
    G_i* via esda on KNN spatial weights. x_j = trust-weighted intensity.
    Returns z-scores and permutation p-values (p_sim = primary test).
    """
    k_eff = min(k, len(junctions) - 1)
    if k_eff < 1:
        raise ValueError("Not enough junctions with valid geo to build KNN graph.")

    coords = junctions[["longitude", "latitude"]].values
    w = KNN.from_array(coords, k=k_eff)
    w.transform = "r"

    x = junctions["weighted_intensity"].values.astype(float)
    g = G_Local(x, w, star=True, permutations=permutations, seed=42)

    result = junctions.copy()
    result["z_score"] = g.Zs
    result["p_sim"] = g.p_sim
    result["is_significant_zscore"] = result["z_score"].abs() > Z_THRESHOLD
    result["is_significant_psim"] = result["p_sim"] < P_SIM_THRESHOLD
    result["is_significant"] = result["is_significant_psim"]

    return result.sort_values("z_score", ascending=False)


def run_layer2() -> pd.DataFrame:
    df = load_data()
    junctions = build_junction_table(df)
    print(f"Junctions with valid geo: {len(junctions)}")

    result = compute_getis_ord(junctions)

    n_sig_z = int(result["is_significant_zscore"].sum())
    n_sig_p = int(result["is_significant_psim"].sum())
    max_z = float(result["z_score"].abs().max())

    print(f"\nMax |z-score| across all junctions: {max_z:.2f}")
    print(f"Significant by z > {Z_THRESHOLD} cutoff: {n_sig_z}")
    print(f"Significant by permutation p_sim < {P_SIM_THRESHOLD}: {n_sig_p}")

    if max_z < Z_THRESHOLD and n_sig_p > 0:
        print(
            "NOTE: asymptotic z-score cutoff finds nothing significant, but "
            "permutation p_sim does. Known failure mode under KNN weights on "
            "small/irregular samples — using p_sim as primary criterion."
        )

    print("\n=== Top 15 hotspots by z-score ===")
    cols = [
        "junction", "weighted_intensity", "raw_count", "mean_trust",
        "z_score", "p_sim", "is_significant",
    ]
    print(result[cols].head(15).to_string(index=False))

    out_path = OUT_DIR / "layer2_hotspots.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return result


if __name__ == "__main__":
    run_layer2()
