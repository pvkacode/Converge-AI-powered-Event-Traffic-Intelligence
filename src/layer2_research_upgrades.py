"""
Layer 2 research-grade upgrades (additive only — does not retrain base pipeline models).

3. Multi-scale network Gi*: SPS + NHI (replaces collapsed MSHI)
4. Monte Carlo OBI rank stability
5. Hawkes branching-ratio validation

Run after layer2_hotspots.py:
    python src/layer2_research_upgrades.py
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
import pandas as pd
from esda.getisord import G_Local
from libpysal.weights import W

warnings.filterwarnings("ignore")

from layer2_hotspots import (
    N_PERM,
    OUT_DIR,
    _entropy_weights,
    _equal_weights,
    _network_kernel_weights,
    _pca_weights,
    build_junction_graph,
    compute_severity_hotspots,
    load_data,
)

NETWORK_SCALES = [1, 2, 3, 5]
OBI_MC_SIMS = 1000
OBI_MC_SIGMA = 0.05
OBI_COMPONENTS = [
    "hawkes_norm",
    "severity_norm",
    "persistence_norm",
    "future_risk_norm",
    "frailty_norm",
]


def _cascade_class(branching_ratio: float) -> str:
    if branching_ratio < 0.3:
        return "weak"
    if branching_ratio <= 0.7:
        return "moderate"
    return "strong"


def compute_multiscale_network_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale Persistence Score (SPS) and Network Hotspot Index (NHI).

    SPS_i = mean_h G_i*(h)           — averages raw Gi* values (preserves variance)
    NHI_i = mean_h Percentile(G_i*(h)) — averages percentile ranks (rank-based)
    """
    severity = compute_severity_hotspots(df)
    G = build_junction_graph(df)
    if G is None or G.number_of_nodes() < 10:
        print("WARNING: insufficient graph for multi-scale network index.")
        return pd.DataFrame()

    nodes = [n for n in G.nodes if n in set(severity["junction"])]
    sev = severity.set_index("junction")["severity_hotspot_score"].to_dict()
    x_full = np.array([sev.get(n, 0.0) for n in nodes], dtype=float)
    base_neighbors, _, _ = _network_kernel_weights(G, nodes)
    valid_idx = [i for i in range(len(nodes)) if base_neighbors[i]]
    if len(valid_idx) < 10:
        return pd.DataFrame()
    nodes = [nodes[i] for i in valid_idx]
    x = x_full[valid_idx]

    gi_by_scale: dict[int, np.ndarray] = {}
    for h in NETWORK_SCALES:
        neighbors, kernel_weights, _ = _network_kernel_weights(G, nodes, h=float(h))
        if not any(neighbors[i] for i in range(len(nodes))):
            gi_by_scale[h] = np.zeros(len(nodes))
            continue
        w_obj = W(neighbors, weights=kernel_weights, id_order=list(range(len(nodes))))
        w_obj.transform = "r"
        g = G_Local(x, w_obj, star=True, permutations=N_PERM, seed=42)
        gi_by_scale[h] = np.array(g.Zs, dtype=float)

    gi_matrix = np.column_stack([gi_by_scale[h] for h in NETWORK_SCALES])
    rank_matrix = np.column_stack([
        pd.Series(gi_by_scale[h]).rank(pct=True).values for h in NETWORK_SCALES
    ])
    sps = gi_matrix.mean(axis=1)
    nhi = rank_matrix.mean(axis=1)

    rows = []
    for i, junction in enumerate(nodes):
        row = {"junction": junction}
        for h in NETWORK_SCALES:
            row[f"gi_star_h{h}"] = float(gi_by_scale[h][i])
        row["sps"] = float(sps[i])
        row["nhi"] = float(nhi[i])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("nhi", ascending=False)
    out.to_csv(OUT_DIR / "layer2_multiscale_hotspots.csv", index=False)
    print(
        f"Network index: {len(out)} junctions, "
        f"SPS σ={out['sps'].std():.3f}, NHI σ={out['nhi'].std():.3f}"
    )
    return out


def run_hawkes_validation() -> pd.DataFrame:
    """Branching ratio R = α/β with cascade classification (reads existing Hawkes fit)."""
    path = OUT_DIR / "layer2_hawkes_cascade_risk.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run layer2_hotspots.py first.")

    hawkes = pd.read_csv(path)
    hawkes["branching_ratio"] = (
        hawkes["alpha_excitation"] / hawkes["beta_decay"].clip(lower=1e-6)
    )
    hawkes["cascade_class"] = hawkes["branching_ratio"].map(_cascade_class)
    out = hawkes[["junction", "branching_ratio", "cascade_class"]].copy()
    if "alpha_excitation" in hawkes.columns:
        out["alpha"] = hawkes["alpha_excitation"]
        out["beta"] = hawkes["beta_decay"]
    out = out.sort_values("branching_ratio", ascending=False)
    out.to_csv(OUT_DIR / "layer2_hawkes_validation.csv", index=False)

    counts = out["cascade_class"].value_counts()
    print(
        f"Hawkes validation: {len(out)} junctions — "
        + ", ".join(f"{k}={int(v)}" for k, v in counts.items())
    )
    return out


def _obi_from_components(mat: np.ndarray) -> np.ndarray:
    """Ensemble OBI (entropy + equal + PCA weights), same logic as layer2 compute_obi."""
    w_entropy = _entropy_weights(mat)
    w_equal = _equal_weights(mat.shape[1])
    w_pca = _pca_weights(mat)
    return (mat @ w_entropy + mat @ w_equal + mat @ w_pca) / 3.0


def run_obi_stability() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo perturbation of normalized OBI components (σ=0.05, 1000 sims)."""
    obi_path = OUT_DIR / "layer2_operational_burden_index.csv"
    if not obi_path.exists():
        raise FileNotFoundError(f"Missing {obi_path}. Run layer2_hotspots.py first.")

    obi = pd.read_csv(obi_path)
    col_map = {
        "hawkes_norm": "hawkes_norm",
        "severity_norm": "severity_norm",
        "persistence_norm": "persistence_norm",
        "future_risk_norm": "duration_risk_norm",
        "frailty_norm": "frailty_norm",
    }
    for target, source in col_map.items():
        if source not in obi.columns:
            obi[target] = 0.5
        else:
            obi[target] = obi[source]

    junctions = obi["junction"].values
    components = obi[list(col_map.keys())].values.astype(float)
    n_junc = len(junctions)

    rng = np.random.default_rng(42)
    top10_counts = np.zeros(n_junc, dtype=np.int64)
    top25_counts = np.zeros(n_junc, dtype=np.int64)
    rank_sum = np.zeros(n_junc, dtype=np.float64)
    rank_sq_sum = np.zeros(n_junc, dtype=np.float64)

    batch = 100
    for start in range(0, OBI_MC_SIMS, batch):
        end = min(start + batch, OBI_MC_SIMS)
        n_batch = end - start
        noise = rng.normal(0.0, OBI_MC_SIGMA, size=(n_batch, n_junc, len(OBI_COMPONENTS)))
        perturbed = np.clip(components[None, :, :] + noise, 0.0, 1.0)
        for b in range(n_batch):
            scores = _obi_from_components(perturbed[b])
            order = np.argsort(-scores)
            ranks = np.empty(n_junc, dtype=np.float64)
            ranks[order] = np.arange(1, n_junc + 1)
            rank_sum += ranks
            rank_sq_sum += ranks ** 2
            top10_counts[order[:10]] += 1
            top25_counts[order[:25]] += 1
        if start % 200 == 0:
            gc.collect()

    mean_rank = rank_sum / OBI_MC_SIMS
    rank_var = np.maximum(rank_sq_sum / OBI_MC_SIMS - mean_rank ** 2, 0.0)
    rank_std = np.sqrt(rank_var)

    stability = pd.DataFrame({
        "junction": junctions,
        "mean_rank": mean_rank,
        "rank_std": rank_std,
        "prob_top10": top10_counts / OBI_MC_SIMS,
        "prob_top25": top25_counts / OBI_MC_SIMS,
    }).sort_values("prob_top25", ascending=False)

    stability.to_csv(OUT_DIR / "layer2_obi_stability.csv", index=False)

    stable_top25 = stability.nlargest(25, "prob_top25")[
        ["junction", "mean_rank", "rank_std", "prob_top10", "prob_top25"]
    ]
    stable_top25.to_csv(OUT_DIR / "layer2_obi_stable_top25.csv", index=False)

    print(
        f"OBI stability: {OBI_MC_SIMS} simulations, "
        f"most stable top-25 member: {stable_top25.iloc[0]['junction']} "
        f"(P top-25={stable_top25.iloc[0]['prob_top25']:.2f})"
    )
    return stability, stable_top25


def run_layer2_upgrades() -> None:
    print("=== Layer 2 Research Upgrades (SPS/NHI + Hawkes validation + OBI stability) ===\n")
    df = load_data()
    compute_multiscale_network_index(df)
    run_hawkes_validation()
    run_obi_stability()
    print(f"\nNew outputs → {OUT_DIR}/layer2_multiscale_hotspots.csv, "
          f"layer2_hawkes_validation.csv, layer2_obi_stability.csv")


if __name__ == "__main__":
    run_layer2_upgrades()
