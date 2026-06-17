"""
Layer 3 methodology upgrades (additive — no Hawkes / DIS / LP retraining).

4. Bootstrap PCA loading stability for DIS (B=500)
5. Log fragility score on existing corridor fragility output

Run after layer3_corridor_fragility.py and layer3_resource_optimization.py:
    python src/layer3_methodology_upgrades.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
DATA = ROOT / "data"

PCA_BOOTSTRAP = 500
FRAGILITY_EPS = 0.01

FEATURE_COLS = [
    "operational_burden_index",
    "cascade_risk",
    "future_risk_score",
    "rmst_mean",
    "hotspot_persistence_index",
]
FEATURE_LABELS = {
    "operational_burden_index": "OBI",
    "cascade_risk": "Cascade Risk",
    "future_risk_score": "Future Risk",
    "rmst_mean": "RMST",
    "hotspot_persistence_index": "Persistence",
}


def _load_dis_features() -> pd.DataFrame:
    obi = pd.read_csv(OUT / "layer2_operational_burden_index.csv")
    hawkes = pd.read_csv(OUT / "layer2_hawkes_cascade_risk.csv")
    future = pd.read_csv(OUT / "layer2_future_hotspot_risk.csv")
    rmst = pd.read_csv(OUT / "layer1_rmst_summary.csv")
    persist = pd.read_csv(OUT / "layer2_hotspot_persistence.csv")
    events = pd.read_parquet(DATA / "events_clean.parquet")

    ev_jc = events[events["junction"].notna()][["junction", "corridor"]].drop_duplicates()
    rmst_by_corr = rmst.groupby("corridor")["rmst_360min"].mean().reset_index()
    rmst_by_corr.columns = ["corridor", "rmst_360min_mean"]
    junc_rmst = (
        ev_jc.merge(rmst_by_corr, on="corridor", how="left")
        .groupby("junction")["rmst_360min_mean"].mean()
        .reset_index()
    )
    junc_rmst.columns = ["junction", "rmst_mean"]

    feat = obi[["junction", "operational_burden_index"]].merge(
        hawkes[["junction", "cascade_risk"]], on="junction", how="left"
    ).merge(
        future[["junction", "future_risk_score"]], on="junction", how="left"
    ).merge(junc_rmst, on="junction", how="left").merge(
        persist[["junction", "hotspot_persistence_index"]], on="junction", how="left"
    )

    for col in FEATURE_COLS[1:]:
        feat[col] = feat[col].fillna(feat[col].median())
    return feat.dropna(subset=FEATURE_COLS)


def run_pca_loading_stability() -> pd.DataFrame:
    feat = _load_dis_features()
    X = feat[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n = len(X_scaled)
    rng = np.random.default_rng(42)

    loadings = {col: [] for col in FEATURE_COLS}
    for _ in range(PCA_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        Xb = X_scaled[idx]
        if Xb.shape[0] < 5:
            continue
        pca = PCA(n_components=1)
        pca.fit(Xb)
        comp = pca.components_[0].copy()
        if np.corrcoef(Xb @ comp, Xb[:, 0])[0, 1] < 0:
            comp = -comp
        for col, val in zip(FEATURE_COLS, comp):
            loadings[col].append(float(val))

    rows = []
    for col in FEATURE_COLS:
        vals = np.array(loadings[col])
        rows.append({
            "feature": FEATURE_LABELS[col],
            "mean_loading": float(vals.mean()),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        })
    stability = pd.DataFrame(rows).sort_values("mean_loading", key=abs, ascending=False)
    stability.to_csv(OUT / "layer3_pca_loading_stability.csv", index=False)

    top = stability.iloc[0]
    lines = [
        "=== DIS PCA Loading Stability (Bootstrap B=500) ===",
        "",
        f"Top driver: {top['feature']} "
        f"(mean loading={top['mean_loading']:.4f}, "
        f"95% CI [{top['ci_lower']:.4f}, {top['ci_upper']:.4f}])",
        "",
        "Interpretation:",
    ]
    for _, row in stability.iterrows():
        lines.append(
            f"  {row['feature']:16s}: {row['mean_loading']:+.4f} "
            f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]"
        )
    lines.append("")
    lines.append(
        "Variables whose CI excludes zero consistently drive DIS (PC1). "
        "Use this to defend PCA-based DIS in methodology reviews."
    )
    (OUT / "layer3_pca_stability_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"PCA stability: top driver {top['feature']} (B={PCA_BOOTSTRAP})")
    return stability


def run_fragility_log_score() -> pd.DataFrame:
    path = OUT / "layer3_corridor_fragility.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run layer3_corridor_fragility.py first.")

    df = pd.read_csv(path)
    mu = df["mu"].astype(float).values
    lam = df["current_intensity"].astype(float).values
    df["fragility_raw"] = df["current_fragility"].astype(float)
    df["fragility_log"] = np.log1p(np.maximum(lam - mu, 0.0) / (mu + FRAGILITY_EPS))
    df.to_csv(path, index=False)
    print(f"Fragility log score: {len(df)} corridors (ε={FRAGILITY_EPS})")
    return df


def run_layer3_methodology_upgrades() -> None:
    print("=== Layer 3 Methodology Upgrades (PCA stability + log fragility) ===\n")
    run_pca_loading_stability()
    run_fragility_log_score()
    print(f"\nOutputs → {OUT}/layer3_pca_loading_stability.csv, "
          f"layer3_pca_stability_summary.txt, layer3_corridor_fragility.csv (enriched)")


if __name__ == "__main__":
    run_layer3_methodology_upgrades()
