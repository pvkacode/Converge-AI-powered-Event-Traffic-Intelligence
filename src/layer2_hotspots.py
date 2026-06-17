"""
Layer 2 — Spatial hotspot intelligence (baseline + advanced)
==============================================================
Baseline: trust-weighted Getis-Ord Gi* (Euclidean KNN)
Advanced: severity hotspots, spatiotemporal Gi*, network-kernel Gi*,
           spatio-temporal Hawkes, persistence, future risk,
           entropy-weighted Operational Burden Index (OBI)

Run: python src/layer2_hotspots.py
Outputs → outputs/layer2_*.csv
"""

from __future__ import annotations

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from esda.getisord import G_Local
from libpysal.weights import KNN, W
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "events_clean.parquet"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

K_NEIGHBORS = 6
N_PERMUTATIONS = 999
N_PERM = 499
P_THRESHOLD = 0.05
HOTSPOT_BOOTSTRAP = 50
BOOTSTRAP_PERMUTATIONS = 99
GI_STAR_THRESHOLD = 1.96
NETWORK_H_CANDIDATES = [1, 2, 3, 5, 10]
PRIORITY_WEIGHTS = {"High": 3, "Medium": 2, "Low": 1}
EARTH_RADIUS_KM = 6371.0

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def _priority_weight(s: pd.Series) -> pd.Series:
    return s.map(PRIORITY_WEIGHTS).fillna(1).astype(float)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _entropy_weights(matrix: np.ndarray) -> np.ndarray:
    n, _ = matrix.shape
    if n < 2:
        return np.ones(matrix.shape[1]) / matrix.shape[1]
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums <= 0, 1.0, col_sums)
    p_ij = matrix / col_sums
    entropy = -np.sum(p_ij * np.log(p_ij + 1e-12), axis=0) / np.log(n)
    divergence = 1.0 - entropy
    if divergence.sum() <= 0:
        return np.ones(matrix.shape[1]) / matrix.shape[1]
    return divergence / divergence.sum()


def _equal_weights(n_metrics: int) -> np.ndarray:
    return np.ones(n_metrics) / n_metrics


def _pca_weights(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 3:
        return _equal_weights(matrix.shape[1])
    pca = PCA(n_components=1)
    pca.fit(matrix)
    loadings = np.abs(pca.components_[0])
    if loadings.sum() <= 0:
        return _equal_weights(matrix.shape[1])
    return loadings / loadings.sum()


def _severity_contribution(df: pd.DataFrame) -> pd.Series:
    duration = df["duration_min"].fillna(df["duration_min"].median())
    return df["trust_score"] * np.log1p(duration) * _priority_weight(df["priority"])


# --- Baseline Gi* ------------------------------------------------------------

def build_junction_table(df: pd.DataFrame) -> pd.DataFrame:
    geo = df[df["geo_valid"] & df["junction"].notna()].copy()
    geo = geo[geo["junction"].astype(str).str.strip() != ""]
    if geo.empty:
        raise ValueError("No usable junction rows with valid geo.")
    return geo.groupby("junction").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        raw_count=("trust_score", "size"),
        weighted_intensity=("trust_score", "sum"),
        mean_trust=("trust_score", "mean"),
    ).reset_index()


def compute_getis_ord(
    junctions: pd.DataFrame,
    intensity_col: str = "weighted_intensity",
    permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    k_eff = min(K_NEIGHBORS, len(junctions) - 1)
    coords = junctions[["longitude", "latitude"]].values
    w = KNN.from_array(coords, k=k_eff)
    w.transform = "r"
    g = G_Local(junctions[intensity_col].values.astype(float), w, star=True,
                permutations=permutations, seed=42)
    out = junctions.copy()
    out["z_score"] = g.Zs
    out["p_sim"] = g.p_sim
    out["is_significant"] = out["p_sim"] < P_THRESHOLD
    return out.sort_values("z_score", ascending=False)


# --- Advanced ----------------------------------------------------------------

def compute_severity_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    geo = df[df["geo_valid"] & df["junction"].notna()].copy()
    geo["duration_fill"] = geo["duration_min"].fillna(geo["duration_min"].median())
    geo["severity_contrib"] = _severity_contribution(geo)
    return geo.groupby("junction").agg(
        severity_hotspot_score=("severity_contrib", "sum"),
        raw_count=("trust_score", "size"),
        mean_duration=("duration_fill", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    ).reset_index().sort_values("severity_hotspot_score", ascending=False)


def compute_spatiotemporal_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    geo = df[df["geo_valid"] & df["junction"].notna()].copy()
    geo["severity"] = _severity_contribution(geo)
    rows = []
    for (hour, dow), sl in geo.groupby(["hour_local", "dow_local"]):
        if len(sl) < 50:
            continue
        junc = sl.groupby("junction").agg(
            x=("severity", "sum"), latitude=("latitude", "mean"), longitude=("longitude", "mean")
        ).reset_index()
        if len(junc) < 10:
            continue
        k = min(K_NEIGHBORS, len(junc) - 1)
        w = KNN.from_array(junc[["longitude", "latitude"]].values, k=k)
        w.transform = "r"
        g = G_Local(junc["x"].values, w, star=True, permutations=min(N_PERM, 199), seed=42)
        sig = g.p_sim < P_THRESHOLD
        if sig.sum() == 0 and len(junc) >= 15:
            sig = g.Zs >= np.quantile(g.Zs, 0.90)
        for idx, (_, row) in enumerate(junc.iterrows()):
            rows.append({"junction": row["junction"], "hour": int(hour), "dow": int(dow),
                         "gi_star": float(g.Zs[idx]), "p_sim": float(g.p_sim[idx]),
                         "is_significant": bool(sig[idx]), "weighted_intensity": float(row["x"])})
    return pd.DataFrame(rows)


def build_junction_graph(df: pd.DataFrame):
    if not HAS_NX:
        return None
    geo = df[df["geo_valid"] & df["junction"].notna() & df["corridor"].notna()]
    G = nx.Graph()
    for corridor, grp in geo.groupby("corridor"):
        js = grp["junction"].unique()
        for i, j1 in enumerate(js):
            for j2 in js[i + 1:]:
                G.add_edge(j1, j2)
    return G


def _network_kernel_weights(
    G, nodes: list[str], h: float | None = None,
) -> tuple[dict[int, list[int]], dict[int, dict[int, float]], float]:
    distances = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i >= j:
                continue
            if nx.has_path(G, n1, n2):
                distances.append(nx.shortest_path_length(G, n1, n2))
    if h is None:
        h = float(np.median(distances)) if distances else 1.0
    h = max(float(h), 1.0)
    max_dist = 3.0 * h

    neighbors: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    weights: dict[int, dict[int, float]] = {i: {} for i in range(len(nodes))}
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i == j:
                continue
            if nx.has_path(G, n1, n2):
                d_ij = nx.shortest_path_length(G, n1, n2)
                if d_ij <= max_dist:
                    w_ij = float(np.exp(-d_ij / h))
                    if w_ij > 1e-6:
                        neighbors[i].append(j)
                        weights[i][j] = w_ij
    return neighbors, weights, h


def _network_gi_variance(x: np.ndarray, neighbors, kernel_weights) -> float:
    w_obj = W(neighbors, weights=kernel_weights, id_order=list(range(len(x))))
    w_obj.transform = "r"
    g = G_Local(x, w_obj, star=True, permutations=99, seed=42)
    z = g.Zs[np.isfinite(g.Zs)]
    return float(np.var(z)) if len(z) else 0.0


def _poisson_loglik(mu: float, times: np.ndarray, t_max: float) -> float:
    if mu <= 0 or len(times) == 0:
        return -np.inf
    return len(times) * np.log(mu) - mu * t_max


def compute_network_hotspots(df: pd.DataFrame, severity: pd.DataFrame) -> pd.DataFrame:
    G = build_junction_graph(df)
    if G is None or G.number_of_nodes() < 10:
        return pd.DataFrame()
    nodes = [n for n in G.nodes if n in set(severity["junction"])]
    sev = severity.set_index("junction")["severity_hotspot_score"].to_dict()
    x_full = np.array([sev.get(n, 0) for n in nodes])
    base_neighbors, _, _ = _network_kernel_weights(G, nodes)
    valid_idx = [i for i in range(len(nodes)) if base_neighbors[i]]
    if len(valid_idx) < 10:
        return pd.DataFrame()
    nodes = [nodes[i] for i in valid_idx]
    x = x_full[valid_idx]

    best_h, best_var, best_g = None, -np.inf, None
    for h_try in NETWORK_H_CANDIDATES:
        neighbors, kernel_weights, h_val = _network_kernel_weights(G, nodes, h=float(h_try))
        valid = [i for i in range(len(nodes)) if neighbors[i]]
        if len(valid) < 10:
            continue
        var_g = _network_gi_variance(x, neighbors, kernel_weights)
        if var_g > best_var:
            best_var = var_g
            best_h = h_val
            best_g = (neighbors, kernel_weights)

    if best_g is None:
        return pd.DataFrame()
    neighbors, kernel_weights = best_g
    h = best_h
    w_obj = W(neighbors, weights=kernel_weights, id_order=list(range(len(nodes))))
    w_obj.transform = "r"
    g = G_Local(x, w_obj, star=True, permutations=N_PERM, seed=42)
    rows = [{
        "junction": nodes[i],
        "network_gi_star": float(g.Zs[i]),
        "p_sim": float(g.p_sim[i]),
        "is_significant": bool(g.p_sim[i] < P_THRESHOLD),
        "severity_hotspot_score": float(x[i]),
        "kernel_bandwidth_h": h,
    } for i in range(len(nodes))]
    return pd.DataFrame(rows).sort_values("network_gi_star", ascending=False)


def _spatiotemporal_hawkes_nll(params, times, spatial_w, t_max):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e12
    n = len(times)
    ll = 0.0
    excitation = 0.0
    last_t = 0.0
    for i in range(n):
        dt_step = times[i] - last_t
        if dt_step > 0:
            excitation *= np.exp(-beta * dt_step)
        lam = mu + excitation
        if lam <= 1e-12:
            return 1e12
        ll += np.log(lam)
        excitation += alpha * spatial_w[i]
        last_t = times[i]
    compensator = mu * t_max + (alpha / beta) * np.sum(spatial_w * (1.0 - np.exp(-beta * (t_max - times))))
    return -(ll - compensator)


def fit_spatiotemporal_hawkes(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    geo = df[df["geo_valid"] & df["junction"].notna()].copy().sort_values("start_datetime")
    if geo.empty:
        return pd.DataFrame()
    t0 = geo["start_datetime"].min()
    all_times = ((geo["start_datetime"] - t0).dt.total_seconds() / 60.0).values.astype(float)
    all_lat = geo["latitude"].values
    all_lon = geo["longitude"].values
    t_max = float(all_times.max()) + 1.0
    rows = []
    for junction in geo["junction"].value_counts().head(top_n).index:
        sub = geo[geo["junction"] == junction]
        if len(sub) < 10:
            continue
        lat0 = float(sub["latitude"].mean())
        lon0 = float(sub["longitude"].mean())
        dist_km = _haversine_km(all_lat, all_lon, lat0, lon0)
        sigma = max(float(np.median(dist_km[dist_km > 0])), 0.5) if np.any(dist_km > 0) else 2.0
        spatial_w = np.exp(-dist_km / sigma)
        keep = spatial_w >= 0.05
        if keep.sum() < 20:
            keep = spatial_w >= np.quantile(spatial_w, 0.5)
        times = all_times[keep]
        spatial_w = spatial_w[keep]
        order = np.argsort(times)
        times = times[order]
        spatial_w = spatial_w[order]
        x0 = [max(len(times) / t_max, 1e-4), 0.05, 1.0]
        res = minimize(
            _spatiotemporal_hawkes_nll,
            x0,
            args=(times, spatial_w, t_max),
            bounds=[(1e-5, None), (0.0, 5.0), (0.01, None)],
            method="L-BFGS-B",
        )
        mu, alpha, beta = res.x if res.success else x0
        ll_hawkes = -_spatiotemporal_hawkes_nll([mu, alpha, beta], times, spatial_w, t_max)
        junc_times = ((sub["start_datetime"] - t0).dt.total_seconds() / 60.0).values.astype(float)
        mu_poisson = max(len(junc_times) / t_max, 1e-5)
        ll_poisson = _poisson_loglik(mu_poisson, junc_times, t_max)
        lr_stat = max(0.0, 2.0 * (ll_hawkes - ll_poisson))
        lr_pvalue = float(1.0 - chi2.cdf(lr_stat, df=2))
        rows.append({
            "junction": junction,
            "mu_baseline": float(mu),
            "alpha_excitation": float(alpha),
            "beta_decay": float(beta),
            "sigma_km": float(sigma),
            "cascade_risk": float(alpha / (beta + 1e-6)),
            "loglik_poisson": float(ll_poisson),
            "loglik_hawkes": float(ll_hawkes),
            "lr_statistic": float(lr_stat),
            "lr_pvalue": lr_pvalue,
            "hawkes_preferred": lr_pvalue < P_THRESHOLD and alpha > 0,
            "n_events": int(len(sub)),
            "n_spatial_contributors": int(len(times)),
        })
    return pd.DataFrame(rows).sort_values("cascade_risk", ascending=False)


def compute_hotspot_persistence(df: pd.DataFrame) -> pd.DataFrame:
    geo = df[df["geo_valid"] & df["junction"].notna()].copy()
    geo["week"] = geo["start_local"].dt.to_period("W").astype(str)
    weeks = sorted(geo["week"].unique())
    sig_counts: dict[str, int] = {}
    gi_sums: dict[str, float] = {}
    for week in weeks:
        junc = geo[geo["week"] == week].groupby("junction").agg(
            x=("trust_score", "sum"), latitude=("latitude", "mean"), longitude=("longitude", "mean")
        ).reset_index()
        if len(junc) < 10:
            continue
        w = KNN.from_array(junc[["longitude", "latitude"]].values, k=min(K_NEIGHBORS, len(junc) - 1))
        w.transform = "r"
        g = G_Local(junc["x"].values, w, star=True, permutations=199, seed=42)
        for i, row in junc.iterrows():
            gi_pos = max(0.0, float(g.Zs[i]))
            gi_sums[row["junction"]] = gi_sums.get(row["junction"], 0.0) + gi_pos
            if g.p_sim[i] < P_THRESHOLD:
                sig_counts[row["junction"]] = sig_counts.get(row["junction"], 0) + 1
    n = len(weeks)
    all_j = set(sig_counts) | set(gi_sums)
    whpi_vals = [gi_sums.get(j, 0.0) / n for j in all_j]
    q40 = np.quantile(whpi_vals, 0.40) if whpi_vals else 0.0
    q75 = np.quantile(whpi_vals, 0.75) if whpi_vals else 0.0
    rows = []
    for j in all_j:
        hpi = sig_counts.get(j, 0) / n
        whpi = gi_sums.get(j, 0.0) / n
        rows.append({
            "junction": j,
            "hotspot_persistence_index": hpi,
            "weighted_hotspot_persistence_index": whpi,
            "significant_weeks": sig_counts.get(j, 0),
            "total_weeks": n,
            "persistence_class": "chronic" if whpi >= q75 else
            "recurring" if whpi >= q40 else "transient",
        })
    return pd.DataFrame(rows).sort_values("weighted_hotspot_persistence_index", ascending=False)


def bootstrap_hotspot_probability(df: pd.DataFrame, n_boot: int = HOTSPOT_BOOTSTRAP) -> pd.DataFrame:
    geo = df[df["geo_valid"] & df["junction"].notna()]
    if len(geo) < 50:
        return pd.DataFrame()
    geo_lat = geo["latitude"].to_numpy()
    geo_lon = geo["longitude"].to_numpy()
    geo_trust = geo["trust_score"].to_numpy()
    geo_junc = geo["junction"].to_numpy()
    hot_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    rng = np.random.default_rng(42)
    n = len(geo)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pd.DataFrame({
            "junction": geo_junc[idx],
            "latitude": geo_lat[idx],
            "longitude": geo_lon[idx],
            "trust_score": geo_trust[idx],
        })
        junctions = sample.groupby("junction", as_index=False).agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            weighted_intensity=("trust_score", "sum"),
        )
        if len(junctions) < 15:
            continue
        g = compute_getis_ord(junctions, permutations=BOOTSTRAP_PERMUTATIONS)
        sig = (g["z_score"].values >= GI_STAR_THRESHOLD) | (g["p_sim"].values < P_THRESHOLD)
        for j, is_hot in zip(g["junction"].values, sig):
            total_counts[j] = total_counts.get(j, 0) + 1
            if is_hot:
                hot_counts[j] = hot_counts.get(j, 0) + 1
        del sample, junctions, g
        if b % 10 == 0:
            gc.collect()
    rows = []
    for j, tot in total_counts.items():
        rows.append({
            "junction": j,
            "hotspot_probability": hot_counts.get(j, 0) / tot,
            "bootstrap_samples": tot,
        })
    return pd.DataFrame(rows).sort_values("hotspot_probability", ascending=False)


def predict_future_hotspot_risk(df: pd.DataFrame, severity: pd.DataFrame, hawkes: pd.DataFrame) -> pd.DataFrame:
    if not HAS_NX:
        out = severity[["junction", "severity_hotspot_score"]].rename(
            columns={"severity_hotspot_score": "future_risk_score"})
        out["percentile_rank"] = out["future_risk_score"].rank(pct=True) * 100
        return out
    G = build_junction_graph(df)
    if G is None:
        return severity[["junction", "severity_hotspot_score"]].rename(
            columns={"severity_hotspot_score": "future_risk_score"})
    geo = df[df["geo_valid"] & df["junction"].notna()].copy()
    geo["week"] = geo["start_local"].dt.to_period("W")
    weeks = sorted(geo["week"].unique())
    if len(weeks) < 6:
        out = severity[["junction", "severity_hotspot_score"]].rename(
            columns={"severity_hotspot_score": "future_risk_score"})
        out["percentile_rank"] = out["future_risk_score"].rank(pct=True) * 100
        return out
    sev = severity.set_index("junction")["severity_hotspot_score"].to_dict()
    hk = hawkes.set_index("junction")["cascade_risk"].to_dict() if not hawkes.empty else {}
    X, y = [], []
    for i in range(len(weeks) - 3):
        nxt = geo[geo["week"] == weeks[i + 1]].groupby("junction")["trust_score"].sum()
        if nxt.empty:
            continue
        thr = nxt.quantile(0.75)
        for j in G.nodes:
            X.append([sev.get(j, 0), hk.get(j, 0), G.degree(j)])
            y.append(1 if nxt.get(j, 0) >= thr else 0)
    if len(X) < 30:
        out = severity[["junction", "severity_hotspot_score"]].rename(
            columns={"severity_hotspot_score": "future_risk_score"})
        out["percentile_rank"] = out["future_risk_score"].rank(pct=True) * 100
        return out
    model = (xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
             if HAS_XGB else GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
    model.fit(np.array(X), np.array(y))
    nodes = list(G.nodes)
    Xa = np.array([[sev.get(j, 0), hk.get(j, 0), G.degree(j)] for j in nodes])
    risk = model.predict_proba(Xa)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xa).astype(float)
    out = pd.DataFrame({"junction": nodes, "future_risk_score": risk})
    out["percentile_rank"] = out["future_risk_score"].rank(pct=True) * 100
    return out.sort_values("future_risk_score", ascending=False)


def _load_junction_frailty(df: pd.DataFrame) -> pd.DataFrame:
    path = OUT_DIR / "layer1_frailty_scores.csv"
    if not path.exists():
        return pd.DataFrame()
    frailty = pd.read_csv(path)[["corridor", "frailty_effect"]]
    geo = df[df["junction"].notna() & df["corridor"].notna()][["junction", "corridor"]]
    junc_corr = geo.groupby("junction")["corridor"].agg(lambda s: s.mode().iloc[0]).reset_index()
    merged = junc_corr.merge(frailty, on="corridor", how="left")
    merged["frailty_difficulty"] = 1.0 / merged["frailty_effect"].clip(0.1, 10.0)
    return merged[["junction", "frailty_difficulty"]]


def compute_obi(severity, persistence, hawkes, future_risk, df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    obi = severity[["junction", "severity_hotspot_score"]].rename(columns={"severity_hotspot_score": "severity_raw"})

    def merge_norm(left, right, col, name):
        if right.empty or col not in right.columns:
            left[name] = 0.5
            return left
        m = right[["junction", col]].drop_duplicates("junction")
        left = left.merge(m, on="junction", how="left")
        v = left[col].fillna(left[col].median()).values.reshape(-1, 1)
        left[name] = scaler.fit_transform(v).flatten()
        return left.drop(columns=[col], errors="ignore")

    obi = merge_norm(obi, persistence, "weighted_hotspot_persistence_index", "persistence_norm")
    if obi["persistence_norm"].nunique() <= 1:
        obi = merge_norm(obi, persistence, "hotspot_persistence_index", "persistence_norm")
    obi = merge_norm(obi, hawkes, "cascade_risk", "hawkes_norm")
    obi = merge_norm(obi, future_risk, "future_risk_score", "duration_risk_norm")
    obi["severity_norm"] = scaler.fit_transform(obi["severity_raw"].values.reshape(-1, 1)).flatten()
    frailty = _load_junction_frailty(df)
    obi = merge_norm(obi, frailty, "frailty_difficulty", "frailty_norm")

    metric_cols = ["hawkes_norm", "severity_norm", "persistence_norm", "duration_risk_norm", "frailty_norm"]
    mat = obi[metric_cols].values
    w_entropy = _entropy_weights(mat)
    w_equal = _equal_weights(len(metric_cols))
    w_pca = _pca_weights(mat)
    obi_entropy = mat @ w_entropy
    obi_equal = mat @ w_equal
    obi_pca = mat @ w_pca
    obi["obi_entropy"] = obi_entropy
    obi["obi_equal"] = obi_equal
    obi["obi_pca"] = obi_pca
    obi["operational_burden_index"] = (obi_entropy + obi_equal + obi_pca) / 3.0
    for i, col in enumerate(metric_cols):
        obi[f"entropy_weight_{col.replace('_norm', '')}"] = w_entropy[i]
        obi[f"pca_weight_{col.replace('_norm', '')}"] = w_pca[i]
    return obi.sort_values("operational_burden_index", ascending=False)


def run_layer2() -> pd.DataFrame:
    print("=== Layer 2: Hotspots (baseline + advanced) ===\n")
    df = load_data()

    print("--- Baseline: trust-weighted Getis-Ord Gi* ---")
    junctions = build_junction_table(df)
    baseline = compute_getis_ord(junctions)
    hotspot_prob = bootstrap_hotspot_probability(df)
    if not hotspot_prob.empty:
        baseline = baseline.merge(hotspot_prob, on="junction", how="left")
        baseline["hotspot_probability"] = baseline["hotspot_probability"].fillna(0)
    print(f"Significant (p_sim<{P_THRESHOLD}): {baseline['is_significant'].sum()} / {len(baseline)}")
    print(baseline[["junction", "weighted_intensity", "z_score", "p_sim", "is_significant"]].head(10).to_string(index=False))
    baseline.to_csv(OUT_DIR / "layer2_hotspots.csv", index=False)
    if not hotspot_prob.empty:
        hotspot_prob.to_csv(OUT_DIR / "layer2_hotspot_probability.csv", index=False)

    print("\n--- Advanced: severity, spatiotemporal, network-kernel, Hawkes, persistence, OBI ---")
    severity = compute_severity_hotspots(df)
    severity.to_csv(OUT_DIR / "layer2_severity_hotspots.csv", index=False)

    st = compute_spatiotemporal_hotspots(df)
    st.to_csv(OUT_DIR / "layer2_spatiotemporal_hotspots.csv", index=False)
    print(f"Spatiotemporal significant pairs: {st['is_significant'].sum() if not st.empty else 0}")

    network = compute_network_hotspots(df, severity)
    if not network.empty:
        network.to_csv(OUT_DIR / "layer2_network_hotspots.csv", index=False)
        print(f"Network kernel Gi* significant: {network['is_significant'].sum()} (h={network['kernel_bandwidth_h'].iloc[0]:.2f})")

    hawkes = fit_spatiotemporal_hawkes(df)
    hawkes.to_csv(OUT_DIR / "layer2_hawkes_cascade_risk.csv", index=False)
    n_pref = int(hawkes["hawkes_preferred"].sum()) if "hawkes_preferred" in hawkes.columns else 0
    print(f"Spatio-temporal Hawkes fitted for {len(hawkes)} junctions ({n_pref} prefer Hawkes over Poisson, p<{P_THRESHOLD})")

    persistence = compute_hotspot_persistence(df)
    if not persistence.empty:
        persistence.to_csv(OUT_DIR / "layer2_hotspot_persistence.csv", index=False)

    future = predict_future_hotspot_risk(df, severity, hawkes)
    future.to_csv(OUT_DIR / "layer2_future_hotspot_risk.csv", index=False)

    obi = compute_obi(severity, persistence, hawkes, future, df)
    obi.to_csv(OUT_DIR / "layer2_operational_burden_index.csv", index=False)
    obi.head(25).to_csv(OUT_DIR / "layer2_operational_burden_top25.csv", index=False)
    w = obi.filter(like="entropy_weight_").iloc[0].to_dict()
    print(f"Ensemble OBI (entropy/equal/PCA mean). Entropy weights: {w}")
    print(f"OBI top junction: {obi.iloc[0]['junction']} ({obi.iloc[0]['operational_burden_index']:.3f})")

    print(f"\nAll Layer 2 outputs → {OUT_DIR}/layer2_*")
    return baseline


if __name__ == "__main__":
    run_layer2()
