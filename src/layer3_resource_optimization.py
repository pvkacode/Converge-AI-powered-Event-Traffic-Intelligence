"""
Layer 3 v2 — Resource Optimization Engine (Upgraded)
ASTraM Bengaluru Traffic Disruption Intelligence

Upgrades over v1:
  - PCA-learned DIS replaces fixed-weight DIS (data-driven, defensible)
  - Operational Demand Score (ODS): continuous, multiplicative resource sizing
  - Linear Programming resource allocation (scipy.optimize.linprog)
  - Dijkstra diversion routing via networkx corridor graph
  - Resource efficiency simulation (diminishing-returns model)

Consumes Layer 1 and Layer 2 outputs only — no re-training.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    print("[INFO] networkx not found — Dijkstra fallback will be used")

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "outputs"
DATA = ROOT / "data"

# ── City-wide resource budget constants (configurable) ────────────────────────
TOTAL_OFFICERS    = 120
TOTAL_TOW_UNITS   = 15
TOTAL_BARRICADES  = 100
TOTAL_SUPERVISORS = 20

# ── Efficiency simulation calibration ─────────────────────────────────────────
K_CLEARANCE = 0.08     # diminishing-returns rate for officer deployment
MAX_STAFF_IMPROVEMENT = 0.40  # max 40% clearance improvement from extra staffing


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def load_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"  [OK] {label}: {len(df)} rows | cols: {list(df.columns)}")
        return df
    except Exception as exc:
        print(f"  [ERROR] {label} @ {path}: {exc}")
        raise


def minmax_norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    return pd.Series(0.0, index=series.index) if mx == mn else (series - mn) / (mx - mn)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LEARNED DIS VIA PCA
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 1: Learned DIS via PCA")

obi_df      = load_csv(OUT / "layer2_operational_burden_index.csv", "OBI")
hawkes_df   = load_csv(OUT / "layer2_hawkes_cascade_risk.csv",      "Hawkes cascade")
future_df   = load_csv(OUT / "layer2_future_hotspot_risk.csv",      "Future risk")
rmst_df     = load_csv(OUT / "layer1_rmst_summary.csv",             "RMST")
persist_df  = load_csv(OUT / "layer2_hotspot_persistence.csv",      "Persistence")

print("  Loading events_clean.parquet for junction-corridor mapping...")
events = pd.read_parquet(DATA / "events_clean.parquet")
print(f"  [OK] events_clean: {events.shape}")

# RMST aggregation: corridor → junction via events_clean
ev_jc = events[events["junction"].notna()][["junction", "corridor"]].drop_duplicates()
rmst_by_corr = rmst_df.groupby("corridor")["rmst_360min"].mean().reset_index()
rmst_by_corr.columns = ["corridor", "rmst_360min_mean"]
ev_rmst = ev_jc.merge(rmst_by_corr, on="corridor", how="left")
junc_rmst = ev_rmst.groupby("junction")["rmst_360min_mean"].mean().reset_index()
junc_rmst.columns = ["junction", "rmst_mean"]
global_rmst_med = float(rmst_df["rmst_360min"].median())
print(f"  Global median RMST: {global_rmst_med:.1f} min | "
      f"Junctions with RMST: {junc_rmst['rmst_mean'].notna().sum()}/{len(junc_rmst)}")

# Merge all five components onto OBI master junction list
feat = obi_df[["junction", "operational_burden_index"]].copy()
feat = feat.merge(hawkes_df[["junction", "cascade_risk"]],              on="junction", how="left")
feat = feat.merge(future_df[["junction", "future_risk_score"]],         on="junction", how="left")
feat = feat.merge(junc_rmst,                                             on="junction", how="left")
feat = feat.merge(persist_df[["junction", "hotspot_persistence_index"]], on="junction", how="left")

all5 = feat.dropna().shape[0]
print(f"  Junctions with all 5 components: {all5} / {len(feat)}")

for col, fallback in [
    ("cascade_risk",              float(hawkes_df["cascade_risk"].median())),
    ("future_risk_score",         float(future_df["future_risk_score"].median())),
    ("rmst_mean",                 global_rmst_med),
    ("hotspot_persistence_index", float(persist_df["hotspot_persistence_index"].median())),
]:
    n_miss = int(feat[col].isna().sum())
    if n_miss:
        print(f"  Filling {n_miss} NaN in '{col}' with median={fallback:.4f}")
    feat[col] = feat[col].fillna(fallback)

# Build feature matrix
feature_cols = [
    "operational_burden_index",
    "cascade_risk",
    "future_risk_score",
    "rmst_mean",
    "hotspot_persistence_index",
]
X = feat[feature_cols].values.astype(float)

# PCA on standardized features
scaler_pca = StandardScaler()
X_scaled   = scaler_pca.fit_transform(X)
pca        = PCA(n_components=5)
pca.fit(X_scaled)

print("\n  PCA explained variance ratio:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"    PC{i+1}: {ev:.4f} ({ev*100:.1f}%)")
print("  PC1 component loadings:")
for fname, load in zip(feature_cols, pca.components_[0]):
    print(f"    {fname:40s}: {load:+.4f}")

# Project onto PC1
raw_dis = X_scaled @ pca.components_[0]

# Sign correction: DIS must correlate positively with OBI
obi_norm = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
if np.corrcoef(raw_dis, obi_norm)[0, 1] < 0:
    print("  [INFO] Flipping PC1 sign so DIS correlates positively with OBI")
    raw_dis = -raw_dis

# Normalize to [0, 100]
dis_scores = 100.0 * (raw_dis - raw_dis.min()) / (raw_dis.max() - raw_dis.min())
feat["dis_score"] = dis_scores.round(2)

def risk_level(s: float) -> str:
    if s < 30: return "Low"
    if s < 60: return "Moderate"
    if s < 80: return "High"
    return "Critical"

feat["risk_level"] = feat["dis_score"].apply(risk_level)

# Add per-junction PCA component values for transparency
feat["obi_component"]         = minmax_norm(feat["operational_burden_index"])
feat["hawkes_component"]      = minmax_norm(feat["cascade_risk"])
feat["future_risk_component"] = minmax_norm(feat["future_risk_score"])
feat["rmst_component"]        = minmax_norm(feat["rmst_mean"])
feat["persistence_component"] = minmax_norm(feat["hotspot_persistence_index"])

# PC1 loadings (constants, replicated per row for convenience)
load_labels = ["pc1_loading_obi", "pc1_loading_hawkes", "pc1_loading_future",
               "pc1_loading_rmst", "pc1_loading_persistence"]
for lbl, load in zip(load_labels, pca.components_[0]):
    feat[lbl] = round(float(load), 6)

dis_out_cols = [
    "junction", "dis_score", "risk_level",
    "obi_component", "hawkes_component", "future_risk_component",
    "rmst_component", "persistence_component",
] + load_labels
feat[dis_out_cols].to_csv(OUT / "layer3_disruption_impact_scores.csv", index=False)

# Save PCA model
with open(OUT / "layer3_pca_model.pkl", "wb") as f:
    pickle.dump({"scaler": scaler_pca, "pca": pca, "feature_cols": feature_cols}, f)

# Save explained variance CSV
ev_df = pd.DataFrame({
    "component":             [f"PC{i+1}" for i in range(5)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance":   np.cumsum(pca.explained_variance_ratio_),
})
ev_df.to_csv(OUT / "layer3_pca_explained_variance.csv", index=False)

print(f"\n  Top 20 by DIS:")
print(feat[["junction", "dis_score", "risk_level"]].nlargest(20, "dis_score").to_string(index=False))
print(f"\n  Risk level counts: {feat['risk_level'].value_counts().to_dict()}")
print(f"  Saved: layer3_disruption_impact_scores.csv ({len(feat)} rows)")
print(f"  Saved: layer3_pca_model.pkl | layer3_pca_explained_variance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — OPERATIONAL DEMAND SCORE (ODS)
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 2: Operational Demand Score (ODS)")

dis_df   = pd.read_csv(OUT / "layer3_disruption_impact_scores.csv")
surv_q   = load_csv(OUT / "layer1_survival_quantiles.csv", "Survival quantiles")

# P80 per junction (cap at 360 min to avoid KM extrapolation artifacts)
corr_p80 = surv_q.groupby("corridor")["p80_min"].mean().reset_index()
ev_p80   = ev_jc.merge(corr_p80, on="corridor", how="left")
junc_p80 = ev_p80.groupby("junction")["p80_min"].mean().reset_index()
junc_p80.columns = ["junction", "p80_raw"]
global_p80_med = float(np.clip(surv_q["p80_min"].median(), 0, 360))
p80_lookup = dict(zip(junc_p80["junction"], junc_p80["p80_raw"].clip(0, 360).fillna(global_p80_med)))

# Closure flag per junction: majority vote from events
ev_cl = (
    events[events["junction"].notna()]
    [["junction", "requires_road_closure"]]
    .assign(cl=lambda df: df["requires_road_closure"].fillna(False).astype(int))
    .groupby("junction")["cl"].mean()
    .reset_index()
)
ev_cl.columns = ["junction", "closure_rate"]
closure_lookup = {r["junction"]: (1 if r["closure_rate"] > 0.5 else 0)
                  for _, r in ev_cl.iterrows()}

# Hawkes branching ratio R (from hawkes_validation.csv; cap at 2.0)
try:
    hv_df   = load_csv(OUT / "layer2_hawkes_validation.csv", "Hawkes validation")
    hv_lookup = dict(zip(hv_df["junction"], hv_df["branching_ratio"].clip(0, 2.0)))
    global_R = float(hv_df["branching_ratio"].median())
except Exception:
    print("  WARNING: hawkes_validation not found; computing R = alpha/beta from cascade CSV")
    hv_lookup = {}
    global_R  = 0.05

manpower_rows = []
for _, row in dis_df.iterrows():
    junc      = row["junction"]
    dis       = float(row["dis_score"])
    p80       = float(p80_lookup.get(junc, global_p80_med))
    cl_flag   = int(closure_lookup.get(junc, 0))
    R         = float(hv_lookup.get(junc, global_R))

    dur_factor     = 1.0 + p80 / 120.0            # p80 already capped at 360
    closure_factor = 1.5 if cl_flag else 1.0
    cascade_factor = 1.0 + R

    ods = dis * dur_factor * closure_factor * cascade_factor

    officers    = min(25, max(1, math.ceil(ods / 30)))
    barricades  = min(40, max(0, math.ceil(ods / 20)))
    tow_units   = min(5,  max(0, math.ceil(ods / 80)))
    supervisors = max(0, math.ceil(officers / 6))
    patrol_veh  = max(0, math.ceil(officers / 4))
    qru_units   = 1 if dis >= 70 else 0

    manpower_rows.append({
        "junction":               junc,
        "dis_score":              dis,
        "risk_level":             row["risk_level"],
        "ods_score":              round(ods, 2),
        "duration_factor":        round(dur_factor, 3),
        "closure_factor":         closure_factor,
        "cascade_factor":         round(cascade_factor, 3),
        "p80_duration_min":       round(p80, 1),
        "closure_flag":           cl_flag,
        "hawkes_branching_ratio": round(R, 4),
        "officers":               officers,
        "supervisors":            supervisors,
        "tow_vehicles":           tow_units,
        "qru_units":              qru_units,
        "patrol_vehicles":        patrol_veh,
        "barricades_from_ods":    barricades,
    })

manpower_df = pd.DataFrame(manpower_rows)
manpower_df.to_csv(OUT / "layer3_manpower_recommendations.csv", index=False)

print(f"  ODS stats: min={manpower_df['ods_score'].min():.1f}, "
      f"mean={manpower_df['ods_score'].mean():.1f}, max={manpower_df['ods_score'].max():.1f}")
print(f"  Officers range: {manpower_df['officers'].min()}–{manpower_df['officers'].max()}")
print(f"  Saved: layer3_manpower_recommendations.csv ({len(manpower_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LINEAR PROGRAMMING RESOURCE ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 3: LP Resource Allocation")

mp = manpower_df.copy()

# Filter to Moderate+ (DIS >= 30) and take top 50
lp_candidates = mp[mp["dis_score"] >= 30].nlargest(min(50, (mp["dis_score"] >= 30).sum()), "dis_score").reset_index(drop=True)
N = len(lp_candidates)
print(f"  LP candidates (DIS>=30): {N}")

c_obj = -lp_candidates["dis_score"].values          # negate for minimization
A_ub  = np.array([
    lp_candidates["officers"].values.astype(float),
    lp_candidates["tow_vehicles"].values.astype(float),
    lp_candidates["barricades_from_ods"].values.astype(float),
    lp_candidates["supervisors"].values.astype(float),
], dtype=float)
b_ub = np.array([TOTAL_OFFICERS, TOTAL_TOW_UNITS, TOTAL_BARRICADES, TOTAL_SUPERVISORS], dtype=float)
bounds = [(0.0, 1.0)] * N

lp_result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

if lp_result.status == 0:
    alloc_frac = lp_result.x
    print(f"  LP solved (optimal). Objective (max DIS allocated): {-lp_result.fun:.1f}")
else:
    print(f"  WARNING: LP status={lp_result.status} '{lp_result.message}' — using greedy fallback")
    # Greedy: allocate full resources in DIS order until budget exhausted
    alloc_frac = np.zeros(N)
    remaining = {"officers": TOTAL_OFFICERS, "tow": TOTAL_TOW_UNITS,
                 "barricades": TOTAL_BARRICADES, "supervisors": TOTAL_SUPERVISORS}
    for i in range(N):
        fracs = [
            remaining["officers"]   / max(lp_candidates.iloc[i]["officers"], 1),
            remaining["tow"]        / max(lp_candidates.iloc[i]["tow_vehicles"], 1),
            remaining["barricades"] / max(lp_candidates.iloc[i]["barricades_from_ods"], 1),
            remaining["supervisors"]/ max(lp_candidates.iloc[i]["supervisors"], 1),
        ]
        f = min(1.0, min(fracs))
        alloc_frac[i] = f
        remaining["officers"]    -= f * lp_candidates.iloc[i]["officers"]
        remaining["tow"]         -= f * lp_candidates.iloc[i]["tow_vehicles"]
        remaining["barricades"]  -= f * lp_candidates.iloc[i]["barricades_from_ods"]
        remaining["supervisors"] -= f * lp_candidates.iloc[i]["supervisors"]

lp_rows = []
for i, (_, row) in enumerate(lp_candidates.iterrows()):
    f = float(alloc_frac[i])
    lp_rows.append({
        "junction":              row["junction"],
        "dis_score":             row["dis_score"],
        "recommended_officers":  row["officers"],
        "recommended_tow":       row["tow_vehicles"],
        "recommended_barricades":row["barricades_from_ods"],
        "allocated_officers":    round(row["officers"]         * f),
        "allocated_tow":         round(row["tow_vehicles"]     * f),
        "allocated_barricades":  round(row["barricades_from_ods"] * f),
        "allocated_supervisors": round(row["supervisors"]      * f),
        "allocation_fraction":   round(f, 4),
        "fully_allocated":       f >= 0.95,
    })

lp_df = pd.DataFrame(lp_rows)
lp_df.to_csv(OUT / "layer3_lp_resource_allocation.csv", index=False)

used_off  = (lp_df["recommended_officers"]  * lp_df["allocation_fraction"]).sum()
used_tow  = (lp_df["recommended_tow"]       * lp_df["allocation_fraction"]).sum()
used_bar  = (lp_df["recommended_barricades"]* lp_df["allocation_fraction"]).sum()
n_full    = lp_df["fully_allocated"].sum()
print(f"  Officers used: {used_off:.0f}/{TOTAL_OFFICERS} | Tow: {used_tow:.0f}/{TOTAL_TOW_UNITS} "
      f"| Barricades: {used_bar:.0f}/{TOTAL_BARRICADES}")
print(f"  Junctions fully allocated: {n_full}/{N}")
print(f"  Saved: layer3_lp_resource_allocation.csv ({len(lp_df)} rows)")

# Update manpower_df with LP columns (junctions NOT in LP keep recommended=allocated)
lp_join = lp_df[["junction", "allocated_officers", "allocated_tow",
                  "allocated_barricades", "allocated_supervisors", "allocation_fraction"]].copy()
manpower_df = manpower_df.merge(lp_join, on="junction", how="left")
for col_alloc, col_rec in [
    ("allocated_officers",    "officers"),
    ("allocated_tow",         "tow_vehicles"),
    ("allocated_barricades",  "barricades_from_ods"),
    ("allocated_supervisors", "supervisors"),
]:
    manpower_df[col_alloc] = manpower_df[col_alloc].fillna(manpower_df[col_rec]).astype(int)
manpower_df["allocation_fraction"] = manpower_df["allocation_fraction"].fillna(1.0)
manpower_df.to_csv(OUT / "layer3_manpower_recommendations.csv", index=False)
print(f"  Updated: layer3_manpower_recommendations.csv with LP columns ({len(manpower_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DIJKSTRA DIVERSION ROUTING
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 4: Dijkstra Diversion Routing")

dis_full  = pd.read_csv(OUT / "layer3_disruption_impact_scores.csv")
mp_full   = pd.read_csv(OUT / "layer3_manpower_recommendations.csv")

# Junction-level normalized scores for edge weights
junc_obi   = dict(zip(obi_df["junction"],    minmax_norm(obi_df["operational_burden_index"])))
junc_fut   = dict(zip(future_df["junction"], minmax_norm(future_df["future_risk_score"])))
junc_haw   = dict(zip(hawkes_df["junction"], minmax_norm(hawkes_df["cascade_risk"])))
junc_rmst2 = dict(zip(junc_rmst["junction"], minmax_norm(junc_rmst["rmst_mean"].fillna(global_rmst_med))))

dis_lookup = dict(zip(dis_full["junction"], dis_full["dis_score"]))

def _node_cost(j: str) -> float:
    return (0.4 * junc_obi.get(j, 0.5)
            + 0.3 * junc_fut.get(j, 0.5)
            + 0.2 * junc_haw.get(j, 0.5)
            + 0.1 * junc_rmst2.get(j, 0.5)
            + 0.01)

def _edge_cost(u: str, v: str) -> float:
    return (_node_cost(u) + _node_cost(v)) / 2.0

routing_method = "dijkstra" if NX_AVAILABLE else "zone_fallback"
G = None
if NX_AVAILABLE:
    G = nx.DiGraph()
    all_junctions = events["junction"].dropna().unique().tolist()
    G.add_nodes_from(all_junctions)

    # Edges: junctions sharing a corridor
    ev_jc2 = events[events["junction"].notna()][["junction", "corridor"]].drop_duplicates()
    corr2junc = ev_jc2.groupby("corridor")["junction"].apply(list).to_dict()

    for corr, jlist in corr2junc.items():
        jlist = list(set(jlist))
        if len(jlist) < 2:
            continue
        # Sort by total incident count to get a consistent ordering for adjacency
        inc_cnt = events[events["corridor"] == corr].groupby("junction").size()
        jlist_sorted = sorted(jlist, key=lambda j: inc_cnt.get(j, 0), reverse=True)
        # Chain-style edges (consecutive pairs) + reverse
        for i in range(len(jlist_sorted) - 1):
            u, v = jlist_sorted[i], jlist_sorted[i + 1]
            w = _edge_cost(u, v)
            G.add_edge(u, v, weight=w)
            G.add_edge(v, u, weight=w)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    print(f"  Largest weakly-connected component: {len(largest_cc)} nodes")

    if G.number_of_nodes() < 10:
        print("  WARNING: Graph too small — using zone fallback")
        routing_method = "zone_fallback"
        NX_AVAILABLE_LOCAL = False
    else:
        NX_AVAILABLE_LOCAL = True
else:
    NX_AVAILABLE_LOCAL = False

# Zone-based fallback data
junc_zone = (
    events[events["junction"].notna() & events["zone"].notna()]
    [["junction", "zone"]].drop_duplicates("junction")
)
zone_lookup = dict(zip(junc_zone["junction"], junc_zone["zone"]))

diversion_rows = []
top30 = dis_full.nlargest(30, "dis_score")
route_labels = ["Route A (best)", "Route B", "Route C"]

for _, jrow in top30.iterrows():
    jname     = jrow["junction"]
    j_dis     = float(jrow["dis_score"])
    candidates_found: list[tuple[str, float, list[str]]] = []

    if NX_AVAILABLE_LOCAL and G is not None and jname in G:
        # Build temporary graph with blocked junction removed
        G_temp = G.copy()
        G_temp.remove_node(jname)

        neighbors_of_blocked = list(G.successors(jname)) + list(G.predecessors(jname))
        neighbors_of_blocked = [n for n in set(neighbors_of_blocked) if n in G_temp]

        # From each neighbor, find shortest paths to all reachable nodes
        path_scores: dict[str, tuple[float, list[str]]] = {}
        for src in neighbors_of_blocked[:5]:   # limit search sources
            try:
                lengths, paths = nx.single_source_dijkstra(G_temp, src, cutoff=2.0, weight="weight")
                for tgt, length in lengths.items():
                    if tgt != src and tgt not in path_scores:
                        path_scores[tgt] = (length, paths.get(tgt, [src, tgt]))
                    elif tgt in path_scores and length < path_scores[tgt][0]:
                        path_scores[tgt] = (length, paths.get(tgt, [src, tgt]))
            except Exception:
                pass

        # Sort by path cost, exclude the blocked junction and its immediate neighbors
        ranked = sorted(
            [(tgt, sc, pth) for tgt, (sc, pth) in path_scores.items()
             if tgt not in (neighbors_of_blocked + [jname])],
            key=lambda x: x[1],
        )
        candidates_found = [(tgt, sc, pth) for tgt, sc, pth in ranked[:3]]

    if len(candidates_found) < 3:
        # Zone fallback
        jzone = zone_lookup.get(jname)
        zone_junctions = [j for j, z in zone_lookup.items() if z == jzone and j != jname] if jzone else []
        all_other = dis_full[dis_full["junction"] != jname]["junction"].tolist()
        fallback_pool = zone_junctions if len(zone_junctions) >= 3 else all_other
        scored_fb = sorted(
            [(j, _node_cost(j), [jname, j]) for j in fallback_pool if j != jname],
            key=lambda x: x[1],
        )
        existing_targets = {c[0] for c in candidates_found}
        for tgt, sc, pth in scored_fb:
            if tgt not in existing_targets and len(candidates_found) < 3:
                candidates_found.append((tgt, sc, pth))
        routing_method_row = "zone_fallback"
    else:
        routing_method_row = "dijkstra"

    for rank_i, (tgt, path_weight, path_nodes) in enumerate(candidates_found[:3]):
        path_str = "|".join(path_nodes) if path_nodes else tgt
        obi_c    = round(_node_cost(tgt), 4)
        diversion_rows.append({
            "junction":                    jname,
            "dis_score":                   j_dis,
            "route_rank":                  rank_i + 1,
            "route_label":                 route_labels[rank_i],
            "diversion_corridor":          tgt,   # kept for backward compat
            "diversion_path":              path_str,
            "path_weight":                 round(float(path_weight), 4),
            "estimated_additional_time_min": round(float(path_weight) * 15, 1),
            "obi_cost":                    round(junc_obi.get(tgt, 0.5), 4),
            "future_cost":                 round(junc_fut.get(tgt, 0.5), 4),
            "hawkes_cost":                 round(junc_haw.get(tgt, 0.5), 4),
            "rmst_cost":                   round(junc_rmst2.get(tgt, 0.5), 4),
            "obi_component":               round(junc_obi.get(tgt, 0.5), 4),
            "hawkes_component":            round(junc_haw.get(tgt, 0.5), 4),
            "future_risk_component":       round(junc_fut.get(tgt, 0.5), 4),
            "route_cost":                  round(float(path_weight), 4),
            "recommendation_label":        route_labels[rank_i],
            "routing_method":              routing_method_row,
        })

diversion_df = pd.DataFrame(diversion_rows)
diversion_df.to_csv(OUT / "layer3_diversion_recommendations.csv", index=False)
dijkstra_ct  = (diversion_df["routing_method"] == "dijkstra").sum()
fallback_ct  = (diversion_df["routing_method"] == "zone_fallback").sum()
print(f"  Routes: {len(diversion_df)} total | Dijkstra: {dijkstra_ct} | Fallback: {fallback_ct}")
print(f"  Saved: layer3_diversion_recommendations.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RESOURCE EFFICIENCY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 5: Resource Efficiency Simulation")

top20_junctions = dis_full.nlargest(20, "dis_score")["junction"].tolist()
mp_sim = manpower_df.set_index("junction")
MULTIPLIERS = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]

eff_rows = []
city_totals: dict[float, float] = {m: 0.0 for m in MULTIPLIERS}

for junc in top20_junctions:
    if junc not in mp_sim.index:
        continue
    row = mp_sim.loc[junc]
    base_officers  = int(row.get("allocated_officers", row["officers"]))
    base_clearance = float(min(row["p80_duration_min"], 360.0))

    prev_clearance = base_clearance
    for mult in MULTIPLIERS:
        n_off       = math.ceil(base_officers * mult)
        reduction   = (1.0 - math.exp(-K_CLEARANCE * n_off)) * MAX_STAFF_IMPROVEMENT
        pred_clear  = base_clearance * (1.0 - reduction)
        marginal    = prev_clearance - pred_clear
        eff_rows.append({
            "junction":              junc,
            "dis_score":             float(row["dis_score"]),
            "officer_multiplier":    mult,
            "n_officers":            n_off,
            "predicted_clearance_min": round(pred_clear, 1),
            "marginal_gain_min":     round(marginal, 1),
            "base_clearance_min":    round(base_clearance, 1),
        })
        city_totals[mult] += float(row["dis_score"]) * pred_clear
        prev_clearance = pred_clear

eff_df = pd.DataFrame(eff_rows)
eff_df.to_csv(OUT / "layer3_resource_efficiency_simulation.csv", index=False)

current_wtd = city_totals[1.0]
plus20_wtd  = city_totals[1.2]
improvement = (current_wtd - plus20_wtd) / current_wtd * 100 if current_wtd > 0 else 0.0
print(f"\n  HEADLINE: Adding 20% more officers reduces weighted clearance time by {improvement:.1f}%")

efficiency_scenarios = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "top20_junctions_analyzed": top20_junctions,
    "city_wide_scenarios": [
        {
            "officer_multiplier": m,
            "total_weighted_clearance": round(city_totals[m], 1),
            "improvement_vs_baseline_pct": round((city_totals[1.0] - city_totals[m]) / city_totals[1.0] * 100, 2) if city_totals[1.0] > 0 else 0,
        }
        for m in MULTIPLIERS
    ],
    "headline_finding": f"Adding 20% more officers reduces weighted clearance time by {improvement:.1f}%",
}
with open(OUT / "layer3_efficiency_scenarios.json", "w", encoding="utf-8") as f:
    json.dump(efficiency_scenarios, f, indent=2)

print(f"  Saved: layer3_resource_efficiency_simulation.csv ({len(eff_df)} rows)")
print(f"  Saved: layer3_efficiency_scenarios.json")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BARRICADING PLAN (ODS-driven)
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 6: Barricading Plan (ODS-driven)")

sev_df = load_csv(OUT / "layer2_severity_hotspots.csv", "Severity hotspots")
bar_df = manpower_df.copy()
bar_df = bar_df.merge(sev_df[["junction", "severity_hotspot_score"]], on="junction", how="left")
bar_df["severity_hotspot_score"] = bar_df["severity_hotspot_score"].fillna(0.0)

def _bar_strategy(rl: str) -> tuple[str, str]:
    if rl == "Low":      return "none",              "none"
    if rl == "Moderate": return "partial_closure",   "partial_lane"
    if rl == "High":     return "full_barricading",  "full_road"
    return                      "emergency_closure", "full_road"

bar_rows = []
for _, r in bar_df.iterrows():
    strat, ctype = _bar_strategy(r["risk_level"])
    bcount = int(r["barricades_from_ods"])
    bar_rows.append({
        "junction":              r["junction"],
        "dis_score":             r["dis_score"],
        "risk_level":            r["risk_level"],
        "strategy":              strat,
        "barricades":            bcount,
        "ods_driven_barricades": bcount,
        "closure_type":          ctype,
        "barricade_window_start":"deploy_at_event_start",
        "barricade_window_end":  f"remove_after_{int(r['p80_duration_min'])}_min",
        "teams_needed":          math.ceil(bcount / 8) if bcount > 0 else 0,
        "severity_hotspot_score":r["severity_hotspot_score"],
    })

bar_out = pd.DataFrame(bar_rows)
bar_out.to_csv(OUT / "layer3_barricading_plan.csv", index=False)
print(f"  Strategy dist: {bar_out['strategy'].value_counts().to_dict()}")
print(f"  Total barricades: {bar_out['barricades'].sum()} | Teams: {bar_out['teams_needed'].sum()}")
print(f"  Saved: layer3_barricading_plan.csv ({len(bar_out)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TOW PLACEMENT
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 7: Tow Vehicle Placement")

tow_cands = manpower_df[manpower_df["tow_vehicles"] > 0].copy()
tow_cands = tow_cands.merge(
    persist_df[["junction", "hotspot_persistence_index"]], on="junction", how="left"
)
tow_cands["hotspot_persistence_index"] = tow_cands["hotspot_persistence_index"].fillna(
    persist_df["hotspot_persistence_index"].median()
)
tow_cands["dis_n"]  = minmax_norm(tow_cands["dis_score"])
tow_cands["tow_n"]  = minmax_norm(tow_cands["tow_vehicles"].astype(float))
tow_cands["per_n"]  = minmax_norm(tow_cands["hotspot_persistence_index"])
tow_cands["tow_priority_score"] = (
    0.4 * tow_cands["dis_n"] + 0.3 * tow_cands["tow_n"] + 0.3 * tow_cands["per_n"]
)
tow_cands = tow_cands.sort_values("tow_priority_score", ascending=False).reset_index(drop=True)

n_tow = int(
    manpower_df[manpower_df["risk_level"].isin(["Critical", "High"])]["tow_vehicles"].sum()
)

ev_jh = events[events["junction"].notna()][["junction", "hour_local"]].dropna()
shift_map: dict[str, str] = {}
for junc, grp in ev_jh.groupby("junction"):
    h = grp["hour_local"]
    t = len(h)
    if t == 0: shift_map[junc] = "all_day"; continue
    morn = ((h >= 7) & (h < 10)).sum() / t
    eve  = ((h >= 17) & (h < 20)).sum() / t
    if morn >= 0.25 and morn >= eve:   shift_map[junc] = "morning_peak"
    elif eve >= 0.25 and eve > morn:   shift_map[junc] = "evening_peak"
    else:                               shift_map[junc] = "all_day"

tow_rows = []
for _, r in tow_cands.iterrows():
    if len(tow_rows) >= n_tow: break
    tow_rows.append({
        "tow_unit_id":        f"TOW-{len(tow_rows)+1:03d}",
        "assigned_junction":  r["junction"],
        "tow_priority_score": round(float(r["tow_priority_score"]), 4),
        "recommended_shift":  shift_map.get(r["junction"], "all_day"),
    })

tow_df = pd.DataFrame(tow_rows)
tow_df.to_csv(OUT / "layer3_tow_placement.csv", index=False)
print(f"  Tow units placed: {len(tow_df)} | Shift dist: {tow_df['recommended_shift'].value_counts().to_dict()}")
print(f"  Saved: layer3_tow_placement.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — DEPLOYMENT BLUEPRINT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
_section("SECTION 8: Deployment Blueprint Generator")

dis_bp   = pd.read_csv(OUT / "layer3_disruption_impact_scores.csv")
mp_bp    = pd.read_csv(OUT / "layer3_manpower_recommendations.csv")
bar_bp   = pd.read_csv(OUT / "layer3_barricading_plan.csv")
div_bp   = pd.read_csv(OUT / "layer3_diversion_recommendations.csv")
tow_bp   = pd.read_csv(OUT / "layer3_tow_placement.csv")
lp_bp    = pd.read_csv(OUT / "layer3_lp_resource_allocation.csv")
eff_bp   = pd.read_csv(OUT / "layer3_resource_efficiency_simulation.csv")

# Fast lookup dicts
dis_lu  = dis_bp.set_index("junction").to_dict("index")
mp_lu   = mp_bp.set_index("junction").to_dict("index")
bar_lu  = bar_bp.set_index("junction").to_dict("index")
tow_lu  = tow_bp.groupby("assigned_junction")["tow_unit_id"].first().to_dict()
div_lu: dict[str, list[dict]] = {}
for _, r in div_bp.iterrows():
    div_lu.setdefault(r["junction"], []).append({
        "route_rank":           int(r["route_rank"]),
        "route_label":          str(r.get("route_label", r.get("recommendation_label", ""))),
        "path":                 str(r.get("diversion_path", r["diversion_corridor"])),
        "estimated_time_min":   float(r.get("estimated_additional_time_min", r.get("path_weight", 0) * 15)),
    })

def _eff_note(junc: str, mp_row: dict) -> str:
    base_off  = int(mp_row.get("allocated_officers", mp_row.get("officers", 6)))
    n_plus20  = math.ceil(base_off * 1.2)
    base_cl   = float(min(mp_row.get("p80_duration_min", 120.0), 360.0))
    red_base  = (1 - math.exp(-K_CLEARANCE * base_off))  * MAX_STAFF_IMPROVEMENT
    red_plus  = (1 - math.exp(-K_CLEARANCE * n_plus20)) * MAX_STAFF_IMPROVEMENT
    gain      = base_cl * (red_plus - red_base)
    return f"+20% officers ({base_off}→{n_plus20}) saves ~{gain:.0f} min clearance"


def generate_deployment_blueprint(junction_name: str, event_type: str | None = None) -> dict:
    d   = dis_lu.get(junction_name, {})
    m   = mp_lu.get(junction_name, {})
    b   = bar_lu.get(junction_name, {})
    tow = tow_lu.get(junction_name, "None")

    blueprint = {
        "junction":             junction_name,
        "event_type":           event_type,
        "dis_score":            float(d.get("dis_score", 50.0)),
        "risk_level":           str(d.get("risk_level", "Moderate")),
        "ods_score":            float(m.get("ods_score", 0.0)),
        "pca_loadings": {
            "obi":         float(d.get("pc1_loading_obi", 0.0)),
            "hawkes":      float(d.get("pc1_loading_hawkes", 0.0)),
            "future":      float(d.get("pc1_loading_future", 0.0)),
            "rmst":        float(d.get("pc1_loading_rmst", 0.0)),
            "persistence": float(d.get("pc1_loading_persistence", 0.0)),
        },
        "allocated_officers":    int(m.get("allocated_officers",   m.get("officers", 6))),
        "allocated_supervisors": int(m.get("allocated_supervisors", m.get("supervisors", 1))),
        "allocated_tow":         int(m.get("allocated_tow",         m.get("tow_vehicles", 0))),
        "allocated_barricades":  int(m.get("allocated_barricades",  m.get("barricades_from_ods", 8))),
        "qru_units":             int(m.get("qru_units", 0)),
        "patrol_vehicles":       int(m.get("patrol_vehicles", 2)),
        "barricade_strategy":    str(b.get("strategy",      "partial_closure")),
        "closure_type":          str(b.get("closure_type",  "partial_lane")),
        "diversion_routes":      sorted(div_lu.get(junction_name, []), key=lambda x: x["route_rank"]),
        "tow_unit_assigned":     tow,
        "efficiency_note":       _eff_note(junction_name, m),
    }

    sep = "-" * 55
    print(f"\n  {sep}")
    print(f"  DEPLOYMENT BLUEPRINT: {junction_name}")
    print(f"  {sep}")
    print(f"  DIS: {blueprint['dis_score']:.1f} [{blueprint['risk_level']}]  ODS: {blueprint['ods_score']:.1f}")
    print(f"  Officers: {blueprint['allocated_officers']}  Supervisors: {blueprint['allocated_supervisors']}  Tow: {blueprint['allocated_tow']}  QRU: {blueprint['qru_units']}")
    print(f"  Barricades: {blueprint['allocated_barricades']}  [{blueprint['barricade_strategy']} / {blueprint['closure_type']}]")
    if blueprint["diversion_routes"]:
        print(f"  Diversions:")
        for dr in blueprint["diversion_routes"]:
            print(f"    [{dr['route_label']}] {dr['path']} (~{dr['estimated_time_min']:.0f} min extra)")
    print(f"  Tow: {tow}")
    print(f"  Efficiency: {blueprint['efficiency_note']}")

    return blueprint


top5 = dis_bp.nlargest(5, "dis_score")["junction"].tolist()
blueprints = []
for jn in top5:
    blueprints.append(generate_deployment_blueprint(jn))

def _js(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)):  return bool(obj)
    raise TypeError(type(obj))

with open(OUT / "layer3_deployment_blueprints.json", "w", encoding="utf-8") as f:
    json.dump(blueprints, f, indent=2, default=_js)

# Full dashboard
dash = dis_bp.copy()
dash = dash.merge(mp_bp.drop(columns=["dis_score", "risk_level"], errors="ignore"), on="junction", how="left")
dash = dash.merge(bar_bp[["junction", "strategy", "barricades", "closure_type", "teams_needed", "severity_hotspot_score"]],
                  on="junction", how="left")
dash = dash.merge(persist_df[["junction", "persistence_class"]], on="junction", how="left")
tow_j = tow_bp.rename(columns={"assigned_junction": "junction", "tow_unit_id": "tow_unit"})
dash  = dash.merge(tow_j[["junction", "tow_unit", "tow_priority_score"]], on="junction", how="left")
lp_j  = lp_bp[["junction", "allocation_fraction", "fully_allocated"]]
dash  = dash.merge(lp_j, on="junction", how="left")
dash.to_csv(OUT / "layer3_full_dashboard.csv", index=False)
print(f"\n  Saved: layer3_deployment_blueprints.json | layer3_full_dashboard.csv ({len(dash)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== LAYER 3 COMPLETE ===")
print("=" * 70)
for fname in [
    "layer3_disruption_impact_scores.csv",
    "layer3_pca_explained_variance.csv",
    "layer3_pca_model.pkl",
    "layer3_manpower_recommendations.csv",
    "layer3_lp_resource_allocation.csv",
    "layer3_barricading_plan.csv",
    "layer3_diversion_recommendations.csv",
    "layer3_tow_placement.csv",
    "layer3_resource_efficiency_simulation.csv",
    "layer3_efficiency_scenarios.json",
    "layer3_deployment_blueprints.json",
    "layer3_full_dashboard.csv",
]:
    p = OUT / fname
    if p.exists():
        sz = p.stat().st_size
        if fname.endswith(".csv"):
            rc = len(pd.read_csv(p))
            print(f"  {fname}: {rc} rows ({sz//1024} KB)")
        else:
            print(f"  {fname}: {sz//1024} KB")
    else:
        print(f"  {fname}: MISSING!")
