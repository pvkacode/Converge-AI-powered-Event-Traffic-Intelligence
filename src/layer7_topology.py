"""
Layer 7 — M7B.1 Part C: Road Network Topology.

Builds a reusable spatial topology over the Layer 5 active sites (preparation for
FUTURE MPC / GNN / propagation models). Read-only on inputs; additive outputs only.
Coordinates come from data/events_raw.csv (id renamed event_id); corridor/junction
from the L4.5 as-of matrix.

    distance_km via haversine
    adjacency_weight = exp(-distance_km / D0)  in [0,1]

ADDITIVE ONLY. Writes:
  outputs/layer7_sensor_topology.csv   (pairwise edges)
  outputs/layer7_topology_metrics.csv  (per-node connectivity)
  outputs/layer7_topology_summary.csv  (network-level metrics)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
ROOT = Path(__file__).resolve().parent.parent

D0_KM = 5.0            # adjacency decay length (configurable)
EDGE_WEIGHT_MIN = 0.05  # ignore negligible edges for degree/density


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def _site_coords() -> pd.DataFrame:
    active = pd.read_csv(OUT / "layer7_active_site_state.csv")
    active["event_id"] = active["event_id"].astype(str)
    raw = pd.read_csv(ROOT / "data" / "events_raw.csv")
    idcol = "event_id" if "event_id" in raw.columns else "id"
    raw = raw.rename(columns={idcol: "event_id"})
    raw["event_id"] = raw["event_id"].astype(str)
    geo = raw[["event_id", "latitude", "longitude", "corridor", "junction"]].drop_duplicates("event_id")
    df = active[["event_id"]].merge(geo, on="event_id", how="left")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    # valid coords only (drop 0/0 placeholders and NaN)
    df["geo_valid"] = (df["latitude"].notna() & df["longitude"].notna()
                       & ~((df["latitude"].abs() < 1e-6) & (df["longitude"].abs() < 1e-6)))
    return df


def build_topology(d0: float = D0_KM):
    sites = _site_coords()
    valid = sites[sites["geo_valid"]].reset_index(drop=True)
    edges = []
    n = len(valid)
    for i in range(n):
        a = valid.iloc[i]
        for j in range(i + 1, n):
            b = valid.iloc[j]
            dist = _haversine_km(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            w = float(np.clip(np.exp(-dist / d0), 0.0, 1.0))
            edges.append({
                "site_a": a["event_id"], "site_b": b["event_id"],
                "distance_km": round(dist, 4),
                "same_corridor": bool(a["corridor"] == b["corridor"]),
                "adjacency_weight": round(w, 6),
                "generated_at": _NOW_ISO,
            })
    edges_df = pd.DataFrame(edges)

    # C3 connectivity metrics (per node)
    nodes = list(valid["event_id"])
    corridor_of = dict(zip(valid["event_id"], valid["corridor"]))
    deg, wdeg, corr_conn = {}, {}, {}
    for nid in nodes:
        deg[nid] = wdeg[nid] = corr_conn[nid] = 0
    if len(edges_df):
        sig = edges_df[edges_df["adjacency_weight"] >= EDGE_WEIGHT_MIN]
        for _, e in edges_df.iterrows():
            wdeg[e["site_a"]] += e["adjacency_weight"]
            wdeg[e["site_b"]] += e["adjacency_weight"]
        for _, e in sig.iterrows():
            deg[e["site_a"]] += 1
            deg[e["site_b"]] += 1
            if e["same_corridor"]:
                corr_conn[e["site_a"]] += 1
                corr_conn[e["site_b"]] += 1
    metrics = pd.DataFrame([{
        "event_id": nid, "corridor": corridor_of[nid],
        "node_degree": int(deg[nid]), "weighted_degree": round(float(wdeg[nid]), 6),
        "corridor_connectivity": int(corr_conn[nid]), "generated_at": _NOW_ISO,
    } for nid in nodes])

    # network-level summary
    max_edges = n * (n - 1) / 2 if n > 1 else 0
    sig_edges = int((edges_df["adjacency_weight"] >= EDGE_WEIGHT_MIN).sum()) if len(edges_df) else 0
    density = (sig_edges / max_edges) if max_edges else 0.0
    summary = pd.DataFrame([{
        "n_active_sites": int(len(sites)),
        "n_geo_valid_nodes": int(n),
        "n_pairs": int(len(edges_df)),
        "n_significant_edges": sig_edges,
        "edge_weight_threshold": EDGE_WEIGHT_MIN,
        "d0_km": d0,
        "network_density": round(float(density), 6),
        "avg_node_degree": round(float(metrics["node_degree"].mean()), 4) if len(metrics) else 0.0,
        "avg_weighted_degree": round(float(metrics["weighted_degree"].mean()), 4) if len(metrics) else 0.0,
        "mean_distance_km": round(float(edges_df["distance_km"].mean()), 4) if len(edges_df) else 0.0,
        "same_corridor_edges": int(edges_df["same_corridor"].sum()) if len(edges_df) else 0,
        "generated_at": _NOW_ISO,
    }])
    return edges_df, metrics, summary


def run(write: bool = True) -> tuple[dict, list[dict]]:
    edges, metrics, summary = build_topology()
    if write:
        edges.to_csv(OUT / "layer7_sensor_topology.csv", index=False)
        metrics.to_csv(OUT / "layer7_topology_metrics.csv", index=False)
        summary.to_csv(OUT / "layer7_topology_summary.csv", index=False)
    w_ok = bool(len(edges) == 0 or ((edges["adjacency_weight"] >= 0)
                                    & (edges["adjacency_weight"] <= 1)).all())
    fin = bool(len(edges) == 0 or np.isfinite(edges[["distance_km", "adjacency_weight"]].to_numpy()).all())
    checks = [{
        "check_id": "m7b1_topology_weights_bounded", "phase": "topology",
        "passed": w_ok and fin,
        "detail": f"{len(edges)} edges; weights in [0,1] and finite; "
                  f"density={float(summary['network_density'].iloc[0]):.4f}; "
                  f"avg_degree={float(summary['avg_node_degree'].iloc[0]):.3f}",
        "severity": "critical" if not (w_ok and fin) else "info",
    }]
    return {"edges": edges, "metrics": metrics, "summary": summary}, checks


if __name__ == "__main__":
    tables, checks = run(write=True)
    print(tables["summary"].T.to_string())
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
