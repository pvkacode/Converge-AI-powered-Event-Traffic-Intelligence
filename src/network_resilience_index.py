#!/usr/bin/env python3
"""
Network Resilience Index (NRI) — additive post-processing.

Reads existing Layer 2, 3, and 7 outputs.
Computes: NRI = 1 - (w1·H + w2·F + w3·S) / (w1 + w2 + w3)

H = normalized hotspot burden (from Layer 2 OBI)
F = normalized fragility (from Layer 3 Hawkes)
S = normalized spillover centrality (from Layer 7)

High NRI = resilient city. Low NRI = fragile, high-burden network.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FRONTEND = OUT / "frontend"

W_BASE = {"H": 0.40, "F": 0.35, "S": 0.25}

WEIGHT_VARIANTS = [
    {"variant_name": "H_dominant", "H": 0.50, "F": 0.30, "S": 0.20},
    {"variant_name": "base", "H": 0.40, "F": 0.35, "S": 0.25},
    {"variant_name": "F_dominant", "H": 0.30, "F": 0.45, "S": 0.25},
    {"variant_name": "S_dominant", "H": 0.35, "F": 0.30, "S": 0.35},
    {"variant_name": "equal", "H": 0.33, "F": 0.33, "S": 0.34},
]


def classify_nri(nri: float) -> tuple[str, str]:
    if nri >= 0.80:
        return "RESILIENT", "#28A745"
    if nri >= 0.65:
        return "MODERATE", "#E8A53D"
    if nri >= 0.50:
        return "VULNERABLE", "#FF8C00"
    return "CRITICAL", "#B0413E"


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_h_component() -> dict | None:
    path = OUT / "layer2_operational_burden_index.csv"
    if not path.exists():
        print("[NRI] WARNING: layer2_operational_burden_index.csv not found — H unavailable")
        return None

    df = pd.read_csv(path)
    col = _pick_col(df, ["operational_burden_index", "obi", "obi_pca", "obi_equal"])
    if col is None:
        print("[NRI] WARNING: no OBI column in Layer 2 file — H unavailable")
        return None

    obi = pd.to_numeric(df[col], errors="coerce").dropna()
    if obi.empty:
        print("[NRI] WARNING: OBI column empty — H unavailable")
        return None

    h_min = float(obi.min())
    h_max = float(obi.max())
    h_raw = float(obi.mean())
    denom = h_max - h_min
    h_norm = _clip01((h_raw - h_min) / denom if denom > 1e-12 else 0.5)

    return {
        "raw": h_raw,
        "normalized": h_norm,
        "min_observed": h_min,
        "max_observed": h_max,
        "p90": float(obi.quantile(0.90)),
        "n_critical": int((obi > 0.5).sum()),
        "max": h_max,
    }


def load_f_component() -> dict | None:
    path = OUT / "layer3_corridor_fragility.csv"
    if not path.exists():
        print("[NRI] WARNING: layer3_corridor_fragility.csv not found — F unavailable")
        return None

    df = pd.read_csv(path)
    if "fragility_reliable" in df.columns:
        reliable = df["fragility_reliable"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[reliable]
        if df.empty:
            df = pd.read_csv(path)

    col = _pick_col(df, ["fragility_practical", "current_fragility", "fragility_raw"])
    if col is None:
        print("[NRI] WARNING: no fragility column in Layer 3 file — F unavailable")
        return None

    frag = pd.to_numeric(df[col], errors="coerce").dropna()
    if frag.empty:
        print("[NRI] WARNING: fragility column empty — F unavailable")
        return None

    f_raw = float(frag.mean())
    f_max = float(frag.max())
    f_norm = _clip01(f_raw / f_max if f_max > 1e-12 else 0.5)

    return {
        "raw": f_raw,
        "normalized": f_norm,
        "max_observed": f_max,
        "n_elevated": int((frag > 1.0).sum()),
    }


def load_s_component() -> dict | None:
    graph_path = OUT / "layer7_graph_centrality.csv"
    spill_path = OUT / "layer7_spillover_centrality.csv"

    if graph_path.exists():
        df = pd.read_csv(graph_path)
        col = _pick_col(df, ["hub_score", "ssc", "pagerank_normalized"])
        zone_col = _pick_col(df, ["zone", "receiver_zone", "target_zone"])
        if col and zone_col:
            vals = pd.to_numeric(df[col], errors="coerce")
            zones = df[zone_col].astype(str)
            valid = vals.notna()
            if valid.any():
                v = vals[valid]
                z = zones[valid]
                s_min = float(v.min())
                s_max = float(v.max())
                s_raw = float(v.mean())
                denom = s_max - s_min
                s_norm = _clip01((s_raw - s_min) / denom if denom > 1e-12 else 0.5)
                idx = v.idxmax()
                return {
                    "raw": s_raw,
                    "normalized": s_norm,
                    "min_observed": s_min,
                    "max_observed": s_max,
                    "max_zone": str(z.loc[idx]),
                    "source": "layer7_graph_centrality",
                }
        print("[NRI] WARNING: layer7_graph_centrality.csv missing usable columns — trying SSC file")

    if spill_path.exists():
        df = pd.read_csv(spill_path)
        col = _pick_col(df, ["SSC_centrality", "ssc", "hub_score"])
        zone_col = _pick_col(df, ["zone", "receiver_zone", "target_zone"])
        if col and zone_col:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                zones = df.loc[vals.index, zone_col].astype(str)
                idx = vals.idxmax()
                s_min = float(vals.min())
                s_max = float(vals.max())
                s_raw = float(vals.mean())
                denom = s_max - s_min
                s_norm = _clip01((s_raw - s_min) / denom if denom > 1e-12 else 0.5)
                return {
                    "raw": s_raw,
                    "normalized": s_norm,
                    "min_observed": s_min,
                    "max_observed": s_max,
                    "max_zone": str(zones.loc[idx]),
                    "source": "layer7_spillover_centrality",
                }

    print("[NRI] WARNING: no Layer 7 centrality file found — S unavailable")
    return None


def compute_burden(
    h_norm: float | None,
    f_norm: float | None,
    s_norm: float | None,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    active: dict[str, float] = {}
    if h_norm is not None and "H" in weights:
        active["H"] = weights["H"]
    if f_norm is not None and "F" in weights:
        active["F"] = weights["F"]
    if s_norm is not None and "S" in weights:
        active["S"] = weights["S"]

    if not active:
        return 0.0, {}

    total = sum(active.values())
    w_adj = {k: v / total for k, v in active.items()}
    burden = 0.0
    if "H" in w_adj and h_norm is not None:
        burden += w_adj["H"] * h_norm
    if "F" in w_adj and f_norm is not None:
        burden += w_adj["F"] * f_norm
    if "S" in w_adj and s_norm is not None:
        burden += w_adj["S"] * s_norm
    return burden, w_adj


def main() -> int:
    print("\n=== NETWORK RESILIENCE INDEX ===")

    h_data = load_h_component()
    f_data = load_f_component()
    s_data = load_s_component()

    available = []
    if h_data is not None:
        available.append("H")
    if f_data is not None:
        available.append("F")
    if s_data is not None:
        available.append("S")

    if not available:
        err_path = OUT / "network_resilience_index_error.txt"
        msg = "No NRI components available — Layer 2, 3, and 7 outputs all missing."
        err_path.write_text(msg + "\n", encoding="utf-8")
        print(f"[NRI] ERROR: {msg}")
        return 1

    missing = [c for c in ["H", "F", "S"] if c not in available]
    if missing:
        print(f"[NRI] WARNING: missing components {missing} — using adjusted weights")

    h_norm = h_data["normalized"] if h_data else None
    f_norm = f_data["normalized"] if f_data else None
    s_norm = s_data["normalized"] if s_data else None

    burden_score, w_adj = compute_burden(h_norm, f_norm, s_norm, W_BASE)
    nri = 1.0 - burden_score
    nri_class, nri_color = classify_nri(nri)
    weights_source = "base" if len(available) == 3 else "adjusted_for_missing"

    sensitivity_rows = []
    nri_values = []
    for wv in WEIGHT_VARIANTS:
        w = {"H": wv["H"], "F": wv["F"], "S": wv["S"]}
        bs, _ = compute_burden(h_norm, f_norm, s_norm, w)
        nri_v = 1.0 - bs
        nri_values.append(nri_v)
        v_class, _ = classify_nri(nri_v)
        sensitivity_rows.append({
            "variant_name": wv["variant_name"],
            "w_H": wv["H"],
            "w_F": wv["F"],
            "w_S": wv["S"],
            "NRI": round(nri_v, 4),
            "NRI_class": v_class,
        })

    nri_range_low = min(nri_values)
    nri_range_high = max(nri_values)

    component_scores = {
        "H": h_norm if h_norm is not None else float("nan"),
        "F": f_norm if f_norm is not None else float("nan"),
        "S": s_norm if s_norm is not None else float("nan"),
    }
    most_vulnerable = max(
        ((k, v) for k, v in component_scores.items() if pd.notna(v)),
        key=lambda x: x[1],
    )[0]

    snapshot = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "NRI": round(nri, 4),
        "NRI_class": nri_class,
        "NRI_color": nri_color,
        "NRI_range_low": round(nri_range_low, 4),
        "NRI_range_high": round(nri_range_high, 4),
        "H_raw": round(h_data["raw"], 6) if h_data else "",
        "H_normalized": round(h_norm, 6) if h_norm is not None else "",
        "H_weight": round(w_adj.get("H", 0.0), 4),
        "F_raw": round(f_data["raw"], 6) if f_data else "",
        "F_normalized": round(f_norm, 6) if f_norm is not None else "",
        "F_weight": round(w_adj.get("F", 0.0), 4),
        "S_raw": round(s_data["raw"], 6) if s_data else "",
        "S_normalized": round(s_norm, 6) if s_norm is not None else "",
        "S_weight": round(w_adj.get("S", 0.0), 4),
        "burden_score": round(burden_score, 6),
        "H_n_critical_junctions": h_data["n_critical"] if h_data else "",
        "F_n_elevated_corridors": f_data["n_elevated"] if f_data else "",
        "S_max_zone": s_data["max_zone"] if s_data else "",
        "weights_source": weights_source,
        "components_available": ",".join(available),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    FRONTEND.mkdir(parents=True, exist_ok=True)

    index_df = pd.DataFrame([snapshot])
    sens_df = pd.DataFrame(sensitivity_rows)

    index_df.to_csv(OUT / "network_resilience_index.csv", index=False)
    sens_df.to_csv(OUT / "network_resilience_sensitivity.csv", index=False)
    index_df.to_csv(FRONTEND / "network_resilience_index.csv", index=False)
    sens_df.to_csv(FRONTEND / "network_resilience_sensitivity.csv", index=False)

    print("\n[Network Resilience Index]")
    print(f"NRI = {nri:.4f} ({nri_class})")
    print(f"Sensitivity range: [{nri_range_low:.4f}, {nri_range_high:.4f}]")
    print(
        f"Components: H={h_norm if h_norm is not None else float('nan'):.3f} "
        f"F={f_norm if f_norm is not None else float('nan'):.3f} "
        f"S={s_norm if s_norm is not None else float('nan'):.3f}"
    )
    print(
        f"Weights: H={w_adj.get('H', 0):.2f} "
        f"F={w_adj.get('F', 0):.2f} "
        f"S={w_adj.get('S', 0):.2f}"
    )
    print(f"Most vulnerable component: {most_vulnerable}")
    print("  Wrote network_resilience_index.csv")
    print("  Wrote network_resilience_sensitivity.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
