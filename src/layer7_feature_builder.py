"""
Layer 7 Feature Builder — Zone × Time Panel Dataset
ASTraM Bengaluru Traffic Disruption Intelligence

Additive. Does NOT modify Layer 1-6 or Part A outputs.

Builds an hourly zone × time panel over Nov 10, 2023 – Jan 19, 2024.
Lambda_v(t) uses a fresh Nov-Dec-only Hawkes refit (identical to
layer7b_spillover_forecast_eval.py Section 4) — NOT Part A's full
Nov-Jan fit, which would leak Jan data into the eval features.

Train/eval split for the panel:
  Train rows : grid_time in [Nov 10 2023, Dec 31 2023]
  Eval  rows : grid_time in [Jan 01 2024, Jan 19 2024]

Outputs:
  outputs/layer7_panel_dataset.csv
  outputs/layer7_model_artifacts/layer7_hawkes_params_novdec.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUTS = Path("outputs")
ARTIFACTS = OUTPUTS / "layer7_model_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
MARK_WEIGHTS   = {"DIS": 0.40, "OBI": 0.30, "SEV": 0.20, "CONF": 0.10}
BETA_GRID      = np.logspace(-3, 1, 60)
KAPPA_SHRINK   = 5.0
MIN_ZONE_EVENTS = 5
N_RESTARTS     = 4
GRID_STEP_H    = 1.0   # hourly grid
HORIZONS       = [3, 6, 9]

DATA_START        = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
TRAIN_END         = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
EVAL_START        = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
EVAL_END          = pd.Timestamp("2024-01-19 23:59:59", tz="UTC")

ZONE_ADJACENCY_UNDIRECTED = [
    ("Central Zone 1", "Central Zone 2"),
    ("Central Zone 1", "North Zone 1"),
    ("Central Zone 1", "North Zone 2"),
    ("Central Zone 1", "South Zone 1"),
    ("Central Zone 1", "West Zone 1"),
    ("Central Zone 2", "North Zone 2"),
    ("Central Zone 2", "South Zone 2"),
    ("Central Zone 2", "East Zone 2"),
    ("North Zone 1",  "North Zone 2"),
    ("North Zone 1",  "West Zone 1"),
    ("North Zone 2",  "East Zone 1"),
    ("South Zone 1",  "South Zone 2"),
    ("South Zone 1",  "West Zone 2"),
    ("South Zone 2",  "East Zone 1"),
    ("South Zone 2",  "East Zone 2"),
    ("East Zone 1",   "East Zone 2"),
    ("West Zone 1",   "West Zone 2"),
]


def build_adjacency(zones):
    zone_set = set(zones)
    pairs = set()
    for a, b in ZONE_ADJACENCY_UNDIRECTED:
        if a in zone_set and b in zone_set:
            pairs.add((a, b)); pairs.add((b, a))
    return sorted(pairs)


def minmax_scale_train(series, train_mask):
    mn, mx = series[train_mask].min(), series[train_mask].max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return ((series - mn) / (mx - mn)).clip(0.0, 1.0)


# ── Hawkes kernel helpers (identical to layer7b) ──────────────────────────────

def compute_self_A(times, marks, beta):
    n, A = len(times), np.zeros(len(times))
    for i in range(1, n):
        A[i] = np.exp(-beta * (times[i] - times[i-1])) * (A[i-1] + marks[i-1])
    return A


def compute_cross_A(times_recv, times_src, marks_src, beta):
    A = np.zeros(len(times_recv))
    for j in range(len(times_recv)):
        idx = np.searchsorted(times_src, times_recv[j], side="left")
        if idx > 0:
            dts = times_recv[j] - times_src[:idx]
            A[j] = np.dot(marks_src[:idx], np.exp(-beta * dts))
    return A


def compensator_term(times_src, marks_src, alpha, beta, T_obs):
    if alpha == 0.0 or len(times_src) == 0:
        return 0.0
    return (alpha / beta) * float(np.sum(marks_src * (1.0 - np.exp(-beta * (T_obs - times_src)))))


def zone_loglik(params, times_v, marks_v, A_self, cross_A_list, T_obs, beta, l2_pen=0.0):
    mu, alpha_self, alpha_cross = params[0], params[1], params[2:]
    if mu <= 0 or alpha_self < 0 or np.any(alpha_cross < 0):
        return 1e12
    intensity = np.maximum(mu + alpha_self * A_self
                           + sum(alpha_cross[k] * A_uv for k, (A_uv, _) in enumerate(cross_A_list)),
                           1e-10)
    comp = (mu * T_obs + compensator_term(times_v, marks_v, alpha_self, beta, T_obs)
            + sum(alpha_cross[k] * cu for k, (_, cu) in enumerate(cross_A_list)))
    return -(float(np.sum(np.log(intensity))) - comp - l2_pen * float(np.sum(alpha_cross**2)))


def fit_zone_hawkes(times_v, marks_v, A_self, cross_A_list, T_obs, beta, kappa):
    n_cross = len(cross_A_list)
    rate_est = len(times_v) / max(T_obs, 1.0)
    l2_pen = kappa / max(len(times_v), 1)
    rng = np.random.default_rng(42)
    best_val, best_x = np.inf, None
    for _ in range(N_RESTARTS):
        x0 = np.concatenate([[rng.uniform(max(rate_est*0.05,1e-5), max(rate_est*1.5,0.01)),
                               rng.uniform(0.001, 0.4)],
                              rng.uniform(0.0, 0.1, size=n_cross)])
        bounds = [(1e-8, None), (0.0, None)] + [(0.0, None)] * n_cross
        try:
            res = minimize(zone_loglik, x0=x0,
                           args=(times_v, marks_v, A_self, cross_A_list, T_obs, beta, l2_pen),
                           method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 400, "ftol": 1e-9})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        except Exception:
            continue
    if best_x is None:
        best_x = np.array([rate_est, 0.0] + [0.0]*n_cross)
    return best_x, best_val < 1e9


def profile_fit_hawkes(zones, zone_data, adj_pairs, T_obs, kappa=KAPPA_SHRINK):
    recv_pairs = {v: [] for v in zones}
    for i, (u, v) in enumerate(adj_pairs):
        recv_pairs[v].append((u, i))

    best_total, best_beta = -np.inf, BETA_GRID[0]
    best_params = {}

    for beta in BETA_GRID:
        self_As = {v: compute_self_A(zone_data[v]["times"], zone_data[v]["marks"], beta)
                   if len(zone_data[v]["times"]) > 0 else np.array([])
                   for v in zones}
        cross_As, cross_comps = {}, {}
        for (u, v) in adj_pairs:
            zu, zv = zone_data[u], zone_data[v]
            if len(zv["times"]) == 0 or len(zu["times"]) == 0:
                cross_As[(u,v)] = np.zeros(len(zv["times"]))
                cross_comps[(u,v)] = 0.0
            else:
                cross_As[(u,v)] = compute_cross_A(zv["times"], zu["times"], zu["marks"], beta)
                cross_comps[(u,v)] = (1.0/beta)*float(np.sum(zu["marks"]*(1.-np.exp(-beta*(T_obs-zu["times"])))))

        total_ll = 0.0
        params_beta = {}
        for v in zones:
            zd = zone_data[v]
            if len(zd["times"]) < MIN_ZONE_EVENTS:
                params_beta[v] = None; continue
            cross_A_list = [(cross_As.get((u,v), np.zeros(len(zd["times"]))),
                             cross_comps.get((u,v), 0.0))
                            for (u, _) in recv_pairs[v]]
            p, conv = fit_zone_hawkes(zd["times"], zd["marks"], self_As[v],
                                      cross_A_list, T_obs, beta, kappa)
            ll = -zone_loglik(p, zd["times"], zd["marks"], self_As[v], cross_A_list, T_obs, beta, 0.0)
            total_ll += ll
            params_beta[v] = {
                "mu": float(p[0]), "alpha_self": float(p[1]),
                "alpha_cross": {adj_pairs[recv_pairs[v][k][1]][0]: float(p[2+k])
                                for k in range(len(recv_pairs[v]))},
                "converged": conv,
            }
        if total_ll > best_total:
            best_total, best_beta, best_params = total_ll, beta, params_beta

    return best_beta, best_params, best_total


# ── vectorized lambda computation ─────────────────────────────────────────────

def compute_lambda_grid(T_grid, all_times, all_zones, all_marks,
                        zone_v, params, beta, adj_pairs):
    """
    Compute lambda_v(t) for each t in T_grid (numpy array, hours).
    Uses only events strictly before each t.
    Vectorized over T_grid.
    """
    zp = params.get(zone_v)
    if zp is None:
        return np.full(len(T_grid), np.nan)

    mu_v = zp["mu"]
    result = np.full(len(T_grid), mu_v)

    sources = [(zone_v, zp["alpha_self"])] + [
        (u, zp["alpha_cross"].get(u, 0.0))
        for (u, v2) in adj_pairs if v2 == zone_v
    ]

    for src_zone, alpha in sources:
        if alpha == 0.0:
            continue
        mask = all_zones == src_zone
        t_src = all_times[mask]
        m_src = all_marks[mask]
        if len(t_src) == 0:
            continue
        # diff[g, i] = T_grid[g] - t_src[i]; positive means event is in the past
        diff = T_grid[:, None] - t_src[None, :]   # (N_grid, N_src)
        valid = diff > 0
        contrib = np.where(valid, m_src[None, :] * np.exp(-beta * np.where(valid, diff, 0.0)), 0.0)
        result += alpha * contrib.sum(axis=1)

    return result


# ── vectorized lookback counts/burden ────────────────────────────────────────

def compute_lookbacks(T_grid, t_events, marks, windows_h):
    """
    For each grid time t_g and each lookback window w,
    compute count and burden of events in (t_g - w, t_g).
    Returns dict: w -> (count_arr, burden_arr), each shape (N_grid,).
    """
    if len(t_events) == 0:
        out = {}
        for w in windows_h:
            out[w] = (np.zeros(len(T_grid), dtype=int), np.zeros(len(T_grid)))
        return out

    diff = T_grid[:, None] - t_events[None, :]   # (N_grid, N_events)
    out = {}
    for w in windows_h:
        in_win = (diff > 0) & (diff <= w)
        out[w] = (in_win.sum(axis=1), (in_win * marks[None, :]).sum(axis=1))
    return out


# ── target computation ────────────────────────────────────────────────────────

def compute_targets(T_grid, t_events, horizons):
    """
    For each grid time t_g and horizon h,
    count events in (t_g, t_g + h].
    Returns dict: h -> (Y_count, Y_bin).
    """
    if len(t_events) == 0:
        return {h: (np.zeros(len(T_grid), dtype=int), np.zeros(len(T_grid), dtype=int))
                for h in horizons}
    diff_fwd = t_events[None, :] - T_grid[:, None]   # (N_grid, N_events)
    out = {}
    for h in horizons:
        in_win = (diff_fwd > 0) & (diff_fwd <= h)
        counts = in_win.sum(axis=1).astype(int)
        out[h] = (counts, (counts > 0).astype(int))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: DATA LOADING ===")

feat = pd.read_csv(OUTPUTS / "layer45_asof_feature_matrix.csv")
feat["start_local"] = pd.to_datetime(feat["start_local"], utc=True, errors="coerce")
feat = feat.dropna(subset=["start_local", "zone"]).copy()
feat = feat.sort_values("start_local").reset_index(drop=True)
print(f"  Feature matrix: {feat.shape}")

hotspots = pd.read_csv(OUTPUTS / "layer2_hotspots.csv")
print(f"  Layer 2 hotspots: {hotspots.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MARK CONSTRUCTION  (Nov-Dec normalization only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: MARK CONSTRUCTION ===")

train_mask = (feat["start_local"] >= DATA_START) & (feat["start_local"] <= TRAIN_END)
panel_mask = (feat["start_local"] >= DATA_START) & (feat["start_local"] <= EVAL_END)

priority_map = {"High": 1.0, "Low": 0.333, "Unknown": 0.667}
for raw_col, src_col, out_col in [
    ("DIS_asof_raw",  "asof_fragility_proxy",       "DIS_asof"),
    ("OBI_asof_raw",  "asof_obi_proxy",              "OBI_asof"),
    ("conf_raw",      "asof_retrieval_confidence",   "confidence_i"),
]:
    feat[raw_col] = pd.to_numeric(feat[src_col], errors="coerce").fillna(0.0)
    feat[out_col] = minmax_scale_train(feat[raw_col], train_mask)

feat["severity_raw"] = feat["priority"].astype(str).map(priority_map).fillna(0.667)
feat["severity_i"] = minmax_scale_train(feat["severity_raw"], train_mask)

feat["mark"] = (MARK_WEIGHTS["DIS"] * feat["DIS_asof"]
                + MARK_WEIGHTS["OBI"] * feat["OBI_asof"]
                + MARK_WEIGHTS["SEV"] * feat["severity_i"]
                + MARK_WEIGHTS["CONF"] * feat["confidence_i"]).clip(0.0, 1.0)

mark_std = feat.loc[train_mask, "mark"].std()
if mark_std < 1e-4:
    print("  WARNING: mark variance near zero, using m_i=1")
    feat["mark"] = 1.0
print(f"  Mark train std={mark_std:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: NOV-DEC HAWKES REFIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: NOV-DEC HAWKES REFIT ===")
print("  Fitting on Nov 10 – Dec 31 only (same window as layer7b eval train).")

feat_train_hk = feat[train_mask].copy()
t0 = feat_train_hk["start_local"].min()
feat_train_hk["t_hours"] = (feat_train_hk["start_local"] - t0).dt.total_seconds() / 3600.0

observed_zones = sorted(feat["zone"].dropna().unique().tolist())
adj_pairs = build_adjacency(observed_zones)

zone_data_train = {}
for z in observed_zones:
    sub = feat_train_hk[feat_train_hk["zone"] == z].sort_values("t_hours")
    zone_data_train[z] = {
        "times": sub["t_hours"].values.astype(float),
        "marks": sub["mark"].values.astype(float),
    }

T_train_hk = float(feat_train_hk["t_hours"].max())

print(f"  Training events: {train_mask.sum()}  T_train={T_train_hk:.1f} h")
print(f"  Fitting profile likelihood ({len(BETA_GRID)} beta candidates)...")
beta_fit, params_fit, ll_train = profile_fit_hawkes(
    observed_zones, zone_data_train, adj_pairs, T_train_hk)

half_life = np.log(2.0) / beta_fit
print(f"  Best beta={beta_fit:.5f} h^-1  half_life={half_life:.2f} h  LL={ll_train:.3f}")
converged_n = sum(1 for v in observed_zones if params_fit.get(v) and params_fit[v]["converged"])
print(f"  Converged: {converged_n}/{len(observed_zones)} zones")

# Save params
hk_artifact = {
    "beta": beta_fit, "half_life_hours": half_life,
    "t0_utc": str(t0), "T_train_hours": T_train_hk,
    "train_period": "Nov 10 2023 – Dec 31 2023",
    "note": "Used for lambda_v(t) feature in layer7_zone_forecast. NOT Part A params.",
    "zone_params": params_fit,
}
with open(ARTIFACTS / "layer7_hawkes_params_novdec.json", "w") as fh:
    json.dump(hk_artifact, fh, indent=2, default=str)
print("  Saved layer7_hawkes_params_novdec.json")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: STATIC ZONE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: STATIC ZONE FEATURES ===")

# Layer 2 hotspot burden aggregated to zone via junction->zone mapping
jz_map = (feat[["junction", "zone"]].dropna()
          .drop_duplicates("junction")
          .set_index("junction")["zone"].to_dict())

hotspots["zone"] = hotspots["junction"].map(jz_map)
zone_hotspot = (hotspots.dropna(subset=["zone"])
                .groupby("zone")
                .agg(hotspot_n_sig=("is_significant", "sum"),
                     hotspot_max_z=("z_score", "max"),
                     hotspot_mean_burden=("weighted_intensity", "mean"))
                .reset_index())
print(f"  Mapped {hotspots['zone'].notna().sum()}/{len(hotspots)} hotspots to zones")

# Zone historical rate (events/hour) from Nov-Dec training
zone_hist_rate = {}
citywide_rate = train_mask.sum() / max(T_train_hk, 1.0)
for z in observed_zones:
    n_z = len(zone_data_train[z]["times"])
    zone_hist_rate[z] = n_z / max(T_train_hk, 1.0)
    if n_z < 10:  # thin history: blend with citywide
        zone_hist_rate[z] = 0.5 * zone_hist_rate[z] + 0.5 * citywide_rate / len(observed_zones)

# Zone mean fragility from training events (as-of, so point-in-time safe)
zone_frag = (feat[train_mask].groupby("zone")["asof_fragility_proxy"]
             .mean().rename("zone_mean_fragility").reset_index())
zone_frag["asof_fragility_proxy"] = pd.to_numeric(zone_frag["zone_mean_fragility"], errors="coerce").fillna(0.0)

static_df = pd.DataFrame({"zone": observed_zones})
static_df = static_df.merge(zone_hotspot, on="zone", how="left")
static_df = static_df.merge(zone_frag[["zone", "zone_mean_fragility"]], on="zone", how="left")
static_df["zone_hist_rate"] = static_df["zone"].map(zone_hist_rate)
static_df = static_df.fillna(0.0)
print(f"  Static features: {static_df.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BUILD TEMPORAL GRID
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: TEMPORAL GRID ===")

# Grid from DATA_START to EVAL_END in hourly steps
# t_hours are measured from t0 (first training event time)
t0_utc = t0
grid_start_h = (DATA_START - t0_utc).total_seconds() / 3600.0
grid_end_h   = (EVAL_END   - t0_utc).total_seconds() / 3600.0
T_grid = np.arange(grid_start_h, grid_end_h, GRID_STEP_H)

train_end_h  = (TRAIN_END  - t0_utc).total_seconds() / 3600.0
eval_start_h = (EVAL_START - t0_utc).total_seconds() / 3600.0

print(f"  Grid: {len(T_grid)} hourly points ({grid_start_h:.1f} to {grid_end_h:.1f} h)")
print(f"  Train rows (per zone): {(T_grid <= train_end_h).sum()}")
print(f"  Eval rows  (per zone): {((T_grid > train_end_h) & (T_grid <= grid_end_h)).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ALL-EVENTS ARRAY (for lambda + lookback computation)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: ALL-EVENTS ARRAY ===")

feat_all = feat[panel_mask].copy()
feat_all["t_hours"] = (feat_all["start_local"] - t0_utc).dt.total_seconds() / 3600.0
feat_all = feat_all.sort_values("t_hours").reset_index(drop=True)

all_t = feat_all["t_hours"].values.astype(float)
all_z = feat_all["zone"].values
all_m = feat_all["mark"].values
print(f"  All panel events: {len(all_t)} (Nov 10 – Jan 19)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: COMPUTE FEATURES PER ZONE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: FEATURE COMPUTATION ===")

LOOKBACK_WINDOWS = [1, 4, 24]  # hours

grid_utc = pd.to_datetime(t0_utc) + pd.to_timedelta(T_grid, unit="h")

# Time-of-day features (shared across zones)
hour_of_day = (T_grid % 24)
hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
dow = np.array([ts.dayofweek for ts in grid_utc])
is_weekend = (dow >= 5).astype(int)
is_peak = (((hour_of_day >= 7) & (hour_of_day < 10)) |
           ((hour_of_day >= 17) & (hour_of_day < 20))).astype(int)

panel_rows = []
for z in observed_zones:
    print(f"  Building features for {z}...")
    z_mask = all_z == z
    t_z = all_t[z_mask]
    m_z = all_m[z_mask]

    # Lambda_v(t) from Nov-Dec Hawkes refit
    lam = compute_lambda_grid(T_grid, all_t, all_z, all_m, z, params_fit, beta_fit, adj_pairs)

    # Count/burden lookbacks
    lb = compute_lookbacks(T_grid, t_z, m_z, LOOKBACK_WINDOWS)

    # Targets for each horizon
    tgt = compute_targets(T_grid, t_z, HORIZONS)

    # Static features for this zone
    st = static_df[static_df["zone"] == z].iloc[0] if (static_df["zone"] == z).any() else {}

    for gi, tg in enumerate(T_grid):
        row = {
            "grid_time_utc": grid_utc[gi].isoformat(),
            "t_hours": tg,
            "zone": z,
            "split": "train" if tg <= train_end_h else "eval",
            # Time features
            "hour_sin": round(hour_sin[gi], 6),
            "hour_cos": round(hour_cos[gi], 6),
            "dow": int(dow[gi]),
            "is_weekend": int(is_weekend[gi]),
            "is_peak": int(is_peak[gi]),
            # Hawkes intensity (Nov-Dec refit)
            "lambda_v": round(float(lam[gi]), 6) if not np.isnan(lam[gi]) else 0.0,
            "log_lambda_v": round(float(np.log(max(lam[gi], 1e-8))), 6) if not np.isnan(lam[gi]) else -18.42,
            # Recent activity
            "count_1h":   int(lb[1][0][gi]),
            "count_4h":   int(lb[4][0][gi]),
            "count_24h":  int(lb[24][0][gi]),
            "burden_1h":  round(float(lb[1][1][gi]), 6),
            "burden_4h":  round(float(lb[4][1][gi]), 6),
            # Static zone features
            "hotspot_n_sig":      float(st.get("hotspot_n_sig", 0.0)),
            "hotspot_max_z":      float(st.get("hotspot_max_z", 0.0)),
            "hotspot_mean_burden": float(st.get("hotspot_mean_burden", 0.0)),
            "zone_mean_fragility": float(st.get("zone_mean_fragility", 0.0)),
            "zone_hist_rate":     round(float(zone_hist_rate.get(z, 0.0)), 6),
        }
        # Targets
        for h in HORIZONS:
            cnt, bn = tgt[h][0][gi], tgt[h][1][gi]
            # Mark NaN if target window extends beyond EVAL_END
            if tg + h > grid_end_h:
                row[f"Y_count_{h}h"] = None
                row[f"Y_bin_{h}h"]   = None
            else:
                row[f"Y_count_{h}h"] = int(cnt)
                row[f"Y_bin_{h}h"]   = int(bn)
        panel_rows.append(row)

df_panel = pd.DataFrame(panel_rows)
print(f"  Panel shape: {df_panel.shape}")
print(f"  Train rows: {(df_panel['split']=='train').sum()}, Eval rows: {(df_panel['split']=='eval').sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 8: SAVE ===")

df_panel.to_csv(OUTPUTS / "layer7_panel_dataset.csv", index=False)
print(f"  Wrote layer7_panel_dataset.csv ({len(df_panel)} rows, {len(df_panel.columns)} cols)")

# Save t0 and split boundaries for the forecast script
meta = {
    "t0_utc": str(t0_utc),
    "train_end_h": train_end_h,
    "eval_start_h": eval_start_h,
    "grid_end_h": grid_end_h,
    "horizons": HORIZONS,
    "zones": observed_zones,
    "grid_step_h": GRID_STEP_H,
    "south_zone_2_ks_note": (
        "South Zone 2 showed KS p=0.021 in Part B spillover eval — "
        "weakest calibration among zones. Widen confidence band in ERI."
    ),
}
with open(ARTIFACTS / "layer7_panel_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print("  Wrote layer7_panel_meta.json")
print("\n=== layer7_feature_builder.py complete ===")
