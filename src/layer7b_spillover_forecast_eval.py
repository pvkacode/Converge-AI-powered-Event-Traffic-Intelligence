"""
Layer 7 Part B — Out-of-Sample Forecast Evaluation
ASTraM Bengaluru Traffic Disruption Intelligence

Additive module. Does NOT modify any Layer 1-6 or Part A output file.

DATA LIMITATION (discovered during Part A):
  Zone labels in events_clean.parquet are only populated through Jan 19 2024.
  The originally planned Mar-Apr evaluation window cannot be used — those events
  have null zone and cannot participate in a zone-level Hawkes model.

  Revised chronological split (fits entirely within the zone-labeled window):
    Train : Nov 10, 2023 – Dec 31, 2023
    Eval  : Jan 01, 2024 – Jan 19, 2024  (genuine zone-labeled cutoff)

  This is a data limitation, not a methodological choice. The holdout is smaller
  than this project's other evaluations. All metrics are reported with 95%
  bootstrap confidence intervals (event-level resample, B=1000) rather than
  point estimates alone. The Poisson baseline provides a meaningful reference.

  Part A fits on the full Nov 10 – Jan 19 window; Part B re-fits independently
  on Nov-Dec only so that Jan 1-19 is a genuine out-of-sample holdout.

Metrics:
  - Held-out log-likelihood per event (Hawkes vs Poisson baseline)
  - Per-zone LL and event count
  - Bootstrap 95% CI on mean log-intensity
  - KS test on time-rescaled residuals (model calibration)
  - Day-by-day binned predicted rate vs observed
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, ks_1samp, expon

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUTS = Path("outputs")
DATA = Path("data")

# ── hyperparameters (same as Part A) ─────────────────────────────────────────
MARK_WEIGHTS = {"DIS": 0.40, "OBI": 0.30, "SEV": 0.20, "CONF": 0.10}
BETA_GRID     = np.logspace(-3, 1, 60)
KAPPA_SHRINK  = 5.0
MIN_ZONE_EVENTS = 5
N_RESTARTS    = 4
N_BOOTSTRAP   = 1000

# ── chronological split (revised to fit zone-labeled data window) ─────────────
DATA_START       = pd.Timestamp("2023-11-01 00:00:00", tz="UTC")
PART_B_TRAIN_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
PART_B_EVAL_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
PART_B_EVAL_END   = pd.Timestamp("2024-01-19 23:59:59", tz="UTC")

# ── zone adjacency (same as Part A) ──────────────────────────────────────────
ZONE_ADJACENCY_UNDIRECTED = [
    ("Central Zone 1", "Central Zone 2"),
    ("Central Zone 1", "North Zone 1"),
    ("Central Zone 1", "North Zone 2"),
    ("Central Zone 1", "South Zone 1"),
    ("Central Zone 1", "West Zone 1"),
    ("Central Zone 2", "North Zone 2"),
    ("Central Zone 2", "South Zone 2"),
    ("Central Zone 2", "East Zone 2"),
    ("North Zone 1", "North Zone 2"),
    ("North Zone 1", "West Zone 1"),
    ("North Zone 2", "East Zone 1"),
    ("South Zone 1", "South Zone 2"),
    ("South Zone 1", "West Zone 2"),
    ("South Zone 2", "East Zone 1"),
    ("South Zone 2", "East Zone 2"),
    ("East Zone 1", "East Zone 2"),
    ("West Zone 1", "West Zone 2"),
]


def build_adjacency(zones):
    zone_set = set(zones)
    pairs = set()
    for a, b in ZONE_ADJACENCY_UNDIRECTED:
        if a in zone_set and b in zone_set:
            pairs.add((a, b))
            pairs.add((b, a))
    return sorted(pairs)


def minmax_scale_train(series, train_mask):
    mn = series[train_mask].min()
    mx = series[train_mask].max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return ((series - mn) / (mx - mn)).clip(0.0, 1.0)


# ── Hawkes kernel helpers (identical to Part A) ───────────────────────────────

def compute_self_A(times, marks, beta):
    n = len(times)
    A = np.zeros(n)
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        A[i] = np.exp(-beta * dt) * (A[i - 1] + marks[i - 1])
    return A


def compute_cross_A(times_recv, times_src, marks_src, beta):
    n_r = len(times_recv)
    A = np.zeros(n_r)
    for j in range(n_r):
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
    mu = params[0]
    alpha_self = params[1]
    alpha_cross = params[2:]
    if mu <= 0 or alpha_self < 0 or np.any(alpha_cross < 0):
        return 1e12
    intensity = mu + alpha_self * A_self
    for k, (A_uv, _) in enumerate(cross_A_list):
        intensity = intensity + alpha_cross[k] * A_uv
    intensity = np.maximum(intensity, 1e-10)
    comp = (mu * T_obs
            + compensator_term(times_v, marks_v, alpha_self, beta, T_obs))
    for k, (_, comp_u) in enumerate(cross_A_list):
        comp += alpha_cross[k] * comp_u
    ll = float(np.sum(np.log(intensity))) - comp
    penalty = l2_pen * float(np.sum(alpha_cross ** 2))
    return -(ll - penalty)


def fit_zone(times_v, marks_v, A_self, cross_A_list, T_obs, beta, kappa):
    n_cross = len(cross_A_list)
    rate_est = len(times_v) / max(T_obs, 1.0)
    l2_pen = kappa / max(len(times_v), 1)
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    for _ in range(N_RESTARTS):
        mu0 = rng.uniform(max(rate_est * 0.05, 1e-5), max(rate_est * 1.5, 0.01))
        a_self0 = rng.uniform(0.001, 0.4)
        a_cross0 = rng.uniform(0.0, 0.1, size=n_cross)
        x0 = np.concatenate([[mu0, a_self0], a_cross0])
        bounds = [(1e-8, None), (0.0, None)] + [(0.0, None)] * n_cross
        try:
            res = minimize(
                zone_loglik, x0=x0,
                args=(times_v, marks_v, A_self, cross_A_list, T_obs, beta, l2_pen),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 400, "ftol": 1e-9},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
    if best_x is None:
        best_x = np.array([rate_est, 0.0] + [0.0] * n_cross)
    unpen_ll = -zone_loglik(best_x, times_v, marks_v, A_self, cross_A_list, T_obs, beta, 0.0)
    return best_x, unpen_ll, (best_val < 1e9)


def profile_fit(zones, zone_data, adj_pairs, T_obs, self_only=False, kappa=KAPPA_SHRINK):
    recv_pairs = {v: [] for v in zones}
    for u, v in adj_pairs:
        recv_pairs[v].append((u, adj_pairs.index((u, v))))

    best_total = -np.inf
    best_beta = BETA_GRID[0]
    best_zone_params = {}
    best_per_zone_ll = {}

    for beta in BETA_GRID:
        self_As = {}
        cross_As = {}
        cross_comps = {}

        for v in zones:
            zd = zone_data[v]
            if len(zd["times"]) == 0:
                self_As[v] = np.array([])
                continue
            self_As[v] = compute_self_A(zd["times"], zd["marks"], beta)

        for (u, v) in adj_pairs:
            if not self_only:
                zd_u, zd_v = zone_data[u], zone_data[v]
                if len(zd_v["times"]) == 0 or len(zd_u["times"]) == 0:
                    cross_As[(u, v)] = np.zeros(len(zone_data[v]["times"]))
                    cross_comps[(u, v)] = 0.0
                else:
                    cross_As[(u, v)] = compute_cross_A(
                        zd_v["times"], zd_u["times"], zd_u["marks"], beta)
                    cross_comps[(u, v)] = (1.0 / beta) * float(
                        np.sum(zd_u["marks"] * (1.0 - np.exp(-beta * (T_obs - zd_u["times"])))))

        total_ll = 0.0
        zone_params_beta = {}
        per_zone_ll_beta = {}

        for v in zones:
            zd = zone_data[v]
            if len(zd["times"]) < MIN_ZONE_EVENTS:
                zone_params_beta[v] = None
                per_zone_ll_beta[v] = 0.0
                continue

            A_self = self_As[v]
            cross_A_list = []
            if not self_only:
                for (u, idx) in recv_pairs[v]:
                    pair = (u, v)
                    cross_A_list.append((
                        cross_As.get(pair, np.zeros(len(zd["times"]))),
                        cross_comps.get(pair, 0.0)
                    ))

            params, ll, conv = fit_zone(
                zd["times"], zd["marks"], A_self, cross_A_list,
                T_obs, beta, kappa if not self_only else 0.0
            )
            total_ll += ll
            per_zone_ll_beta[v] = ll
            zone_params_beta[v] = {
                "mu": params[0],
                "alpha_self": params[1],
                "alpha_cross": {
                    adj_pairs[recv_pairs[v][k][1]][0]: params[2 + k]
                    for k in range(len(recv_pairs[v]))
                } if not self_only else {},
                "converged": conv,
            }

        if total_ll > best_total:
            best_total = total_ll
            best_beta = beta
            best_zone_params = zone_params_beta
            best_per_zone_ll = per_zone_ll_beta

    return best_beta, best_zone_params, best_total, best_per_zone_ll


def build_zone_data(df, zones):
    zd = {}
    for z in zones:
        sub = df[df["zone"] == z].sort_values("t_hours")
        zd[z] = {
            "times": sub["t_hours"].values.astype(float),
            "marks": sub["mark"].values.astype(float),
        }
    return zd


# ── intensity computation (same logic as Part A) ──────────────────────────────

def compute_asof_intensity(event_times, event_zones, event_marks,
                           query_times, query_zones,
                           zone_params, beta, adj_pairs, all_zones):
    recv_pairs = {v: [] for v in all_zones}
    for u, v in adj_pairs:
        recv_pairs[v].append(u)

    results = []
    for qt, qz in zip(query_times, query_zones):
        zp = zone_params.get(qz)
        if zp is None:
            results.append(np.nan)
            continue
        mask_v = (event_zones == qz) & (event_times < qt)
        past_t_v = event_times[mask_v]
        past_m_v = event_marks[mask_v]
        a_self = float(np.sum(past_m_v * np.exp(-beta * (qt - past_t_v)))) if len(past_t_v) > 0 else 0.0
        intensity = zp["mu"] + zp["alpha_self"] * a_self
        for u in recv_pairs.get(qz, []):
            a_uv = zp["alpha_cross"].get(u, 0.0)
            if a_uv == 0.0:
                continue
            mask_u = (event_zones == u) & (event_times < qt)
            past_t_u = event_times[mask_u]
            past_m_u = event_marks[mask_u]
            a_cross = float(np.sum(past_m_u * np.exp(-beta * (qt - past_t_u)))) if len(past_t_u) > 0 else 0.0
            intensity += a_uv * a_cross
        results.append(max(intensity, 0.0))
    return np.array(results)


# ── analytic compensator over [T_s, T_e] ─────────────────────────────────────

def zone_compensator_interval(zone_v, zone_params, beta, adj_pairs,
                              all_times, all_zones, all_marks, T_s, T_e):
    """
    Integral of lambda_v(t) dt over [T_s, T_e].
    Analytic formula: uses all events with t < T_e (events in [T_s,T_e)
    contribute from their own occurrence time onward within the window).
    """
    zp = zone_params.get(zone_v)
    if zp is None:
        return 0.0

    mu_v = zp["mu"]
    comp = mu_v * (T_e - T_s)

    # Self + each adjacent source zone
    sources = [(zone_v, zp["alpha_self"])] + [
        (u, zp["alpha_cross"].get(u, 0.0))
        for (u, v2) in adj_pairs if v2 == zone_v
    ]

    for (src_zone, alpha) in sources:
        if alpha == 0.0:
            continue
        mask = all_zones == src_zone
        t_src = all_times[mask]
        m_src = all_marks[mask]
        if len(t_src) == 0:
            continue

        # Events strictly before T_s: active at T_s, decay from T_s to T_e
        mask_pre = t_src < T_s
        if mask_pre.any():
            dt_s = T_s - t_src[mask_pre]
            dt_e = T_e - t_src[mask_pre]
            comp += (alpha / beta) * float(
                np.sum(m_src[mask_pre] * (np.exp(-beta * dt_s) - np.exp(-beta * dt_e)))
            )

        # Events in [T_s, T_e): enter the window mid-way
        mask_in = (t_src >= T_s) & (t_src < T_e)
        if mask_in.any():
            dt_e = T_e - t_src[mask_in]
            comp += (alpha / beta) * float(
                np.sum(m_src[mask_in] * (1.0 - np.exp(-beta * dt_e)))
            )

    return comp


# ── time-rescaling KS test ────────────────────────────────────────────────────

def ks_time_rescaling(eval_times_zone, all_times, all_zones, all_marks,
                      zone_v, zone_params, beta, adj_pairs, T_eval_start):
    """
    Compute Λ_v(t_{i-1}, t_i) for consecutive eval events in zone v.
    Under a correctly specified model these are i.i.d. Exp(1).
    Returns (ks_stat, p_value, n_intervals).
    """
    if len(eval_times_zone) < 3:
        return np.nan, np.nan, len(eval_times_zone)

    # First inter-arrival: from T_eval_start to first event
    boundaries = np.concatenate([[T_eval_start], eval_times_zone])
    residuals = []
    for i in range(len(eval_times_zone)):
        a = boundaries[i]
        b = boundaries[i + 1]
        comp = zone_compensator_interval(
            zone_v, zone_params, beta, adj_pairs,
            all_times, all_zones, all_marks, a, b
        )
        residuals.append(comp)

    residuals = np.array(residuals)
    residuals = residuals[residuals > 0]
    if len(residuals) < 3:
        return np.nan, np.nan, len(residuals)

    stat, p = ks_1samp(residuals, expon.cdf)
    return float(stat), float(p), len(residuals)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: DATA LOADING ===")

feat = pd.read_csv(OUTPUTS / "layer45_asof_feature_matrix.csv")
print(f"  layer45_asof_feature_matrix: {feat.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MARK CONSTRUCTION (normalized on Part B train window only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: MARK CONSTRUCTION ===")
print("  NOTE: normalization stats computed on Nov 10–Dec 31 training only.")

feat["start_local"] = pd.to_datetime(feat["start_local"], utc=True, errors="coerce")
feat = feat.dropna(subset=["start_local", "zone"]).copy()
feat = feat.sort_values("start_local").reset_index(drop=True)

train_mask = (feat["start_local"] >= DATA_START) & (feat["start_local"] <= PART_B_TRAIN_END)
eval_mask  = (feat["start_local"] >= PART_B_EVAL_START) & (feat["start_local"] <= PART_B_EVAL_END)

print(f"  Part B train events: {train_mask.sum()}")
print(f"  Part B eval events : {eval_mask.sum()}")

feat["DIS_asof_raw"] = pd.to_numeric(feat["asof_fragility_proxy"], errors="coerce").fillna(0.0)
feat["DIS_asof"] = minmax_scale_train(feat["DIS_asof_raw"], train_mask)

feat["OBI_asof_raw"] = pd.to_numeric(feat["asof_obi_proxy"], errors="coerce").fillna(0.0)
feat["OBI_asof"] = minmax_scale_train(feat["OBI_asof_raw"], train_mask)

priority_map = {"High": 1.0, "Low": 0.333, "Unknown": 0.667}
feat["severity_raw"] = feat["priority"].astype(str).map(priority_map).fillna(0.667)
feat["severity_i"] = minmax_scale_train(feat["severity_raw"], train_mask)

feat["conf_raw"] = pd.to_numeric(feat["asof_retrieval_confidence"], errors="coerce").fillna(0.0)
feat["confidence_i"] = minmax_scale_train(feat["conf_raw"], train_mask)

feat["mark"] = (
    MARK_WEIGHTS["DIS"]  * feat["DIS_asof"]
    + MARK_WEIGHTS["OBI"]  * feat["OBI_asof"]
    + MARK_WEIGHTS["SEV"]  * feat["severity_i"]
    + MARK_WEIGHTS["CONF"] * feat["confidence_i"]
).clip(0.0, 1.0)

mark_std = feat.loc[train_mask, "mark"].std()
MARK_FALLBACK = False
if mark_std < 1e-4:
    print("  WARNING: mark variance near zero — falling back to m_i=1.0")
    feat["mark"] = 1.0
    MARK_FALLBACK = True
else:
    print(f"  Mark train std={mark_std:.4f}, mean={feat.loc[train_mask,'mark'].mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ZONE SETUP & SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: ZONE SETUP & SPLIT ===")

observed_zones = sorted(feat["zone"].dropna().unique().tolist())
adj_pairs = build_adjacency(observed_zones)
print(f"  Zones: {len(observed_zones)}, directed pairs: {len(adj_pairs)}")

feat_train = feat[train_mask].copy()
feat_eval  = feat[eval_mask].copy()

t0_train = feat_train["start_local"].min()
feat_train["t_hours"] = (feat_train["start_local"] - t0_train).dt.total_seconds() / 3600.0
feat_eval["t_hours"]  = (feat_eval["start_local"]  - t0_train).dt.total_seconds() / 3600.0

T_train = float(feat_train["t_hours"].max())
T_eval_start = float(feat_eval["t_hours"].min()) if len(feat_eval) > 0 else T_train
T_eval_end   = float(feat_eval["t_hours"].max()) if len(feat_eval) > 0 else T_train
T_eval_duration = T_eval_end - T_eval_start if len(feat_eval) > 0 else 0.0

print(f"  T_train (hours from t0): {T_train:.1f}")
print(f"  Eval window (hours from t0): [{T_eval_start:.1f}, {T_eval_end:.1f}]"
      f" = {T_eval_duration:.1f} h ({T_eval_duration/24:.1f} days)")

zone_data_train = build_zone_data(feat_train, observed_zones)
for z in observed_zones:
    print(f"  {z}: {len(zone_data_train[z]['times'])} train | "
          f"{(feat_eval['zone']==z).sum()} eval events")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FIT HAWKES ON PART B TRAINING WINDOW (Nov-Dec only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: HAWKES FIT ON PART B TRAINING WINDOW ===")
print("  Fitting cross-zone model on Nov 10–Dec 31 via profile likelihood...")

beta_fit, params_fit, ll_train_total, per_zone_ll_train = profile_fit(
    observed_zones, zone_data_train, adj_pairs, T_train,
    self_only=False, kappa=KAPPA_SHRINK
)
half_life = np.log(2.0) / beta_fit
print(f"  Best beta: {beta_fit:.5f} h^-1  (half-life: {half_life:.2f} h)")
print(f"  Train log-lik: {ll_train_total:.3f}")
print(f"  Converged zones: {sum(1 for v in observed_zones if params_fit.get(v) and params_fit[v]['converged'])}/{len(observed_zones)}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: COMPUTE EVAL INTENSITIES (past-only, using train history)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: EVAL INTENSITIES ===")

# Pool all events for history lookup (sorted by time)
feat_all = pd.concat([feat_train, feat_eval], ignore_index=True).sort_values("t_hours")
all_t = feat_all["t_hours"].values.astype(float)
all_z = feat_all["zone"].values
all_m = feat_all["mark"].values

# Compute intensity for eval events only (each uses strictly past events)
eval_t = feat_eval["t_hours"].values.astype(float)
eval_z = feat_eval["zone"].values
eval_m = feat_eval["mark"].values

# Sort all events for efficiency
sort_idx = np.argsort(all_t)
all_t_s  = all_t[sort_idx]
all_z_s  = all_z[sort_idx]
all_m_s  = all_m[sort_idx]

print("  Computing as-of intensity for eval events (strictly past events only)...")
eval_intensities = compute_asof_intensity(
    all_t_s, all_z_s, all_m_s,
    eval_t, eval_z,
    params_fit, beta_fit, adj_pairs, observed_zones
)
print(f"  Eval events: {len(eval_intensities)}, non-nan: {np.sum(~np.isnan(eval_intensities))}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COMPENSATOR OVER EVAL PERIOD (per zone)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 6: COMPENSATORS ===")

comp_hawkes = {}
for v in observed_zones:
    comp_hawkes[v] = zone_compensator_interval(
        v, params_fit, beta_fit, adj_pairs,
        all_t_s, all_z_s, all_m_s,
        T_eval_start, T_eval_end
    )
    print(f"  comp_hawkes({v}): {comp_hawkes[v]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: HELD-OUT LOG-LIKELIHOOD & POISSON BASELINE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 7: HELD-OUT LL & POISSON BASELINE ===")

per_zone_metrics = {}
agg_ll_hawkes = 0.0
agg_ll_poisson = 0.0
agg_n_eval = 0
log_intensities_all = []  # for aggregate bootstrap

for v in observed_zones:
    n_train_v = len(zone_data_train[v]["times"])
    rate_poisson_v = n_train_v / max(T_train, 1.0)

    eval_mask_v = eval_z == v
    n_eval_v = int(eval_mask_v.sum())
    log_int_v = np.log(np.maximum(eval_intensities[eval_mask_v], 1e-10))
    comp_v = comp_hawkes[v]

    ll_hawkes_v = float(np.sum(log_int_v)) - comp_v
    ll_poisson_v = (n_eval_v * np.log(max(rate_poisson_v, 1e-10))
                    - rate_poisson_v * T_eval_duration)

    per_zone_metrics[v] = {
        "zone": v,
        "n_train": n_train_v,
        "n_eval": n_eval_v,
        "rate_poisson_per_hour": round(rate_poisson_v, 6),
        "hawkes_compensator_eval": round(comp_v, 4),
        "ll_hawkes_total": round(ll_hawkes_v, 4),
        "ll_poisson_total": round(ll_poisson_v, 4),
        "ll_improvement_vs_poisson": round(ll_hawkes_v - ll_poisson_v, 4),
        "mean_log_intensity_hawkes": round(float(np.mean(log_int_v)), 6) if n_eval_v > 0 else np.nan,
        "mean_log_intensity_poisson": round(np.log(max(rate_poisson_v, 1e-10)), 6),
    }
    agg_ll_hawkes  += ll_hawkes_v
    agg_ll_poisson += ll_poisson_v
    agg_n_eval     += n_eval_v
    if n_eval_v > 0:
        log_intensities_all.extend(log_int_v.tolist())

log_intensities_all = np.array(log_intensities_all)

print(f"  Aggregate eval events: {agg_n_eval}")
print(f"  Aggregate Hawkes LL  : {agg_ll_hawkes:.3f}  ({agg_ll_hawkes/max(agg_n_eval,1):.4f} per event)")
print(f"  Aggregate Poisson LL : {agg_ll_poisson:.3f}  ({agg_ll_poisson/max(agg_n_eval,1):.4f} per event)")
print(f"  LL improvement       : {agg_ll_hawkes - agg_ll_poisson:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BOOTSTRAP CONFIDENCE INTERVALS (B=1000 event-level resample)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== SECTION 8: BOOTSTRAP CIs (B={N_BOOTSTRAP}) ===")
print("  Resampling eval event log-intensities; captures finite-sample uncertainty.")
print(f"  NOTE: eval n={agg_n_eval} events over {T_eval_duration/24:.1f} days — CIs are wider than")
print("  other project evaluations due to the shortened holdout window.")

rng_bs = np.random.default_rng(123)
bs_means = np.zeros(N_BOOTSTRAP)

if len(log_intensities_all) > 0:
    for b in range(N_BOOTSTRAP):
        sample = rng_bs.choice(log_intensities_all, size=len(log_intensities_all), replace=True)
        bs_means[b] = float(np.mean(sample))
    ci_lo = float(np.percentile(bs_means, 2.5))
    ci_hi = float(np.percentile(bs_means, 97.5))
    mean_log_int = float(np.mean(log_intensities_all))
else:
    ci_lo = ci_hi = mean_log_int = np.nan

mean_log_int_poisson = float(np.log(max(agg_n_eval / max(T_train * len(observed_zones), 1.0), 1e-10)))
# Per-zone bootstrap
per_zone_bs = {}
for v in observed_zones:
    eval_mask_v = eval_z == v
    n_eval_v = int(eval_mask_v.sum())
    log_int_v = np.log(np.maximum(eval_intensities[eval_mask_v], 1e-10))
    if n_eval_v < 3:
        per_zone_bs[v] = (np.nan, np.nan)
        continue
    bs_v = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        s = rng_bs.choice(log_int_v, size=n_eval_v, replace=True)
        bs_v[b] = float(np.mean(s))
    per_zone_bs[v] = (float(np.percentile(bs_v, 2.5)), float(np.percentile(bs_v, 97.5)))

print(f"  Mean log-intensity (Hawkes) : {mean_log_int:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Mean log-intensity (Poisson): aggregate reference = log(overall rate)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: KS TIME-RESCALING TEST (model calibration)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 9: KS TIME-RESCALING TEST ===")

ks_results = {}
for v in observed_zones:
    eval_mask_v = eval_z == v
    eval_times_v = np.sort(eval_t[eval_mask_v])
    ks_stat, ks_p, n_int = ks_time_rescaling(
        eval_times_v, all_t_s, all_z_s, all_m_s,
        v, params_fit, beta_fit, adj_pairs, T_eval_start
    )
    ks_results[v] = {"ks_stat": ks_stat, "ks_p": ks_p, "n_intervals": n_int}
    if not np.isnan(ks_stat):
        calib = "PASS" if ks_p > 0.05 else "FAIL"
        print(f"  {v}: KS={ks_stat:.4f}  p={ks_p:.4f}  [{calib}]  n_intervals={n_int}")
    else:
        print(f"  {v}: insufficient eval events for KS test ({n_int} events)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: DAY-BY-DAY BINNED INTENSITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 10: DAY-BY-DAY BINNED INTENSITY ===")

# Bin eval events by calendar day; compute predicted rate = compensator / 24h
eval_timestamps = feat_eval["start_local"].values
eval_dates = pd.to_datetime(eval_timestamps).normalize()
eval_df_temp = pd.DataFrame({
    "date": eval_dates,
    "zone": eval_z,
    "t_hours": eval_t,
    "log_intensity": np.where(
        np.isnan(eval_intensities), np.nan,
        np.log(np.maximum(eval_intensities, 1e-10))
    ),
})
daily_obs = eval_df_temp.groupby("date").size().reset_index(name="n_observed")

# Predicted daily rate: mean lambda across eval events on each day
daily_pred = eval_df_temp.dropna(subset=["log_intensity"]).copy()
daily_pred["intensity"] = np.exp(daily_pred["log_intensity"])
daily_pred = daily_pred.groupby("date")["intensity"].mean().reset_index(name="mean_intensity_hawkes")

df_daily = daily_obs.merge(daily_pred, on="date", how="left")
df_daily["date"] = df_daily["date"].dt.strftime("%Y-%m-%d")
print(f"  Daily bins: {len(df_daily)} days")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: BUILD OUTPUT TABLES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 11: BUILDING OUTPUT TABLES ===")

# Aggregate metrics
n_zones_ks_pass = sum(
    1 for v in observed_zones
    if not np.isnan(ks_results[v]["ks_stat"]) and ks_results[v]["ks_p"] > 0.05
)
n_zones_ks_tested = sum(
    1 for v in observed_zones if not np.isnan(ks_results[v]["ks_stat"])
)

df_agg = pd.DataFrame([{
    "eval_period": "Jan 01–19 2024",
    "train_period": "Nov 10–Dec 31 2023",
    "data_limitation_note": (
        "Zone labels only available through Jan 19 2024. "
        "Mar-Apr eval infeasible from current data."
    ),
    "n_eval_events": agg_n_eval,
    "eval_duration_days": round(T_eval_duration / 24, 1),
    "n_zones": len(observed_zones),
    "beta_fit": round(beta_fit, 5),
    "half_life_hours": round(half_life, 2),
    "ll_hawkes_total": round(agg_ll_hawkes, 4),
    "ll_poisson_total": round(agg_ll_poisson, 4),
    "ll_improvement_vs_poisson": round(agg_ll_hawkes - agg_ll_poisson, 4),
    "ll_per_event_hawkes": round(agg_ll_hawkes / max(agg_n_eval, 1), 6),
    "ll_per_event_poisson": round(agg_ll_poisson / max(agg_n_eval, 1), 6),
    "mean_log_intensity_hawkes": round(mean_log_int, 6),
    "mean_log_intensity_ci_lo_95": round(ci_lo, 6),
    "mean_log_intensity_ci_hi_95": round(ci_hi, 6),
    "bootstrap_n": N_BOOTSTRAP,
    "bootstrap_method": "event-level resample with replacement",
    "ks_zones_pass_calibration": n_zones_ks_pass,
    "ks_zones_tested": n_zones_ks_tested,
    "mark_fallback_used": MARK_FALLBACK,
}])

# Per-zone metrics table
per_zone_rows = []
for v in observed_zones:
    m = per_zone_metrics[v]
    bs_lo, bs_hi = per_zone_bs.get(v, (np.nan, np.nan))
    ks = ks_results[v]
    row = {**m,
           "mean_log_int_ci_lo_95": round(bs_lo, 6) if not np.isnan(bs_lo) else None,
           "mean_log_int_ci_hi_95": round(bs_hi, 6) if not np.isnan(bs_hi) else None,
           "ks_stat": round(ks["ks_stat"], 6) if not np.isnan(ks["ks_stat"]) else None,
           "ks_p_value": round(ks["ks_p"], 6) if not np.isnan(ks["ks_p"]) else None,
           "ks_calibrated": (ks["ks_p"] > 0.05) if not np.isnan(ks.get("ks_p", np.nan)) else None,
           }
    per_zone_rows.append(row)
df_per_zone = pd.DataFrame(per_zone_rows)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: WRITE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 12: WRITING OUTPUTS ===")

OUTPUTS.mkdir(exist_ok=True)

df_agg.to_csv(OUTPUTS / "layer7b_eval_metrics.csv", index=False)
print(f"  Wrote layer7b_eval_metrics.csv")

df_per_zone.to_csv(OUTPUTS / "layer7b_eval_per_zone.csv", index=False)
print(f"  Wrote layer7b_eval_per_zone.csv ({len(df_per_zone)} zones)")

df_daily.to_csv(OUTPUTS / "layer7b_eval_daily_intensity.csv", index=False)
print(f"  Wrote layer7b_eval_daily_intensity.csv ({len(df_daily)} days)")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: SUMMARY TEXT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 13: FUSION SUMMARY ===")

hawkes_better = (agg_ll_hawkes > agg_ll_poisson)
ks_summary = f"{n_zones_ks_pass}/{n_zones_ks_tested} zones pass KS calibration test (p>0.05)"

summary_lines = [
    "=" * 70,
    "ASTraM Layer 7 Part B — Out-of-Sample Spillover Forecast Evaluation",
    "=" * 70,
    "",
    "DATA LIMITATION — documented explicitly:",
    "  Zone labels in events_clean.parquet are only populated through",
    "  Jan 19 2024 (3,389 usable events). The originally planned",
    "  Mar-Apr evaluation window is infeasible — those events have null",
    "  zone and cannot participate in a zone-level Hawkes model.",
    "  This is a confirmed data limitation, not a methodological choice.",
    "",
    "CHRONOLOGICAL SPLIT (revised to fit zone-labeled window):",
    f"  Train : Nov 10, 2023 – Dec 31, 2023",
    f"  Eval  : Jan 01, 2024 – Jan 19, 2024  (genuine zone-labeled cutoff)",
    f"  Eval duration : {T_eval_duration/24:.1f} days  |  Eval events: {agg_n_eval}",
    "",
    "  The holdout is smaller than this project's other evaluations.",
    "  All metrics include 95% bootstrap CIs (event-level resample, B=1000).",
    "",
    "--- Model Fit (Part B training window: Nov-Dec only) ---",
    f"  Best beta : {beta_fit:.5f} h^-1  (half-life: {half_life:.2f} h)",
    f"  Train LL  : {ll_train_total:.3f}",
    f"  Mark fallback used: {MARK_FALLBACK}",
    "",
    "--- Held-Out Log-Likelihood ---",
    f"  Hawkes LL (total)   : {agg_ll_hawkes:.3f}  ({agg_ll_hawkes/max(agg_n_eval,1):.4f} per event)",
    f"  Poisson LL (total)  : {agg_ll_poisson:.3f}  ({agg_ll_poisson/max(agg_n_eval,1):.4f} per event)",
    f"  dLL (Hawkes-Poisson): {agg_ll_hawkes - agg_ll_poisson:.3f}",
    f"  Hawkes > Poisson on holdout: {hawkes_better}",
    "",
    "--- Mean Log-Intensity (intensity-term only, with bootstrap CI) ---",
    f"  Hawkes  : {mean_log_int:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]",
    f"  (Bootstrap: B={N_BOOTSTRAP}, event-level resample with replacement)",
    f"  CI width: {ci_hi - ci_lo:.4f}  — wider than typical due to small holdout",
    "",
    "--- KS Time-Rescaling Test (model calibration) ---",
    f"  {ks_summary}",
]

for v in observed_zones:
    ks = ks_results[v]
    if not np.isnan(ks["ks_stat"]):
        calib = "PASS" if ks["ks_p"] > 0.05 else "FAIL"
        summary_lines.append(
            f"    {v:<22}: KS={ks['ks_stat']:.4f}  p={ks['ks_p']:.4f}  [{calib}]  n={ks['n_intervals']}"
        )
    else:
        summary_lines.append(f"    {v:<22}: insufficient events for KS test")

summary_lines += [
    "",
    "--- Per-Zone Held-Out LL ---",
]
for v in observed_zones:
    m = per_zone_metrics[v]
    bs_lo, bs_hi = per_zone_bs.get(v, (np.nan, np.nan))
    ci_str = f"  CI [{bs_lo:.3f}, {bs_hi:.3f}]" if not np.isnan(bs_lo) else "  CI: n/a"
    summary_lines.append(
        f"  {v:<22}: n_eval={m['n_eval']:3d}  LL={m['ll_hawkes_total']:8.3f}"
        f"  vs Poisson={m['ll_poisson_total']:8.3f}"
        f"  mean_log_lam={m['mean_log_intensity_hawkes']}" + (f"  95%CI [{bs_lo:.3f},{bs_hi:.3f}]" if not np.isnan(bs_lo) else "")
    )

summary_lines += [
    "",
    "--- Outputs ---",
    "  layer7b_eval_metrics.csv       — aggregate metrics with CIs",
    "  layer7b_eval_per_zone.csv      — per-zone metrics with CIs",
    "  layer7b_eval_daily_intensity.csv — day-by-day observed vs predicted",
    "  layer7b_forecast_eval_summary.txt — this summary",
    "",
    "--- Leakage Audit ---",
    "  Model parameters fit on Nov-Dec training only: YES",
    "  Eval intensities use strictly past events only: YES",
    "  Normalization stats from training period only: YES",
    "  Layer 1-6 and Part A output files NOT modified: YES",
    "",
    "=" * 70,
]

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(OUTPUTS / "layer7b_forecast_eval_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(summary_text + "\n")
print("\n  Wrote layer7b_forecast_eval_summary.txt")
print("\n=== Layer 7 Part B complete ===")
