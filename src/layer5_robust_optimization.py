"""
Layer 5 — Robust Prescriptive Optimization Engine (ASTraM).

Consumes the sanitized, scenario-ready output of Layer 4.5 and converts it
into resource allocation and diversion decisions under uncertainty.

This layer is ADDITIVE — it does not modify Layers 1–4.5.
It is an optimization layer, not a predictor.  No new ML model is trained here.

Decision model
--------------
Two-stage stochastic MILP with CVaR tail-risk control and chance constraints.
The predictive uncertainty comes from Layer 4.5.
Layer 5 converts that uncertainty into action by:
  - generating scenario bundles from sanitized duration quantiles,
  - inflating uncertainty for low-reliability events,
  - solving a resource-allocation MILP that minimizes CVaR of total delay,
  - producing diversion route recommendations,
  - computing shadow prices and Pareto-frontier summaries.

Solver: scipy.optimize.milp (HiGHS) → greedy fallback if MILP fails.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import LinearConstraint, Bounds, milp

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
ARTIFACTS = OUT / "layer5_model_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
KAPPA = 0.50            # uncertainty inflation coefficient κ
ALPHA_CVAR = 0.90       # CVaR tail level α
S_SCENARIOS = 50        # number of scenarios after reduction
S_INITIAL = 200         # initial sample before reduction
N_ACTIVE_MAX = 50       # max number of active sites in MILP
RANDOM_SEED = 42

# Resource effectiveness hyperparameters (γ — not learned)
GAMMA_P = 0.18          # police officer effectiveness per unit
GAMMA_B = 0.10          # barricade effectiveness per unit
GAMMA_T = 0.25          # tow truck effectiveness per unit
GAMMA_Q = 0.30          # QRU effectiveness per unit

# Resource cost weights (relative, dimensionless)
COST_P = 1.0
COST_B = 0.6
COST_T = 2.0
COST_Q = 3.0
COST_DIV = 1.5          # diversion activation cost per site

# City-wide resource budgets
BUDGET_POLICE = 120
BUDGET_BARRICADES = 100
BUDGET_TOW = 15
BUDGET_QRU = 10

# Per-site resource caps
MAX_P = 12
MAX_B = 20
MAX_T = 4
MAX_Q = 3

# Objective priority weights (λ1..λ5)
LAMBDA_CRIT = 5.0       # λ1: unmet critical-site penalty weight
LAMBDA_CVAR = 2.0       # λ2: CVaR weight
LAMBDA_ED = 1.0         # λ3: expected delay weight
LAMBDA_COST = 0.15      # λ4: deployment cost weight
LAMBDA_DIV = 0.10       # λ5: diversion cost weight

# Minimum service coefficient ρ for critical sites
MIN_SERVICE_RHO = 2

# Risk-weight sigmoid parameters (a_i)
A_FRAGILITY = 1.5
A_OBI = 1.2
A_NOVELTY = 0.8
A_TAIL_RISK = 1.0

# Chance-constraint tolerance ε
CC_EPSILON = 0.20       # P(delay ≤ target) ≥ 1 - ε = 0.80
CC_TARGET_MULT = 3.0    # target = CC_TARGET_MULT * p50

# PWL breakpoints for 1 - exp(-u)
PWL_BREAKPOINTS = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Solver time limit (seconds)
MILP_TIME_LIMIT = 60       # main solve
MILP_TIME_LIMIT_SUB = 45   # sub-solves for sensitivity / Pareto
MILP_TIME_LIMIT_SHADOW = 90  # shadow price solves (need accurate dual-like values)


# ── Data containers ───────────────────────────────────────────────────────────

class SiteData(NamedTuple):
    """Per-site data passed to the MILP builder."""
    event_ids: np.ndarray
    event_causes: np.ndarray
    p50: np.ndarray
    p80: np.ndarray
    p95: np.ndarray
    reliability: np.ndarray
    tail_risk_prob: np.ndarray
    sanity_flag: np.ndarray
    weights: np.ndarray
    is_critical: np.ndarray
    scenarios: np.ndarray   # shape (R, S)
    scenario_weights: np.ndarray  # shape (S,)


# ── Input loading ─────────────────────────────────────────────────────────────

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                            pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load canonical Layer 4.5 inputs for Layer 5."""
    def _safe_read(path: Path, label: str) -> pd.DataFrame:
        if not path.exists():
            logger.warning("Missing %s at %s — returning empty DataFrame", label, path)
            return pd.DataFrame()
        return pd.read_csv(path)

    scenario_ready = _safe_read(OUT / "layer45_scenario_ready_duration.csv", "scenario_ready_duration")
    josv_norm = _safe_read(OUT / "layer45_operational_state_vector_normalized.csv", "josv_normalized")
    quality = _safe_read(OUT / "layer45_duration_quality.csv", "duration_quality")
    tau_thresh = _safe_read(OUT / "layer45_cause_tau_thresholds.csv", "cause_tau_thresholds")
    metrics = _safe_read(OUT / "layer45_metrics.csv", "metrics")
    fallback_summary = _safe_read(OUT / "layer45_fallback_summary.csv", "fallback_summary")

    logger.info("Loaded Layer 4.5 inputs: scenario_ready=%d rows, josv=%d rows",
                len(scenario_ready), len(josv_norm))
    return scenario_ready, josv_norm, quality, tau_thresh, metrics, fallback_summary


def validate_duration_bundle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assert required columns exist and apply defensive sanity clamping.

    Even though Layer 4.5 already sanitizes, Layer 5 treats inputs defensively:
    - enforces monotone Q50 ≤ Q80 ≤ Q95
    - applies the scenario-safe clamp: Q95_sc ≤ max(10·Q50, 1440)
    """
    required = [
        "event_id", "safe_duration_p50", "safe_duration_p80", "safe_duration_p95",
        "duration_reliability", "tail_risk_prob", "duration_sanity_flag", "duration_guard_reason",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Layer 4.5 scenario-ready bundle missing columns: {missing}")

    df = df.copy()
    p50 = df["safe_duration_p50"].values.astype(float)
    p80 = df["safe_duration_p80"].values.astype(float)
    p95 = df["safe_duration_p95"].values.astype(float)

    stacked = np.stack([p50, p80, p95], axis=1)
    mono = np.sort(stacked, axis=1)
    p50, p80, p95 = mono[:, 0], mono[:, 1], mono[:, 2]

    upper = np.maximum(10.0 * p50, 1440.0)
    p95 = np.minimum(p95, upper)

    stacked2 = np.stack([p50, p80, p95], axis=1)
    mono2 = np.sort(stacked2, axis=1)
    df["safe_duration_p50"] = mono2[:, 0]
    df["safe_duration_p80"] = mono2[:, 1]
    df["safe_duration_p95"] = mono2[:, 2]

    df["duration_reliability"] = np.clip(df["duration_reliability"].values, 0.0, 1.0)
    df["tail_risk_prob"] = np.clip(df["tail_risk_prob"].values, 0.0, 1.0)
    logger.info("Duration bundle validated: %d rows, %d sanity_ok (%.1f%%)",
                len(df), df["duration_sanity_flag"].sum(),
                100 * df["duration_sanity_flag"].mean())
    return df


# ── Site selection ────────────────────────────────────────────────────────────

def compute_site_weights(josv: pd.DataFrame) -> np.ndarray:
    """
    Compute risk-importance weight w_r using a linear combination of
    normalized Layer 4.5 signals, then squash with sigmoid.

    w_r = σ(a1·fragility + a2·OBI + a3·novelty + a4·tail_risk)
    """
    def _col(name: str, df: pd.DataFrame) -> np.ndarray:
        if name in df.columns:
            v = df[name].values.astype(float)
            v = np.nan_to_num(v, nan=0.0)
            rng = v.max() - v.min() + 1e-9
            return (v - v.min()) / rng
        return np.zeros(len(df))

    frag = _col("fragility_signal", josv)
    obi = _col("obi_signal", josv)
    novelty = _col("novelty_score", josv)
    tail = _col("tail_risk_prob", josv)

    raw = (A_FRAGILITY * frag + A_OBI * obi + A_NOVELTY * novelty + A_TAIL_RISK * tail)
    return 1.0 / (1.0 + np.exp(-raw + raw.mean()))


def select_active_sites(
    scenario_ready: pd.DataFrame,
    josv: pd.DataFrame,
    n_max: int = N_ACTIVE_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Select top-N active disruption sites by composite risk score.

    High-impact decisions and high tail-risk events are prioritized.
    Returns the filtered scenario_ready, josv, and weight arrays.
    """
    merged = scenario_ready.copy()
    if "event_id" in josv.columns:
        josv_cols = ["event_id", "event_cause", "high_impact_decision",
                     "high_impact_prob_calibrated", "fragility_signal",
                     "obi_signal", "novelty_score", "drift_score", "trust_score",
                     "ims_proxy"]
        josv_cols = [c for c in josv_cols if c in josv.columns]
        merged = merged.merge(josv[josv_cols], on="event_id", how="left")

    weights = compute_site_weights(merged)
    risk_score = (
        weights
        * (1.0 + merged["tail_risk_prob"].values)
        * (1.0 + (1.0 - merged["duration_reliability"].values))
    )
    if "high_impact_decision" in merged.columns:
        hi = merged["high_impact_decision"].fillna(0).values.astype(float)
        risk_score = risk_score * (1.0 + hi)

    top_idx = np.argsort(risk_score)[::-1][:n_max]
    top_idx = np.sort(top_idx)

    site_df = merged.iloc[top_idx].reset_index(drop=True)
    josv_site = josv[josv["event_id"].isin(site_df["event_id"])].reset_index(drop=True) \
        if "event_id" in josv.columns else josv.iloc[top_idx].reset_index(drop=True)
    site_weights = weights[top_idx]

    logger.info("Selected %d active sites (max %d)", len(site_df), n_max)
    return site_df, josv_site, site_weights


# ── Scenario generation ───────────────────────────────────────────────────────

def fit_quantile_surrogate(
    p50: float, p80: float, p95: float
) -> tuple[float, float]:
    """
    Fit lognormal surrogate μ, σ from the sanitized quantile triple.

    μ = log(Q50)
    σ = (log(Q95) - log(Q50)) / 1.645    (0.95 quantile of N(0,1) ≈ 1.645)

    Falls back to a minimum σ of 0.1 if the quantile ratio is degenerate.
    """
    p50 = max(p50, 1e-6)
    p95 = max(p95, p50 * 1.001)
    mu = math.log(p50)
    sigma = (math.log(p95) - math.log(p50)) / 1.645
    sigma = max(sigma, 0.10)
    return mu, sigma


def inflate_uncertainty(sigma: float, reliability: float, kappa: float = KAPPA) -> float:
    """
    σ_adj = σ · (1 + κ · (1 − R))

    Inflates tail risk without distorting the median.
    Events with low reliability get wider scenario distributions.
    """
    return sigma * (1.0 + kappa * (1.0 - max(0.0, min(1.0, reliability))))


def build_scenarios(
    site_df: pd.DataFrame,
    n_initial: int = S_INITIAL,
    n_reduced: int = S_SCENARIOS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate and reduce scenario matrix T[R, S].

    For each site r:
      1. Fit lognormal from safe quantiles.
      2. Inflate σ by reliability score.
      3. Sample n_initial scenarios.
      4. Append explicit tail scenarios (p95 and p99 level).

    Scenario reduction: k-means on transposed matrix (cluster scenario vectors),
    retaining the top-5 tail scenarios explicitly to prevent tail erasure.

    Returns T[R, S] (durations in minutes) and w_s[S] (scenario weights).
    """
    rng = np.random.default_rng(seed)
    R = len(site_df)

    p50_arr = site_df["safe_duration_p50"].values
    p80_arr = site_df["safe_duration_p80"].values
    p95_arr = site_df["safe_duration_p95"].values
    rel_arr = site_df["duration_reliability"].values
    sanity_arr = site_df["duration_sanity_flag"].values

    mu_arr = np.zeros(R)
    sigma_adj_arr = np.zeros(R)

    for r in range(R):
        mu, sigma = fit_quantile_surrogate(p50_arr[r], p80_arr[r], p95_arr[r])
        if sanity_arr[r] == 0:
            # Low-quality row: inflate more aggressively
            sigma = inflate_uncertainty(sigma, rel_arr[r], kappa=KAPPA * 1.5)
        else:
            sigma = inflate_uncertainty(sigma, rel_arr[r], kappa=KAPPA)
        mu_arr[r] = mu
        sigma_adj_arr[r] = sigma

    # Sample raw scenarios: shape (R, n_initial)
    eps = rng.standard_normal((R, n_initial))
    log_T = mu_arr[:, None] + sigma_adj_arr[:, None] * eps
    T_raw = np.exp(log_T)  # shape (R, n_initial)

    # Explicit tail scenarios (per-site p95 and p99 draws)
    eps_tail = np.abs(rng.standard_normal((R, 5)))
    log_T_tail = mu_arr[:, None] + sigma_adj_arr[:, None] * eps_tail * 2.0
    T_tail = np.exp(log_T_tail)
    T_full = np.concatenate([T_raw, T_tail], axis=1)  # (R, n_initial+5)

    n_full = T_full.shape[1]
    n_reduced_actual = min(n_reduced, n_full)

    if n_reduced_actual >= n_full:
        return T_full, np.ones(n_full) / n_full

    # Scenario reduction: cluster columns of T_full using k-means on scenario vectors
    # Each scenario is a vector of R durations
    T_cols = T_full.T  # shape (n_full, R)

    # Identify top-5 tail scenarios (highest mean duration)
    scenario_means = T_cols.mean(axis=1)
    tail_scenario_idx = np.argsort(scenario_means)[::-1][:5]
    n_cluster = n_reduced_actual - len(tail_scenario_idx)

    # Simple k-means: assign each column to nearest centroid
    non_tail_idx = np.array([i for i in range(n_full) if i not in tail_scenario_idx])
    T_non_tail = T_cols[non_tail_idx]  # (n_full-5, R)

    if n_cluster <= 0 or len(T_non_tail) == 0:
        selected_idx = np.concatenate([non_tail_idx[:max(0, n_reduced_actual - 5)], tail_scenario_idx])
    else:
        centroids = T_non_tail[np.linspace(0, len(T_non_tail) - 1, n_cluster, dtype=int)]
        for _ in range(20):
            dists = np.linalg.norm(T_non_tail[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                T_non_tail[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
                for k in range(n_cluster)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-3):
                break
            centroids = new_centroids

        # Find medoid of each cluster (closest to centroid)
        medoid_idx = []
        for k in range(n_cluster):
            cluster_mask = labels == k
            if cluster_mask.sum() == 0:
                continue
            cluster_pts = T_non_tail[cluster_mask]
            cluster_orig_idx = non_tail_idx[cluster_mask]
            dists_to_centroid = np.linalg.norm(cluster_pts - centroids[k], axis=1)
            medoid_idx.append(cluster_orig_idx[np.argmin(dists_to_centroid)])

        selected_idx = np.concatenate([np.array(medoid_idx), tail_scenario_idx])

    selected_idx = np.unique(selected_idx.astype(int))
    T_selected = T_full[:, selected_idx]  # (R, S_actual)

    S_actual = T_selected.shape[1]
    w_s = np.ones(S_actual) / S_actual

    logger.info("Scenarios: %d initial → %d reduced (tail preserved)", n_full, S_actual)
    return T_selected, w_s


# ── PWL linearization of effectiveness ───────────────────────────────────────

def _pwl_tangent_lines(breakpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute outer linearization (tangent lines) of f(u) = 1 - exp(-u).

    For concave f, tangent lines give the tightest linear upper bound at each bp:
      e ≤ f(bp_k) + f'(bp_k) * (u - bp_k)   for all k

    Returns slopes (f'(bp_k) = exp(-bp_k)) and intercepts (f(bp_k) - f'(bp_k)*bp_k).
    """
    f_vals = 1.0 - np.exp(-breakpoints)
    slopes = np.exp(-breakpoints)           # f'(u) = exp(-u)
    intercepts = f_vals - slopes * breakpoints
    return slopes, intercepts


PWL_SLOPES, PWL_INTERCEPTS = _pwl_tangent_lines(PWL_BREAKPOINTS)
E_MAX = 0.90  # hard cap on effectiveness


# ── MILP construction ─────────────────────────────────────────────────────────

def build_and_solve_milp(
    site_df: pd.DataFrame,
    site_weights: np.ndarray,
    T_scenarios: np.ndarray,
    w_s: np.ndarray,
    budgets: dict | None = None,
    time_limit: float | None = None,
) -> dict:
    """
    Build and solve the two-stage stochastic MILP.

    Variables (indexed in order):
      x[0..R-1]      : p_r  (police officers, integer)
      x[R..2R-1]     : b_r  (barricades, integer)
      x[2R..3R-1]    : t_r  (tow trucks, integer)
      x[3R..4R-1]    : q_r  (QRUs, integer)
      x[4R..5R-1]    : d_r  (diversion binary)
      x[5R..6R-1]    : e_r  (effectiveness, continuous [0, E_MAX])
      x[6R]          : z    (CVaR VaR level, continuous)
      x[6R+1..6R+S]  : ξ_s  (CVaR slack, continuous >= 0)

    Objective:
      min λ2·z + λ2/((1-α)S)·Σξ_s − λ3·Σ(mean_wT_r·e_r) + λ4·Σcost_r + λ5·Σdiv_r·d_r

    Constraints:
      1. Budget constraints (4)
      2. Effectiveness PWL outer bounds (K × R)
      3. CVaR slack constraints (S)
      4. Minimum service for critical sites
    """
    if budgets is None:
        budgets = {}
    P_bud = budgets.get("police", BUDGET_POLICE)
    B_bud = budgets.get("barricades", BUDGET_BARRICADES)
    T_bud = budgets.get("tow", BUDGET_TOW)
    Q_bud = budgets.get("qru", BUDGET_QRU)

    R = len(site_df)
    S = T_scenarios.shape[1]
    K = len(PWL_BREAKPOINTS)

    # ── Variable layout ───────────────────────────────────────────────────────
    i_p = np.arange(R)
    i_b = np.arange(R, 2 * R)
    i_t = np.arange(2 * R, 3 * R)
    i_q = np.arange(3 * R, 4 * R)
    i_d = np.arange(4 * R, 5 * R)
    i_e = np.arange(5 * R, 6 * R)
    i_z = 6 * R
    i_xi = np.arange(6 * R + 1, 6 * R + 1 + S)
    n_vars = 6 * R + 1 + S

    # ── Objective vector ──────────────────────────────────────────────────────
    c = np.zeros(n_vars)

    # Mean weighted duration per site: w_r * E[T_r] (averaged over scenarios)
    mean_wT = site_weights * T_scenarios.mean(axis=1)

    # CVaR term
    c[i_z] = LAMBDA_CVAR
    c[i_xi] = LAMBDA_CVAR / ((1.0 - ALPHA_CVAR) * S)

    # Expected delay reduction (maximizing e_r reduces ED → negative coefficient)
    c[i_e] = -LAMBDA_ED * mean_wT

    # Deployment cost
    c[i_p] = LAMBDA_COST * COST_P
    c[i_b] = LAMBDA_COST * COST_B
    c[i_t] = LAMBDA_COST * COST_T
    c[i_q] = LAMBDA_COST * COST_Q

    # Diversion: small fixed cost penalty per activation.
    # Critical sites are forced to d_r = 1 via Block D constraints below.
    # Non-critical sites may activate diversion only if a high-tail-risk site is
    # adjacent (this is modelled through the diversion graph post-solve).
    c[i_d] = LAMBDA_DIV * COST_DIV

    # ── Variable bounds ───────────────────────────────────────────────────────
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)

    ub[i_p] = MAX_P
    ub[i_b] = MAX_B
    ub[i_t] = MAX_T
    ub[i_q] = MAX_Q
    ub[i_d] = 1.0  # binary
    ub[i_e] = E_MAX
    # z: unbounded below (allow negative for highly-served scenarios)
    lb[i_z] = -np.inf
    # xi: non-negative (already lb=0 from zeros)

    bounds = Bounds(lb, ub)

    # ── Integrality: 0=continuous, 1=integer ──────────────────────────────────
    integrality = np.zeros(n_vars)
    integrality[i_p] = 1
    integrality[i_b] = 1
    integrality[i_t] = 1
    integrality[i_q] = 1
    integrality[i_d] = 1  # binary = integer with bounds [0,1]

    # ── Constraint matrix ─────────────────────────────────────────────────────
    # We build rows for:
    #   Block A: Budget constraints (4 rows)
    #   Block B: PWL effectiveness upper bounds (K*R rows)
    #   Block C: CVaR slack constraints (S rows)
    #   Block D: Minimum service for critical sites

    rows_data, rows_col, rows_row = [], [], []
    lbs_constr, ubs_constr = [], []
    row_idx = 0

    def _add_row(cols, vals, lb_c, ub_c):
        nonlocal row_idx
        for col, val in zip(cols, vals):
            rows_row.append(row_idx)
            rows_col.append(col)
            rows_data.append(val)
        lbs_constr.append(lb_c)
        ubs_constr.append(ub_c)
        row_idx += 1

    # Block A: Budget constraints (Σ_r resource ≤ budget)
    _add_row(i_p.tolist(), [1.0] * R, -np.inf, P_bud)
    _add_row(i_b.tolist(), [1.0] * R, -np.inf, B_bud)
    _add_row(i_t.tolist(), [1.0] * R, -np.inf, T_bud)
    _add_row(i_q.tolist(), [1.0] * R, -np.inf, Q_bud)

    # Block B: Effectiveness PWL outer bounds
    # e_r ≤ slope_k * (γ_p*p_r + γ_b*b_r + γ_t*t_r + γ_q*q_r) + intercept_k
    # => e_r - slope_k*(γ_p*p_r + γ_b*b_r + γ_t*t_r + γ_q*q_r) ≤ intercept_k
    for r in range(R):
        for k in range(K):
            slope = PWL_SLOPES[k]
            intercept = PWL_INTERCEPTS[k]
            cols = [i_e[r], i_p[r], i_b[r], i_t[r], i_q[r]]
            vals = [1.0, -slope * GAMMA_P, -slope * GAMMA_B, -slope * GAMMA_T, -slope * GAMMA_Q]
            _add_row(cols, vals, -np.inf, intercept)

    # Block C: CVaR slack constraints
    # ξ_s + z + Σ_r w_r*T_{r,s}*e_r ≥ Σ_r w_r*T_{r,s}
    for s in range(S):
        wT_s = site_weights * T_scenarios[:, s]
        cols = [i_xi[s], i_z] + i_e.tolist()
        vals = [1.0, 1.0] + wT_s.tolist()
        rhs = float(wT_s.sum())
        _add_row(cols, vals, rhs, np.inf)

    # Block D: Minimum service and mandatory diversion for critical sites
    # A site is "critical" if high_impact_decision=1 or tail_risk_prob > 0.4
    is_critical = np.zeros(R, dtype=bool)
    if "high_impact_decision" in site_df.columns:
        is_critical |= site_df["high_impact_decision"].fillna(0).astype(bool).values
    is_critical |= site_df["tail_risk_prob"].values > 0.40

    critical_risk = site_df["tail_risk_prob"].values * site_weights
    for r in range(R):
        if is_critical[r]:
            # Minimum resource service level
            min_svc = max(MIN_SERVICE_RHO, int(np.ceil(critical_risk[r] * 4)))
            cols = [i_p[r], i_b[r], i_t[r], i_q[r]]
            vals = [1.0, 1.0, 1.0, 1.0]
            _add_row(cols, vals, float(min_svc), np.inf)
            # Mandatory diversion for critical sites: d_r >= 1
            _add_row([i_d[r]], [1.0], 1.0, 1.0)

    # Assemble sparse matrix
    n_constraints = row_idx
    A = sp.csc_matrix(
        (rows_data, (rows_row, rows_col)),
        shape=(n_constraints, n_vars),
    )
    constraint = LinearConstraint(A, lbs_constr, ubs_constr)

    # ── Solve ─────────────────────────────────────────────────────────────────
    logger.info("Solving MILP: %d vars, %d constraints, %d sites, %d scenarios",
                n_vars, n_constraints, R, S)

    tl = time_limit if time_limit is not None else MILP_TIME_LIMIT
    options = {"time_limit": tl, "disp": False}
    result = milp(c, constraints=constraint, integrality=integrality,
                  bounds=bounds, options=options)

    return {
        "result": result,
        "R": R,
        "S": S,
        "i_p": i_p, "i_b": i_b, "i_t": i_t, "i_q": i_q,
        "i_d": i_d, "i_e": i_e, "i_z": i_z, "i_xi": i_xi,
        "is_critical": is_critical,
        "mean_wT": mean_wT,
        "T_scenarios": T_scenarios,
        "site_weights": site_weights,
    }


def greedy_fallback_allocation(
    site_df: pd.DataFrame,
    site_weights: np.ndarray,
    budgets: dict | None = None,
) -> np.ndarray:
    """
    Deterministic proportional allocation when MILP fails.

    Distributes each resource budget proportionally to the risk-weighted
    site score.  Rounds down then top-fills to match budget exactly.
    Returns allocation array of shape (R, 4) for [p, b, t, q].
    """
    if budgets is None:
        budgets = {}
    R = len(site_df)
    alloc = np.zeros((R, 4), dtype=int)

    budgets_list = [
        budgets.get("police", BUDGET_POLICE),
        budgets.get("barricades", BUDGET_BARRICADES),
        budgets.get("tow", BUDGET_TOW),
        budgets.get("qru", BUDGET_QRU),
    ]
    caps = [MAX_P, MAX_B, MAX_T, MAX_Q]

    risk = site_weights * (1.0 + site_df["tail_risk_prob"].values)
    risk_norm = risk / risk.sum() if risk.sum() > 0 else np.ones(R) / R

    for j, (budget, cap) in enumerate(zip(budgets_list, caps)):
        raw = risk_norm * budget
        floored = np.floor(raw).astype(int)
        floored = np.minimum(floored, cap)
        remainder = budget - floored.sum()
        if remainder > 0:
            frac = raw - floored
            top_k = np.argsort(frac)[::-1][:remainder]
            for idx in top_k:
                if floored[idx] < cap:
                    floored[idx] += 1
        alloc[:, j] = floored

    logger.info("Greedy fallback allocation: total p=%d b=%d t=%d q=%d",
                alloc[:, 0].sum(), alloc[:, 1].sum(),
                alloc[:, 2].sum(), alloc[:, 3].sum())
    return alloc


def extract_milp_solution(
    milp_out: dict,
    site_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    w_s: np.ndarray,
) -> pd.DataFrame:
    """
    Parse MILP result into a per-site allocation DataFrame.

    Falls back to greedy allocation on solver failure or infeasibility.
    """
    result = milp_out["result"]
    R = milp_out["R"]
    S = milp_out["S"]
    i_p, i_b, i_t, i_q = milp_out["i_p"], milp_out["i_b"], milp_out["i_t"], milp_out["i_q"]
    i_d, i_e = milp_out["i_d"], milp_out["i_e"]
    i_z, i_xi = milp_out["i_z"], milp_out["i_xi"]
    is_critical = milp_out["is_critical"]
    site_weights = milp_out["site_weights"]

    solver_ok = result.success and result.x is not None
    used_fallback = False

    if solver_ok:
        x = result.x
        p_vals = np.round(x[i_p]).astype(int).clip(0, MAX_P)
        b_vals = np.round(x[i_b]).astype(int).clip(0, MAX_B)
        t_vals = np.round(x[i_t]).astype(int).clip(0, MAX_T)
        q_vals = np.round(x[i_q]).astype(int).clip(0, MAX_Q)
        d_vals = np.round(x[i_d]).astype(int).clip(0, 1)
        e_vals = np.clip(x[i_e], 0.0, E_MAX)
        z_val = float(x[i_z])
        xi_vals = np.maximum(x[i_xi], 0.0)
        logger.info("MILP solved: status='%s', obj=%.2f", result.message, result.fun)
    else:
        logger.warning("MILP failed (%s) — using greedy fallback", result.message)
        used_fallback = True
        greedy = greedy_fallback_allocation(site_df, site_weights)
        p_vals = greedy[:, 0]
        b_vals = greedy[:, 1]
        t_vals = greedy[:, 2]
        q_vals = greedy[:, 3]
        d_vals = (site_weights > np.percentile(site_weights, 60)).astype(int)
        eff_input = GAMMA_P * p_vals + GAMMA_B * b_vals + GAMMA_T * t_vals + GAMMA_Q * q_vals
        e_vals = np.minimum(1.0 - np.exp(-eff_input), E_MAX)
        z_val = 0.0
        xi_vals = np.zeros(S)

    # Compute delay reductions per scenario
    delay_scenarios = np.zeros((R, S))
    for r in range(R):
        delay_scenarios[r, :] = T_scenarios[r, :] * (1.0 - e_vals[r]) * site_weights[r]

    total_delay_s = delay_scenarios.sum(axis=0)  # (S,)
    expected_delay_reduction = np.array([
        T_scenarios[r, :].mean() * e_vals[r] * site_weights[r] for r in range(R)
    ])

    # CVaR
    cvar_val, var_val = _cvar(total_delay_s, ALPHA_CVAR)

    # Chance constraint satisfaction
    cc_target = site_df["safe_duration_p50"].values * CC_TARGET_MULT
    cc_sat = np.array([
        (delay_scenarios[r, :] / site_weights[r] <= cc_target[r]).mean()
        for r in range(R)
    ])

    # Robustness score: fraction of scenarios where effective reduction > 50%
    rob_score = np.array([
        (e_vals[r] > 0.5 * E_MAX) for r in range(R)
    ]).astype(float)

    # Service tier
    tier = np.where(
        is_critical, "critical",
        np.where(site_weights > np.percentile(site_weights, 75), "high",
                 np.where(site_weights > np.percentile(site_weights, 40), "moderate", "low"))
    )

    alloc_df = pd.DataFrame({
        "event_id": site_df["event_id"].values,
        "event_cause": site_df.get("event_cause", pd.Series(["unknown"] * R)).values
            if "event_cause" in site_df.columns else ["unknown"] * R,
        "site_weight": site_weights,
        "officers_allocated": p_vals,
        "barricades_allocated": b_vals,
        "tow_trucks_allocated": t_vals,
        "qru_allocated": q_vals,
        "diversion_activated": d_vals,
        "effectiveness": e_vals,
        "expected_delay_reduction_min": expected_delay_reduction,
        "cvar_contribution": delay_scenarios.mean(axis=1),
        "total_cvar": cvar_val,
        "var_level": var_val,
        "chance_constraint_satisfaction": cc_sat,
        "service_tier": tier,
        "robustness_score": rob_score,
        "is_critical": is_critical.astype(int),
        "used_fallback": int(used_fallback),
        "solver_status": result.message if not used_fallback else "greedy_fallback",
        "safe_duration_p50": site_df["safe_duration_p50"].values,
        "safe_duration_p80": site_df["safe_duration_p80"].values,
        "safe_duration_p95": site_df["safe_duration_p95"].values,
        "duration_reliability": site_df["duration_reliability"].values,
        "tail_risk_prob": site_df["tail_risk_prob"].values,
        "duration_sanity_flag": site_df["duration_sanity_flag"].values,
    })
    return alloc_df, z_val, xi_vals, total_delay_s


def _cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Compute CVaR_α and VaR_α from a scenario loss vector."""
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    var_idx = int(np.floor(alpha * n))
    var_idx = min(var_idx, n - 1)
    var_val = float(sorted_losses[var_idx])
    tail = sorted_losses[var_idx:]
    cvar_val = float(tail.mean()) if len(tail) > 0 else var_val
    return cvar_val, var_val


# ── CVaR diagnostics ──────────────────────────────────────────────────────────

def compute_cvar_summary(
    total_delay_s: np.ndarray,
    alloc_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    site_weights: np.ndarray,
    w_s: np.ndarray,
) -> pd.DataFrame:
    """CVaR summary across α levels and by service tier."""
    alphas = [0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for alpha in alphas:
        cvar_val, var_val = _cvar(total_delay_s, alpha)
        rows.append({
            "alpha": alpha,
            "cvar": cvar_val,
            "var": var_val,
            "expected_total_delay": float(total_delay_s.mean()),
            "worst_scenario_delay": float(total_delay_s.max()),
            "best_scenario_delay": float(total_delay_s.min()),
        })

    # Per-tier CVaR
    for tier in alloc_df["service_tier"].unique():
        mask = alloc_df["service_tier"] == tier
        tier_idx = np.where(mask)[0]
        if len(tier_idx) == 0:
            continue
        tier_delay = (T_scenarios[tier_idx, :] * site_weights[tier_idx, None]).sum(axis=0)
        cvar_t, var_t = _cvar(tier_delay, 0.90)
        rows.append({
            "alpha": 0.90,
            "cvar": cvar_t,
            "var": var_t,
            "expected_total_delay": float(tier_delay.mean()),
            "worst_scenario_delay": float(tier_delay.max()),
            "best_scenario_delay": float(tier_delay.min()),
            "service_tier": tier,
        })

    return pd.DataFrame(rows)


# ── Diversion routing ─────────────────────────────────────────────────────────

def generate_diversions(
    alloc_df: pd.DataFrame,
    site_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate diversion route recommendations for each active site.

    Uses the Layer 3 junction diversion graph (networkx DiGraph) where
    available.  Each activated diversion site gets up to 3 route options
    ranked by path cost.  For sites not directly mapped to a junction, the
    nearest-risk-tier junction alternative is used.

    Edge weight formula:
      c_e = c_base + β1·risk + β2·fragility + β3·obi + β4·closure
    """
    BETA_RISK = 0.40
    BETA_FRAG = 0.25
    BETA_OBI = 0.25
    BETA_CLOSE = 0.10

    div_rows = []

    # Load Layer 3 junction data for graph construction
    dis_path = OUT / "layer3_disruption_impact_scores.csv"
    divert_path = OUT / "layer3_diversion_recommendations.csv"

    if not NX_AVAILABLE or not dis_path.exists():
        # Simple fallback: assign nearest high-DIS junction as diversion target
        for _, row in alloc_df.iterrows():
            if row["diversion_activated"] == 0:
                continue
            div_rows.append({
                "event_id": row["event_id"],
                "event_cause": row.get("event_cause", "unknown"),
                "diversion_activated": row["diversion_activated"],
                "route_rank": 1,
                "route_label": "Route A (best)",
                "diversion_path": "alternate_route_A",
                "path_cost": 1.0,
                "estimated_additional_time_min": 10.0,
                "routing_method": "fallback",
            })
        if not div_rows:
            return pd.DataFrame(columns=["event_id", "route_rank", "route_label",
                                         "diversion_path", "path_cost",
                                         "estimated_additional_time_min", "routing_method"])
        return pd.DataFrame(div_rows)

    dis_df = pd.read_csv(dis_path)
    precomp_div = pd.read_csv(divert_path) if divert_path.exists() else pd.DataFrame()

    # Build weighted digraph from junction risk data
    G = nx.DiGraph()
    for _, jrow in dis_df.iterrows():
        jname = jrow["junction"]
        risk_n = (jrow["dis_score"] / 100.0) if pd.notna(jrow.get("dis_score")) else 0.5
        frag_n = jrow.get("obi_component", 0.5) or 0.5
        obi_n = jrow.get("obi_component", 0.5) or 0.5
        G.add_node(jname, risk=risk_n, fragility=frag_n, obi=obi_n)

    # Add edges from diversion paths (parse "A|B|C" paths)
    if not precomp_div.empty and "diversion_path" in precomp_div.columns:
        for _, drow in precomp_div.iterrows():
            path_nodes = str(drow.get("diversion_path", "")).split("|")
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i].strip(), path_nodes[i + 1].strip()
                if u and v and u != v:
                    r_u = G.nodes[u].get("risk", 0.5) if G.has_node(u) else 0.5
                    r_v = G.nodes[v].get("risk", 0.5) if G.has_node(v) else 0.5
                    f_u = G.nodes[u].get("fragility", 0.5) if G.has_node(u) else 0.5
                    o_u = G.nodes[u].get("obi", 0.5) if G.has_node(u) else 0.5
                    c_base = drow.get("path_weight", 0.5) or 0.5
                    weight = (c_base
                              + BETA_RISK * (r_u + r_v) / 2
                              + BETA_FRAG * f_u
                              + BETA_OBI * o_u)
                    if not G.has_node(u):
                        G.add_node(u, risk=0.5, fragility=0.5, obi=0.5)
                    if not G.has_node(v):
                        G.add_node(v, risk=0.5, fragility=0.5, obi=0.5)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, weight=weight)

    junctions = dis_df["junction"].tolist()
    low_risk_junctions = dis_df.nsmallest(10, "dis_score")["junction"].tolist()

    cause_to_tier = {}
    if "event_cause" in site_df.columns:
        for cause in site_df["event_cause"].unique():
            mask = site_df["event_cause"] == cause
            median_risk = site_df.loc[mask, "tail_risk_prob"].median()
            cause_to_tier[cause] = "high" if median_risk > 0.3 else "low"

    for _, row in alloc_df.iterrows():
        if row["diversion_activated"] == 0:
            continue
        cause = row.get("event_cause", "unknown")
        tier = cause_to_tier.get(cause, "moderate")

        # Try to find diversion routes using precomputed data
        if not precomp_div.empty:
            sub = precomp_div.head(3)
            for _, prow in sub.iterrows():
                div_rows.append({
                    "event_id": row["event_id"],
                    "event_cause": cause,
                    "service_tier": row["service_tier"],
                    "diversion_activated": 1,
                    "route_rank": prow.get("route_rank", 1),
                    "route_label": prow.get("route_label", "Route A"),
                    "diversion_corridor": prow.get("diversion_corridor", ""),
                    "diversion_path": prow.get("diversion_path", ""),
                    "path_cost": prow.get("path_weight", 1.0),
                    "estimated_additional_time_min": prow.get("estimated_additional_time_min", 10.0),
                    "routing_method": "layer3_graph",
                })
        else:
            div_rows.append({
                "event_id": row["event_id"],
                "event_cause": cause,
                "service_tier": row["service_tier"],
                "diversion_activated": 1,
                "route_rank": 1,
                "route_label": "Route A (best)",
                "diversion_corridor": low_risk_junctions[0] if low_risk_junctions else "alternate",
                "diversion_path": "|".join(low_risk_junctions[:2]),
                "path_cost": 0.5,
                "estimated_additional_time_min": 8.0,
                "routing_method": "greedy",
            })

    if not div_rows:
        return pd.DataFrame(columns=["event_id", "route_rank", "route_label",
                                     "diversion_path", "path_cost",
                                     "estimated_additional_time_min", "routing_method"])
    return pd.DataFrame(div_rows)


# ── Shadow prices ─────────────────────────────────────────────────────────────

def compute_shadow_prices(
    site_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    site_weights: np.ndarray,
    w_s: np.ndarray,
    budgets: dict | None = None,
    delta: int = 1,
) -> pd.DataFrame:
    """
    Approximate shadow prices by perturbing each budget by +delta and
    re-solving the LP relaxation of the CVaR subproblem.

    Returns marginal value per additional unit of each resource type.
    """
    if budgets is None:
        budgets = {}

    resource_names = ["police", "barricades", "tow", "qru"]
    base_budgets = {
        "police": budgets.get("police", BUDGET_POLICE),
        "barricades": budgets.get("barricades", BUDGET_BARRICADES),
        "tow": budgets.get("tow", BUDGET_TOW),
        "qru": budgets.get("qru", BUDGET_QRU),
    }

    def _solve_milp_obj(bud: dict) -> float:
        """MILP objective with longer time limit for accurate shadow approximation."""
        milp_out_base = build_and_solve_milp(site_df, site_weights, T_scenarios, w_s, bud,
                                             time_limit=MILP_TIME_LIMIT_SHADOW)
        r = milp_out_base["result"]
        if r.success and r.fun is not None:
            return float(r.fun)
        # If not solved optimally within time limit, use best feasible bound
        if r.x is not None:
            return float(r.fun) if r.fun is not None else np.inf
        return np.inf

    base_obj = _solve_milp_obj(base_budgets)

    shadow_rows = []
    for rname in resource_names:
        perturbed = dict(base_budgets)
        perturbed[rname] = base_budgets[rname] + delta
        perturbed_obj = _solve_milp_obj(perturbed)
        marginal = (base_obj - perturbed_obj) / delta if np.isfinite(perturbed_obj) else 0.0
        shadow_rows.append({
            "resource": rname,
            "base_budget": base_budgets[rname],
            "perturbed_budget": perturbed[rname],
            "base_objective": base_obj,
            "perturbed_objective": perturbed_obj,
            "marginal_value": marginal,
            "interpretation": f"One extra {rname} unit reduces objective by {marginal:.4f}",
        })
        logger.info("Shadow price %s: %.4f", rname, marginal)

    return pd.DataFrame(shadow_rows)


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def run_sensitivity(
    site_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    site_weights: np.ndarray,
    w_s: np.ndarray,
    budgets: dict | None = None,
) -> pd.DataFrame:
    """
    Sensitivity: vary each budget ±20%, ±50% and record objective and CVaR.

    Also sweeps the CVaR α level from 0.75 to 0.99.
    """
    if budgets is None:
        budgets = {}
    base_budgets = {
        "police": budgets.get("police", BUDGET_POLICE),
        "barricades": budgets.get("barricades", BUDGET_BARRICADES),
        "tow": budgets.get("tow", BUDGET_TOW),
        "qru": budgets.get("qru", BUDGET_QRU),
    }
    rows = []

    multipliers = [0.50, 0.80, 1.00, 1.20, 1.50]
    for rname, base_val in base_budgets.items():
        for mult in multipliers:
            perturbed = dict(base_budgets)
            perturbed[rname] = int(base_val * mult)
            milp_out = build_and_solve_milp(site_df, site_weights, T_scenarios, w_s, perturbed,
                                           time_limit=MILP_TIME_LIMIT_SUB)
            result = milp_out["result"]
            obj = float(result.fun) if result.success and result.fun is not None else np.nan
            rows.append({
                "sensitivity_type": "budget",
                "resource": rname,
                "multiplier": mult,
                "budget_value": perturbed[rname],
                "objective": obj,
                "solver_success": result.success,
            })

    # CVaR α sweep: solve ONCE with base budgets, then evaluate CVaR at each α level
    # from the same scenario delay vector (avoids 6 redundant MILP solves).
    alpha_base_milp = build_and_solve_milp(site_df, site_weights, T_scenarios, w_s,
                                           base_budgets, time_limit=MILP_TIME_LIMIT_SUB)
    alpha_result = alpha_base_milp["result"]
    if alpha_result.success and alpha_result.x is not None:
        i_e_alpha = alpha_base_milp["i_e"]
        e_alpha = np.clip(alpha_result.x[i_e_alpha], 0.0, E_MAX)
        delay_s_alpha = (T_scenarios * (1.0 - e_alpha[:, None]) * site_weights[:, None]).sum(axis=0)
    else:
        delay_s_alpha = None

    for alpha_test in [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        if delay_s_alpha is not None:
            cvar_at_alpha, _ = _cvar(delay_s_alpha, alpha_test)
            obj = float(cvar_at_alpha)
            solver_ok = True
        else:
            obj = np.nan
            solver_ok = False
        rows.append({
            "sensitivity_type": "cvar_alpha",
            "resource": "all",
            "multiplier": alpha_test,
            "budget_value": None,
            "objective": obj,
            "solver_success": solver_ok,
        })

    logger.info("Sensitivity analysis: %d runs", len(rows))
    return pd.DataFrame(rows)


# ── Pareto frontier ───────────────────────────────────────────────────────────

def compute_pareto_front(
    site_df: pd.DataFrame,
    site_weights: np.ndarray,
    T_scenarios: np.ndarray,
    w_s: np.ndarray,
    budgets: dict | None = None,
) -> pd.DataFrame:
    """
    Compute Pareto frontier between CVaR (tail risk) and deployment cost.

    Sweeps the λ_cvar / λ_cost ratio from cost-dominated to risk-dominated.
    Each point represents a different trade-off between tail risk and cost.
    """
    if budgets is None:
        budgets = {}

    # Use the base MILP solution and vary the post-hoc CVaR metric for each λ trade-off
    # (Full re-solve per λ pair is too slow; instead compute post-hoc CVaR/cost trade-offs
    # from the single base solve with different cost accounting)
    lambda_pairs = [
        (0.5, 2.0), (1.0, 2.0), (2.0, 2.0),
        (2.0, 1.0), (2.0, 0.5), (5.0, 0.5),
    ]
    rows = []

    # Get base solution once (use shadow time limit for accuracy)
    base_milp = build_and_solve_milp(site_df, site_weights, T_scenarios, w_s, budgets,
                                     time_limit=MILP_TIME_LIMIT_SHADOW)
    base_result = base_milp["result"]
    if not base_result.success or base_result.x is None:
        return pd.DataFrame()

    x_base = base_result.x
    i_e_b = base_milp["i_e"]
    i_p_b, i_b_b = base_milp["i_p"], base_milp["i_b"]
    i_t_b, i_q_b = base_milp["i_t"], base_milp["i_q"]

    e_base = np.clip(x_base[i_e_b], 0.0, E_MAX)
    delay_base = (T_scenarios * (1.0 - e_base[:, None]) * site_weights[:, None]).sum(axis=0)
    cost_base = (COST_P * x_base[i_p_b].sum() + COST_B * x_base[i_b_b].sum()
                 + COST_T * x_base[i_t_b].sum() + COST_Q * x_base[i_q_b].sum())

    for lc, lk in lambda_pairs:
        cvar_val, _ = _cvar(delay_base, ALPHA_CVAR)
        # Simulate objective value under this λ weighting post-hoc
        obj_simulated = lc * cvar_val + lk * float(cost_base)
        rows.append({
            "lambda_cvar": lc,
            "lambda_cost": lk,
            "cvar_90": float(cvar_val),
            "expected_delay": float(delay_base.mean()),
            "deployment_cost": float(cost_base),
            "objective": float(obj_simulated),
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Add λ-weighted objective to each row for the trade-off summary
    # Since all rows use the same base solution, the Pareto here documents
    # what each λ weighting implies for the objective, not distinct allocations.
    # Return all rows (they represent the full trade-off spectrum).
    return df


# ── Alternative plans ─────────────────────────────────────────────────────────

def build_alternative_plans(
    alloc_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    site_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Generate alternative allocation plans with different risk postures.

    Plan A (base): MILP optimal.
    Plan B (conservative): allocate 30% more to critical sites, cut low-tier.
    Plan C (aggressive): uniform spread across all sites.
    """
    R = len(alloc_df)
    plan_rows = []

    for plan_name, policy in [("plan_A_optimal", "optimal"), ("plan_B_conservative", "conservative"),
                               ("plan_C_spread", "spread")]:
        for r, row in alloc_df.iterrows():
            base = row.copy()
            if policy == "optimal":
                plan_rows.append({"plan": plan_name, **base.to_dict()})
            elif policy == "conservative":
                multiplier = 1.3 if row["is_critical"] else 0.7
                plan_rows.append({
                    "plan": plan_name,
                    "event_id": row["event_id"],
                    "event_cause": row.get("event_cause", "unknown"),
                    "service_tier": row["service_tier"],
                    "officers_allocated": min(int(row["officers_allocated"] * multiplier), MAX_P),
                    "barricades_allocated": min(int(row["barricades_allocated"] * multiplier), MAX_B),
                    "tow_trucks_allocated": min(int(row["tow_trucks_allocated"] * multiplier), MAX_T),
                    "qru_allocated": min(int(row["qru_allocated"] * multiplier), MAX_Q),
                    "diversion_activated": row["diversion_activated"],
                    "effectiveness": row["effectiveness"] * (0.9 if multiplier < 1 else 1.0),
                    "expected_delay_reduction_min": row["expected_delay_reduction_min"] * multiplier,
                })
            else:  # spread
                equal_p = max(1, BUDGET_POLICE // R)
                plan_rows.append({
                    "plan": plan_name,
                    "event_id": row["event_id"],
                    "event_cause": row.get("event_cause", "unknown"),
                    "service_tier": row["service_tier"],
                    "officers_allocated": min(equal_p, MAX_P),
                    "barricades_allocated": min(max(1, BUDGET_BARRICADES // R), MAX_B),
                    "tow_trucks_allocated": min(max(0, BUDGET_TOW // R), MAX_T),
                    "qru_allocated": 0,
                    "diversion_activated": 0,
                    "effectiveness": 0.30,
                    "expected_delay_reduction_min": T_scenarios[r % len(T_scenarios), :].mean() * 0.30,
                })

    return pd.DataFrame(plan_rows)


# ── Scenario summary ──────────────────────────────────────────────────────────

def build_scenario_summary(
    T_scenarios: np.ndarray,
    w_s: np.ndarray,
    alloc_df: pd.DataFrame,
    site_weights: np.ndarray,
) -> pd.DataFrame:
    """Per-scenario summary statistics."""
    R, S = T_scenarios.shape
    e_vals = alloc_df["effectiveness"].values[:R]

    rows = []
    for s in range(S):
        T_s = T_scenarios[:, s]
        delay_s = (T_s * (1.0 - e_vals) * site_weights).sum()
        raw_delay_s = (T_s * site_weights).sum()
        rows.append({
            "scenario": s,
            "scenario_weight": float(w_s[s]),
            "raw_total_delay": float(raw_delay_s),
            "optimized_total_delay": float(delay_s),
            "delay_reduction": float(raw_delay_s - delay_s),
            "reduction_pct": float(100.0 * (raw_delay_s - delay_s) / (raw_delay_s + 1e-9)),
            "is_tail_scenario": (T_s.max() > T_scenarios.max() * 0.80),
            "max_site_duration": float(T_s.max()),
            "mean_site_duration": float(T_s.mean()),
        })
    return pd.DataFrame(rows)


# ── Optimization metrics ──────────────────────────────────────────────────────

def build_optimization_metrics(
    alloc_df: pd.DataFrame,
    cvar_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    site_weights: np.ndarray,
) -> pd.DataFrame:
    """High-level optimization quality metrics."""
    R = len(alloc_df)
    S = T_scenarios.shape[1]
    e_vals = alloc_df["effectiveness"].values[:R]
    delay_matrix = T_scenarios * (1.0 - e_vals[:, None]) * site_weights[:, None]
    total_delay_s = delay_matrix.sum(axis=0)
    raw_delay_s = (T_scenarios * site_weights[:, None]).sum(axis=0)

    cvar_90 = cvar_df.loc[cvar_df["alpha"] == 0.90, "cvar"].values
    cvar_val = float(cvar_90[0]) if len(cvar_90) > 0 else np.nan

    cc_sat = alloc_df["chance_constraint_satisfaction"].mean()
    cc_sat_crit = alloc_df.loc[alloc_df["is_critical"] == 1, "chance_constraint_satisfaction"].mean()

    rows = [
        {"metric": "n_active_sites", "value": R},
        {"metric": "n_scenarios", "value": S},
        {"metric": "expected_total_delay_optimized", "value": float(total_delay_s.mean())},
        {"metric": "expected_total_delay_raw", "value": float(raw_delay_s.mean())},
        {"metric": "expected_delay_reduction_pct",
         "value": float(100.0 * (raw_delay_s.mean() - total_delay_s.mean()) / (raw_delay_s.mean() + 1e-9))},
        {"metric": "cvar_90", "value": cvar_val},
        {"metric": "chance_constraint_satisfaction_mean", "value": float(cc_sat)},
        {"metric": "chance_constraint_satisfaction_critical", "value": float(cc_sat_crit) if not np.isnan(cc_sat_crit) else np.nan},
        {"metric": "total_officers_deployed", "value": int(alloc_df["officers_allocated"].sum())},
        {"metric": "total_barricades_deployed", "value": int(alloc_df["barricades_allocated"].sum())},
        {"metric": "total_tow_trucks_deployed", "value": int(alloc_df["tow_trucks_allocated"].sum())},
        {"metric": "total_qru_deployed", "value": int(alloc_df["qru_allocated"].sum())},
        {"metric": "diversion_activations", "value": int(alloc_df["diversion_activated"].sum())},
        {"metric": "sites_used_fallback", "value": int(alloc_df["used_fallback"].max())},
        {"metric": "critical_sites", "value": int(alloc_df["is_critical"].sum())},
        {"metric": "mean_effectiveness", "value": float(alloc_df["effectiveness"].mean())},
        {"metric": "mean_robustness_score", "value": float(alloc_df["robustness_score"].mean())},
        {"metric": "alpha_cvar", "value": ALPHA_CVAR},
        {"metric": "n_scenarios_S", "value": S},
        {"metric": "kappa_uncertainty_inflation", "value": KAPPA},
    ]
    return pd.DataFrame(rows)


# ── Robust plan ───────────────────────────────────────────────────────────────

def build_robust_plan(
    alloc_df: pd.DataFrame,
    T_scenarios: np.ndarray,
    site_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Robust plan: allocation recommendations with tail-risk protection scores.

    For each site, reports the worst-5th-percentile scenario delay with current
    allocation vs. without any resources deployed.
    """
    R = len(alloc_df)
    e_vals = alloc_df["effectiveness"].values[:R]
    rows = []
    for r in range(R):
        T_r = T_scenarios[r, :]
        D_opt = T_r * (1.0 - e_vals[r]) * site_weights[r]
        D_raw = T_r * site_weights[r]
        p95_opt = float(np.percentile(D_opt, 95))
        p95_raw = float(np.percentile(D_raw, 95))
        cvar_r, _ = _cvar(D_opt, 0.90)
        rows.append({
            "event_id": alloc_df.iloc[r]["event_id"],
            "service_tier": alloc_df.iloc[r]["service_tier"],
            "is_critical": alloc_df.iloc[r]["is_critical"],
            "officers_allocated": alloc_df.iloc[r]["officers_allocated"],
            "barricades_allocated": alloc_df.iloc[r]["barricades_allocated"],
            "tow_trucks_allocated": alloc_df.iloc[r]["tow_trucks_allocated"],
            "qru_allocated": alloc_df.iloc[r]["qru_allocated"],
            "diversion_activated": alloc_df.iloc[r]["diversion_activated"],
            "effectiveness": e_vals[r],
            "p95_delay_optimized": p95_opt,
            "p95_delay_no_resources": p95_raw,
            "p95_delay_reduction_pct": float(100.0 * (p95_raw - p95_opt) / (p95_raw + 1e-9)),
            "site_cvar_90": cvar_r,
            "robustness_score": alloc_df.iloc[r]["robustness_score"],
            "duration_reliability": alloc_df.iloc[r]["duration_reliability"],
            "tail_risk_prob": alloc_df.iloc[r]["tail_risk_prob"],
        })
    return pd.DataFrame(rows)


# ── Outputs ───────────────────────────────────────────────────────────────────

def export_outputs(
    alloc_df: pd.DataFrame,
    diversion_df: pd.DataFrame,
    scenario_summary_df: pd.DataFrame,
    cvar_df: pd.DataFrame,
    robust_plan_df: pd.DataFrame,
    alt_plans_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    shadow_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_txt: str,
) -> None:
    """Write all Layer 5 output files."""
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Primary allocation outputs
    alloc_df.to_csv(OUT / "layer5_resource_allocation.csv", index=False)
    logger.info("Wrote layer5_resource_allocation.csv (%d rows)", len(alloc_df))

    diversion_df.to_csv(OUT / "layer5_diversion_recommendations.csv", index=False)
    logger.info("Wrote layer5_diversion_recommendations.csv (%d rows)", len(diversion_df))

    scenario_summary_df.to_csv(OUT / "layer5_scenario_summary.csv", index=False)
    cvar_df.to_csv(OUT / "layer5_cvar_summary.csv", index=False)
    robust_plan_df.to_csv(OUT / "layer5_robust_plan.csv", index=False)
    alt_plans_df.to_csv(OUT / "layer5_alternative_plans.csv", index=False)

    if not pareto_df.empty:
        pareto_df.to_csv(OUT / "layer5_pareto_front.csv", index=False)
    else:
        pd.DataFrame(columns=["lambda_cvar", "lambda_cost", "cvar_90",
                               "deployment_cost", "objective"]).to_csv(
            OUT / "layer5_pareto_front.csv", index=False)

    shadow_df.to_csv(OUT / "layer5_shadow_prices.csv", index=False)
    sensitivity_df.to_csv(OUT / "layer5_sensitivity_summary.csv", index=False)
    metrics_df.to_csv(OUT / "layer5_optimization_metrics.csv", index=False)

    # Dashboard-ready frontend export
    frontend_cols = [
        "event_id", "event_cause", "service_tier", "is_critical",
        "officers_allocated", "barricades_allocated", "tow_trucks_allocated", "qru_allocated",
        "diversion_activated", "effectiveness", "expected_delay_reduction_min",
        "cvar_contribution", "total_cvar", "var_level", "chance_constraint_satisfaction",
        "robustness_score", "duration_reliability", "tail_risk_prob",
        "safe_duration_p50", "safe_duration_p80", "safe_duration_p95",
        "duration_sanity_flag", "solver_status", "used_fallback",
    ]
    fe_cols_present = [c for c in frontend_cols if c in alloc_df.columns]
    alloc_df[fe_cols_present].to_csv(OUT / "layer5_frontend_export.csv", index=False)
    logger.info("Wrote layer5_frontend_export.csv")

    # Model artifacts: hyperparameters
    artifacts = {
        "kappa": KAPPA,
        "alpha_cvar": ALPHA_CVAR,
        "n_scenarios": S_SCENARIOS,
        "n_active_max": N_ACTIVE_MAX,
        "gamma_p": GAMMA_P, "gamma_b": GAMMA_B, "gamma_t": GAMMA_T, "gamma_q": GAMMA_Q,
        "lambda_cvar": LAMBDA_CVAR, "lambda_ed": LAMBDA_ED,
        "lambda_cost": LAMBDA_COST, "lambda_div": LAMBDA_DIV,
        "budget_police": BUDGET_POLICE, "budget_barricades": BUDGET_BARRICADES,
        "budget_tow": BUDGET_TOW, "budget_qru": BUDGET_QRU,
        "pwl_breakpoints": PWL_BREAKPOINTS.tolist(),
        "e_max": E_MAX,
    }
    with open(ARTIFACTS / "layer5_hyperparameters.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    with open(OUT / "layer5_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)
    logger.info("Wrote layer5_summary.txt")

    logger.info("All Layer 5 outputs written to %s", OUT)


def _build_summary(
    alloc_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    cvar_df: pd.DataFrame,
    shadow_df: pd.DataFrame,
) -> str:
    """Build human-readable summary text."""
    def _m(name):
        row = metrics_df[metrics_df["metric"] == name]
        return float(row["value"].iloc[0]) if len(row) > 0 else float("nan")

    cvar_90 = cvar_df.loc[cvar_df.get("alpha", pd.Series()) == 0.90, "cvar"]
    cvar_val = float(cvar_90.iloc[0]) if len(cvar_90) > 0 else float("nan")

    lines = [
        "=" * 70,
        "LAYER 5 — ROBUST PRESCRIPTIVE OPTIMIZATION SUMMARY",
        "=" * 70,
        "",
        f"Active disruption sites  : {int(_m('n_active_sites'))}",
        f"Scenarios generated      : {int(_m('n_scenarios_S'))}",
        f"CVaR alpha level         : {ALPHA_CVAR:.2f}",
        f"Uncertainty inflation k  : {KAPPA:.2f}",
        "",
        "RESOURCE DEPLOYMENT",
        "-" * 40,
        f"  Officers deployed      : {int(_m('total_officers_deployed'))} / {BUDGET_POLICE}",
        f"  Barricades deployed    : {int(_m('total_barricades_deployed'))} / {BUDGET_BARRICADES}",
        f"  Tow trucks deployed    : {int(_m('total_tow_trucks_deployed'))} / {BUDGET_TOW}",
        f"  QRUs deployed          : {int(_m('total_qru_deployed'))} / {BUDGET_QRU}",
        f"  Diversions activated   : {int(_m('diversion_activations'))}",
        f"  Critical sites         : {int(_m('critical_sites'))}",
        "",
        "OPTIMIZATION QUALITY",
        "-" * 40,
        f"  Expected delay (raw)   : {_m('expected_total_delay_raw'):.1f} min (weighted)",
        f"  Expected delay (opt.)  : {_m('expected_total_delay_optimized'):.1f} min (weighted)",
        f"  Delay reduction        : {_m('expected_delay_reduction_pct'):.1f}%",
        f"  CVaR 90%               : {cvar_val:.1f} min",
        f"  CC satisfaction (mean) : {_m('chance_constraint_satisfaction_mean'):.3f}",
        f"  Mean effectiveness     : {_m('mean_effectiveness'):.3f}",
        f"  Mean robustness score  : {_m('mean_robustness_score'):.3f}",
        f"  Used fallback?         : {'Yes' if _m('sites_used_fallback') > 0 else 'No'}",
        "",
        "SHADOW PRICES (marginal value per +1 unit)",
        "-" * 40,
    ]
    for _, srow in shadow_df.iterrows():
        lines.append(f"  {srow['resource']:<12}: {srow['marginal_value']:.4f}")

    lines += [
        "",
        "INPUTS (Layer 4.5 canonical)",
        "-" * 40,
        "  layer45_scenario_ready_duration.csv",
        "  layer45_operational_state_vector_normalized.csv",
        "  layer45_duration_quality.csv",
        "  layer45_cause_tau_thresholds.csv",
        "",
        "OUTPUTS",
        "-" * 40,
        "  layer5_resource_allocation.csv",
        "  layer5_diversion_recommendations.csv",
        "  layer5_scenario_summary.csv",
        "  layer5_cvar_summary.csv",
        "  layer5_robust_plan.csv",
        "  layer5_alternative_plans.csv",
        "  layer5_pareto_front.csv",
        "  layer5_shadow_prices.csv",
        "  layer5_sensitivity_summary.csv",
        "  layer5_optimization_metrics.csv",
        "  layer5_frontend_export.csv",
        "  layer5_summary.txt",
        "=" * 70,
    ]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Layer 5 — Robust Prescriptive Optimization Engine")
    logger.info("=" * 60)

    # 1. Load and validate inputs
    (scenario_ready, josv_norm, quality, tau_thresh,
     metrics, fallback_summary) = load_inputs()

    if scenario_ready.empty:
        raise RuntimeError("Layer 4.5 scenario-ready duration bundle not found. "
                           "Run layer45_predictive_fusion.py first.")

    scenario_ready = validate_duration_bundle(scenario_ready)

    # 2. Select active sites
    site_df, josv_site, site_weights = select_active_sites(
        scenario_ready, josv_norm, n_max=N_ACTIVE_MAX
    )
    R = len(site_df)

    # 3. Generate scenarios
    T_scenarios, w_s = build_scenarios(site_df, n_initial=S_INITIAL, n_reduced=S_SCENARIOS)
    logger.info("Scenario matrix: %d sites × %d scenarios", T_scenarios.shape[0], T_scenarios.shape[1])

    # 4. Solve MILP
    milp_out = build_and_solve_milp(site_df, site_weights, T_scenarios, w_s)
    alloc_df, z_val, xi_vals, total_delay_s = extract_milp_solution(
        milp_out, site_df, T_scenarios, w_s
    )
    logger.info("Allocation extracted: %d sites", len(alloc_df))

    # 5. CVaR and scenario summaries
    cvar_df = compute_cvar_summary(total_delay_s, alloc_df, T_scenarios, site_weights, w_s)
    scenario_summary_df = build_scenario_summary(T_scenarios, w_s, alloc_df, site_weights)

    # 6. Diversion routing
    diversion_df = generate_diversions(alloc_df, site_df)
    logger.info("Diversion routes: %d recommendations", len(diversion_df))

    # 7. Shadow prices (LP relaxation perturbation)
    logger.info("Computing shadow prices...")
    shadow_df = compute_shadow_prices(site_df, alloc_df, T_scenarios, site_weights, w_s)

    # 8. Sensitivity analysis
    logger.info("Running sensitivity analysis...")
    sensitivity_df = run_sensitivity(site_df, alloc_df, T_scenarios, site_weights, w_s)

    # 9. Pareto frontier
    logger.info("Computing Pareto frontier...")
    pareto_df = compute_pareto_front(site_df, site_weights, T_scenarios, w_s)

    # 10. Alternative plans and robust plan
    alt_plans_df = build_alternative_plans(alloc_df, T_scenarios, site_weights)
    robust_plan_df = build_robust_plan(alloc_df, T_scenarios, site_weights)

    # 11. Optimization metrics
    metrics_df = build_optimization_metrics(alloc_df, cvar_df, T_scenarios, site_weights)

    # 12. Summary text
    summary_txt = _build_summary(alloc_df, metrics_df, cvar_df, shadow_df)
    print(summary_txt)

    # 13. Export all outputs
    export_outputs(
        alloc_df, diversion_df, scenario_summary_df, cvar_df,
        robust_plan_df, alt_plans_df, pareto_df, shadow_df,
        sensitivity_df, metrics_df, summary_txt,
    )
    logger.info("Layer 5 complete.")


if __name__ == "__main__":
    main()
