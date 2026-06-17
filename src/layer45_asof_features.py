"""
Layer 4.5 — point-in-time (as-of) surrogate features from raw incident history.

Uses daily snapshot cadence: all rows on day D share feature state built from
incidents with start_local strictly before D 00:00:00.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FALLBACK_LEVELS = ("cause_corridor", "cause_only", "corridor_only", "global")
MIN_STRATUM = 5
DECAY_LAMBDA = 0.05
HAWKES_BETA = 0.1
RETRIEVAL_MIN_SUPPORT = 3

RETRIEVAL_NUM_COLS = ["hour_local", "dow_local", "requires_road_closure"]
RETRIEVAL_CAT_COLS = ["event_cause", "corridor", "zone", "priority"]


@dataclass
class FittedParams:
    """Parameters estimated from training window only."""
    decay_lambda: float = DECAY_LAMBDA
    hawkes_mu: float = 0.01
    hawkes_alpha: float = 0.05
    hawkes_beta: float = HAWKES_BETA
    high_impact_tau: float = 60.0
    cause_tau: dict[str, float] = field(default_factory=dict)
    corridor_mu: dict[str, float] = field(default_factory=dict)
    corridor_alpha: dict[str, float] = field(default_factory=dict)
    global_mu: float = 0.01
    global_alpha: float = 0.05
    snapshot_cadence: str = "daily"


def _resolved_hist(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    if "is_censored" in h.columns:
        h = h[~h["is_censored"].astype(bool)]
    h = h[h["duration_min"].notna() & (h["duration_min"] > 0)]
    return h


def apply_fallback_chain(
    cause: str,
    corridor: str,
    durations: pd.Series,
    causes: pd.Series,
    corridors: pd.Series,
    min_n: int = MIN_STRATUM,
) -> tuple[float, float, float, str, int]:
    """Return p50, p80, p95, fallback_level, support_count."""
    masks = [
        ("cause_corridor", (causes == cause) & (corridors == corridor)),
        ("cause_only", causes == cause),
        ("corridor_only", corridors == corridor),
        ("global", pd.Series(True, index=durations.index)),
    ]
    for level, mask in masks:
        sub = durations[mask]
        if len(sub) >= min_n:
            return (
                float(sub.quantile(0.50)),
                float(sub.quantile(0.80)),
                float(sub.quantile(0.95)),
                level,
                int(len(sub)),
            )
    sub = durations
    if len(sub) == 0:
        return (60.0, 120.0, 180.0, "global", 0)
    return (
        float(sub.quantile(0.50)),
        float(sub.quantile(0.80)),
        float(sub.quantile(0.95)),
        "global",
        int(len(sub)),
    )


def _survival_prob(durations: np.ndarray, horizon: float) -> float:
    if len(durations) == 0:
        return 0.5
    return float(np.mean(durations > horizon))


def _rmst(durations: np.ndarray, horizon: float = 180.0) -> float:
    if len(durations) == 0:
        return horizon / 2
    capped = np.minimum(durations, horizon)
    return float(np.mean(capped))


def _gower_row(
    row: pd.Series,
    proto: pd.DataFrame,
    num_ranges: dict[str, float],
) -> np.ndarray:
    if proto.empty:
        return np.array([])
    n = len(proto)
    sims = np.zeros(n)
    n_feat = len(RETRIEVAL_NUM_COLS) + len(RETRIEVAL_CAT_COLS)
    for j, (_, p) in enumerate(proto.iterrows()):
        match = 0.0
        for col in RETRIEVAL_NUM_COLS:
            rv = row.get(col, np.nan)
            pv = p.get(col, np.nan)
            if pd.isna(rv) or pd.isna(pv):
                continue
            try:
                rv_f, pv_f = float(rv), float(pv)
            except (TypeError, ValueError):
                rv_f, pv_f = float(bool(rv)), float(bool(pv))
            denom = num_ranges.get(col, 1.0) or 1.0
            match += 1.0 - min(abs(rv_f - pv_f) / denom, 1.0)
        for col in RETRIEVAL_CAT_COLS:
            rv = str(row.get(col, ""))
            pv = str(p.get(col, ""))
            match += 1.0 if rv == pv else 0.0
        sims[j] = match / max(n_feat, 1)
    return sims


def _fit_hawkes_on_train(train_df: pd.DataFrame) -> FittedParams:
    """Estimate global and per-corridor Hawkes-like parameters from training only."""
    params = FittedParams()
    resolved = _resolved_hist(train_df)
    if resolved.empty:
        return params

    span_days = max(
        (resolved["start_local"].max() - resolved["start_local"].min()).total_seconds() / 86400,
        1.0,
    )
    params.hawkes_mu = len(resolved) / (span_days * max(resolved["corridor"].nunique(), 1))
    params.global_mu = params.hawkes_mu
    params.global_alpha = min(0.2, params.hawkes_mu * 0.5)
    params.high_impact_tau = float(resolved["duration_min"].quantile(0.75))
    for cause, grp in resolved.groupby("event_cause"):
        c = str(cause)
        if len(grp) >= MIN_STRATUM:
            params.cause_tau[c] = float(grp["duration_min"].quantile(0.75))
        else:
            params.cause_tau[c] = params.high_impact_tau

    for corridor, grp in resolved.groupby("corridor"):
        n = len(grp)
        mu_c = n / max(span_days, 1.0)
        params.corridor_mu[str(corridor)] = mu_c
        params.corridor_alpha[str(corridor)] = min(0.3, mu_c * 0.4)

    return params


def _corridor_burden(
    hist: pd.DataFrame,
    t_ref: pd.Timestamp,
    decay_lambda: float,
) -> dict[str, float]:
    burdens: dict[str, float] = {}
    if hist.empty:
        return burdens
    dt_days = (t_ref - hist["start_local"]).dt.total_seconds() / 86400.0
    weights = np.exp(-decay_lambda * np.maximum(dt_days.values, 0))
    for corridor, w in zip(hist["corridor"].astype(str), weights):
        burdens[corridor] = burdens.get(corridor, 0.0) + float(w)
    return burdens


def _hawkes_intensity(
    hist: pd.DataFrame,
    t_ref: pd.Timestamp,
    corridor: str,
    params: FittedParams,
) -> float:
    sub = hist[hist["corridor"].astype(str) == str(corridor)]
    mu = params.corridor_mu.get(str(corridor), params.global_mu)
    alpha = params.corridor_alpha.get(str(corridor), params.global_alpha)
    if sub.empty:
        return mu
    dt = (t_ref - sub["start_local"]).dt.total_seconds() / 3600.0
    excitation = alpha * np.exp(-params.hawkes_beta * np.maximum(dt.values, 0)).sum()
    return float(mu + excitation)


def _burstiness(hist: pd.DataFrame, corridor: str, window_h: float = 24.0) -> float:
    sub = hist[hist["corridor"].astype(str) == str(corridor)]
    if len(sub) < 2:
        return 0.0
    t_max = sub["start_local"].max()
    recent = sub[sub["start_local"] >= t_max - pd.Timedelta(hours=window_h)]
    if len(recent) < 2:
        return 0.0
    gaps = recent["start_local"].sort_values().diff().dt.total_seconds() / 60.0
    gaps = gaps.dropna()
    if gaps.empty or gaps.mean() == 0:
        return 0.0
    return float(gaps.std() / (gaps.mean() + 1e-6))


def _zone_neighbor_corridors(df: pd.DataFrame) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for zone, grp in df.groupby("zone"):
        corridors = grp["corridor"].astype(str).unique().tolist()
        for c in corridors:
            mapping[c] = [x for x in corridors if x != c]
    return mapping


def _snapshot_state(
    hist: pd.DataFrame,
    snap: pd.Timestamp,
    params: FittedParams,
    zone_nbrs: dict[str, list[str]],
) -> dict[str, Any]:
    resolved = _resolved_hist(hist)
    burdens = _corridor_burden(hist, snap, params.decay_lambda)
    j_burden: dict[str, float] = {}
    if not hist.empty:
        dt_days = (snap - hist["start_local"]).dt.total_seconds() / 86400.0
        weights = np.exp(-params.decay_lambda * np.maximum(dt_days.values, 0))
        for junc, w in zip(hist["junction"].astype(str), weights):
            j_burden[junc] = j_burden.get(junc, 0.0) + float(w)

    dur_by_cc: dict[tuple, np.ndarray] = {}
    if not resolved.empty:
        for (c, r), grp in resolved.groupby(["event_cause", "corridor"]):
            dur_by_cc[(str(c), str(r))] = grp["duration_min"].values

    planned = hist[hist.get("is_true_planned_event", False).astype(bool)] if "is_true_planned_event" in hist.columns else pd.DataFrame()

    num_ranges = {}
    for col in RETRIEVAL_NUM_COLS:
        if col not in hist.columns or not hist[col].notna().any():
            num_ranges[col] = 1.0
            continue
        vals = pd.to_numeric(hist[col], errors="coerce").astype(float)
        if vals.notna().sum() == 0:
            num_ranges[col] = 1.0
        else:
            num_ranges[col] = float(vals.max() - vals.min()) or 1.0

    corridor_durations = (
        resolved.groupby("corridor")["duration_min"].apply(lambda x: x.values).to_dict()
        if not resolved.empty
        else {}
    )
    corridor_rates = hist.groupby("corridor").size().to_dict() if not hist.empty else {}

    return {
        "resolved": resolved,
        "burdens": burdens,
        "j_burden": j_burden,
        "dur_by_cc": dur_by_cc,
        "planned": planned,
        "num_ranges": num_ranges,
        "corridor_durations": corridor_durations,
        "corridor_rates": corridor_rates,
        "zone_nbrs": zone_nbrs,
        "snap": snap,
    }


def _row_features(
    row: pd.Series,
    state: dict[str, Any],
    params: FittedParams,
) -> dict[str, Any]:
    cause = str(row.get("event_cause", ""))
    corridor = str(row.get("corridor", ""))
    junction = str(row.get("junction", ""))
    snap = state["snap"]
    resolved: pd.DataFrame = state["resolved"]
    hist_for_hawkes = resolved  # prefix-only resolved events

    if not resolved.empty:
        p50, p80, p95, fb_level, support = apply_fallback_chain(
            cause,
            corridor,
            resolved["duration_min"],
            resolved["event_cause"].astype(str),
            resolved["corridor"].astype(str),
        )
        cc_durs = resolved[
            (resolved["event_cause"].astype(str) == cause)
            & (resolved["corridor"].astype(str) == corridor)
        ]["duration_min"].values
        if len(cc_durs) < MIN_STRATUM:
            for level, mask_fn in [
                ("cause_only", lambda d: d["event_cause"].astype(str) == cause),
                ("corridor_only", lambda d: d["corridor"].astype(str) == corridor),
                ("global", lambda d: pd.Series(True, index=d.index)),
            ]:
                sub = resolved[mask_fn(resolved)]["duration_min"].values
                if len(sub) >= MIN_STRATUM:
                    cc_durs = sub
                    break
    else:
        p50, p80, p95, fb_level, support = 60.0, 120.0, 180.0, "global", 0
        cc_durs = np.array([])

    surv60 = _survival_prob(cc_durs, 60)
    surv120 = _survival_prob(cc_durs, 120)
    surv180 = _survival_prob(cc_durs, 180)
    rmst180 = _rmst(cc_durs, 180)

    burden = state["burdens"].get(corridor, 0.0)
    j_burden = state["j_burden"].get(junction, 0.0)
    obi_proxy = burden / (1.0 + j_burden)
    event_rate = state["corridor_rates"].get(corridor, 0)

    hawkes_i = _hawkes_intensity(hist_for_hawkes, snap, corridor, params)
    fragility = params.corridor_mu.get(corridor, params.global_mu) + 0.5 * hawkes_i
    burst = _burstiness(hist_for_hawkes, corridor)
    branching = min(hawkes_i / (params.corridor_mu.get(corridor, params.global_mu) + 1e-6), 3.0)

    planned: pd.DataFrame = state["planned"]
    sims = _gower_row(row, planned, state["num_ranges"])
    if len(sims) > 0:
        mean_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))
        n_eff = float(len(sims))
        conf = (n_eff / (n_eff + 2.0)) * mean_sim * max_sim
        ims = mean_sim * np.log1p(n_eff)
        planned_support = int(len(sims))
    else:
        mean_sim = max_sim = conf = ims = 0.0
        n_eff = 0.0
        planned_support = 0

    retrieval_fb = "exact" if planned_support >= RETRIEVAL_MIN_SUPPORT else "low_support"

    zone_nbrs = state["zone_nbrs"].get(corridor, [])
    nbr_burdens = [state["burdens"].get(c, 0.0) for c in zone_nbrs]
    nbr_durs = []
    for c in zone_nbrs:
        d = state["corridor_durations"].get(c, np.array([]))
        if len(d):
            nbr_durs.append(float(np.median(d)))
    nbr_mean_burden = float(np.mean(nbr_burdens)) if nbr_burdens else 0.0
    nbr_mean_duration = float(np.mean(nbr_durs)) if nbr_durs else p50
    nbr_mean_severity = nbr_mean_duration * nbr_mean_burden

    trust = float(row.get("trust_score", 0.5) or 0.5)
    tau_c = params.cause_tau.get(cause, params.high_impact_tau)
    high_impact_proxy = 1.0 if p80 > tau_c else 0.0

    out = {
        "snapshot_date": snap.date().isoformat(),
        "asof_p50_duration": p50,
        "asof_p80_duration": p80,
        "asof_p95_duration": p95,
        "asof_surv_prob_60": surv60,
        "asof_surv_prob_120": surv120,
        "asof_surv_prob_180": surv180,
        "asof_rmst_180": rmst180,
        "asof_quantile_fallback_level": fb_level,
        "asof_quantile_support": support,
        "asof_corridor_burden": burden,
        "asof_junction_burden": j_burden,
        "asof_obi_proxy": obi_proxy,
        "asof_corridor_event_rate": float(event_rate),
        "asof_nbr_mean_burden": nbr_mean_burden,
        "asof_nbr_mean_duration": nbr_mean_duration,
        "asof_fragility_proxy": fragility,
        "asof_hawkes_intensity": hawkes_i,
        "asof_burstiness": burst,
        "asof_branching_ratio_proxy": branching,
        "asof_retrieval_confidence": conf,
        "asof_retrieval_n_eff": n_eff,
        "asof_retrieval_mean_sim": mean_sim,
        "asof_retrieval_max_sim": max_sim,
        "asof_planned_support": planned_support,
        "asof_ims_proxy": ims,
        "asof_retrieval_fallback": retrieval_fb,
        "obi_x_fragility": obi_proxy * fragility,
        "retrieval_x_risk": conf * high_impact_proxy,
        "fragility_x_duration": fragility * p50,
        "trust_x_confidence": trust * conf,
        "hotspot_x_duration": burden * p50,
        "hawkes_x_obi": hawkes_i * obi_proxy,
        "nbr_mean_obi": nbr_mean_burden,
        "nbr_mean_fragility": fragility,
        "nbr_mean_duration": nbr_mean_duration,
        "nbr_mean_severity": nbr_mean_severity,
        "coverage_flag": fb_level != "global" or support >= MIN_STRATUM,
    }
    return out


def build_asof_feature_matrix(
    df: pd.DataFrame,
    train_mask: pd.Series,
    snapshot_cadence: str = "daily",
) -> tuple[pd.DataFrame, FittedParams, list[dict]]:
    """
    Build leak-free as-of features for all rows.
    Parameters (Hawkes, tau, decay) fit on train_mask rows only.
    """
    sort_idx = df.sort_values("start_local").index
    work = df.loc[sort_idx].reset_index(drop=True).copy()
    work["start_local"] = pd.to_datetime(work["start_local"], errors="coerce")
    if "month" not in work.columns:
        work["month"] = work["start_local"].dt.month

    train_on_work = train_mask.reindex(sort_idx).fillna(False).values
    train_df = work.loc[train_on_work]
    params = _fit_hawkes_on_train(train_df)
    params.snapshot_cadence = snapshot_cadence

    zone_nbrs = _zone_neighbor_corridors(work)

    min_d = work["start_local"].min().normalize()
    max_d = work["start_local"].max().normalize()
    if snapshot_cadence == "weekly":
        snap_dates = pd.date_range(min_d, max_d, freq="W-MON", tz=work["start_local"].dt.tz)
    else:
        snap_dates = pd.date_range(min_d, max_d + pd.Timedelta(days=1), freq="D", tz=work["start_local"].dt.tz)

    if len(snap_dates) == 0:
        snap_dates = pd.DatetimeIndex([min_d])

    # Map each row -> snapshot (last snap strictly before row's calendar day start)
    row_day = work["start_local"].dt.normalize()
    snap_list = list(snap_dates)
    row_snap_idx = np.searchsorted(
        [s.value for s in snap_list],
        [d.value for d in row_day],
        side="left",
    ) - 1
    row_snap_idx = np.clip(row_snap_idx, 0, len(snap_list) - 1)

    # Precompute snapshot states (one per calendar day; avoids cache eviction bugs)
    unique_sis = sorted(set(int(x) for x in row_snap_idx))
    snap_states: dict[int, dict] = {}
    for si in unique_sis:
        snap = snap_list[si]
        hist = work[work["start_local"] < snap]
        snap_states[si] = _snapshot_state(hist, snap, params, zone_nbrs)
    logger.info("Precomputed %d daily snapshot states", len(snap_states))

    fallback_log: list[dict] = []
    feature_rows: list[dict] = []

    for pos, (_, row) in enumerate(work.iterrows()):
        si = int(row_snap_idx[pos])
        feats = _row_features(row, snap_states[si], params)
        feats["event_id"] = row.get("event_id", pos)
        feats["start_local"] = row["start_local"]
        fallback_log.append({
            "event_id": feats["event_id"],
            "snapshot_date": feats["snapshot_date"],
            "quantile_fallback": feats["asof_quantile_fallback_level"],
            "retrieval_fallback": feats["asof_retrieval_fallback"],
            "quantile_support": feats["asof_quantile_support"],
            "planned_support": feats["asof_planned_support"],
            "event_cause": row.get("event_cause"),
            "corridor": row.get("corridor"),
            "coverage_flag": feats["coverage_flag"],
        })
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    base_cols = [
        "event_cause", "corridor", "zone", "junction", "priority",
        "requires_road_closure", "event_type", "hour_local", "dow_local",
        "month", "is_weekend", "trust_score", "geo_valid", "duration_anomaly",
        "iso_flagged", "mnar_predicted_prob", "is_true_planned_event",
    ]

    meta = work[["event_id", "start_local", "duration_min", "is_censored"]].copy()
    for c in base_cols:
        if c in work.columns:
            meta[c] = work[c].values
    asof_only = [c for c in feat_df.columns if c not in base_cols and c not in meta.columns]
    out = pd.concat([meta, feat_df[asof_only]], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    logger.info("Built as-of matrix: %d rows, %d snapshots", len(out), len(snap_list))
    return out, params, fallback_log


def update_running_statistics(
    hist: pd.DataFrame,
    params: FittedParams,
) -> dict[str, Any]:
    """Expose prefix state builder for deployment inference."""
    snap = hist["start_local"].max() + pd.Timedelta(seconds=1) if not hist.empty else pd.Timestamp.now(tz="UTC")
    zone_nbrs = _zone_neighbor_corridors(hist)
    return _snapshot_state(hist, snap, params, zone_nbrs)
