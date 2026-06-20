"""
Layer 7 — M7B Step 9: state consistency checks + Kalman stability metrics.

Detects: negative speeds, negative queues, travel-time inconsistency, capacity > 1,
variance explosions; aggregates Kalman stability across sites. No I/O of its own;
returns a diagnostics DataFrame for the orchestrator to write.

ADDITIVE ONLY.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_VAR_EXPLOSION = 1e6


def build_diagnostics(est: pd.DataFrame, cap: pd.DataFrame, stab: list[dict]) -> pd.DataFrame:
    rows = []

    def add(group, metric, value, flag=""):
        rows.append({"diagnostic_group": group, "metric": metric, "value": value,
                     "flag": flag, "generated_at": _NOW_ISO})

    wide = est.pivot_table(index="event_id", columns="state_name",
                           values="state_value", aggfunc="first")

    n_neg_speed = int((wide.get("speed", pd.Series(dtype=float)) < 0).sum())
    n_neg_queue = int((wide.get("queue_length", pd.Series(dtype=float)) < 0).sum())
    # travel-time inconsistency: very low speed yet very low travel time (physically odd)
    if "speed" in wide and "travel_time" in wide:
        inconsist = int(((wide["speed"] < 15) & (wide["travel_time"] < 10)).sum())
    else:
        inconsist = 0
    n_cap_over = int((cap["capacity_utilization"] > 1.0 + 1e-9).sum())
    n_var_explosion = int((est["posterior_variance"] > _VAR_EXPLOSION).sum())

    add("consistency", "negative_speeds", n_neg_speed, "OK" if n_neg_speed == 0 else "VIOLATION")
    add("consistency", "negative_queues", n_neg_queue, "OK" if n_neg_queue == 0 else "VIOLATION")
    add("consistency", "travel_time_inconsistencies", inconsist, "OK" if inconsist == 0 else "WARN")
    add("consistency", "capacity_over_1", n_cap_over, "OK" if n_cap_over == 0 else "VIOLATION")
    add("consistency", "variance_explosions", n_var_explosion, "OK" if n_var_explosion == 0 else "VIOLATION")

    sr = pd.DataFrame(stab)
    add("kalman_stability", "n_sites", int(len(sr)))
    add("kalman_stability", "all_stable", bool(sr["stable"].all()),
        "OK" if bool(sr["stable"].all()) else "UNSTABLE")
    add("kalman_stability", "max_spectral_radius_A", round(float(sr["spectral_radius_A"].max()), 6))
    add("kalman_stability", "max_posterior_variance", round(float(sr["max_posterior_variance"].max()), 4))
    add("kalman_stability", "min_P_eigenvalue", round(float(sr["min_P_eigenvalue"].min()), 8))
    add("kalman_stability", "max_innovation_cov_cond", round(float(sr["max_innovation_cov_cond"].max()), 4))
    add("kalman_stability", "kalman_updates_per_site", int(sr["update_count"].max()))

    add("state_finiteness", "all_state_values_finite",
        bool(np.isfinite(est["state_value"]).all()),
        "OK" if bool(np.isfinite(est["state_value"]).all()) else "VIOLATION")
    add("state_finiteness", "all_variances_nonneg",
        bool((est["posterior_variance"] >= 0).all()),
        "OK" if bool((est["posterior_variance"] >= 0).all()) else "VIOLATION")

    return pd.DataFrame(rows)
