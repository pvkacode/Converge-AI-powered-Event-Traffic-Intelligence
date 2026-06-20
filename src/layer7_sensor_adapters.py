"""
Layer 7 — M7A Step 3 + Future Live Mode: Sensor Adapters.

Two adapters behind one interface so the framework runs with zero real sensors:

  ReplaySensorAdapter  — generates SIMULATED observations from Layer 5 / Layer 6
                         operational signals (sensor_mode=REPLAY). Never real.
  LiveSensorAdapter    — future real-time seam; poll() returns no observations today.

Replay observations are derived from a per-site "truth" (operational risk, alert
activity, duration estimates, diversion) that each sensor observes with reliability-
scaled noise. OFFLINE sensors emit nothing (drives fallback downstream).

ADDITIVE ONLY. This module writes nothing on import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from layer7_config import OUT
from layer7_sensor_registry import SENSOR_QUANTITIES

_SEED = 101

QUANTITIES = ["traffic_speed", "queue_length", "travel_time",
              "incident_indicator", "lane_availability"]
# natural scale (range) of each quantity, used to map normalized sensor sigma -> units
Q_SCALE = {"traffic_speed": 65.0, "queue_length": 600.0, "travel_time": 240.0,
           "incident_indicator": 1.0, "lane_availability": 1.0}


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_site_truth() -> pd.DataFrame:
    """Per active-site latent value for each fused quantity, from L5/L6 signals.
    Also used by the fallback path when a site has no usable sensors."""
    st = _read("layer7_operational_state.csv")
    st["event_id"] = st["event_id"].astype(str)
    active = _read("layer7_active_site_state.csv")
    active["event_id"] = active["event_id"].astype(str)
    df = active[["event_id"]].merge(
        st[["event_id", "operational_risk_score"]], on="event_id", how="left")
    df["r"] = (pd.to_numeric(df["operational_risk_score"], errors="coerce").fillna(0) / 100.0).clip(0, 1)

    alerts = _read("layer7_prioritized_alerts.csv")
    if len(alerts):
        alerts["affected_event_id"] = alerts["affected_event_id"].astype(str)
        nc = alerts.groupby("affected_event_id").size()
        df["alert_density"] = df["event_id"].map(lambda e: min(1.0, int(nc.get(e, 0)) / 3.0))
    else:
        df["alert_density"] = 0.0

    sr = _read("layer45_scenario_ready_duration.csv")
    if len(sr):
        sr["event_id"] = sr["event_id"].astype(str)
        df = df.merge(sr[["event_id", "safe_duration_p50"]], on="event_id", how="left")
    if "safe_duration_p50" not in df.columns:
        df["safe_duration_p50"] = 0.0
    df["safe_duration_p50"] = pd.to_numeric(df["safe_duration_p50"], errors="coerce").fillna(0.0)

    alloc = _read("layer5_resource_allocation.csv")
    div = {}
    if len(alloc):
        alloc["event_id"] = alloc["event_id"].astype(str)
        div = dict(zip(alloc["event_id"],
                       alloc["diversion_activated"].astype(str).str.lower().isin(["1", "true", "yes"])))
    df["diversion"] = df["event_id"].map(lambda e: 1.0 if div.get(e, False) else 0.0)

    r = df["r"].to_numpy()
    df["traffic_speed"] = np.clip(60 - 45 * r, 5, 70)
    df["queue_length"] = np.clip(50 + 450 * r, 0, 600)
    df["travel_time"] = np.clip(5 + 0.02 * df["safe_duration_p50"].to_numpy() + 30 * r, 1, 240)
    df["incident_indicator"] = np.clip(0.05 + 0.7 * r + 0.2 * df["alert_density"].to_numpy(), 0, 1)
    df["lane_availability"] = np.clip(1 - 0.7 * r - 0.2 * df["diversion"].to_numpy(), 0.1, 1)
    return df[["event_id"] + QUANTITIES]


class LiveSensorAdapter:
    """Future real-time seam. No live sensors exist today -> returns no observations."""

    mode = "LIVE"

    def poll(self, registry: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "sensor_id", "event_id", "sensor_type", "quantity", "observed_value",
            "sensor_reliability", "sensor_obs_variance", "sensor_mode", "status"])


class ReplaySensorAdapter:
    """Generates simulated observations from L5/L6 truth + registry/health."""

    mode = "REPLAY"

    def poll(self, registry: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(_SEED)
        truth = build_site_truth().set_index("event_id")
        hmap = health.set_index("sensor_id")[["sensor_reliability", "sensor_sigma"]].to_dict("index")

        rows = []
        for _, s in registry.iterrows():
            if str(s["status"]).upper() == "OFFLINE":
                continue  # offline sensors emit nothing -> fallback driver
            eid = str(s["event_id"])
            if eid not in truth.index:
                continue
            h = hmap.get(s["sensor_id"], {"sensor_reliability": 0.5, "sensor_sigma": 0.2})
            R = float(h["sensor_reliability"])
            sigma_norm = float(h["sensor_sigma"])
            for q in SENSOR_QUANTITIES.get(str(s["sensor_type"]), []):
                true_v = float(truth.loc[eid, q])
                sigma_u = sigma_norm * Q_SCALE[q]
                obs = true_v + float(rng.normal(0, sigma_u))
                if q in ("incident_indicator", "lane_availability"):
                    obs = float(np.clip(obs, 0, 1))
                else:
                    obs = float(max(0.0, obs))
                rows.append({
                    "sensor_id": s["sensor_id"], "event_id": eid,
                    "sensor_type": s["sensor_type"], "quantity": q,
                    "observed_value": round(obs, 4),
                    "sensor_reliability": round(R, 4),
                    "sensor_obs_variance": round(sigma_u ** 2, 8),
                    "sensor_mode": "REPLAY", "status": s["status"],
                })
        return pd.DataFrame(rows)
