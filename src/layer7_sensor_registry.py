"""
Layer 7 — M7A Step 1: Canonical Sensor Registry.

Defines the supported sensor taxonomy and builds a deterministic SIMULATED registry
mapped onto the Layer 5 active sites (corridor/junction from the L4.5 as-of matrix).
No live sensors exist, so every entry is replay/simulated; status is varied to exercise
the downstream health model.

ADDITIVE ONLY. Writes only outputs/layer7_sensor_registry.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_SEED = 7  # deterministic registry

SENSOR_TYPES = [
    "CCTV", "ANPR", "Loop Detector", "Radar", "LiDAR", "Bluetooth Probe",
    "GPS Probe", "Signal Controller", "Weather Feed", "Incident Dispatch Feed",
    "Roadwork Feed", "Event Feed",
]

STATUSES = ["ACTIVE", "DEGRADED", "STALE", "OFFLINE", "SIMULATED"]
# status sampling weights (mostly ACTIVE; a few OFFLINE to exercise fallback)
_STATUS_P = [0.55, 0.15, 0.12, 0.08, 0.10]

# which traffic quantities each sensor type observes
SENSOR_QUANTITIES = {
    "CCTV": ["queue_length", "incident_indicator", "lane_availability"],
    "ANPR": ["travel_time"],
    "Loop Detector": ["traffic_speed", "queue_length"],
    "Radar": ["traffic_speed"],
    "LiDAR": ["queue_length", "lane_availability"],
    "Bluetooth Probe": ["travel_time"],
    "GPS Probe": ["traffic_speed", "travel_time"],
    "Signal Controller": ["queue_length", "lane_availability"],
    "Weather Feed": ["incident_indicator"],
    "Incident Dispatch Feed": ["incident_indicator"],
    "Roadwork Feed": ["lane_availability"],
    "Event Feed": ["incident_indicator", "lane_availability"],
}


def build_registry() -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)
    astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    asof = pd.read_csv(OUT / "layer45_asof_feature_matrix.csv")[["event_id", "corridor", "junction"]]
    asof["event_id"] = asof["event_id"].astype(str)
    asof = asof.drop_duplicates("event_id")
    sites = astate[["event_id"]].merge(asof, on="event_id", how="left")
    sites["corridor"] = sites["corridor"].fillna("Non-corridor")
    sites["junction"] = sites["junction"].fillna("unknown")

    rows = []
    for _, s in sites.iterrows():
        # 3-6 sensors per site, distinct types
        n = int(rng.integers(3, 7))
        types = list(rng.choice(SENSOR_TYPES, size=n, replace=False))
        for j, st in enumerate(types):
            status = str(rng.choice(STATUSES, p=_STATUS_P))
            rows.append({
                "sensor_id": f"S-{s['event_id']}-{j:02d}",
                "sensor_type": st,
                "corridor": s["corridor"],
                "junction": s["junction"],
                "event_id": s["event_id"],
                "source": "SIMULATED",
                "status": status,
                "sensor_mode": "REPLAY",
                "generated_at": _NOW_ISO,
            })
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_registry()
    if write:
        df.to_csv(OUT / "layer7_sensor_registry.csv", index=False)
    checks = [{
        "check_id": "m7a_registry_built", "phase": "sensor_registry",
        "passed": len(df) > 0 and set(df["status"]) <= set(STATUSES)
                  and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} sensors across {df['event_id'].nunique()} sites; "
                  f"{df['sensor_type'].nunique()} types; status={df['status'].value_counts().to_dict()}",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
