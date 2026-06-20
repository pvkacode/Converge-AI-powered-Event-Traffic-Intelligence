"""
Layer 7 — M7 Part C: Real-Time Ingestion Simulator.

Generates a SYNTHETIC feed of future live events for offline testing. NO external
data, NO network, NO streaming — deterministic synthetic generation only (fixed
seed + synthetic timestamps), so output is reproducible.

Feed types: sensor_update, incident_update, operator_update, drift_alert, resource_change.

ADDITIVE ONLY. Writes only outputs/layer7_simulated_feed.csv.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_BASE = datetime(2026, 6, 19, 0, 0, 0, tzinfo=timezone.utc)  # synthetic clock (deterministic)

FEED_TYPES = ["sensor_update", "incident_update", "operator_update",
              "drift_alert", "resource_change"]
_SENSOR_TYPES = ["traffic_camera", "gps_probe", "roadside_unit", "weather_feed"]


def build_feed(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)  # fixed seed -> reproducible
    astate = OUT / "layer7_active_site_state.csv"
    site_ids = (pd.read_csv(astate)["event_id"].astype(str).tolist()
                if astate.exists() else [f"SITE-{i:03d}" for i in range(50)])

    rows = []
    for i in range(n):
        ftype = FEED_TYPES[i % len(FEED_TYPES)]
        ts = (_BASE + timedelta(minutes=5 * i)).isoformat()
        eid = site_ids[int(rng.integers(0, len(site_ids)))]
        conf = round(float(rng.uniform(0.5, 0.99)), 4)
        if ftype == "sensor_update":
            st = _SENSOR_TYPES[int(rng.integers(0, len(_SENSOR_TYPES)))]
            payload = f"sensor_type={st};reading={round(float(rng.uniform(0,100)),1)}"
        elif ftype == "incident_update":
            payload = f"status={'cleared' if rng.random() > 0.5 else 'ongoing'};severity={int(rng.integers(1,5))}"
        elif ftype == "operator_update":
            payload = f"operator=OP{int(rng.integers(1,9))};action=acknowledge"
        elif ftype == "drift_alert":
            payload = f"variable=log_duration;magnitude={round(float(rng.uniform(0,5)),2)}"
        else:  # resource_change
            payload = f"resource={['police','tow','barricades','qru'][int(rng.integers(0,4))]};delta={int(rng.integers(-3,4))}"
        rows.append({
            "feed_event_id": f"FEED-{i:04d}",
            "feed_type": ftype,
            "event_id": eid,
            "synthetic_timestamp": ts,
            "simulated_confidence": conf,
            "payload_summary": payload,
            "source": "ingestion_simulator",
            "is_synthetic": True,
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


def run(write: bool = True, n: int = 60) -> tuple[pd.DataFrame, list[dict]]:
    df = build_feed(n)
    if write:
        df.to_csv(OUT / "layer7_simulated_feed.csv", index=False)
    checks = [{
        "check_id": "m7_ingestion_simulator_ran", "phase": "ingestion_simulator",
        "passed": len(df) == n and set(df["feed_type"]) == set(FEED_TYPES)
                  and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} synthetic feed events; types={sorted(set(df['feed_type']))}",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head(6).to_string(index=False))
    print("type counts:", df["feed_type"].value_counts().to_dict())
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
