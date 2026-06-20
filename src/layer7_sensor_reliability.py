"""
Layer 7 — M7A Step 2: Sensor Health / Reliability Model.

Per-sensor reliability R_i in [0,1] from uptime / missingness / latency / trust, plus
a health tier and the measurement variance used by Bayesian fusion. Synthetic but
deterministic, and anchored to the sensor's registry status so health aligns with status.

    R_i = w1*u + w2*(1-m) + w3*(1-l) + w4*t

ADDITIVE ONLY. Writes only outputs/layer7_sensor_health.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_SEED = 23

# reliability component weights (sum to 1)
W_UPTIME, W_MISS, W_LAT, W_TRUST = 0.30, 0.25, 0.20, 0.25

# per-status nominal profile: (uptime, missingness, latency, trust) baselines in [0,1]
_STATUS_PROFILE = {
    "ACTIVE":    (0.97, 0.03, 0.08, 0.92),
    "DEGRADED":  (0.80, 0.20, 0.35, 0.70),
    "STALE":     (0.60, 0.45, 0.55, 0.55),
    "OFFLINE":   (0.02, 0.98, 0.95, 0.10),
    "SIMULATED": (0.85, 0.12, 0.20, 0.75),
}
# base measurement std per sensor type (units are per-quantity-normalized; scaled by 1-R)
_TYPE_BASE_SIGMA = {
    "CCTV": 0.12, "ANPR": 0.10, "Loop Detector": 0.08, "Radar": 0.09,
    "LiDAR": 0.07, "Bluetooth Probe": 0.13, "GPS Probe": 0.11,
    "Signal Controller": 0.09, "Weather Feed": 0.20, "Incident Dispatch Feed": 0.15,
    "Roadwork Feed": 0.16, "Event Feed": 0.18,
}


def _tier(r: float) -> str:
    if r >= 0.75:
        return "HIGH"
    if r >= 0.50:
        return "MEDIUM"
    if r >= 0.25:
        return "LOW"
    return "FAILED"


def build_health(registry: pd.DataFrame | None = None) -> pd.DataFrame:
    if registry is None:
        registry = pd.read_csv(OUT / "layer7_sensor_registry.csv")
    rng = np.random.default_rng(_SEED)

    rows = []
    for _, s in registry.iterrows():
        prof = _STATUS_PROFILE.get(str(s["status"]), _STATUS_PROFILE["SIMULATED"])
        jit = rng.normal(0, 0.03, size=4)  # small deterministic jitter
        u = float(np.clip(prof[0] + jit[0], 0, 1))
        m = float(np.clip(prof[1] + jit[1], 0, 1))
        lat = float(np.clip(prof[2] + jit[2], 0, 1))
        t = float(np.clip(prof[3] + jit[3], 0, 1))
        R = float(np.clip(W_UPTIME * u + W_MISS * (1 - m) + W_LAT * (1 - lat) + W_TRUST * t, 0, 1))
        base_sigma = _TYPE_BASE_SIGMA.get(str(s["sensor_type"]), 0.15)
        # measurement variance grows as reliability falls
        sigma = base_sigma * (1.0 + 2.0 * (1.0 - R))
        rows.append({
            "sensor_id": s["sensor_id"], "sensor_type": s["sensor_type"],
            "event_id": s["event_id"], "status": s["status"],
            "uptime": round(u, 4), "missingness": round(m, 4),
            "latency": round(lat, 4), "trust": round(t, 4),
            "sensor_reliability": round(R, 4), "sensor_health_tier": _tier(R),
            "sensor_sigma": round(sigma, 6), "sensor_variance": round(sigma ** 2, 8),
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_health()
    if write:
        df.to_csv(OUT / "layer7_sensor_health.csv", index=False)
    r = df["sensor_reliability"]
    checks = [{
        "check_id": "m7a_health_built", "phase": "sensor_reliability",
        "passed": len(df) > 0 and bool(((r >= 0) & (r <= 1)).all())
                  and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} sensors; R in [{r.min():.3f},{r.max():.3f}]; "
                  f"tiers={df['sensor_health_tier'].value_counts().to_dict()}",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
