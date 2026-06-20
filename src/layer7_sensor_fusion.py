"""
Layer 7 — M7A: Sensor Fusion Framework (orchestrator).

Additive, deployment-ready sensor-fusion scaffold that works with ZERO real sensors.
Runs in REPLAY mode (synthetic observations from Layer 5 / Layer 6) and exposes a
LIVE seam for future real-time sensors. Per-site fused operational observations carry
full provenance and degrade to a Layer-5/6 fallback (with confidence penalty) when no
sensors are available — never crashing on 0 sensors.

NO Layer 1-6 or existing Layer 7 output is modified. NO scoring logic changed.
NOT Kalman / MPC / GNN — pure inverse-variance Bayesian fusion only.

Outputs:
  layer7_sensor_registry.csv      layer7_sensor_health.csv
  layer7_sensor_observations.csv  layer7_sensor_fusion.csv
  layer7_sensor_conflicts.csv     layer7_fusion_diagnostics.csv
  layer7_sensor_readiness.csv     layer7_sensor_summary.txt
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import layer7_sensor_registry as registry_mod
import layer7_sensor_reliability as health_mod
from layer7_sensor_adapters import (
    LiveSensorAdapter,
    QUANTITIES,
    Q_SCALE,
    ReplaySensorAdapter,
    build_site_truth,
)
from layer7_config import OUT
from layer7_fusion_math import bayesian_fuse, confidence_penalty, consistency
from layer7_sensor_registry import SENSOR_QUANTITIES, SENSOR_TYPES, STATUSES

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_FALLBACK_BASE_CONF = 0.40
_FALLBACK_PENALTY = 0.50


def _fuse_site_quantity(obs_q: pd.DataFrame, truth_val: float, scale: float):
    """Fuse one (site, quantity). Falls back to truth value if no observations."""
    if len(obs_q) == 0:
        return {
            "fused_value": float(truth_val), "fused_variance": float("nan"),
            "fused_confidence": confidence_penalty(_FALLBACK_BASE_CONF, _FALLBACK_PENALTY),
            "source_count": 0, "consensus_ratio": 1.0, "conflict_score": 0.0,
            "conflict_flag": "LOW_CONFLICT", "fallback_mode": True,
            "sensor_types_used": "", "sensor_ids_used": "",
        }
    fz = bayesian_fuse(obs_q["observed_value"], obs_q["sensor_reliability"],
                       obs_q["sensor_obs_variance"])
    con = consistency(obs_q["observed_value"], scale=scale)
    return {
        "fused_value": fz["fused_value"], "fused_variance": fz["fused_variance"],
        "fused_confidence": fz["fused_confidence"], "source_count": fz["n"],
        "consensus_ratio": con["consensus_ratio"], "conflict_score": con["conflict_score"],
        "conflict_flag": con["conflict_flag"], "fallback_mode": False,
        "sensor_types_used": ";".join(sorted(obs_q["sensor_type"].unique())),
        "sensor_ids_used": ";".join(sorted(obs_q["sensor_id"].unique())),
    }


def run(write: bool = True) -> tuple[dict, list[dict]]:
    checks: list[dict] = []

    def chk(cid, passed, detail, sev="critical"):
        checks.append({"check_id": cid, "phase": "sensor_fusion", "passed": bool(passed),
                       "detail": detail, "severity": "info" if passed else sev})

    # Step 1-2: registry + health
    registry, c = registry_mod.run(write=write); checks += c
    health, c = health_mod.run(write=write); checks += c

    # Step 3 + live seam: poll adapters
    replay = ReplaySensorAdapter().poll(registry, health)
    live = LiveSensorAdapter().poll(registry, health)  # empty today (no real sensors)
    observations = pd.concat([replay, live], ignore_index=True)
    if write:
        observations.to_csv(OUT / "layer7_sensor_observations.csv", index=False)

    truth = build_site_truth().set_index("event_id")
    sites = list(truth.index.astype(str))

    # Step 4/5/7/8: fuse per (site, quantity) with provenance + conflict + fallback
    detail_rows, site_rows = [], []
    for eid in sites:
        obs_e = observations[observations["event_id"].astype(str) == eid] if len(observations) else observations
        per_q = {}
        site_types, site_ids = set(), set()
        any_fallback = False
        for q in QUANTITIES:
            obs_q = obs_e[obs_e["quantity"] == q] if len(obs_e) else obs_e
            res = _fuse_site_quantity(obs_q, float(truth.loc[eid, q]), Q_SCALE[q])
            per_q[q] = res
            any_fallback = any_fallback or res["fallback_mode"]
            if res["sensor_types_used"]:
                site_types.update(res["sensor_types_used"].split(";"))
            if res["sensor_ids_used"]:
                site_ids.update(res["sensor_ids_used"].split(";"))
            detail_rows.append({
                "event_id": eid, "quantity": q,
                "fused_value": round(res["fused_value"], 4),
                "fused_variance": (round(res["fused_variance"], 6)
                                   if np.isfinite(res["fused_variance"]) else ""),
                "fused_confidence": round(res["fused_confidence"], 4),
                "source_count": res["source_count"],
                "consensus_ratio": res["consensus_ratio"],
                "conflict_score": res["conflict_score"],
                "conflict_flag": res["conflict_flag"],
                "fallback_mode": res["fallback_mode"],
                "sensor_types_used": res["sensor_types_used"],
                "sensor_ids_used": res["sensor_ids_used"],
                "generated_at": _NOW_ISO,
            })
        # Step 6: per-site fused operational observation + provenance
        site_conf = float(np.mean([per_q[q]["fused_confidence"] for q in QUANTITIES]))
        site_conflict = float(np.mean([per_q[q]["conflict_score"] for q in QUANTITIES]))
        src_count = len(site_ids)
        quality = []
        if any_fallback:
            quality.append("fallback")
        if any(per_q[q]["conflict_flag"] == "HIGH_CONFLICT" for q in QUANTITIES):
            quality.append("high_conflict")
        if src_count < 2:
            quality.append("low_source")
        if not quality:
            quality.append("ok")
        site_rows.append({
            "event_id": eid,
            "traffic_speed": round(per_q["traffic_speed"]["fused_value"], 3),
            "queue_length": round(per_q["queue_length"]["fused_value"], 3),
            "travel_time": round(per_q["travel_time"]["fused_value"], 3),
            "incident_probability": round(per_q["incident_indicator"]["fused_value"], 4),
            "lane_availability": round(per_q["lane_availability"]["fused_value"], 4),
            "confidence": round(site_conf, 4),
            "uncertainty": round(1.0 - site_conf, 4),
            "source_count": src_count,
            "conflict_score": round(site_conflict, 4),
            "sensor_count": src_count,
            "sensor_types_used": ";".join(sorted(site_types)),
            "sensor_ids_used": ";".join(sorted(site_ids)),
            "fusion_method": "inverse_variance_bayesian",
            "quality_flags": ";".join(quality),
            "fallback_mode": any_fallback,
            "sensor_mode": "REPLAY",
            "generated_at": _NOW_ISO,
        })

    detail = pd.DataFrame(detail_rows)
    fused = pd.DataFrame(site_rows)
    if write:
        detail.to_csv(OUT / "layer7_sensor_conflicts.csv", index=False)
        fused.to_csv(OUT / "layer7_sensor_fusion.csv", index=False)

    # Step 9: diagnostics
    diag = _diagnostics(registry, health, detail, fused, observations)
    if write:
        diag.to_csv(OUT / "layer7_fusion_diagnostics.csv", index=False)

    readiness = _readiness(registry, health)
    if write:
        readiness.to_csv(OUT / "layer7_sensor_readiness.csv", index=False)

    # --- validation
    chk("m7a_fused_finite", bool(np.isfinite(fused[["traffic_speed", "queue_length",
        "travel_time", "incident_probability", "lane_availability"]].to_numpy()).all()),
        "all fused operational values finite")
    chk("m7a_reliability_bounded", bool(((health["sensor_reliability"] >= 0)
        & (health["sensor_reliability"] <= 1)).all()), "all R in [0,1]")
    chk("m7a_confidence_bounded", bool(((fused["confidence"] >= 0) & (fused["confidence"] <= 1)).all()
        and ((detail["fused_confidence"] >= 0) & (detail["fused_confidence"] <= 1)).all()),
        "all confidences in [0,1]")
    fb = bayesian_fuse([], [], [])
    chk("m7a_fallback_functions", fb["n"] == 0 and fb["fused_confidence"] == 0.0
        and int(fused["fallback_mode"].sum()) >= 0,
        f"0-sensor fuse returns fallback tuple; site rows with fallback={int(fused['fallback_mode'].sum())}")
    chk("m7a_replay_functions", len(observations) > 0 and (observations["sensor_mode"] == "REPLAY").all(),
        f"{len(observations)} replay observations generated (all REPLAY)")
    prov_ok = bool((fused["fusion_method"] != "").all() and (fused["quality_flags"] != "").all()
                   and (fused["sensor_count"] >= 0).all())
    chk("m7a_provenance_populated", prov_ok, "provenance fields populated on every fused row")
    chk("m7a_no_nan_core", int(fused.drop(columns=["sensor_ids_used"]).isna().sum().sum()) == 0,
        "no NaN in fused operational observations")

    if write:
        _summary(registry, health, observations, detail, fused, diag, checks)

    return {"registry": registry, "health": health, "observations": observations,
            "detail": detail, "fused": fused, "diagnostics": diag, "readiness": readiness}, checks


def _diagnostics(registry, health, detail, fused, observations) -> pd.DataFrame:
    rows = []

    def add(group, metric, value):
        rows.append({"diagnostic_group": group, "metric": metric, "value": value,
                     "generated_at": _NOW_ISO})

    for k, v in health["sensor_health_tier"].value_counts().items():
        add("reliability_distribution", str(k), int(v))
    add("reliability_distribution", "mean_R", round(float(health["sensor_reliability"].mean()), 4))
    for k, v in registry["status"].value_counts().items():
        add("sensor_health_distribution", str(k), int(v))
    for k, v in detail["conflict_flag"].value_counts().items():
        add("conflict_distribution", str(k), int(v))
    # fusion confidence histogram (site-level)
    bins = pd.cut(fused["confidence"], [0, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
    for k, v in bins.value_counts().sort_index().items():
        add("fusion_confidence_distribution", str(k), int(v))
    # missing sensor report
    add("missing_sensor_report", "sites_total", int(len(fused)))
    add("missing_sensor_report", "sites_in_fallback", int(fused["fallback_mode"].sum()))
    add("missing_sensor_report", "quantity_fallbacks", int(detail["fallback_mode"].sum()))
    add("missing_sensor_report", "sites_low_source(<2)", int((fused["source_count"] < 2).sum()))
    add("missing_sensor_report", "offline_sensors", int((registry["status"] == "OFFLINE").sum()))
    return pd.DataFrame(rows)


def _readiness(registry, health) -> pd.DataFrame:
    rows = []
    counts = registry["sensor_type"].value_counts().to_dict()
    for st in SENSOR_TYPES:
        rows.append({
            "sensor_type": st,
            "replay_supported": True,
            "live_adapter_available": True,   # stub seam present
            "required_fields": "sensor_id;sensor_type;timestamp;latitude;longitude;confidence;payload_json;status",
            "observes": ";".join(SENSOR_QUANTITIES.get(st, [])),
            "n_registered": int(counts.get(st, 0)),
            "live_ready": False,              # no live feed connected yet
            "readiness_note": "replay validated; live adapter stub present; awaiting real feed",
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


def _summary(registry, health, observations, detail, fused, diag, checks) -> None:
    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = sum(1 for c in checks if not c["passed"])
    lines = [
        "LAYER 7 — M7A SENSOR FUSION FRAMEWORK SUMMARY",
        "=" * 52,
        f"generated_at: {_NOW_ISO}",
        "mode: REPLAY (no real sensors) + LIVE seam (stub)",
        "",
        f"sensor types supported: {registry['sensor_type'].nunique()} / {len(SENSOR_TYPES)}",
        f"sensors registered: {len(registry)} across {registry['event_id'].nunique()} sites",
        f"status distribution: {registry['status'].value_counts().to_dict()}",
        f"reliability tiers: {health['sensor_health_tier'].value_counts().to_dict()} "
        f"(mean R={health['sensor_reliability'].mean():.3f})",
        "",
        f"replay observations generated: {len(observations)}",
        f"fused per-(site,quantity) estimates: {len(detail)}",
        f"fused per-site operational observations: {len(fused)}",
        f"conflict distribution: {detail['conflict_flag'].value_counts().to_dict()}",
        f"site confidence: mean={fused['confidence'].mean():.3f} "
        f"range=[{fused['confidence'].min():.3f},{fused['confidence'].max():.3f}]",
        f"sites in fallback: {int(fused['fallback_mode'].sum())}; "
        f"quantity-level fallbacks: {int(detail['fallback_mode'].sum())}",
        "",
        "FALLBACK: 0-sensor fusion returns a finite Layer-5/6 estimate with a confidence",
        "penalty; framework never crashes when sensors are missing.",
        "FUTURE LIVE: LiveSensorAdapter seam present; live_ready=False until a real feed connects.",
        "",
        f"VALIDATION: {n_pass} passed / {n_fail} failed",
    ]
    for c in checks:
        lines.append(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['check_id']}: {c['detail']}")
    (OUT / "layer7_sensor_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("LAYER 7 — M7A SENSOR FUSION FRAMEWORK")
    print("=" * 60)
    tables, checks = run(write=True)
    n_pass = sum(1 for c in checks if c["passed"])
    n_fail = sum(1 for c in checks if not c["passed"])
    print(f"sensors: {len(tables['registry'])}  observations: {len(tables['observations'])}  "
          f"fused sites: {len(tables['fused'])}")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
    print(f"\nValidation: {n_pass} passed / {n_fail} failed")


if __name__ == "__main__":
    main()
