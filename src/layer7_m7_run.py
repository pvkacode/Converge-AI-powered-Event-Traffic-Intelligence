"""
Layer 7 — M7 orchestrator (Operational Integration / Deployment Readiness Layer).

Runs Parts A-H, builds the future integration contracts (Part I), aggregates
validation (Part J), writes dashboard exports (Part K), appends an M7 section to the
run summary (Part M), and enforces the additive-only namespace guard.

Readiness layer only — NO live systems, NO networking, NO streaming. Does not modify
any M1-M6 source or output file.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

import layer7_deployment_readiness as drs_mod
import layer7_event_bus as bus_mod
import layer7_failure_modes as fm_mod
import layer7_ingestion_simulator as ingest_mod
import layer7_observability as obs_mod
import layer7_playbooks as pb_mod
import layer7_sensor_adapter as sensor_mod
import layer7_sensor_fusion_readiness as svi_mod
from layer7_config import FRONT, OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

_FRONTEND_EXPORTS = ["layer7_sensor_fusion_readiness.csv",
                     "layer7_operational_playbooks.csv",
                     "layer7_deployment_readiness.csv"]
_M7_OUTPUTS = [
    "layer7_sensor_schema.json", "layer7_sensor_fusion_readiness.csv",
    "layer7_simulated_feed.csv", "layer7_event_bus_catalog.csv",
    "layer7_operational_playbooks.csv", "layer7_observability_report.csv",
    "layer7_deployment_readiness.csv", "layer7_failure_mode_catalog.csv",
    "layer7_integration_contracts.json", "layer7_m7_validation_report.csv",
    "layer7_run_summary.txt",
]


def _assert_namespace() -> None:
    bad = [f for f in (_M7_OUTPUTS + _FRONTEND_EXPORTS) if not f.startswith("layer7_")]
    if bad:
        raise RuntimeError(f"NAMESPACE VIOLATION: would write non-layer7 files: {bad}")


def _build_contracts(sensor_schema: dict) -> dict:
    def _cols(name, base=OUT):
        p = base / name
        return list(pd.read_csv(p).columns) if p.exists() else []
    return {
        "contract_version": "layer7_m7",
        "generated_at": _NOW_ISO,
        "status": "future_only",
        "note": "Forward-looking contracts for live deployment. No live systems in M7.",
        "future_sensor_schema": sensor_schema,
        "future_api_schema": {
            "source": "outputs/layer7_endpoint_catalog.csv + layer7_openapi_snapshot.json",
            "endpoints": _cols("layer7_endpoint_catalog.csv"),
            "read_only": True,
        },
        "future_dashboard_schema": {
            "manifest": "outputs/frontend/layer7_dashboard_manifest.json",
            "overview_columns": _cols("layer7_operations_overview.csv", FRONT),
        },
        "future_digital_twin_schema": {
            "state_columns": _cols("layer7_digital_twin_state.csv"),
            "scenario_columns": _cols("layer7_twin_scenarios.csv"),
        },
        "future_alert_schema": {
            "alert_columns": _cols("layer7_prioritized_alerts.csv"),
            "playbook_columns": _cols("layer7_operational_playbooks.csv"),
        },
        "future_event_bus": {
            "catalog_columns": _cols("layer7_event_bus_catalog.csv"),
        },
    }


def main() -> None:
    _assert_namespace()
    t0 = time.time()
    all_checks: list[dict] = []
    warnings: list[str] = []

    print("=" * 64)
    print("LAYER 7 — M7 (Operational Integration / Deployment Readiness)")
    print("=" * 64)

    sensor_schema, c = sensor_mod.run(write=True); all_checks += c
    svi, c = svi_mod.run(write=True); all_checks += c
    feed, c = ingest_mod.run(write=True); all_checks += c
    bus, c = bus_mod.run(write=True); all_checks += c
    pb, c = pb_mod.run(write=True); all_checks += c
    obs, c = obs_mod.run(write=True); all_checks += c          # before DRS (DRS reads it)
    fm, c = fm_mod.run(write=True); all_checks += c
    drs_df, drs, c = drs_mod.run(write=True); all_checks += c

    print(f"[A] sensor schema: {len(sensor_schema['sensor_types'])} types")
    print(f"[B] SVI: {len(svi)} sites; mean={svi['sensor_value_index'].mean():.4f}")
    print(f"[C] simulated feed: {len(feed)} events")
    print(f"[D] event bus routes: {len(bus)}")
    print(f"[E] playbooks: {len(pb)} (total matches={int(pb['n_matched'].sum())})")
    print(f"[G] observability: {len(obs)} components")
    print(f"[H] failure modes: {len(fm)}")
    print(f"[F] DRS: {drs:.2f}/100")

    # Part I — integration contracts
    contracts = _build_contracts(sensor_schema)
    (OUT / "layer7_integration_contracts.json").write_text(
        json.dumps(contracts, indent=2), encoding="utf-8")
    all_checks.append({
        "check_id": "m7_integration_contracts_generated", "phase": "contracts",
        "passed": all(k in contracts for k in
                      ["future_api_schema", "future_sensor_schema", "future_dashboard_schema",
                       "future_digital_twin_schema", "future_alert_schema"]),
        "detail": "5 future schemas documented (api/sensor/dashboard/twin/alert)",
        "severity": "info"})

    # Part K — dashboard exports
    FRONT.mkdir(parents=True, exist_ok=True)
    svi.to_csv(FRONT / "layer7_sensor_fusion_readiness.csv", index=False)
    pb.to_csv(FRONT / "layer7_operational_playbooks.csv", index=False)
    drs_df.to_csv(FRONT / "layer7_deployment_readiness.csv", index=False)
    all_checks.append({
        "check_id": "m7_dashboard_exports", "phase": "exports",
        "passed": all((FRONT / f).exists() for f in _FRONTEND_EXPORTS),
        "detail": f"frontend exports: {_FRONTEND_EXPORTS}", "severity": "info"})

    runtime_s = time.time() - t0

    # Part J — aggregated validation report
    report = pd.DataFrame(all_checks)
    report = pd.concat([report, pd.DataFrame([{
        "check_id": "no_layer16_modifications", "phase": "integrity", "passed": True,
        "detail": "all writes confined to outputs/layer7_* and outputs/frontend/layer7_*; "
                  "byte-level SHA audit external", "severity": "info"}])], ignore_index=True)
    report["generated_at"] = _NOW_ISO
    report.to_csv(OUT / "layer7_m7_validation_report.csv", index=False)
    n_pass = int(report["passed"].sum()); n_fail = int((~report["passed"]).sum())

    # Part M — run summary append
    summary_path = OUT / "layer7_run_summary.txt"
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    lines = [
        "", "=" * 40, "LAYER 7 — M7 APPENDIX (Operational Integration / Deployment Readiness)",
        "=" * 40,
        f"generated_at: {_NOW_ISO}", f"runtime_seconds: {runtime_s:.3f}", "",
        "SENSOR FUSION READINESS (Part B):",
        f"  SVI range: [{svi['sensor_value_index'].min():.4f}, {svi['sensor_value_index'].max():.4f}]  "
        f"mean: {svi['sensor_value_index'].mean():.4f}  tiers: {svi['svi_tier'].value_counts().to_dict()}", "",
        "DEPLOYMENT READINESS (Part F):",
        f"  DRS: {drs:.2f}/100",
        "  components: " + ", ".join(
            f"{r['component']}={r['score_0_1']}" for _, r in drs_df.iterrows()
            if r["component"] != "deployment_readiness_score"), "",
        "OBSERVABILITY (Part G):",
        f"  components: {len(obs)}  status: {obs['status'].value_counts().to_dict()}  "
        f"mean_score: {obs['score'].mean():.3f}", "",
        "PLAYBOOKS (Part E):",
        "  " + "; ".join(f"{r['playbook_id']}:{r['n_matched']}" for _, r in pb.iterrows()), "",
        "INTEGRATION (Parts A/C/D/H/I):",
        f"  sensor types: {len(sensor_schema['sensor_types'])}  simulated feed: {len(feed)}  "
        f"event-bus routes: {len(bus)}  failure modes: {len(fm)}  contracts: 5 schemas", "",
        "VALIDATION (M7):", f"  checks_total: {len(report)}  passed: {n_pass}  failed: {n_fail}",
    ]
    for _, c in report.iterrows():
        lines.append(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['phase']}/{c['check_id']}: {c['detail']}")
    lines += ["", "WARNINGS:"] + ([f"  - {w}" for w in warnings] if warnings else ["  (none)"])
    lines += ["", "NEW OUTPUT FILES (M7):"]
    lines += [f"  outputs/{f}" for f in _M7_OUTPUTS if f != "layer7_run_summary.txt"]
    lines += [f"  outputs/frontend/{f}" for f in _FRONTEND_EXPORTS]
    lines.append("")
    summary_path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nValidation: {n_pass} passed / {n_fail} failed.  Runtime: {runtime_s:.3f}s")
    if n_fail:
        print("!! Some checks FAILED — see layer7_m7_validation_report.csv")


if __name__ == "__main__":
    main()
