"""
Layer 7 — M1 orchestrator + diagnostics.

Runs the three M1 engines in order and aggregates validation + run diagnostics:
    1. Shared Loader            (Phase 1)
    2. Operational State Engine (Phase 2)
    3. Alert Prioritization     (Phase 3)

ADDITIVE ONLY. Writes exclusively to outputs/layer7_*. A namespace guard asserts
that no path outside the layer7_ namespace is written (hard stop if violated).

Outputs (beyond the per-engine files):
    outputs/layer7_validation_report.csv
    outputs/layer7_run_summary.txt
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

import layer7_alert_prioritization as alerts_engine
import layer7_operational_state as state_engine
from layer7_config import ALLOWED_WRITE_PREFIXES, OUT
from layer7_loader import audit_inputs, validate as loader_validate

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# Files this run is permitted to (re)write — all in the layer7_ namespace.
_EXPECTED_OUTPUTS = [
    "layer7_input_audit.csv",
    "layer7_schema_inventory.csv",
    "layer7_data_freshness.csv",
    "layer7_operational_state.csv",
    "layer7_site_risk_ranking.csv",
    "layer7_state_summary.csv",
    "layer7_prioritized_alerts.csv",
    "layer7_alert_summary.csv",
    "layer7_alert_diagnostics.csv",
    "layer7_validation_report.csv",
    "layer7_run_summary.txt",
]


def _assert_namespace() -> None:
    bad = [f for f in _EXPECTED_OUTPUTS if not f.startswith(ALLOWED_WRITE_PREFIXES)]
    if bad:
        raise RuntimeError(f"NAMESPACE VIOLATION: Layer 7 would write non-layer7 files: {bad}")


def main() -> None:
    _assert_namespace()
    t0 = time.time()
    warnings: list[str] = []
    all_checks: list[dict] = []

    print("=" * 64)
    print("LAYER 7 — M1 (Loader · Operational State · Alert Prioritization)")
    print("=" * 64)

    # --- Phase 1: loader
    store, tables = audit_inputs(write=True)
    all_checks.extend(loader_validate(tables))
    audit = tables["audit"]
    files_read = int(audit["found"].sum())
    for _, r in audit.iterrows():
        if r["status"] in ("MISSING", "SCHEMA_DRIFT"):
            warnings.append(f"{r['file']}: {r['status']} ({r['missing_required_columns']})")
        elif r["status"] == "EMPTY":
            warnings.append(f"{r['file']}: EMPTY (valid)")
    stale = tables["freshness"]
    stale_files = stale[stale["stale_flag"] == True]["file"].tolist()  # noqa: E712
    if stale_files:
        warnings.append(f"stale inputs: {stale_files}")
    print(f"\n[Phase 1] inputs found: {files_read}/{len(audit)}")

    # --- Phase 2: operational state
    state_tables, state_checks = state_engine.run(store, write=True)
    all_checks.extend(state_checks)
    n_state = len(state_tables["state"])
    print(f"[Phase 2] operational_state rows: {n_state}")

    # --- Phase 3: alert prioritization
    alert_tables, alert_checks = alerts_engine.run(store, write=True)
    all_checks.extend(alert_checks)
    n_alerts = len(alert_tables["alerts"])
    print(f"[Phase 3] prioritized_alerts rows: {n_alerts}")

    runtime_s = time.time() - t0

    # --- validation report
    report = pd.DataFrame(all_checks)
    report["generated_at"] = _NOW_ISO
    report.to_csv(OUT / "layer7_validation_report.csv", index=False)
    n_pass = int(report["passed"].sum())
    n_fail = int((~report["passed"]).sum())

    # --- run summary
    rows_processed = n_state + n_alerts
    lines = [
        "LAYER 7 — M1 RUN SUMMARY",
        "=" * 40,
        f"generated_at: {_NOW_ISO}",
        f"runtime_seconds: {runtime_s:.3f}",
        "",
        f"files_read: {files_read}/{len(audit)}",
        f"rows_processed: {rows_processed} "
        f"(operational_state={n_state}, prioritized_alerts={n_alerts})",
        "",
        "VALIDATION:",
        f"  checks_total: {len(report)}",
        f"  passed: {n_pass}",
        f"  failed: {n_fail}",
    ]
    for _, c in report.iterrows():
        mark = "PASS" if c["passed"] else "FAIL"
        lines.append(f"    [{mark}] {c['phase']}/{c['check_id']}: {c['detail']}")

    lines += ["", "WARNINGS:"]
    if warnings:
        lines += [f"  - {w}" for w in warnings]
    else:
        lines.append("  (none)")

    state_summary = state_tables["summary"].set_index("metric")["value"].to_dict()
    lines += [
        "",
        "OPERATIONAL STATE:",
        f"  ORS mean: {state_summary.get('ors_mean')}",
        f"  ORS variance: {state_summary.get('ors_variance')}",
        f"  ORS range: [{state_summary.get('ors_min')}, {state_summary.get('ors_max')}]",
        f"  tiers: Normal={state_summary.get('tier_normal_count')} "
        f"Elevated={state_summary.get('tier_elevated_count')} "
        f"Critical={state_summary.get('tier_critical_count')} "
        f"Emergency={state_summary.get('tier_emergency_count')}",
    ]
    pr_dist = (
        alert_tables["alerts"]["priority"].value_counts().to_dict() if n_alerts else {}
    )
    lines += ["", "ALERT PRIORITIZATION:", f"  priority distribution: {pr_dist}"]

    lines += ["", "OUTPUT FILES:"]
    lines += [f"  outputs/{f}" for f in _EXPECTED_OUTPUTS]

    summary_text = "\n".join(lines) + "\n"
    (OUT / "layer7_run_summary.txt").write_text(summary_text, encoding="utf-8")

    print("\n" + summary_text)
    print(f"Validation: {n_pass} passed / {n_fail} failed.")
    if n_fail:
        print("!! Some validation checks FAILED — see layer7_validation_report.csv")


if __name__ == "__main__":
    main()
