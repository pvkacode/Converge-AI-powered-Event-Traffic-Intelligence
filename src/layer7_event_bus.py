"""
Layer 7 — M7 Part D: Event Bus Model.

Declarative model of the FUTURE event-routing architecture. NO networking, NO
MQTT/Kafka, NO streaming — a catalog only, describing how event types would route
to consumers with retry/dead-letter policies when ASTraM is deployed live.

ADDITIVE ONLY. Writes only outputs/layer7_event_bus_catalog.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# source, event_type, priority, target_consumer, retry_policy, dead_letter
_ROUTES = [
    ("sensor_adapter", "sensor_update", "P3", "operational_state_engine", "retry_3_backoff_5s", False),
    ("ingestion_simulator", "incident_update", "P2", "operational_state_engine", "retry_3_backoff_5s", False),
    ("operator_console", "operator_update", "P2", "override_engine", "retry_1", False),
    ("drift_monitor", "drift_alert", "P1", "alert_prioritization+playbooks", "retry_1", True),
    ("optimizer_bridge", "resource_change", "P2", "dashboard_backend", "retry_3_backoff_5s", False),
    ("alert_prioritization", "prioritized_alert", "P1", "playbooks+dashboard", "retry_2", True),
    ("decision_confidence", "confidence_update", "P3", "dashboard_backend", "retry_2", False),
    ("digital_twin", "scenario_result", "P3", "dashboard_backend", "retry_2", False),
    ("governance", "governance_event", "P2", "observability", "retry_2", False),
    ("playbooks", "operator_action", "P1", "operator_console", "retry_1", True),
]


def build_catalog() -> pd.DataFrame:
    rows = [{
        "event_source": s, "event_type": et, "priority": p,
        "target_consumer": tc, "retry_policy": rp, "dead_letter_flag": dl,
        "generated_at": _NOW_ISO,
    } for (s, et, p, tc, rp, dl) in _ROUTES]
    return pd.DataFrame(rows)


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_catalog()
    if write:
        df.to_csv(OUT / "layer7_event_bus_catalog.csv", index=False)
    checks = [{
        "check_id": "m7_event_bus_catalog", "phase": "event_bus",
        "passed": len(df) >= 5 and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} routes; dead-letter routes={int(df['dead_letter_flag'].sum())}",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df[["event_source", "event_type", "priority", "target_consumer", "dead_letter_flag"]].to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
