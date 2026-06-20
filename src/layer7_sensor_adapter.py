"""
Layer 7 — M7 Part A: Sensor Abstraction Layer.

Defines the CANONICAL sensor schema for future real-time deployment. SCHEMA ONLY —
no live ingestion, no external calls, no networking. This is a readiness artifact.

ADDITIVE ONLY. Writes only outputs/layer7_sensor_schema.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

SENSOR_TYPES = [
    "traffic_camera", "gps_probe", "roadside_unit", "weather_feed",
    "planned_event_feed", "manual_operator_feed", "social_signal_feed",
]

CANONICAL_RECORD = {
    "sensor_id": {"type": "string", "required": True, "description": "Unique sensor identifier"},
    "sensor_type": {"type": "enum", "required": True, "enum": SENSOR_TYPES,
                    "description": "One of the supported canonical sensor types"},
    "timestamp": {"type": "string(iso8601)", "required": True, "description": "Observation time (UTC)"},
    "latitude": {"type": "float", "required": True, "description": "WGS84 latitude"},
    "longitude": {"type": "float", "required": True, "description": "WGS84 longitude"},
    "confidence": {"type": "float[0,1]", "required": True, "description": "Source-declared confidence"},
    "payload_json": {"type": "object", "required": True, "description": "Type-specific payload"},
    "status": {"type": "enum", "required": True, "enum": ["active", "degraded", "offline"],
               "description": "Sensor operational status"},
}

_TYPE_PAYLOAD_HINTS = {
    "traffic_camera": ["vehicle_count", "queue_length_m", "incident_detected"],
    "gps_probe": ["speed_kph", "heading_deg", "trip_id"],
    "roadside_unit": ["occupancy_pct", "flow_vph", "avg_speed_kph"],
    "weather_feed": ["condition", "visibility_m", "precip_mm"],
    "planned_event_feed": ["event_cause", "expected_attendance", "corridor"],
    "manual_operator_feed": ["operator_id", "observation_text", "severity"],
    "social_signal_feed": ["platform", "mention_count", "sentiment"],
}


def build_schema() -> dict:
    return {
        "schema_version": "layer7_m7",
        "generated_at": _NOW_ISO,
        "live_ingestion": False,
        "note": "Schema only. No live ingestion, no external calls (M7 readiness layer).",
        "canonical_record": CANONICAL_RECORD,
        "sensor_types": {
            t: {"description": f"{t.replace('_', ' ')} sensor",
                "typical_payload_fields": _TYPE_PAYLOAD_HINTS.get(t, [])}
            for t in SENSOR_TYPES
        },
    }


def run(write: bool = True) -> tuple[dict, list[dict]]:
    schema = build_schema()
    if write:
        (OUT / "layer7_sensor_schema.json").write_text(
            json.dumps(schema, indent=2), encoding="utf-8")
    checks = [{
        "check_id": "m7_sensor_schema_valid", "phase": "sensor_adapter",
        "passed": len(schema["sensor_types"]) == 7 and len(schema["canonical_record"]) == 8,
        "detail": f"{len(schema['sensor_types'])} sensor types; "
                  f"{len(schema['canonical_record'])} canonical fields",
        "severity": "info",
    }]
    return schema, checks


if __name__ == "__main__":
    schema, checks = run(write=True)
    print(json.dumps({"sensor_types": list(schema["sensor_types"]),
                      "fields": list(schema["canonical_record"])}, indent=2))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
