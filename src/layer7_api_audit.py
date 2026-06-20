"""
Layer 7 — M5 Part D: API audit logging.

Append-only log of API calls: timestamp, endpoint, request_type, status,
response_time_ms. Existing records are never deleted; each sweep appends new rows.

The audit sweep exercises every endpoint provider (timing each) so the log is
populated without a persistent server. Real HTTP calls (M4/M5 server) would append
through log_call() in the same format.

Output: outputs/layer7_api_audit_log.csv  (append-only)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

import layer7_api as api
import layer7_api_extensions as ext
from layer7_config import OUT

_COLS = ["call_id", "timestamp", "endpoint", "request_type", "status",
         "response_time_ms", "source"]
_PATH = OUT / "layer7_api_audit_log.csv"


def _time_call(fn) -> tuple[int, float]:
    t0 = time.perf_counter()
    status = 200
    try:
        out = fn()
        if out in ({}, None):
            status = 404
    except Exception:
        status = 500
    return status, (time.perf_counter() - t0) * 1000.0


def run_audit_sweep(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    now = datetime.now(timezone.utc)
    sweep_id = now.strftime("%Y%m%dT%H%M%S")
    overview = api.get_page("overview")
    overrides = api.get_page("overrides")
    sample_site = str(overview[0]["event_id"]) if overview else "NONE"
    sample_ov = str(overrides[0]["override_id"]) if overrides else "NONE"

    calls = [
        ("/", "GET", lambda: {"ok": True}),
        ("/manifest", "GET", api.get_manifest),
        ("/overview", "GET", lambda: api.get_page("overview")),
        ("/alerts", "GET", lambda: api.get_page("alerts")),
        ("/recommendations", "GET", lambda: api.get_page("recommendations")),
        ("/overrides", "GET", lambda: api.get_page("overrides")),
        ("/health", "GET", lambda: api.get_page("health")),
        ("/counterfactuals", "GET", lambda: api.get_page("counterfactuals")),
        ("/healthz", "GET", ext.get_healthz),
        ("/metrics", "GET", ext.get_metrics),
        (f"/site/{sample_site}", "GET", lambda: ext.get_site(sample_site)),
        ("/alerts/P1", "GET", lambda: ext.get_alerts_by_priority("P1")),
        (f"/override/{sample_ov}", "GET", lambda: ext.get_override(sample_ov)),
    ]

    rows = []
    for i, (endpoint, rtype, fn) in enumerate(calls, start=1):
        status, ms = _time_call(fn)
        rows.append({
            "call_id": f"{sweep_id}-{i:03d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint, "request_type": rtype, "status": status,
            "response_time_ms": round(ms, 4), "source": "audit_sweep",
        })
    new = pd.DataFrame(rows, columns=_COLS)

    if _PATH.exists():
        existing = pd.read_csv(_PATH, dtype=str)
        combined = pd.concat([existing, new.astype(str)], ignore_index=True)
    else:
        combined = new
    if write:
        combined.to_csv(_PATH, index=False)

    checks = [{
        "check_id": "m5_api_audit_logged", "phase": "api_audit",
        "passed": len(new) > 0 and bool((new["status"] == 200).all()),
        "detail": f"{len(new)} calls logged this sweep; "
                  f"all_200={bool((new['status'] == 200).all())}; total_log_rows={len(combined)}",
        "severity": "info" if (new["status"] == 200).all() else "warning",
    }]
    return new, checks


if __name__ == "__main__":
    new, checks = run_audit_sweep(write=True)
    print(new.to_string(index=False))
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
