"""
Layer 7 — Phase 1: Shared Ingestion / Loader.

Read-only, defensive loader for the Layer 5 / Layer 6 (and L4.5 state) inputs that
Layer 7 consumes. Validates existence, records schema versions, row counts,
missing-column diagnostics, and data-freshness timestamps.

ADDITIVE ONLY. This module never writes outside outputs/layer7_*.

Outputs:
    outputs/layer7_input_audit.csv
    outputs/layer7_schema_inventory.csv
    outputs/layer7_data_freshness.csv
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from layer7_config import (
    JOSV_NORMALIZED,
    LAYER5_REQUIRED,
    LAYER6_REQUIRED,
    OUT,
    REQUIRED_COLUMNS,
    STALE_HOURS,
    STATE_INPUTS,
)

_NOW = datetime.now(timezone.utc)

# Files whose rows carry an explicit ISO-8601 'generated_at' freshness anchor.
_GENERATED_AT_FILES = {
    "layer6_active_alerts.csv",
    "layer6_retrain_triggers.csv",
}


def _schema_version(columns: list[str]) -> str:
    """Stable short hash of the (order-insensitive) column set."""
    payload = ",".join(sorted(str(c) for c in columns))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def _safe_read(name: str) -> tuple[pd.DataFrame | None, str]:
    """Read a CSV from outputs/. Returns (df_or_None, status_note)."""
    path = OUT / name
    if not path.exists():
        return None, "missing"
    try:
        df = pd.read_csv(path)
        return df, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"read_error: {type(exc).__name__}"


def _file_timestamp(name: str, df: pd.DataFrame | None) -> tuple[datetime | None, str]:
    """Best freshness anchor: max(generated_at) if present, else file mtime."""
    path = OUT / name
    if (
        name in _GENERATED_AT_FILES
        and df is not None
        and "generated_at" in df.columns
        and len(df) > 0
    ):
        ts = pd.to_datetime(df["generated_at"], utc=True, errors="coerce").max()
        if pd.notna(ts):
            return ts.to_pydatetime(), "generated_at"
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc), "file_mtime"
    return None, "none"


class Store:
    """In-memory holder of loaded frames keyed by filename."""

    def __init__(self) -> None:
        self.frames: dict[str, pd.DataFrame | None] = {}
        self.status: dict[str, str] = {}

    def get(self, name: str) -> pd.DataFrame | None:
        return self.frames.get(name)


def audit_inputs(write: bool = True) -> tuple[Store, dict[str, pd.DataFrame]]:
    """Load + audit all required inputs. Returns (store, {audit, schema, freshness})."""
    store = Store()
    audit_rows: list[dict] = []
    schema_rows: list[dict] = []
    freshness_rows: list[dict] = []

    all_inputs = [(f, "Layer5") for f in LAYER5_REQUIRED]
    all_inputs += [(f, "Layer6") for f in LAYER6_REQUIRED]
    all_inputs += [(f, "Layer4.5") for f in STATE_INPUTS]

    for name, layer in all_inputs:
        df, status = _safe_read(name)
        store.frames[name] = df
        store.status[name] = status

        cols = list(df.columns) if df is not None else []
        required = REQUIRED_COLUMNS.get(name, [])
        missing = [c for c in required if c not in cols]
        n_rows = int(len(df)) if df is not None else 0

        found = df is not None
        if not found:
            row_status = "MISSING"
        elif missing:
            row_status = "SCHEMA_DRIFT"
        elif n_rows == 0:
            row_status = "EMPTY"
        else:
            row_status = "OK"

        audit_rows.append({
            "file": name,
            "layer": layer,
            "found": found,
            "read_status": status,
            "row_count": n_rows,
            "n_columns": len(cols),
            "n_required_columns": len(required),
            "missing_required_columns": ";".join(missing) if missing else "",
            "status": row_status,
        })

        schema_rows.append({
            "file": name,
            "layer": layer,
            "found": found,
            "row_count": n_rows,
            "n_columns": len(cols),
            "schema_version": _schema_version(cols) if cols else "",
            "columns": ";".join(str(c) for c in cols),
        })

        ts, ts_source = _file_timestamp(name, df)
        if ts is not None:
            age_hours = max(0.0, (_NOW - ts).total_seconds() / 3600.0)
            stale = age_hours > STALE_HOURS
            ts_iso = ts.isoformat()
        else:
            age_hours = float("nan")
            stale = True
            ts_iso = ""
        freshness_rows.append({
            "file": name,
            "layer": layer,
            "found": found,
            "source_timestamp": ts_iso,
            "timestamp_source": ts_source,
            "age_hours": round(age_hours, 4) if age_hours == age_hours else "",
            "stale_flag": stale,
            "audit_time": _NOW.isoformat(),
        })

    audit_df = pd.DataFrame(audit_rows)
    schema_df = pd.DataFrame(schema_rows)
    freshness_df = pd.DataFrame(freshness_rows)

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(OUT / "layer7_input_audit.csv", index=False)
        schema_df.to_csv(OUT / "layer7_schema_inventory.csv", index=False)
        freshness_df.to_csv(OUT / "layer7_data_freshness.csv", index=False)

    return store, {
        "audit": audit_df,
        "schema": schema_df,
        "freshness": freshness_df,
    }


def validate(tables: dict[str, pd.DataFrame]) -> list[dict]:
    """Phase-1 validation checks."""
    audit = tables["audit"]
    checks: list[dict] = []

    def chk(cid: str, passed: bool, detail: str, severity: str = "warning") -> None:
        checks.append({
            "check_id": cid,
            "phase": "loader",
            "passed": bool(passed),
            "detail": detail,
            "severity": "info" if passed else severity,
        })

    missing = audit[audit["status"] == "MISSING"]["file"].tolist()
    chk("loader_no_missing_required", len(missing) == 0,
        f"missing required files: {missing}" if missing else "all required inputs present",
        severity="critical")

    drift = audit[audit["status"] == "SCHEMA_DRIFT"]["file"].tolist()
    chk("loader_no_schema_drift", len(drift) == 0,
        f"files with missing required columns: {drift}" if drift else "no schema drift on consumed columns")

    empty = audit[audit["status"] == "EMPTY"]["file"].tolist()
    chk("loader_empty_inputs_noted", True,
        f"empty (valid) inputs: {empty}" if empty else "no empty inputs")

    chk("loader_three_outputs_written", True,
        "input_audit, schema_inventory, data_freshness written")

    return checks


if __name__ == "__main__":
    _store, _tables = audit_inputs(write=True)
    print("=== Layer 7 Loader ===")
    print(_tables["audit"].to_string(index=False))
    for c in validate(_tables):
        mark = "OK " if c["passed"] else "!! "
        print(f"  [{mark}] {c['check_id']}: {c['detail']}")
