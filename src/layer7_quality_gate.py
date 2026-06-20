"""
Layer 7 — PATCH (finding F-014 / L6 integrity gate).

Propagates Layer 4.5 / Layer 6 data-quality flags into a per-event quality weight
that the Operational State Engine consumes as a bounded down-weight on ORS. Read-only
on upstream; additive output only.

Sources (read-only):
  layer45_duration_quality.csv  -> duration_sanity_flag (1=sane, 0=flagged)
  layer6_quarantine_report.csv  -> excluded_from_posterior / quarantined

Per-event:
  quality_weight = clip(1 - P_SANITY*[sanity_flag==0] - P_QUAR*[excluded], FLOOR, 1.0)
  data_quality_flag = 'ok' | 'flagged' | 'unknown'

Penalties are MILD and FLOORED so a flagged event is down-weighted, never zeroed, and
within-subset ordering is preserved when a whole subset shares the same flags.

ADDITIVE ONLY. Writes only outputs/layer7_quality_gate.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

P_SANITY = 0.20   # penalty for duration_sanity_flag == 0
P_QUAR = 0.20     # penalty for excluded_from_posterior / quarantined
FLOOR = 0.60      # weight floor (never zero out an event)


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def build_quality_gate() -> pd.DataFrame:
    dq = _read("layer45_duration_quality.csv")
    qr = _read("layer6_quarantine_report.csv")

    if len(dq):
        base = dq[["event_id", "duration_sanity_flag", "duration_guard_reason"]].copy()
        base["event_id"] = base["event_id"].astype(str)
    else:
        base = pd.DataFrame(columns=["event_id", "duration_sanity_flag", "duration_guard_reason"])

    excluded_ids: set = set()
    if len(qr) and "event_id" in qr.columns:
        col = "excluded_from_posterior" if "excluded_from_posterior" in qr.columns else "quarantined"
        flag = qr[col].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        excluded_ids = set(qr.loc[flag, "event_id"].astype(str))

    base["sanity_bad"] = (pd.to_numeric(base["duration_sanity_flag"], errors="coerce") == 0).astype(int)
    base["excluded_from_posterior"] = base["event_id"].isin(excluded_ids).astype(int)
    base["quality_weight"] = np.clip(
        1.0 - P_SANITY * base["sanity_bad"] - P_QUAR * base["excluded_from_posterior"],
        FLOOR, 1.0,
    )
    base["data_quality_flag"] = np.where(
        (base["sanity_bad"] == 1) | (base["excluded_from_posterior"] == 1), "flagged", "ok")
    base["generated_at"] = _NOW_ISO
    return base[["event_id", "duration_sanity_flag", "excluded_from_posterior",
                 "quality_weight", "data_quality_flag", "duration_guard_reason", "generated_at"]]


def run(write: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    df = build_quality_gate()
    if write:
        df.to_csv(OUT / "layer7_quality_gate.csv", index=False)
    s = df["quality_weight"]
    checks = [{
        "check_id": "patch_quality_gate_built", "phase": "quality_gate",
        "passed": len(df) > 0 and bool(((s >= FLOOR) & (s <= 1.0)).all())
                  and int(df.isna().sum().sum()) == 0,
        "detail": f"{len(df)} events; quality_weight in [{s.min():.2f},{s.max():.2f}]; "
                  f"flagged={int((df['data_quality_flag'] == 'flagged').sum())}",
        "severity": "info",
    }]
    return df, checks


if __name__ == "__main__":
    df, checks = run(write=True)
    print(df.head().to_string(index=False))
    print("flag dist:", df["data_quality_flag"].value_counts().to_dict())
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
