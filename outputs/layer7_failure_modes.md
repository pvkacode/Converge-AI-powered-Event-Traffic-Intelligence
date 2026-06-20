# Layer 7 — Phase 10: Failure Modes

Principle: Layer 7 **never crashes the pipeline and never blocks operators**. Every failure degrades to a flagged, logged, partial result. Detection / mitigation / logging below.

| # | Failure mode | Detection | Mitigation | Logging |
|---|--------------|-----------|------------|---------|
| 1 | **Missing input file** (e.g. L5 not run) | `_safe_read` returns None; manifest `found=False` | Skip dependent engine; emit partial outputs; set `coverage_flag` on affected rows | `[SKIP] <file> not found` + manifest row + validation WARN |
| 2 | **Schema drift** (upstream adds/renames column) | Compare actual columns to expected set in `layer7_config` | Use only intersected columns; missing required field → engine-level skip with reason | `[WARN] schema drift: <file> missing <cols>` + validation_report row |
| 3 | **Stale outputs** (L6 `generated_at` older than L5 mtime by > STALE_HOURS) | Freshness check in feedback_store | Proceed but set `stale_flag`; surface staleness banner on dashboard | manifest `stale_flag=True`, health_banner `stale_inputs_count` |
| 4 | **Empty feed** (`layer5_pareto_front`=0 rows, `chance_constraint_violations`=1) | `len(df)==0` / near-zero | Treat as valid "healthy/no-items"; never error | INFO note only |
| 5 | **All-healthy alert batch** (0 alerts) | empty alert_feed after build | Emit empty feed + summary row `n_alerts=0`; dashboard shows "no active alerts" | INFO |
| 6 | **Sensor failure** (future real source unavailable) | coverage gate $c_{k,r}=0$ in fusion | Inverse-variance fusion drops the source; falls back to JOSV only | `[WARN] sensor <k> offline` |
| 7 | **API failure / fastapi absent** | import error in `layer7_api` | Degrade: file exports still produced; API prints "optional dep missing" | `[SKIP] API disabled: fastapi not installed` |
| 8 | **Override abuse** (operator removes all resources / exceeds budget) | `check_override_violation` vs L5 tier minimums + budget caps | Record override but flag `override_violation_flag=True`; OIS quantifies risk increase; never silently applied to L5 | ledger + impact `violated_constraint` |
| 9 | **Invalid recommendation** (event_id not in any upstream) | left-join yields no match; `in_layer5_flag=False` & no JOSV row | Drop from recommendations with reason; keep in audit | validation_report row |
| 10 | **Duplicate / conflicting alerts** | dedup hash collision with differing severity | Keep max-ASS representative; `merged_count` increments; conflicts logged | alert dedup log |
| 11 | **NaN / inf in score inputs** | numpy isfinite check before scoring | Replace with 0, set `coverage_flag`; never propagate NaN to outputs | `[WARN] non-finite input at <event_id>` |
| 12 | **Write outside namespace** (programming error) | path guard in every writer + compatibility test | Raise immediately in dev (fail-fast); CI test blocks merge | hard error (this is the one place L7 *should* stop) |
| 13 | **Upstream re-run mid-batch** (mixed-version inputs) | `generated_at` spread across feeds exceeds skew threshold | Flag `inconsistent_batch`; proceed with newest, warn | health_banner + manifest |
| 14 | **Digital Twin perturbation out of range** | bounds-check vs L5 `sensitivity_summary` ranges | Clamp to valid range; annotate `clamped=True` | sim note |
| 15 | **Override ledger corruption / truncation** | append-only hash check on load | Refuse to truncate; load last-valid snapshot; new appends continue | validation FAIL (override integrity) |

## 10.1 Detection layers
1. **Ingestion** — `feedback_store` + manifest catch missing/stale/schema issues before any compute.
2. **Compute** — each engine guards inputs (finite checks, coverage flags).
3. **Output** — `layer7_validate.py` schema/PK/range/namespace checks after writing.
4. **Runtime** — orchestrator summary line: `engines_run / skipped / warnings / runtime`.

## 10.2 Logging convention
- Reuse the `[OK]/[SKIP]/[WARN]/[FAIL]` prefix convention from `frontend_exports.py`.
- Structured machine-readable record of every WARN/FAIL in `layer7_validation_report.csv` (so the dashboard can surface system-health independent of the data).

## 10.3 The one hard-stop
Only **failure #12 (write outside the `layer7_` namespace)** is allowed to raise and halt Layer 7 — because it would violate the additive-only freeze guarantee. Every other failure degrades gracefully.
