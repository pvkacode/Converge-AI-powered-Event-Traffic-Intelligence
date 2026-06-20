# Layer 7 — Phase 9: Validation Plan

Validation is itself additive: a new `src/layer7_validate.py` (mirroring `validate_consistency.py`) plus unit tests under a new `tests/layer7/` folder. Nothing existing is modified.

## 9.1 Unit tests (per engine, pure functions)

| Target | Test |
|--------|------|
| `compute_operational_risk_score` | bounded [0,100]; monotone ↑ in tail/hi/frag z; robustness discount lowers score; NaN input → 0 + coverage_flag |
| `compute_alert_severity_score` | base map correct; corroboration escalates; recency decays with Δt; clipped [0,1]; tier thresholds exact at 0.30/0.55/0.80 |
| `normalize_severity` | every upstream label (`critical/moderate/info/warning/none/CRITICAL/healthy`) maps deterministically; unknown label → LOW + warning |
| `compute_override_impact_score` | uses L5 γ verbatim; sign convention (positive = worse); violation flag fires below tier minimum-service / above budget cap |
| `fuse` (sensor) | single-source = identity; inverse-variance weighting correct; coverage gate zeroes absent source |
| `recompute_cvar` (twin) | reproduces L5 `layer5_cvar_summary.csv` values to tolerance when given the same scenario matrix (golden test) |
| `_safe_read` | missing file → None + manifest entry, no raise |

## 9.2 Integration tests
- **End-to-end run** of `layer7_operations.main()` over the committed `outputs/` produces all Group A–J files with expected PKs and row counts (state≈3,498 or 50 active; alert_feed ≤ 37+32+1 pre-dedup).
- **Join integrity:** every `event_id` in `layer7_operational_state` exists in L4.5 JOSV; `in_layer5_flag`/`in_layer6_flag` counts match the 50 / 1,097 overlaps.
- **Idempotency:** two consecutive runs differ only in `generated_at` columns.

## 9.3 Compatibility tests (the freeze guarantee)
- **Write-scope assertion:** monkeypatch/wrap file writes; assert every path written matches regex `outputs/layer7_.*` or `outputs/frontend/layer7_.*`. Any other path = test failure. (This is the single most important test.)
- **Upstream-immutability:** snapshot SHA-256 of every non-`layer7_` file in `outputs/` and `src/` before and after `layer7_operations.main()`; assert unchanged.
- **Re-run of `validate_consistency.py`** after Layer 7 still passes (Layer 7 didn't touch the clean parquet).

## 9.4 Schema validation (`layer7_validate.py`)
- Each output: required columns present, declared PK unique & non-null, `generated_at` parseable ISO-8601.
- Score columns within documented ranges; zero NaN in score columns.
- `ingestion_manifest`: no *required* input missing (optional ones, e.g. empty `pareto_front`, allowed).

## 9.5 Alert validation
- No duplicate `l7_alert_id`; `merged_count ≥ 1`.
- Every alert's `source_ids_json` references a real upstream row id.
- Severity ordering: tier consistent with `alert_severity_score` band.
- Empty-feed case (all-healthy batch) → `alert_feed` may be 0 rows, validator passes with an INFO note.

## 9.6 Override validation
- `override_ledger` append-only (row count never decreases across runs; existing rows immutable — hash check).
- Every `override_impact.override_id` joins to a ledger row.
- Violations correctly flagged against L5 tier minimum-service and budget caps.

## 9.7 Runtime diagnostics
- Each engine logs `[OK]/[SKIP]/[WARN]` lines (à la `frontend_exports.py`) and elapsed time; orchestrator prints a summary table.
- Total runtime budget asserted < 60 s (no model inference, no MILP).

## 9.8 Test infrastructure (new, additive)
- `tests/layer7/` with `pytest`; fixtures load committed `outputs/` as golden inputs.
- Optional CI hook documented in `docs/LAYER7.md` (not wired into existing config without approval).
