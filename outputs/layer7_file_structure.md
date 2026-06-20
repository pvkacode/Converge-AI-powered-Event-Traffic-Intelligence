# Layer 7 — Phase 6: File Structure Design

All new files live under `src/` with the `layer7_` prefix. **No existing file is modified.** A new standalone doc `docs/LAYER7.md` is proposed instead of editing `README.md`/`HANDOFF.md`.

## 6.1 Module map

| File | Tier | Purpose |
|------|------|---------|
| `src/layer7_config.py` | 0 | All fixed constants (ORS/ASS/OIS weights, severity map, staleness thresholds, paths). Single source of tunables. |
| `src/layer7_feedback_store.py` | 0 | Defensive read-only loader for every L3/L4.5/L5/L6 input + ingestion manifest. Built first, used by all. |
| `src/layer7_operational_state.py` | 0 | Operational State Engine + Operational Risk Score (§5.1). |
| `src/layer7_alerting.py` | 0 | Alerting Engine + Alert Severity Score (§5.2); normalize/dedupe/rank. |
| `src/layer7_explainability.py` | 0 | Explainability Engine — trace "why" per site/alert from existing posteriors/robustness/guard reasons. |
| `src/layer7_sensor_fusion.py` | 3 (stub now) | Inverse-variance fusion (§5.4). MVP: identity over JOSV, with adapter seam. |
| `src/layer7_override_engine.py` | 1 | Human Override ledger + Override Impact Score (§5.3). Append-only. |
| `src/layer7_digital_twin.py` | 2 | What-if re-evaluation of L5 published formulas (§5.5). No MILP re-solve. |
| `src/layer7_exports.py` | 0 | Writes `outputs/frontend/layer7_*.csv` (mirrors `frontend_exports.py` pattern). |
| `src/layer7_api.py` | 1 (thin) | Read-only file-backed service (FastAPI optional dep) exposing L7 outputs. |
| `src/layer7_operations.py` | 0 | Orchestrator: runs engines in order, like `layer6_adaptive_learning.main()`. |
| `src/layer7_validate.py` | 0 | Read-only consistency/schema checks over `layer7_*` (mirrors `validate_consistency.py`). |
| `docs/LAYER7.md` | 0 | Documentation (new file; README untouched). |

## 6.2 Per-file detail

### `layer7_config.py`
- **Inputs:** none. **Outputs:** none (constants only).
- **Contents:** `OUT_DIR`, `FRONT_DIR`; `ORS_WEIGHTS`, `ASS_BETAS`, `SEVERITY_MAP`, `OIS_LAMBDAS`, `GAMMA = (0.18,0.10,0.25,0.30)` (read-only mirror of L5), `STALE_HOURS`, `RECENCY_HALFLIFE_DAYS`.
- **Deps:** none. **Runtime:** instant.

### `layer7_feedback_store.py`
- **Inputs:** all files in `layer7_interface_inventory.csv`.
- **Outputs:** in-memory `dict[str,DataFrame]`; `outputs/layer7_ingestion_manifest.csv`.
- **Functions:** `load(name)->DataFrame|None` (safe), `load_all()->Store`, `manifest()->DataFrame`, `freshness()->dict`, `_safe_read()` (à la L5).
- **Class:** `Store` (attribute access to frames + coverage flags).
- **Deps:** pandas, pyarrow. **Runtime:** <2 s.

### `layer7_operational_state.py`
- **Inputs:** L4.5 JOSV(_normalized), L5 resource_allocation/frontend_export, L3 DIS/fragility, L6 model_health.
- **Outputs:** `layer7_operational_state.csv` (per `event_id`), contributes to `layer7_operations_cockpit.csv`.
- **Functions:** `compute_operational_risk_score()`, `build_state_table()`, `attach_layer3_context()`, `attach_health_banner()`.
- **Deps:** feedback_store, config, numpy. **Runtime:** <5 s.

### `layer7_alerting.py`
- **Inputs:** L6 active_alerts/retrain_triggers/drift_report, L5 chance_constraint_violations.
- **Outputs:** `layer7_alert_feed.csv`, `layer7_alert_summary.csv`.
- **Functions:** `normalize_severity()`, `compute_alert_severity_score()`, `corroborate()`, `dedupe()`, `rank_feed()`.
- **Deps:** feedback_store, config. **Runtime:** <2 s.

### `layer7_explainability.py`
- **Inputs:** L6 duration_posterior_summary/feedback_log/calibration, L5 robust_plan/diversion, L4.5 scenario_ready_duration (guard reasons)/feature_registry.
- **Outputs:** `layer7_explanations.csv`, `layer7_explanations.json`.
- **Functions:** `explain_site()`, `explain_alert()`, `trace_duration_belief()`, `trace_optimization_value()`.
- **Deps:** feedback_store, operational_state. **Runtime:** <5 s.

### `layer7_override_engine.py`
- **Inputs:** L5 resource_allocation/alternative_plans; optional operator file `data/layer7_overrides_input.csv` (operator-provided, outside outputs/).
- **Outputs:** `layer7_override_ledger.csv` (append-only), `layer7_override_impact.csv`.
- **Functions:** `record_override()`, `compute_override_impact_score()`, `check_override_violation()`.
- **Deps:** feedback_store, config. **Runtime:** <2 s.

### `layer7_sensor_fusion.py`
- **Inputs:** L4.5 JOSV; future sensor adapters.
- **Outputs:** `layer7_fused_signals.csv` (MVP = JOSV passthrough + coverage).
- **Functions:** `fuse()`, `register_source()` (seam). **Runtime:** <2 s.

### `layer7_digital_twin.py`
- **Inputs:** L5 scenario_summary/cvar/robust_plan, JOSV; operator perturbation spec.
- **Outputs:** `layer7_simulation_results.csv`, `layer7_simulation_summary.json`.
- **Functions:** `evaluate_scenarios()`, `recompute_cvar()` (L5 formula, read-only), `apply_perturbation()`, `greedy_refill()` (L5 fallback mirror).
- **Deps:** numpy. **Runtime:** <10 s (no MILP).

### `layer7_exports.py`
- **Inputs:** all L7 outputs. **Outputs:** `outputs/frontend/layer7_operations_overview.csv`, `layer7_active_alerts.csv`, `layer7_resource_recommendations.csv`, `layer7_override_history.csv`, `layer7_model_health.csv`, `layer7_simulation.csv`.
- **Functions:** `run_layer7_exports()`, `_export()` (à la frontend_exports). **Runtime:** <2 s.

### `layer7_api.py`
- **Inputs:** L7 output files. **Outputs:** HTTP JSON (no files).
- **Endpoints (read-only):** `/state`, `/alerts`, `/recommendations`, `/overrides`, `/health`, `/simulate` (POST perturbation → twin).
- **Deps:** fastapi+uvicorn (optional; degrade to "not installed" message). **Runtime:** service.

### `layer7_operations.py`
- **Orchestrator.** `main()` runs: feedback_store → sensor_fusion → operational_state → alerting → explainability → override → (twin) → exports → validate. Mirrors `layer6_adaptive_learning.main()` ordering. **Runtime:** <30 s end-to-end.

### `layer7_validate.py`
- **Inputs:** all `layer7_*` outputs. **Outputs:** console pass/fail + `layer7_validation_report.csv`.
- Checks: schema presence, PK uniqueness, no NaN in score columns, score ranges, no write outside `layer7_` namespace, ingestion manifest has no unexpected missing files.

## 6.3 Dependency graph (build order)
```
layer7_config ─┬─▶ layer7_feedback_store ─┬─▶ layer7_operational_state ─┬─▶ layer7_explainability
               │                          ├─▶ layer7_alerting           │
               │                          ├─▶ layer7_sensor_fusion       │
               │                          └─▶ layer7_override_engine      │
               └────────────────────────────────────────────────────────┴─▶ layer7_digital_twin
                                                                          └─▶ layer7_exports ─▶ layer7_api
                                                              all ─▶ layer7_operations ─▶ layer7_validate
```

## 6.4 New dependencies
- **Required:** none beyond existing `requirements.txt` (pandas/numpy/scipy/networkx already present).
- **Optional:** `fastapi`, `uvicorn` for `layer7_api.py` — must degrade gracefully if absent (try/except import, like `faiss-cpu` precedent). If added, append to `requirements.txt` only with explicit approval.
