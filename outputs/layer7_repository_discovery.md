# Layer 7 — Phase 0: Repository Discovery

**Branch:** `layer7-operations`  ·  **Date:** 2026-06-19  ·  **Status:** read-only audit, no files modified.

## 0.1 Top-level inventory

| Path | Role |
|------|------|
| `data/events_raw.csv` | Raw ASTraM log (~8,173 rows, Nov 2023–Apr 2024) |
| `data/events_clean.parquet` / `.csv` | Cleaned spine with `trust_score` (produced by `data_pipeline.py`) |
| `src/` | 24 Python modules — `data_pipeline.py`, `layer1..6`, `layer45`, `frontend_exports.py`, `validate_consistency.py` |
| `outputs/` | All layer artifacts, one prefix per layer (`layer1_*` … `layer6_*`) + `frontend/` + `*_model_artifacts/` |
| `outputs/frontend/` | Curated dashboard read-path (7 CSVs) — the only sanctioned UI source for L1–L4 |
| `catboost_info/` | CatBoost training-log side-effect of Layer 4.5 |
| `README.md` | 1,440-line authoritative spec for all layers |
| `HANDOFF.md` | Day-1 design/context note |
| `requirements.txt` | Dependency pins |
| `.claude/settings.json` | Permission allowlist only (no project config). **Note:** paths reference a stale `C:\Users\Dell\Desktop\...` location — harmless, not used by Layer 7 |

**No `CLAUDE.md` exists. No `configs/`, `.toml`, `.ini`, `.yaml` files exist.** Configuration is entirely in-code (module-level constants) plus `requirements.txt`.

## 0.2 Per-layer summary (inputs → outputs → deps → API)

### Layer 1 — Duration Intelligence
- **Files:** `layer1_survival.py`, `layer1_research_upgrades.py`
- **Inputs:** `data/events_clean.parquet`
- **Outputs:** `layer1_survival_quantiles.csv`, `layer1_survival_fallback.csv`, `layer1_cox_summary.txt`, `layer1_frailty_scores.csv`, `layer1_duration_predictions.csv`, `layer1_survival_risk_scores.csv`, `layer1_rmst_summary.csv`, `layer1_incident_archetypes.csv`, `layer1_rsf_*`, `layer1_stacked_*`, `layer1_advanced/*`
- **Public API:** `lookup_expected_duration(cause, corridor, km_table, km_fallback, quantile)` ; `run_layer1()`
- **Deps:** data pipeline only.

### Layer 2 — Spatial Intelligence
- **Files:** `layer2_hotspots.py`, `layer2_research_upgrades.py`
- **Inputs:** `data/events_clean.parquet`
- **Outputs:** `layer2_hotspots.csv`, `layer2_severity_hotspots.csv`, `layer2_network_hotspots.csv`, `layer2_hawkes_cascade_risk.csv`, `layer2_hotspot_persistence.csv`, `layer2_future_hotspot_risk.csv`, `layer2_operational_burden_index.csv`, `layer2_multiscale_hotspots.csv`, `layer2_obi_stable_top25.csv`
- **API:** `run_layer2()`, `run_layer2_upgrades()`, `build_junction_table()`, `build_junction_graph()`
- **Deps:** data pipeline only.

### Layer 3 — Resource Optimization + Corridor Fragility
- **Files:** `layer3_resource_optimization.py`, `layer3_corridor_fragility.py`, `layer3_methodology_upgrades.py`
- **Inputs:** L1 + L2 outputs (DIS built from OBI/cascade/future/RMST/persistence)
- **Outputs:** `layer3_disruption_impact_scores.csv` (294), `layer3_manpower_recommendations.csv` (294), `layer3_lp_resource_allocation.csv` (50), `layer3_barricading_plan.csv` (294), `layer3_diversion_recommendations.csv` (90), `layer3_corridor_fragility.csv` (22), `layer3_full_dashboard.csv` (294), `layer3_deployment_blueprints.json`, `layer3_pca_model.pkl`
- **Public API:** `generate_deployment_blueprint(junction_name, event_type) -> dict`
- **Deps:** L1, L2.

### Layer 4 — Event Intelligence + Prototype Retrieval
- **Files:** `layer4_event_intelligence.py`, `layer4_planned_event_retrieval.py`, `layer4_methodology_upgrades.py`, `layer4_operational_upgrades.py`
- **Inputs:** `events_clean.parquet`, L1 lookup, L3 fallbacks
- **Outputs:** `layer4_event_knowledge_base.json`, `layer4_retrieval_results.csv`, `layer4_institutional_memory_scores.csv`, `layer4_evidence_based_recommendations.csv`, `layer4_planned_event_retrieval.csv`, `layer4_planned_event_prototypes.csv`, `layer4_retrieval_diagnostics.csv`
- **API:** `run_layer4_operational_upgrades()`, `build_prototypes()`
- **Deps:** L1, L3, data pipeline.

### Layer 4.5 — Predictive Fusion (leak-free)
- **Files:** `layer45_time_split.py`, `layer45_asof_features.py`, `layer45_feature_registry.py`, `layer45_duration_guard.py`, `layer45_tail_models.py`, `layer45_predictive_fusion.py`
- **Inputs:** `events_clean.parquet` (rebuilds as-of surrogates — deliberately does NOT consume L1–L4 full-batch CSVs to avoid leakage)
- **Key outputs:** `layer45_operational_state_vector.csv` / `_normalized.csv` (JOSV, 3,498 rows), `layer45_scenario_ready_duration.csv` (canonical L5 duration input), `layer45_deployment_state_vector_normalized.csv`, `layer45_metrics.csv`, `layer45_feature_registry.json`, `layer45_model_artifacts/*.cbm/.joblib`
- **API:** `run_backtest()`, `run_deployment_inference()`, `main()`, `build_asof_feature_matrix()`, `build_scenario_ready_duration_bundle()`
- **Deps:** data pipeline only (by design).

### Layer 5 — Robust Prescriptive Optimization
- **File:** `layer5_robust_optimization.py`
- **Inputs:** `layer45_scenario_ready_duration.csv`, `layer45_operational_state_vector_normalized.csv` (and `_deployment_*` variants), `layer3_disruption_impact_scores.csv`, `layer3_diversion_recommendations.csv`
- **Outputs (13 CSV + summary + artifacts):** `layer5_resource_allocation.csv` (50), `layer5_frontend_export.csv` (50), `layer5_diversion_recommendations.csv` (30), `layer5_cvar_summary.csv`, `layer5_baseline_cvar_summary.csv`, `layer5_pre_post_cvar_comparison.csv`, `layer5_chance_constraint_violations.csv`, `layer5_robust_plan.csv`, `layer5_alternative_plans.csv`, `layer5_pareto_front.csv`, `layer5_shadow_prices.csv`, `layer5_sensitivity_summary.csv`, `layer5_optimization_metrics.csv`, `layer5_scenario_summary.csv`, `layer5_summary.txt`, `layer5_model_artifacts/`
- **API:** `load_inputs()`, `build_and_solve_milp()`, `main()` (no per-call public lookup — batch only)
- **Deps:** L4.5, L3.

### Layer 6 — Adaptive Learning (recommendations only)
- **Files:** `layer6_adaptive_learning.py` (orchestrator), `layer6_feedback_store.py`, `layer6_bayesian_duration.py`, `layer6_calibration_updates.py`, `layer6_drift_detection.py`, `layer6_retrain_triggers.py`
- **Inputs:** `events_clean.parquet` (prior/feedback split), L4.5 + L5 canonical outputs
- **Key outputs (40+):** `layer6_active_alerts.csv` (37), `layer6_retrain_triggers.csv` (32), `layer6_model_health_summary.csv` (20), `layer6_drift_report.csv` (7), `layer6_duration_posterior_summary.csv` (119), `layer6_calibration_posteriors.csv` (10), `layer6_monitoring_diagnostics.csv` (22), `layer6_prototype_trust_updates.csv` (47), `layer6_feedback_log.csv` (1,097), `layer6_versioned_knowledge_base.json`, `layer6_model_artifacts/*.json`
- **API:** `main()` (orchestrator); helpers are private (`_*`)
- **Deps:** L4.5, L5, data pipeline. **Never mutates upstream — invariant.**

### Cross-cutting
- `data_pipeline.py` → `run_pipeline()`, builds `trust_score` spine.
- `frontend_exports.py` → `run_frontend_exports()`, copies canonical L1–L4 outputs to `outputs/frontend/`.
- `validate_consistency.py` → `main()`, sanity gate on the clean parquet.

## 0.3 Existing dashboards / UI contract
- The only existing "dashboard backend" is `outputs/frontend/` (7 CSVs: `duration_lookup`, `risk_scores`, `hotspot_rankings`, `operational_burden`, `top25_locations`, `corridor_fragility`, `planned_event_recommendations`). Plus `layer5_frontend_export.csv`. There is **no web server, no API code** — these are static file contracts. Layer 7's API/dashboard backend is greenfield.

## 0.4 Existing helper utilities reusable by Layer 7 (read-only patterns to imitate, not import)
- `frontend_exports.py::_export()` — safe column-subset CSV copier with `[SKIP]`/`[OK]` logging.
- `layer6_feedback_store.py` — read-only loader pattern over upstream outputs.
- `validate_consistency.py` — standalone read-only invariant checker.
- L5 `load_inputs()::_safe_read()` — defensive CSV reader.

## 0.5 Additive-only confirmation
Every prior layer (incl. all `*_research_upgrades`, `*_methodology_upgrades`, `*_operational_upgrades`, L4.5 guard patch, L6 Parts C–H) was built as **read existing outputs → write new namespace → never retrain/modify upstream**. Layer 7 will follow the identical convention.
