# Layer 7 — Phase 1: Assumption Verification

**Method:** every filename below was verified to physically exist in `outputs/` with header + row count read directly (not assumed). Date: 2026-06-19.

## 1.1 Layer 5 assumptions

| Assumed concept | Actual file | Exists | Rows | Notes |
|-----------------|-------------|--------|------|-------|
| Resource allocation | `layer5_resource_allocation.csv` | ✅ | 50 | per active site (event_id PK) |
| Resource allocation (UI) | `layer5_frontend_export.csv` | ✅ | 50 | adds `baseline_cvar`, `optimized_cvar` |
| Diversion plans | `layer5_diversion_recommendations.csv` | ✅ | 30 | Route A/B/C per diversion site |
| CVaR outputs | `layer5_cvar_summary.csv` | ✅ | 9 | α ∈ {.50,.75,.90,.95,.99} + per-tier |
| CVaR baseline | `layer5_baseline_cvar_summary.csv` | ✅ | 5 | zero-resource baseline |
| CVaR pre/post | `layer5_pre_post_cvar_comparison.csv` | ✅ | 5 | reduction all-sites + per-tier |
| Shadow prices | `layer5_shadow_prices.csv` | ✅ | 4 | one per resource type |
| Optimization metrics | `layer5_optimization_metrics.csv` | ✅ | 20 | long `metric,value` format |
| Chance-constraint violations | `layer5_chance_constraint_violations.csv` | ✅ | **1** | only 1 violation this batch |
| Robust plan | `layer5_robust_plan.csv` | ✅ | 50 | p95 with/without resources |
| Alternative plans | `layer5_alternative_plans.csv` | ✅ | 150 | Plans A/B/C × 50 sites |
| Scenario summary | `layer5_scenario_summary.csv` | ✅ | 50 | per-scenario delay |
| Sensitivity | `layer5_sensitivity_summary.csv` | ✅ | 26 | budget × α sweep |
| Pareto front | `layer5_pareto_front.csv` | ✅ | **0** | ⚠️ **header-only, EMPTY this batch** |
| Summary | `layer5_summary.txt` | ✅ | — | human-readable |

⚠️ **Correction to assumptions:** `layer5_pareto_front.csv` exists but is **empty (0 data rows)** in the current batch. Layer 7 must treat it as optional / possibly-empty and not assume non-empty Pareto points. `layer5_chance_constraint_violations.csv` has only **1 row** — feeds are sometimes near-empty by design (a healthy batch). Layer 7 alerting must handle empty/near-empty feeds gracefully (see Phase 10).

## 1.2 Layer 6 assumptions

| Assumed concept | Actual file | Exists | Rows | Notes |
|-----------------|-------------|--------|------|-------|
| Active alerts | `layer6_active_alerts.csv` | ✅ | 37 | cols: `alert_id, source, severity, affected_layer, description, generated_at` |
| Retrain triggers | `layer6_retrain_triggers.csv` | ✅ | 32 | cols incl. `severity, action, affected_layer, generated_at` |
| Model health | `layer6_model_health_summary.csv` | ✅ | 20 | `metric_group, metric, holdout_value, feedback_value, relative_change, status, overall_health` |
| Drift report | `layer6_drift_report.csv` | ✅ | 7 | PH/PSI/ODS/mean-shift; `alert, severity, retrain_urgency` |
| Calibration diagnostics | `layer6_calibration_posteriors.csv` | ✅ | 10 | decile Beta posteriors + ECE before/after |
| Calibration (recs) | `layer6_recalibration_recommendations.csv` | ✅ | 10 | priority ranking |
| Monitoring diagnostics | `layer6_monitoring_diagnostics.csv` | ✅ | 22 | entropy, KRS, redundancy, urgency |
| Duration posteriors | `layer6_duration_posterior_summary.csv` | ✅ | 119 | per-stratum posterior + CI |
| Prototype trust | `layer6_prototype_trust_updates.csv` | ✅ | 47 | Beta-Binomial + EMA |
| Feedback log | `layer6_feedback_log.csv` | ✅ | 1,097 | actual vs predicted per event |
| Versioned KB | `layer6_versioned_knowledge_base.json` | ✅ | — | timestamped posterior snapshot |

All assumed Layer 6 concepts exist. No filename differs from the README. Severity vocabularies verified:
- `layer6_active_alerts.severity` / `retrain_triggers.severity`: `critical | moderate | info`
- `layer6_model_health_summary.status`: `healthy | warning | critical`; `overall_health`: `CRITICAL` (single batch-level value).
- `layer6_drift_report.severity`: `critical | moderate | none`.

⚠️ **Severity vocabularies are NOT uniform** across feeds (`moderate` vs `warning`; uppercase `overall_health`). Layer 7 must normalize to one ordinal scale (Phase 5 §Alert Severity).

## 1.3 Layer 4.5 assumptions (L7 also consumes JOSV directly)

| Concept | File | Exists | Rows |
|---------|------|--------|------|
| Operational state vector (JOSV) | `layer45_operational_state_vector.csv` | ✅ | 3,498 |
| JOSV normalized | `layer45_operational_state_vector_normalized.csv` | ✅ | 3,498 |
| Scenario-ready duration | `layer45_scenario_ready_duration.csv` | ✅ | 3,498 |
| Deployment state (normalized) | `layer45_deployment_state_vector_normalized.csv` | ✅ | 3,498 |
| Backtest metrics | `layer45_metrics.csv` | ✅ | 66 |
| Feature registry | `layer45_feature_registry.json` | ✅ | — |

## 1.4 Cross-layer join keys (verified)

- **`event_id`** is the universal site/event PK across L4.5 JOSV, L5 allocation/diversion/robust, and L6 feedback/posterior-predictive logs.
- **`junction`** is the PK across L2 hotspots, L3 DIS/manpower/barricading/diversion, and `outputs/frontend/`.
- **`event_cause` × `corridor`** is the stratum key for L1 quantiles and L6 duration posteriors.
- ⚠️ L5 operates on the **50 active sites** (subset of 3,498 deployment events); L6 operates on the **1,097 uncensored feedback events**. Layer 7 joins are therefore **left-joins keyed on `event_id`**, and not every L4.5 event appears in L5 or L6.

## 1.5 Timestamp / freshness signals
- `layer6_*` rows carry `generated_at` (ISO-8601, e.g. `2026-06-19T10:34:30Z`). This is the canonical freshness anchor for Layer 7 staleness detection.
- L5 / L4.5 CSVs have **no embedded timestamp column** → Layer 7 must use filesystem `mtime` for their staleness checks.

## 1.6 Verdict
All assumed Layer 5 and Layer 6 interfaces exist. Two material corrections: **`layer5_pareto_front.csv` is empty** and **`layer5_chance_constraint_violations.csv` has 1 row** in this batch — both are valid states, not defects. **No critical defect blocks Layer 7.** No upstream modification required.
