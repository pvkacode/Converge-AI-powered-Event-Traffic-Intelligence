# Layer 7 — Phase 7: Output Specification

All outputs are written **only** to `outputs/layer7_*` and `outputs/frontend/layer7_*`. No existing file is overwritten.

## Group A — Operational State

### `outputs/layer7_operational_state.csv`
- **Producer:** `layer7_operational_state.py` · **Consumer:** cockpit, API `/state`, dashboard Overview
- **PK:** `event_id`
- **Schema:** `event_id, event_cause, corridor, junction, service_tier, operational_risk_score, ors_tier, ors_components_json, ors_confidence, duration_p50, duration_p80, duration_p95, duration_reliability, tail_risk_prob, high_impact_prob_calibrated, novelty_score, drift_score, fragility_signal, obi_signal, officers_allocated, barricades_allocated, tow_trucks_allocated, qru_allocated, diversion_activated, robustness_score, solver_status, in_layer5_flag, in_layer6_flag, coverage_flag, generated_at`

### `outputs/layer7_operations_cockpit.csv`
- Wide join of operational_state + top-alert-per-site + override status + explanation pointer. PK `event_id`. Consumer: single-call dashboard backend.

### `outputs/layer7_ingestion_manifest.csv`
- **PK:** `file` · Schema: `file, layer, found, row_count, mtime, generated_at_max, stale_flag, expected_pk, schema_ok, notes`

## Group B — Recommendations

### `outputs/layer7_resource_recommendations.csv`
- **Producer:** operational_state + override · **Consumer:** dashboard Recommendations page
- **PK:** `event_id` · Schema: `event_id, event_cause, service_tier, recommended_officers, recommended_barricades, recommended_tow, recommended_qru, diversion_activated, diversion_corridor, diversion_route_label, estimated_additional_time_min, expected_delay_reduction_min, effectiveness, baseline_cvar, optimized_cvar, override_pending_flag, recommendation_source(L5|L5+override), generated_at`

## Group C — Alerts

### `outputs/layer7_alert_feed.csv`
- **Producer:** `layer7_alerting.py` · **Consumer:** dashboard Active Alerts, API `/alerts`
- **PK:** `l7_alert_id` · Schema: `l7_alert_id, source_ids_json, alert_severity_score, severity_tier(CRITICAL|HIGH|MEDIUM|LOW), affected_layer, affected_event_id, affected_variable, corroboration_count, retrain_urgency, recency_factor, description, recommended_action, first_generated_at, last_generated_at, merged_count`

### `outputs/layer7_alert_summary.csv`
- **PK:** `severity_tier` · Schema: `severity_tier, n_alerts, n_critical_sources, top_affected_layer, max_alert_severity_score, generated_at`

## Group D — Overrides

### `outputs/layer7_override_ledger.csv` (append-only)
- **PK:** `override_id` · Schema: `override_id, timestamp, operator_id, event_id, field_changed, old_value, new_value, reason_text, plan_basis(A|B|C|custom)`

### `outputs/layer7_override_impact.csv`
- **PK:** `override_id` · Schema: `override_id, event_id, delta_effectiveness, delta_expected_delay_min, delta_cvar_margin, delta_budget_units, override_impact_score, override_violation_flag, violated_constraint, generated_at`

## Group E — Monitoring / Model Health

### `outputs/layer7_model_health.csv`
- **Producer:** alerting + state (re-shaped from L6) · **Consumer:** dashboard Model Health
- **PK:** `metric_group, metric` · Schema: `metric_group, metric, holdout_value, feedback_value, relative_change, status_normalized(healthy|warning|critical), drives_alert_id, overall_health_banner, generated_at`

### `outputs/layer7_health_banner.csv`
- One row: `overall_health, n_critical_alerts, n_warning_alerts, retrain_urgency_score, knowledge_retention_score, stale_inputs_count, generated_at`

## Group F — Explainability

### `outputs/layer7_explanations.csv`
- **PK:** `event_id` · Schema: `event_id, why_priority(text), duration_belief_source(L1|L6 stratum|fallback_level), duration_belief_shifted_flag, optimization_value_text(p95 with/without), top_risk_driver, guard_reason, calibration_note, generated_at`

### `outputs/layer7_explanations.json`
- Structured per-event narrative: `{event_id: {priority_drivers[], duration_trace{}, optimization_trace{}, alert_links[], provenance{}}}`. Consumer: API `/recommendations` detail, dashboard tooltips.

## Group G — Simulation (Tier 2)

### `outputs/layer7_simulation_results.csv`
- **PK:** `sim_id, scenario` · Schema: `sim_id, perturbation_desc, scenario, baseline_total_delay, perturbed_total_delay, delta_delay, is_tail_scenario`

### `outputs/layer7_simulation_summary.json`
- `{sim_id: {perturbation{}, baseline_cvar_90, perturbed_cvar_90, delta_cvar_90, delta_expected_delay, note}}`

## Group H — API (no files; runtime contract)
- `/state` → operational_state rows; `/alerts` → alert_feed; `/recommendations[/{event_id}]` → recommendations + explanation; `/overrides` GET ledger / POST new override; `/health` → health_banner + model_health; `/simulate` POST perturbation → simulation_summary.

## Group I — Dashboard exports (`outputs/frontend/layer7_*`)
Curated, column-subset copies (mirroring `frontend_exports.py`): `layer7_operations_overview.csv`, `layer7_active_alerts.csv`, `layer7_resource_recommendations.csv`, `layer7_override_history.csv`, `layer7_model_health.csv`, `layer7_simulation.csv`.

## Group J — Validation
### `outputs/layer7_validation_report.csv`
- **PK:** `check_id` · Schema: `check_id, category, target_file, passed, detail, severity, generated_at`

## Schema conventions
- Every output carries `generated_at` (ISO-8601 UTC) for downstream staleness.
- Score columns (`*_score`) are bounded and never NaN (NaN → 0 + `coverage_flag`).
- All joins are left-joins anchored on the L4.5 JOSV; non-overlap recorded via `in_layer5_flag` / `in_layer6_flag`.
