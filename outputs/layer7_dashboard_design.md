# Layer 7 — Phase 8: Dashboard Design

The dashboard is a **read-only consumer** of `outputs/frontend/layer7_*` (and existing `outputs/frontend/*` for L1–L4 context). Layer 7 ships the *backend exports*; the UI itself is out of scope but its data contract is fixed here. Refresh frequency = per Layer 7 batch run (the data is a historical batch, so "real-time" = on-rerun; the API exposes the latest files).

## Page 1 — Operations Overview
- **Data source:** `frontend/layer7_operations_overview.csv` (from `layer7_operational_state.csv` + `layer7_health_banner.csv`)
- **Refresh:** on batch rerun (manual/scheduled); API poll ≤ 60 s
- **KPIs:** total active sites (50), # CRITICAL ORS sites, mean ORS, batch `overall_health` banner, expected_delay_reduction_pct (from L5 metrics = 49.8%), cvar_90, # open critical alerts, stale-input count
- **Visualizations:** city map heat by ORS (junction-located); sortable site table (ORS, tier, allocation); health-banner strip; sparkline of CVaR baseline→optimized

## Page 2 — Active Alerts
- **Data source:** `frontend/layer7_active_alerts.csv` (from `layer7_alert_feed.csv` + `alert_summary`)
- **Refresh:** on rerun; API poll ≤ 30 s (alerts are the most time-sensitive)
- **KPIs:** counts by `severity_tier`; # critical drift triggers; top affected layer; max alert severity score
- **Visualizations:** severity-ranked alert list (color by tier, badge `corroboration_count`); grouping by `affected_layer`; recency decay indicator; click-through to Explainability narrative

## Page 3 — Resource Recommendations
- **Data source:** `frontend/layer7_resource_recommendations.csv` (+ `layer5_diversion_recommendations` for routes)
- **Refresh:** on rerun
- **KPIs:** total officers/barricades/tow/qru deployed (120/100/15/10 — budget-saturated); # diversions activated (10); mean effectiveness (0.52); mean robustness (0.64)
- **Visualizations:** per-site allocation cards (with override-pending badge); diversion route A/B/C panel with `estimated_additional_time_min`; before/after p95 delay bars (from robust_plan); "Override this plan" action → Page 4

## Page 4 — Override History
- **Data source:** `frontend/layer7_override_history.csv` (from `layer7_override_ledger.csv` + `override_impact.csv`)
- **Refresh:** on override submit (append) + on rerun
- **KPIs:** # overrides, # with positive OIS (increased risk), # override violations, net delta-delay across overrides
- **Visualizations:** chronological ledger table; OIS distribution; flagged-violation list (red); per-override before/after allocation diff

## Page 5 — Model Health
- **Data source:** `frontend/layer7_model_health.csv` (from `layer6_model_health_summary` reshaped) + `layer6_drift_report` + `layer6_monitoring_diagnostics`
- **Refresh:** on rerun
- **KPIs:** `overall_health` (CRITICAL this batch); retrain_urgency_score (0.61); knowledge_retention_score (0.97); # critical triggers (8); PH stat (326.8); mean-shift z (3.21); PSI hour_local (0.153)
- **Visualizations:** metric grid color-coded by normalized status; drift-test panel (PH/PSI/ODS/z with thresholds); calibration bin chart (ECE before/after); prototype-trust trajectory; "what should we retrain" list (from retrain_triggers `action`/`affected_layer`)

## Page 6 — Simulation (Tier 2)
- **Data source:** `frontend/layer7_simulation.csv` (from `layer7_simulation_results.csv` + summary) via API `/simulate`
- **Refresh:** on-demand (operator submits perturbation)
- **KPIs:** baseline vs perturbed CVaR-90; delta expected delay; # tail scenarios affected
- **Visualizations:** scenario delay distribution (baseline vs perturbed overlay); CVaR delta gauge; budget/alpha sliders (bounded to L5 `sensitivity_summary` ranges) → live re-eval; caveat banner: "re-evaluation of published L5 formulas; MILP not re-solved"

## Cross-cutting UI requirements
- Every numeric shown must trace to a source file (provenance tooltip) — supports the Explainability mission.
- Empty/near-empty feeds (e.g. `layer5_pareto_front` empty, 1 chance-constraint violation) must render as "no items / healthy", never as an error.
- Stale-input banner driven by `layer7_ingestion_manifest.stale_flag`.
- No write actions except override submission (POST `/overrides`), which appends to the ledger — never edits upstream.
