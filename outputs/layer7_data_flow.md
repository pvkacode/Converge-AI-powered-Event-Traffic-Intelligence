# Layer 7 — Phase 4: Data Flow Design

## 4.1 High-level flow

```
            FROZEN UPSTREAM (read-only)                         LAYER 7 (additive)
 ┌─────────────────────────────────────────┐
 │ data/events_clean.parquet  (8,173 spine) │──┐
 └─────────────────────────────────────────┘  │
 ┌─────────────────────────────────────────┐  │   ┌──────────────────────────────┐
 │ L4.5  operational_state_vector(_norm)    │──┼──▶│ layer7_feedback_store        │
 │       scenario_ready_duration            │  │   │ (defensive read-only loader, │
 │       metrics, feature_registry          │  │   │  staleness + coverage flags) │
 └─────────────────────────────────────────┘  │   └──────────────┬───────────────┘
 ┌─────────────────────────────────────────┐  │                  │
 │ L3   disruption_impact_scores            │──┤                  ▼
 │      diversion_recommendations           │  │   ┌──────────────────────────────┐
 │      corridor_fragility, full_dashboard  │  │   │ 1. Operational State Engine   │
 └─────────────────────────────────────────┘  │   │    per-site fused state +     │
 ┌─────────────────────────────────────────┐  │   │    Operational Risk Score     │
 │ L5   resource_allocation / frontend      │──┤   └──────┬───────────────────────┘
 │      diversion_recommendations           │  │          │ layer7_operational_state.csv
 │      cvar / pre_post / shadow / metrics  │  │          ▼
 │      chance_constraint_violations        │  │   ┌──────────────────────────────┐
 │      robust_plan / alternative_plans     │  │   │ 2. Alerting Engine            │◀── L6 active_alerts
 └─────────────────────────────────────────┘  │   │    normalize+dedupe+rank      │◀── L6 retrain_triggers
 ┌─────────────────────────────────────────┐  │   │    Alert Severity Score       │◀── L5 cc_violations
 │ L6   active_alerts, retrain_triggers     │──┘   └──────┬───────────────────────┘
 │      model_health_summary, drift_report  │             │ layer7_alert_feed.csv
 │      monitoring_diagnostics, calib_*      │             ▼
 │      duration_posterior_summary           │      ┌──────────────────────────────┐
 │      prototype_trust_updates, feedback_log│      │ 3. Explainability Engine      │
 └───────────────────────────────────────────┘     │    trace "why" per site/alert │
                                                    └──────┬───────────────────────┘
                                                           │ layer7_explanations.csv/json
                          ┌────────────────────────────────┼───────────────────────┐
                          ▼                                ▼                        ▼
                 ┌────────────────┐            ┌────────────────────┐     ┌──────────────────┐
                 │4. Override Eng.│            │5. Digital Twin (T2)│     │6. Exports + API  │
                 │ ledger+impact  │            │ what-if re-eval     │     │ frontend/layer7_*│
                 └───────┬────────┘            └─────────┬──────────┘     └────────┬─────────┘
                         │ layer7_override_ledger.csv     │ layer7_simulation_*.csv │
                         ▼                                ▼                        ▼
                 ┌────────────────────────────────────────────────────────────────────┐
                 │ layer7_operations_cockpit.csv  +  outputs/frontend/layer7_*.csv      │
                 │ (consumed by dashboard backend / API)                                │
                 └────────────────────────────────────────────────────────────────────┘
```

## 4.2 Layer 5 → Layer 7 ingestion

| L5 source | Join key | L7 consumer | Use |
|-----------|----------|-------------|-----|
| `layer5_resource_allocation.csv` | `event_id` | Operational State, Override | canonical per-site plan (officers/barricades/tow/qru, tier, robustness, solver_status) |
| `layer5_frontend_export.csv` | `event_id` | Dashboard exports | adds baseline/optimized CVaR per site |
| `layer5_chance_constraint_violations.csv` | `event_id` | Alerting | hard-constraint breach → high-severity alert (may be empty) |
| `layer5_diversion_recommendations.csv` | `event_id,route_rank` | Recommendations, Explainability | route A/B/C detail |
| `layer5_robust_plan.csv` | `event_id` | Explainability, Digital Twin | p95 with/without resources, value-of-optimization |
| `layer5_alternative_plans.csv` | `plan,event_id` | Override Engine | A/B/C plan options to choose between |
| `layer5_optimization_metrics.csv` | `metric` | Operational State, Dashboard | batch KPIs (delay reduction %, cvar_90, cc satisfaction) |
| `layer5_shadow_prices.csv` / `cvar_*` / `sensitivity` | resource/alpha | Dashboard, Digital Twin | budget marginal value, tail-risk, what-if reference |

## 4.3 Layer 6 → Layer 7 ingestion

| L6 source | Join key | L7 consumer | Use |
|-----------|----------|-------------|-----|
| `layer6_active_alerts.csv` | `alert_id` | Alerting | primary alert stream (severity, affected_layer, description, generated_at) |
| `layer6_retrain_triggers.csv` | `trigger_id` | Alerting | action recommendations (action, affected_layer) |
| `layer6_model_health_summary.csv` | `metric_group,metric` | Model Health page, State Engine | per-metric health + batch `overall_health` |
| `layer6_drift_report.csv` | `test,variable` | Model Health, Alerting | drift severity + retrain_urgency |
| `layer6_monitoring_diagnostics.csv` | `diagnostic_group,metric` | Model Health | KRS, entropy, redundancy, urgency score |
| `layer6_calibration_posteriors.csv` / `recalibration_recommendations.csv` | `bin_idx` | Model Health, Recommendations | calibration drift + actions |
| `layer6_duration_posterior_summary.csv` | `cause,corridor` | Explainability | learned duration belief per stratum + fallback provenance |
| `layer6_prototype_trust_updates.csv` | `prototype_id` | Model Health | trust trajectory; degraded prototypes |
| `layer6_feedback_log.csv` | `event_id` | Explainability | actual-vs-predicted residual backing |

## 4.4 Layer 7 processing → Layer 7 outputs

| Engine | Inputs | Output(s) |
|--------|--------|-----------|
| Feedback store | all above | in-memory frames + `layer7_ingestion_manifest.csv` (file, found, rows, mtime, stale_flag) |
| Operational State | L4.5 JOSV + L5 alloc + L3 ctx + L6 health | `layer7_operational_state.csv`, `layer7_operations_cockpit.csv` |
| Alerting | L6 alerts/triggers/drift + L5 violations | `layer7_alert_feed.csv`, `layer7_alert_summary.csv` |
| Explainability | L6 posteriors/feedback + L5 robust + L4.5 guard | `layer7_explanations.csv`, `layer7_explanations.json` |
| Override | L5 alloc/alt_plans + operator input file (optional) | `layer7_override_ledger.csv`, `layer7_override_impact.csv` |
| Digital Twin (T2) | L5 scenario/cvar + JOSV | `layer7_simulation_results.csv` |
| Exports/API | all L7 outputs | `outputs/frontend/layer7_*.csv` |

## 4.5 Critical flow invariants
- **Left-join on `event_id`** anchored on the L4.5 JOSV (3,498 deployment events); L5 (50) and L6 (1,097) are sparse overlays → unmatched rows get explicit `not_in_layer5`/`not_in_layer6` flags, never dropped silently.
- **No write path touches any non-`layer7_` file.** Verified against the `frontend_exports.py` contract (its 7 canonical CSVs are never overwritten; L7 adds new `layer7_*` files beside them).
- **Freshness** = `max(generated_at)` from L6 feeds vs filesystem `mtime` of L5/L4.5; a configurable skew threshold flips a `stale` flag in the ingestion manifest.
