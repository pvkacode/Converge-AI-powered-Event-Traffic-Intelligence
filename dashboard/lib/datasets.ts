// Client-safe dataset registry. Maps a logical key to the real CSV path under
// outputs/, plus presentation metadata (badge columns, label overrides, default
// sort). The `file` strings are only ever used server-side by the API route to
// read from disk; the API validates every request against this whitelist, so no
// arbitrary path can be requested.

export interface DatasetSpec {
  key: string;
  file: string; // path relative to outputs/
  /** server-side augmenter id (adds derived columns before paginating) */
  augment?: "risk_tier";
  /** columns rendered as palette-aware badges */
  badgeCols?: string[];
  /** per-column label overrides (beyond the automatic humanizer) */
  labels?: Record<string, string>;
  /** columns to hide from the table view (kept available for charts elsewhere) */
  hide?: string[];
  /** default sort */
  defaultSort?: { col: string; dir: "asc" | "desc" };
  /** default page size */
  pageSize?: number;
}

export const DATASETS: Record<string, DatasetSpec> = {
  duration_lookup: {
    key: "duration_lookup",
    file: "frontend/duration_lookup.csv",
    defaultSort: { col: "n", dir: "desc" },
    pageSize: 15,
  },
  risk_scores: {
    key: "risk_scores",
    file: "frontend/risk_scores.csv",
    augment: "risk_tier",
    badgeCols: ["risk_tier"],
    defaultSort: { col: "survival_risk_score", dir: "desc" },
    pageSize: 20,
  },
  hotspot_rankings: {
    key: "hotspot_rankings",
    file: "frontend/hotspot_rankings.csv",
    defaultSort: { col: "sps", dir: "desc" },
    pageSize: 15,
  },
  layer2_hotspots_geo: {
    key: "layer2_hotspots_geo",
    file: "layer2_hotspots.csv",
    badgeCols: ["is_significant"],
    defaultSort: { col: "hotspot_probability", dir: "desc" },
    pageSize: 15,
  },
  operational_burden: {
    key: "operational_burden",
    file: "frontend/operational_burden.csv",
    defaultSort: { col: "operational_burden_index", dir: "desc" },
    pageSize: 15,
  },
  top25_locations: {
    key: "top25_locations",
    file: "frontend/top25_locations.csv",
    defaultSort: { col: "mean_rank", dir: "asc" },
    pageSize: 25,
  },
  corridor_fragility: {
    key: "corridor_fragility",
    file: "frontend/corridor_fragility.csv",
    defaultSort: { col: "fragility_log", dir: "desc" },
    pageSize: 22,
  },
  planned_event_recommendations: {
    key: "planned_event_recommendations",
    file: "frontend/planned_event_recommendations.csv",
    badgeCols: ["confidence_band", "operator_warning", "abstain_flag"],
    defaultSort: { col: "confidence", dir: "desc" },
    pageSize: 15,
  },
  layer45_scenario_ready_duration: {
    key: "layer45_scenario_ready_duration",
    file: "layer45_scenario_ready_duration.csv",
    badgeCols: ["duration_sanity_flag"],
    defaultSort: { col: "tail_risk_prob", dir: "desc" },
    pageSize: 15,
  },
  layer45_state_vector: {
    key: "layer45_state_vector",
    file: "layer45_operational_state_vector_normalized.csv",
    badgeCols: ["novelty_flag", "drift_flag", "high_impact_decision", "duration_sanity_flag"],
    hide: [
      // hide the long tail of z-scored mirror columns to keep the table legible
      "duration_pred_z", "duration_p50_z", "duration_p80_z", "duration_p95_z",
      "duration_ci_lower_z", "duration_ci_upper_z", "high_impact_prob_z",
      "high_impact_prob_calibrated_z", "retrieval_confidence_z", "novelty_score_z",
      "drift_score_z", "trust_score_z", "fragility_signal_z", "obi_signal_z",
      "ims_proxy_z", "raw_duration_p50_z", "raw_duration_p80_z", "raw_duration_p95_z",
      "safe_duration_p50_z", "safe_duration_p80_z", "safe_duration_p95_z",
      "duration_reliability_z", "tail_risk_prob_z",
    ],
    defaultSort: { col: "high_impact_prob_calibrated", dir: "desc" },
    pageSize: 12,
  },
  layer45_novelty_drift: {
    key: "layer45_novelty_drift",
    file: "layer45_novelty_drift.csv",
    badgeCols: ["novelty_flag", "drift_flag"],
    defaultSort: { col: "novelty_score", dir: "desc" },
    pageSize: 15,
  },
  layer5_frontend_export: {
    key: "layer5_frontend_export",
    file: "layer5_frontend_export.csv",
    badgeCols: ["service_tier", "is_critical", "violation_flag", "diversion_activated", "duration_sanity_flag"],
    defaultSort: { col: "cvar_contribution", dir: "desc" },
    pageSize: 15,
  },
  layer45_metrics: {
    key: "layer45_metrics",
    file: "layer45_metrics.csv",
    badgeCols: ["subset", "task"],
    defaultSort: { col: "subset", dir: "asc" },
    pageSize: 15,
  },
  layer5_cvar_summary: {
    key: "layer5_cvar_summary",
    file: "layer5_cvar_summary.csv",
    badgeCols: ["service_tier"],
    defaultSort: { col: "alpha", dir: "asc" },
    pageSize: 12,
  },
  layer6_model_health: {
    key: "layer6_model_health",
    file: "layer6_model_health_summary.csv",
    badgeCols: ["status"],
    hide: ["overall_health"],
    defaultSort: { col: "metric_group", dir: "asc" },
    pageSize: 20,
  },
  layer6_active_alerts: {
    key: "layer6_active_alerts",
    file: "layer6_active_alerts.csv",
    badgeCols: ["severity"],
    defaultSort: { col: "severity", dir: "asc" },
    pageSize: 12,
  },
  layer6_drift_report: {
    key: "layer6_drift_report",
    file: "layer6_drift_report.csv",
    badgeCols: ["severity", "alert"],
    defaultSort: { col: "severity", dir: "asc" },
    pageSize: 10,
  },
  layer7_expected_risk_index: {
    key: "layer7_expected_risk_index",
    file: "layer7_expected_risk_index.csv",
    defaultSort: { col: "ERI", dir: "desc" },
    pageSize: 15,
  },
  layer7_top_k_early_warning: {
    key: "layer7_top_k_early_warning",
    file: "layer7_top_k_early_warning.csv",
    badgeCols: ["persistence_class", "confidence_band"],
    defaultSort: { col: "ERI_3h", dir: "desc" },
    pageSize: 12,
  },
  layer7_spillover_centrality: {
    key: "layer7_spillover_centrality",
    file: "layer7_spillover_centrality.csv",
    defaultSort: { col: "SSC_centrality", dir: "desc" },
    pageSize: 12,
  },
  layer7_operational_alerts: {
    key: "layer7_operational_alerts",
    file: "layer7_operational_alerts.csv",
    badgeCols: ["persistence_class", "confidence"],
    defaultSort: { col: "risk_score", dir: "desc" },
    pageSize: 12,
  },
  layer7_metrics: {
    key: "layer7_metrics",
    file: "layer7_metrics.csv",
    badgeCols: ["model"],
    defaultSort: { col: "horizon_h", dir: "asc" },
    pageSize: 12,
  },
  layer7_cross_excitation_matrix: {
    key: "layer7_cross_excitation_matrix",
    file: "layer7_cross_excitation_matrix.csv",
    defaultSort: { col: "alpha", dir: "desc" },
    pageSize: 15,
  },
};

export function getDataset(key: string): DatasetSpec | undefined {
  return DATASETS[key];
}
