import { tryLoadCsv } from "@/lib/csv";
import { nums, quantile, median } from "@/lib/stats";
import { toNum } from "@/lib/format";
import { PageHeader, EmptyState } from "@/components/ui";
import { ScenarioBuilder, type ScenarioData } from "@/components/ScenarioBuilder";


export const revalidate = 30;

export default function ScenarioPage() {
  const dl = tryLoadCsv("frontend/duration_lookup.csv");
  const rs = tryLoadCsv("frontend/risk_scores.csv");
  const cf = tryLoadCsv("frontend/corridor_fragility.csv");
  const pe = tryLoadCsv("frontend/planned_event_recommendations.csv");
  const sp = tryLoadCsv("layer7_spillover_centrality.csv");

  if (!dl || !rs) {
    return (
      <>
        <PageHeader eyebrow="Tools" title="Scenario Builder" />
        <EmptyState message="The duration lookup or risk-score exports are missing from outputs/." />
      </>
    );
  }

  // duration lookup keyed by cause|corridor
  const duration: ScenarioData["duration"] = {};
  for (const r of dl.rows) {
    duration[`${r["event_cause"]}|${r["corridor"]}`] = {
      p50: toNum(r["p50_min"]),
      p80: toNum(r["p80_min"]),
      p95: toNum(r["p95_min"]),
      n: toNum(r["n"]),
    };
  }

  // global risk-score quantiles for tier binning (same logic as the API augmenter)
  const allScores = nums(rs.rows, "survival_risk_score");
  const t50 = quantile(allScores, 0.5);
  const t80 = quantile(allScores, 0.8);
  const t95 = quantile(allScores, 0.95);
  const tierOf = (v: number) =>
    v >= t95 ? "Critical" : v >= t80 ? "High" : v >= t50 ? "Moderate" : "Low";

  const riskGroups: Record<string, number[]> = {};
  for (const r of rs.rows) {
    const k = `${r["event_cause"]}|${r["corridor"]}`;
    const v = toNum(r["survival_risk_score"]);
    if (Number.isNaN(v)) continue;
    (riskGroups[k] ??= []).push(v);
  }
  const risk: ScenarioData["risk"] = {};
  for (const [k, arr] of Object.entries(riskGroups)) {
    const max = Math.max(...arr);
    risk[k] = { count: arr.length, max, median: median(arr), tier: tierOf(max) };
  }

  const frag: ScenarioData["frag"] = {};
  cf?.rows.forEach((r) => {
    frag[r["corridor"]] = {
      branching_ratio: toNum(r["branching_ratio"]),
      fragility_log: toNum(r["fragility_log"]),
      current_intensity: toNum(r["current_intensity"]),
    };
  });

  const planned: ScenarioData["planned"] = {};
  pe?.rows.forEach((r) => {
    const k = `${r["cause"]}|${r["corridor"]}`;
    if (planned[k]) return; // keep first match per pairing
    planned[k] = {
      officers: toNum(r["recommended_officers"]),
      barricades: toNum(r["recommended_barricades"]),
      tow: toNum(r["recommended_tow_units"]),
      supervisors: toNum(r["recommended_supervisors"]),
      qru: toNum(r["recommended_qru_units"]),
      confidence_band: r["confidence_band"] ?? "",
      duration_p50: toNum(r["pred_duration_p50"]),
    };
  });

  const spillover: ScenarioData["spillover"] = {};
  sp?.rows.forEach((r) => {
    spillover[r["zone"]] = {
      ssc: toNum(r["SSC_centrality"]),
      half_life: toNum(r["half_life_hours"]),
      s_source: toNum(r["S_source"]),
      v_receiver: toNum(r["V_receiver"]),
    };
  });

  const causes = Array.from(new Set(dl.rows.map((r) => r["event_cause"]))).filter(Boolean).sort();
  const corridors = Array.from(new Set(dl.rows.map((r) => r["corridor"]))).filter(Boolean).sort();
  const zones = Object.keys(spillover).sort();

  const data: ScenarioData = { causes, corridors, zones, duration, risk, frag, planned, spillover };

  return (
    <>
      <PageHeader
        eyebrow="Tools"
        title="Scenario Builder"
        lede="Pick an incident cause, a corridor and a spillover zone. The dashboard assembles the matching prediction across Layers 1, 3, 4 and 7 from the precomputed pipeline exports, so you can read off the expected duration, risk tier, corridor fragility, recommended deployment and zone spillover for that exact combination. Models are not re-run; this composes the existing outputs."
      />
      <ScenarioBuilder data={data} />
    </>
  );
}
