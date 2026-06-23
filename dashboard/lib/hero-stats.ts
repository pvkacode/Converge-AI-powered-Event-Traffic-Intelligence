// Server-only hero metrics from pipeline CSV exports (live after each run).
import "server-only";

import { tryLoadCsv, tryReadText } from "@/lib/csv";
import { loadEventsCleanStats } from "@/lib/events-clean";
import type { HeroStats } from "@/lib/hero-stats-types";
import { countWhere } from "@/lib/stats";
import { toNum } from "@/lib/format";

export type { HeroStats } from "@/lib/hero-stats-types";

function metricValue(csv: string, key: string): number {
  const f = tryLoadCsv(csv);
  if (!f) return NaN;
  const row = f.rows.find((r) => r.metric === key);
  return row ? toNum(row.value) : NaN;
}

function parseSpilloverP(): number {
  const lrt = tryLoadCsv("layer7_lrt_results.csv");
  const row = lrt?.rows[0];
  if (row) {
    const p = toNum(row.p_value_asymptotic);
    if (p > 0) return p;
    const perm = toNum(row.p_value_permutation);
    if (perm > 0) return perm;
  }
  const txt = tryReadText("layer7_hawkes_fusion_summary.txt");
  const m = txt?.match(/p-?value\s*:\s*([\d.eE+-]+)/i);
  if (m) return toNum(m[1]);
  return NaN;
}

function scientificParts(p: number): { mantissa: number; exponent: number } {
  if (!Number.isFinite(p) || p <= 0) return { mantissa: NaN, exponent: NaN };
  const exp = Math.floor(Math.log10(p));
  const mantissa = p / 10 ** exp;
  return {
    mantissa: Math.round(mantissa * 10) / 10,
    exponent: Math.abs(exp),
  };
}

function fmtPct(n: number, digits = 2): string {
  return Number.isFinite(n) ? `${n.toFixed(digits)}%` : "—";
}

function fmtSci(p: number): string {
  if (!Number.isFinite(p) || p <= 0) return "—";
  const { mantissa, exponent } = scientificParts(p);
  return `${mantissa} × 10⁻${exponent}`;
}

export function loadHeroStats(): HeroStats {
  const events = loadEventsCleanStats();

  const hotspots = tryLoadCsv("layer2_hotspots.csv");
  const hotspotsSignificant = hotspots
    ? countWhere(hotspots.rows, (r) =>
        ["true", "1", "yes"].includes((r.is_significant ?? "").toLowerCase())
      )
    : 0;
  const junctionsTotal = hotspots?.rows.length ?? 0;

  const cvarReductionPct =
    metricValue("layer5_optimization_metrics.csv", "expected_delay_reduction_pct") ||
    (() => {
      const pp = tryLoadCsv("layer5_pre_post_cvar_comparison.csv");
      const all = pp?.rows.find((r) => (r.scope ?? "").includes("all"));
      return all ? toNum(all.percentage_reduction) : NaN;
    })();

  const retrain = tryLoadCsv("layer6_retrain_triggers.csv");
  const criticalRetrainTriggers = retrain
    ? countWhere(retrain.rows, (r) => (r.severity ?? "").toLowerCase() === "critical")
    : 0;

  const health = tryLoadCsv("layer6_model_health_summary.csv");
  const healthCriticalChecks = health
    ? countWhere(health.rows, (r) => (r.status ?? "").toLowerCase() === "critical")
    : 0;
  const healthTotalChecks = health?.rows.length ?? 0;

  const spill = tryLoadCsv("layer7_spillover_centrality.csv");
  let topSpilloverZone = "—";
  if (spill?.rows.length) {
    const sorted = [...spill.rows].sort(
      (a, b) => toNum(b.SSC_centrality) - toNum(a.SSC_centrality)
    );
    topSpilloverZone = sorted[0]?.zone ?? "—";
  }

  const spilloverPValue = parseSpilloverP();
  const { mantissa: spilloverPMantissa, exponent: spilloverPExponent } =
    scientificParts(spilloverPValue);

  const survival = tryLoadCsv("layer1_stacked_survival_metrics.csv");
  const rsfRow = survival?.rows.find((r) => (r.model ?? "").toLowerCase() === "rsf");
  let rsfCIndex = rsfRow ? toNum(rsfRow.c_index) : NaN;
  if (Number.isNaN(rsfCIndex)) {
    const rsfMetrics = tryLoadCsv("layer1_rsf_metrics.csv");
    const oob = rsfMetrics?.rows.find((r) => r.metric === "oob_cindex");
    rsfCIndex = oob ? toNum(oob.value) : NaN;
  }

  const diag = tryLoadCsv("layer4_retrieval_diagnostics.csv");
  const nonAbstain =
    diag?.rows.filter((r) => !["1", "true", "yes"].includes((r.abstain_flag ?? "").toLowerCase())) ??
    [];
  const absErrors = nonAbstain.map((r) => toNum(r.abs_error)).filter((n) => !Number.isNaN(n));
  const plannedEventMae =
    absErrors.length > 0 ? absErrors.reduce((a, b) => a + b, 0) / absErrors.length : NaN;
  const within20Pct =
    absErrors.length > 0
      ? (absErrors.filter((n) => n <= 20).length / absErrors.length) * 100
      : NaN;

  const l3val = tryLoadCsv("layer3_fragility_validation.csv");
  const hawkesSupported = l3val
    ? countWhere(l3val.rows, (r) =>
        ["true", "1", "yes"].includes(String(r.hawkes_supported ?? "").toLowerCase())
      )
    : NaN;
  const l3total = l3val?.rows.length ?? 0;

  const cal = tryLoadCsv("layer45_calibration.csv");
  const ece = cal?.rows.map((r) => toNum(r.ece)).find((n) => !Number.isNaN(n) && n > 0) ?? NaN;

  const layerMetrics: Record<string, string> = {
    L1: Number.isFinite(rsfCIndex) ? `RSF C-index: ${rsfCIndex.toFixed(2)}` : "RSF C-index: —",
    L2:
      junctionsTotal > 0
        ? `${hotspotsSignificant} / ${junctionsTotal} junctions`
        : "—",
    L3:
      l3total > 0 && Number.isFinite(hawkesSupported)
        ? `${hawkesSupported}/${l3total} corridors: Hawkes > Poisson`
        : "—",
    L4: Number.isFinite(plannedEventMae)
      ? `MAE: ${plannedEventMae.toFixed(1)} min`
      : "MAE: —",
    "L4.5": Number.isFinite(ece) ? `ECE: ${ece.toExponential(2)}` : "ECE: —",
    L5: Number.isFinite(cvarReductionPct)
      ? `${cvarReductionPct.toFixed(2)}% CVaR reduction`
      : "CVaR reduction: —",
    L6:
      criticalRetrainTriggers > 0
        ? `${criticalRetrainTriggers} critical retrain triggers`
        : "0 critical retrain triggers",
    L7: fmtSci(spilloverPValue) !== "—" ? `p ≈ ${fmtSci(spilloverPValue)}` : "Spillover LRT: —",
  };

  return {
    incidentsTotal: events.total,
    hotspotsSignificant,
    junctionsTotal,
    cvarReductionPct,
    criticalRetrainTriggers,
    healthCriticalChecks,
    healthTotalChecks,
    topSpilloverZone,
    spilloverPValue,
    spilloverPMantissa,
    spilloverPExponent,
    rsfCIndex,
    plannedEventMae,
    within20Pct,
    censoredRows: events.closedWithoutTimestamp,
    layerMetrics,
  };
}
