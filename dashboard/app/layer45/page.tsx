import { tryLoadCsv } from "@/lib/csv";
import { nums, median, countWhere, mean, topBy } from "@/lib/stats";
import { toNum, fmtNum, fmtMinutes } from "@/lib/format";
import { Kpi, PageHeader, Panel, Note, EmptyState } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { ReliabilityChart, VBar, HBar } from "@/components/charts";

export const dynamic = "force-dynamic";

function truthy(v: string | undefined): boolean {
  return ["1", "true", "yes"].includes((v ?? "").toLowerCase());
}

export default function Layer45Page() {
  const cal = tryLoadCsv("layer45_calibration.csv");
  const fi = tryLoadCsv("layer45_feature_importance.csv");
  const nd = tryLoadCsv("layer45_novelty_drift.csv");
  const sr = tryLoadCsv("layer45_scenario_ready_duration.csv");

  const events = sr?.rows.length ?? 0;
  const novel = nd ? countWhere(nd.rows, (r) => truthy(r["novelty_flag"])) : 0;
  const drift = nd ? countWhere(nd.rows, (r) => truthy(r["drift_flag"])) : 0;
  const reliability = sr ? mean(nums(sr.rows, "duration_reliability")) : NaN;

  const calPoints = cal
    ? cal.rows
        .map((r) => ({ x: toNum(r["mean_pred"]), y: toNum(r["mean_true"]) }))
        .filter((p) => !Number.isNaN(p.x) && !Number.isNaN(p.y))
        .sort((a, b) => a.x - b.x)
    : [];
  const overallEce = cal ? mean(nums(cal.rows, "ece")) : NaN;

  const quantileData =
    sr && sr.rows.length
      ? [
          { q: "P50", minutes: median(nums(sr.rows, "safe_duration_p50")) },
          { q: "P80", minutes: median(nums(sr.rows, "safe_duration_p80")) },
          { q: "P95", minutes: median(nums(sr.rows, "safe_duration_p95")) },
        ]
      : [];

  const fiBars = fi ? topBy(fi.rows, "feature", "importance", 12) : [];

  return (
    <>
      <PageHeader
        eyebrow="Layer 4.5 · Fuse"
        title="Predictive Fusion"
        lede="Layer 4.5 is the calibration and sanity gate. It fuses the duration model, high-impact probability, retrieval confidence and spatial signals into one operational state vector per event, with conformal intervals, novelty and drift detection, and a duration guard that sanitises implausible tails. Downstream optimisation reads only these guarded, calibrated quantiles."
      />

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Events fused" value={fmtNum(events)} sub="rows in state vector" />
        <Kpi
          label="Calibration ECE"
          value={Number.isNaN(overallEce) ? "-" : fmtNum(overallEce)}
          sub="mean expected calibration error"
        />
        <Kpi label="Novelty flags" value={fmtNum(novel)} sub="out-of-distribution events" />
        <Kpi label="Drift flags" value={fmtNum(drift)} sub="distribution-shift events" />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 24, alignItems: "start" }}>
        <Panel title="Reliability diagram" meta="High-impact probability calibration">
          {calPoints.length ? (
            <ReliabilityChart
              series={[{ label: "Calibrated probability", colorIndex: 0, points: calPoints }]}
              height={300}
            />
          ) : (
            <EmptyState message="layer45_calibration.csv not found in outputs/." />
          )}
        </Panel>

        <Panel title="Sanitised duration quantiles" meta="Median guarded duration across events">
          {quantileData.length ? (
            <>
              <VBar data={quantileData} xKey="q" yKey="minutes" height={300} unit=" min" colorIndex={1} />
              <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                Median safe P50 ≈ {fmtMinutes(quantileData[0].minutes)}, with the P80 and P95 planning
                tails after the duration guard removes censored extremes.
              </div>
            </>
          ) : (
            <EmptyState message="layer45_scenario_ready_duration.csv not found in outputs/." />
          )}
        </Panel>
      </div>

      {fiBars.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <Panel title="Feature importance" meta="Top drivers of the fused prediction">
            <HBar data={fiBars} height={340} colorIndex={0} />
          </Panel>
        </div>
      )}

      <div className="stack gap-6">
        <DataTable
          dataset="layer45_metrics"
          title="Backtest metrics"
          subtitle="Holdout vs feedback metrics by task"
          searchPlaceholder="Filter by task or metric…"
        />
        <DataTable
          dataset="layer45_scenario_ready_duration"
          title="Scenario-ready durations"
          subtitle="Guarded P50/P80/P95 with reliability, tail-risk and sanity flag"
          searchPlaceholder="Filter by event id or guard reason…"
        />
        <DataTable
          dataset="layer45_novelty_drift"
          title="Novelty & drift"
          subtitle="Per-event out-of-distribution and shift scores"
          searchPlaceholder="Filter by event id…"
        />
        <DataTable
          dataset="layer45_state_vector"
          title="Operational state vector"
          subtitle="Full fused state per event (z-scored mirror columns hidden)"
          searchPlaceholder="Filter by event id or cause…"
        />
      </div>

      <div style={{ marginTop: 20 }}>
        <Note>
          The reliability diagram plots binned predicted probability against observed frequency; the
          dashed line is perfect calibration. Points below the line mean the model is slightly
          over-confident in that bin.
        </Note>
      </div>
    </>
  );
}
