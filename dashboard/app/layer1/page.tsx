import { tryLoadCsv } from "@/lib/csv";
import { nums, median } from "@/lib/stats";
import { toNum, fmtNum, fmtMinutes } from "@/lib/format";
import { Kpi, PageHeader, Note, Panel } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { DurationExplorer, type DurRow } from "@/components/DurationExplorer";

export const dynamic = "force-dynamic";

export default function Layer1Page() {
  const dl = tryLoadCsv("frontend/duration_lookup.csv");
  const causes = dl ? new Set(dl.rows.map((r) => r["event_cause"])).size : 0;
  const corridors = dl ? new Set(dl.rows.map((r) => r["corridor"])).size : 0;
  const samples = dl ? nums(dl.rows, "n").reduce((a, b) => a + b, 0) : 0;
  const medP50 = dl ? median(nums(dl.rows, "p50_min")) : NaN;
  const extreme = dl ? dl.rows.filter((r) => toNum(r["p50_min"]) > 100000).length : 0;

  return (
    <>
      <PageHeader
        eyebrow="Layer 1 · Measure"
        title="Duration Intelligence"
        lede="How long will a disruption last? Layer 1 fits survival / time-to-event models (Cox, AFT, random survival forests, with frailty and stacking) on historical incidents and exports a lookup of duration quantiles for every cause and corridor combination. P50 is the typical clearance time; P80 and P95 are the planning tails you staff against."
      />

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Cause types" value={fmtNum(causes)} sub="distinct event causes" />
        <Kpi label="Corridors" value={fmtNum(corridors)} sub="distinct corridors covered" />
        <Kpi label="Incidents pooled" value={fmtNum(samples)} sub="summed sample count" />
        <Kpi
          label="Median P50"
          value={Number.isNaN(medP50) ? "-" : fmtMinutes(medP50)}
          sub="across all lookup cells"
        />
      </div>

      {extreme > 0 && (
        <div style={{ marginBottom: 20 }}>
          <Note warn>
            Source-data caveat: {extreme} lookup cell{extreme === 1 ? "" : "s"} carry extreme P50
            values (over 100,000 minutes), where the quantiles collapse to a single large number.
            These are right-censored / never-resolved durations in the underlying incident log, not a
            rendering bug. They are shown verbatim from{" "}
            <span className="mono">duration_lookup.csv</span> rather than silently capped.
          </Note>
        </div>
      )}

      <div style={{ marginBottom: 24 }}>
        <Panel title="Duration explorer" meta="Pick a cause and corridor to plot its P50/P80/P95">
          <DurationExplorer rows={(dl?.rows ?? []) as unknown as DurRow[]} />
        </Panel>
      </div>

      <DataTable
        dataset="duration_lookup"
        title="Duration lookup"
        subtitle="Search a cause (e.g. vehicle_breakdown) or corridor (e.g. Mysore Road) to filter"
        searchPlaceholder="Filter by cause or corridor…"
        pageSize={15}
      />
    </>
  );
}
