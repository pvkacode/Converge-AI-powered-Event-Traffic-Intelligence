import { tryLoadCsv } from "@/lib/csv";
import { nums, median } from "@/lib/stats";
import { fmtNum, fmtMinutes } from "@/lib/format";
import { Kpi, PageHeader, Note, Panel } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { DurationExplorer, type DurRow } from "@/components/DurationExplorer";
import type { FallbackRow } from "@/lib/durationFallback";


export const revalidate = 30;

export default function Layer1Page() {
  const dl = tryLoadCsv("frontend/duration_lookup.csv");
  // Cause-only quantiles used by the live pipeline's fallback when a
  // (cause, corridor) pair doesn't clear MIN_GROUP_SIZE - see
  // src/layer1_survival.py:lookup_expected_duration().
  const dlFallback = tryLoadCsv("layer1_survival_fallback.csv");
  const causes = dl ? new Set(dl.rows.map((r) => r["event_cause"])).size : 0;
  const corridors = dl ? new Set(dl.rows.map((r) => r["corridor"])).size : 0;
  const samples = dl ? nums(dl.rows, "n").reduce((a, b) => a + b, 0) : 0;
  const medP50 = dl ? median(nums(dl.rows, "p50_min")) : NaN;

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

      <div style={{ marginBottom: 24 }}>
        <Panel title="Duration explorer" meta="Pick a cause and corridor to plot its P50/P80/P95">
          <DurationExplorer
            rows={(dl?.rows ?? []) as unknown as DurRow[]}
            fallbackRows={(dlFallback?.rows ?? []) as unknown as FallbackRow[]}
          />
        </Panel>
      </div>

      <DataTable
        dataset="duration_lookup"
        title="Duration lookup"
        subtitle="Search a cause (e.g. vehicle_breakdown) or corridor (e.g. Mysore Road) to filter"
        searchPlaceholder="Filter by cause or corridor…"
        pageSize={15}
        headerNote={
          <Note>
            Rows with P50 &gt; 24 hrs are flagged inline — these are confirmed recurring work zones
            (e.g. ORR East 2 metro construction), not data errors.
          </Note>
        }
      />
    </>
  );
}
