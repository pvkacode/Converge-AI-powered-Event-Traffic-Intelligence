import { tryLoadCsv } from "@/lib/csv";
import { valueCounts, countWhere, nums, mean } from "@/lib/stats";
import { toNum, fmtNum, titleCaseValue } from "@/lib/format";
import { Kpi, PageHeader, Panel, Note } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { VBar } from "@/components/charts";
import { Layer4GeoSection } from "@/components/Layer4GeoSection";
import { Layer4GeoKpi } from "@/components/Layer4GeoKpi";
import { Layer4TemporalSection, type TemporalMetadata } from "@/components/Layer4TemporalSection";
import { tryReadText } from "@/lib/csv";

import { PAGE_REVALIDATE_SECONDS } from "@/lib/page-config";

export const revalidate = PAGE_REVALIDATE_SECONDS;

export default function Layer4Page() {
  const pe = tryLoadCsv("frontend/planned_event_recommendations.csv");
  const diag = tryLoadCsv("frontend/layer4_retrieval_diagnostics.csv");
  const geoMatches = tryLoadCsv("frontend/layer4_geo_radius_matches.csv");
  const temporalMetaRaw = tryReadText("frontend/layer4_temporal_decay_metadata.json");
  const temporalMeta: TemporalMetadata | null = temporalMetaRaw
    ? (JSON.parse(temporalMetaRaw) as TemporalMetadata)
    : null;

  const total = pe?.rows.length ?? 0;
  const bands = pe ? valueCounts(pe.rows, "confidence_band") : {};
  const abstain = pe
    ? countWhere(pe.rows, (r) => ["1", "true", "yes"].includes((r["abstain_flag"] ?? "").toLowerCase()))
    : 0;
  const meanSim = pe ? mean(nums(pe.rows, "mean_similarity")) : NaN;

  const geoCounts = diag ? nums(diag.rows, "geo_radius_2km_count") : [];
  const totalGeoCount = geoCounts.reduce((a, b) => a + b, 0);
  const meanGeoCount = geoCounts.length ? mean(geoCounts) : NaN;
  const nearestVals = diag
    ? nums(diag.rows, "geo_radius_nearest_km").filter((n) => n > 0)
    : [];
  const nearestOverall = nearestVals.length ? Math.min(...nearestVals) : NaN;
  const geoUnavailable = !diag || geoCounts.length === 0 || geoCounts.every((n) => n === 0);

  const bandColorFor = (k: string) => {
    const b = k.toLowerCase();
    if (b.includes("high") || b === "strong") return "var(--accent)";
    if (b.includes("med") || b === "moderate") return "var(--warning)";
    return "var(--ink-3)";
  };
  const bandData = Object.entries(bands).map(([k, v]) => ({
    band: titleCaseValue(k),
    count: v,
    __color: bandColorFor(k),
  }));

  return (
    <>
      <PageHeader
        eyebrow="Layer 4 · Retrieve"
        title="Event Intelligence"
        lede="For a planned event (a procession, public gathering, construction window), Layer 4 retrieves the most similar past incidents from an event knowledge base using Gower similarity over mixed features, then turns that institutional memory into a concrete recommendation: expected duration and impact quantiles plus suggested officers, barricades, tow units, supervisors and QRU units. Every recommendation carries a confidence band and an operator warning when evidence is thin."
      />

      <div className="grid grid-5" style={{ marginBottom: 24 }}>
        <Kpi label="Recommendations" value={fmtNum(total)} sub="planned-event rows" />
        <Kpi
          label="Confidence bands"
          isText
          value={Object.entries(bands).map(([k, v]) => `${v} ${titleCaseValue(k)}`).join(" · ") || "-"}
          sub="distribution across recommendations"
        />
        <Kpi label="Abstained" value={fmtNum(abstain)} sub="model declined to recommend" />
        <Kpi label="Mean similarity" value={Number.isNaN(meanSim) ? "-" : fmtNum(meanSim)} sub="avg Gower match to precedents" />
        <Layer4GeoKpi
          geoUnavailable={geoUnavailable}
          totalGeoCount={totalGeoCount}
          meanGeoCount={meanGeoCount}
          nearestOverall={nearestOverall}
        />
      </div>

      {bandData.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <Panel title="Confidence distribution" meta="Recommendations by evidence strength">
            <VBar data={bandData} xKey="band" yKey="count" height={240} />
            <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
              Bands reflect effective sample size and retrieval similarity. This export contains only
              Low and Medium bands, so no recommendation reaches a high-confidence tier on the current
              data.
            </div>
          </Panel>
        </div>
      )}

      {(diag || geoMatches) && (
        <Layer4GeoSection diagnostics={diag?.rows ?? []} matches={geoMatches?.rows ?? []} />
      )}

      <Layer4TemporalSection metadata={temporalMeta} />

      <DataTable
        dataset="planned_event_recommendations"
        title="Planned-event recommendations"
        subtitle="Browse and filter recommendations · confidence band shown as a badge"
        searchPlaceholder="Filter by cause, corridor or event id…"
      />

      <div style={{ marginTop: 20 }}>
        <Note>
          Confidence bands are exported verbatim as <span className="mono">LOW</span> /{" "}
          <span className="mono">MEDIUM</span>. They are rendered as palette-aware badges rather than
          relabelled, so the UI matches the source file exactly.
        </Note>
      </div>
    </>
  );
}
