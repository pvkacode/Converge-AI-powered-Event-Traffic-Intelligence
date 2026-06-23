import { tryLoadCsv } from "@/lib/csv";
import { buildZoneMapData, buildZoneEdgeData } from "@/lib/map-zones";
import { topBy } from "@/lib/stats";
import { toNum, fmtNum } from "@/lib/format";
import { Kpi, PageHeader, Panel, Note, EmptyState } from "@/components/ui";
import { MapPlaceholder } from "@/components/maps/map-ui";
import { SpilloverZoneMap } from "@/components/maps/DynamicMaps";
import { DataTable } from "@/components/DataTable";
import { HBar, ReliabilityChart } from "@/components/charts";
import { Layer7GraphCentrality } from "@/components/Layer7GraphCentrality";

import { PAGE_REVALIDATE_SECONDS } from "@/lib/page-config";

export const revalidate = PAGE_REVALIDATE_SECONDS;

export default function Layer7Page() {
  const spill = tryLoadCsv("layer7_spillover_centrality.csv");
  const zoneMapData = buildZoneMapData();
  const zoneEdges = buildZoneEdgeData();
  const topk = tryLoadCsv("layer7_top_k_early_warning.csv");
  const metrics = tryLoadCsv("layer7_metrics.csv");
  const rel = tryLoadCsv("layer7_reliability_diagram.csv");
  const graphCentrality = tryLoadCsv("layer7_graph_centrality.csv");

  const zones = spill?.rows.length ?? 0;
  const spillBars = spill ? topBy(spill.rows, "zone", "SSC_centrality", 10) : [];
  const topZone = spillBars[0];

  const auc3 = metrics?.rows.find(
    (r) => r["horizon_h"] === "3" && r["model"] === "catboost_calibrated"
  )?.["auc"];

  // reliability series per horizon for the calibrated model
  const horizons = ["3", "6", "9"];
  const relSeries = rel
    ? horizons
        .map((h, i) => ({
          label: `${h}h horizon`,
          colorIndex: i,
          points: rel.rows
            .filter((r) => r["model"] === "catboost_calibrated" && r["horizon_h"] === h)
            .map((r) => ({ x: toNum(r["mean_predicted_prob"]), y: toNum(r["fraction_positive"]) }))
            .filter((p) => !Number.isNaN(p.x) && !Number.isNaN(p.y))
            .sort((a, b) => a.x - b.x),
        }))
        .filter((s) => s.points.length > 0)
    : [];

  return (
    <>
      <PageHeader
        eyebrow="Layer 7 · Spillover"
        title="Cross-Zone Spillover"
        lede="Disruptions do not stay put. Layer 7 fits a multivariate Hawkes process over city zones to learn how congestion in one zone excites risk in its neighbours, then publishes an Expected Risk Index per zone and horizon, ranks zones by spillover centrality, and emits early-warning and operational alerts before the spillover lands."
      />

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Zones modelled" value={fmtNum(zones)} sub="in cross-excitation graph" />
        <Kpi
          label="Top spillover zone"
          isText
          accent
          value={topZone?.name ?? "-"}
          sub={topZone ? `SSC centrality ${fmtNum(topZone.value)}` : undefined}
        />
        <Kpi label="Early-warning zones" value={fmtNum(topk?.rows.length ?? 0)} sub="top-K watch list" />
        <Kpi
          label="3h forecast AUC"
          value={auc3 ? fmtNum(auc3) : "-"}
          sub="calibrated CatBoost"
        />
      </div>

      <div style={{ marginBottom: 24 }}>
        {zoneMapData ? (
          <SpilloverZoneMap zones={zoneMapData} edges={zoneEdges} />
        ) : (
          <MapPlaceholder height={400} message="Map unavailable — check outputs/frontend/ exports" />
        )}
      </div>

      <div className="grid grid-2" style={{ marginBottom: 24, alignItems: "start" }}>
        <Panel title="Spillover centrality" meta="Zones ranked by source + receiver strength">
          {spillBars.length ? (
            <HBar data={spillBars} height={300} colorIndex={0} />
          ) : (
            <EmptyState message="layer7_spillover_centrality.csv not found." />
          )}
        </Panel>
        <Panel title="Forecast reliability" meta="Calibrated probability vs observed, by horizon">
          {relSeries.length ? (
            <ReliabilityChart series={relSeries} height={300} />
          ) : (
            <EmptyState message="layer7_reliability_diagram.csv not found." />
          )}
        </Panel>
      </div>

      <div className="stack gap-6">
        <DataTable
          dataset="layer7_top_k_early_warning"
          title="Top-K early warning"
          subtitle="Highest-risk zone / time cells with persistence class and ERI by horizon"
          searchPlaceholder="Filter by zone or persistence class…"
        />
        <DataTable
          dataset="layer7_operational_alerts"
          title="Operational alerts"
          subtitle="Recommended actions per zone with confidence and persistence"
          searchPlaceholder="Filter by zone or action…"
        />
        <DataTable
          dataset="layer7_spillover_centrality"
          title="Spillover centrality"
          subtitle="Source strength, receiver vulnerability and half-life per zone"
          searchPlaceholder="Filter by zone…"
        />
        <DataTable
          dataset="layer7_expected_risk_index"
          title="Expected Risk Index"
          subtitle="ERI per zone, time and horizon · 13,500 rows, paginated"
          searchPlaceholder="Filter by zone or time…"
        />
        <DataTable
          dataset="layer7_metrics"
          title="Forecast metrics"
          subtitle="AUC, Brier, ECE and count error by horizon and model"
          searchPlaceholder="Filter by model…"
        />
      </div>

      <Layer7GraphCentrality rows={graphCentrality?.rows ?? []} />

      <div style={{ marginTop: 20 }}>
        <Note>
          The reliability chart compares the calibrated CatBoost spillover probabilities against
          observed positive rates per probability bin. Curves tracking the dashed diagonal indicate
          well-calibrated forecasts across the 3h, 6h and 9h horizons.
        </Note>
      </div>
    </>
  );
}
