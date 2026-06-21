import Link from "next/link";
import { tryLoadCsv } from "@/lib/csv";
import { buildJunctionMapData, topJunctionsByObi } from "@/lib/map-junctions";
import { nums, median, countWhere, valueCounts, topBy } from "@/lib/stats";
import { toNum, fmtNum, fmtMinutes } from "@/lib/format";
import { Kpi, PageHeader, Panel, Badge, MetricLine, Note } from "@/components/ui";
import { MapPlaceholder } from "@/components/maps/map-ui";
import { OverviewMiniMap } from "@/components/maps/DynamicMaps";
import { FlowDiagram } from "@/components/FlowDiagram";
import { HBar } from "@/components/charts";
import { healthVariant } from "@/lib/badges";

export const dynamic = "force-dynamic";

export default function OverviewPage() {
  const hotspots = tryLoadCsv("frontend/hotspot_rankings.csv");
  const mapData = buildJunctionMapData();
  const miniPoints = mapData ? topJunctionsByObi(mapData, 10) : [];
  const scenario = tryLoadCsv("layer45_scenario_ready_duration.csv");
  const health = tryLoadCsv("layer6_model_health_summary.csv");
  const alerts = tryLoadCsv("layer6_active_alerts.csv");
  const spill = tryLoadCsv("layer7_spillover_centrality.csv");

  // KPI 1 - hotspots
  const hsTotal = hotspots?.rows.length ?? 0;
  const giHot = hotspots
    ? countWhere(hotspots.rows, (r) =>
        ["gi_star_h1", "gi_star_h2", "gi_star_h3", "gi_star_h5"].some((c) => toNum(r[c]) > 0)
      )
    : 0;

  // KPI 2 - median sanitized duration
  const safe = scenario ? median(nums(scenario.rows, "safe_duration_p50")) : NaN;

  // KPI 3 - model health
  const overall =
    health?.rows.map((r) => r["overall_health"]).find((v) => v && v.trim() !== "") ?? "-";
  const criticalChecks = health ? countWhere(health.rows, (r) => r["status"]?.toLowerCase() === "critical") : 0;
  const totalChecks = health?.rows.length ?? 0;

  // KPI 4 - active alerts
  const alertTotal = alerts?.rows.length ?? 0;
  const sev = alerts ? valueCounts(alerts.rows, "severity") : {};

  // KPI 5 - top spillover zone
  const spillTop = spill
    ? topBy(spill.rows, "zone", "SSC_centrality", 1)[0]
    : undefined;
  const spillBars = spill ? topBy(spill.rows, "zone", "SSC_centrality", 8) : [];

  const sevOrder = ["critical", "warning", "moderate"];
  const sevEntries = Object.entries(sev).sort(
    (a, b) => (sevOrder.indexOf(a[0]) + 10) - (sevOrder.indexOf(b[0]) + 10) || b[1] - a[1]
  );

  return (
    <>
      <PageHeader
        eyebrow="Converge · ASTraM"
        title="Operations overview"
        lede="A read-only view over the Bengaluru traffic disruption pipeline. Seven model layers turn raw incident data into duration estimates, spatial hotspots, retrieved precedents, robust resource plans, and cross-zone spillover early-warning. Every number below is read live from the pipeline's outputs/ exports."
      />

      <div className="grid grid-5" style={{ marginBottom: 24 }}>
        <Kpi
          label="Ranked hotspots"
          value={fmtNum(hsTotal)}
          sub={`${giHot} with positive Gi* clustering`}
        />
        <Kpi
          label="Median safe duration"
          value={Number.isNaN(safe) ? "-" : fmtMinutes(safe)}
          sub="P50, Layer 4.5 sanitized"
        />
        <Kpi
          label="Model health"
          isText
          value={
            <span style={{ color: `var(--${overall.toLowerCase().includes("crit") ? "critical" : overall.toLowerCase().includes("warn") ? "warning" : "ok"})` }}>
              {overall}
            </span>
          }
          sub={`${criticalChecks} of ${totalChecks} checks critical`}
        />
        <Kpi
          label="Active alerts"
          value={fmtNum(alertTotal)}
          sub={sevEntries.map(([k, v]) => `${v} ${k}`).join(", ") || "none"}
        />
        <Kpi
          label="Top spillover zone"
          isText
          accent
          value={spillTop?.name ?? "-"}
          sub={spillTop ? `SSC centrality ${fmtNum(spillTop.value)}` : undefined}
        />
      </div>

      <div style={{ marginBottom: 24 }}>
        {mapData && miniPoints.length ? (
          <OverviewMiniMap points={miniPoints} stats={mapData.stats} />
        ) : (
          <MapPlaceholder height={260} message="Map unavailable — check outputs/frontend/ exports" />
        )}
      </div>

      <div style={{ marginBottom: 24 }}>
        <Panel title="Pipeline flow" meta="Measure → Predict → Retrieve → Fuse → Optimize → Learn → Spillover">
          <p className="muted" style={{ margin: "0 0 16px", maxWidth: "78ch", fontSize: 13.5 }}>
            Each stage is a separate model layer. Click any node to open that layer with its real
            outputs, methodology, and tables.
          </p>
          <FlowDiagram />
        </Panel>
      </div>

      <div className="grid grid-2" style={{ alignItems: "start" }}>
        <Panel title="System status" meta="Layer 6 monitoring · Layer 7 alerts">
          <div className="row between" style={{ marginBottom: 12 }}>
            <span className="muted">Overall model health</span>
            <Badge variant={healthVariant(overall)} dot>
              {overall}
            </Badge>
          </div>
          <hr className="divider" />
          <div style={{ marginTop: 12 }}>
            <div className="kpi-label" style={{ marginBottom: 6 }}>Active alerts by severity</div>
            {sevEntries.length ? (
              sevEntries.map(([k, v]) => (
                <MetricLine
                  key={k}
                  k={<Badge variant={k === "critical" ? "critical" : k === "warning" ? "warning" : "neutral"}>{k}</Badge>}
                  v={v}
                />
              ))
            ) : (
              <span className="dim">No active alerts file available.</span>
            )}
          </div>
          <div style={{ marginTop: 16 }}>
            <Link href="/layer6" className="btn btn-sm">Open monitoring panel</Link>
          </div>
        </Panel>

        <Panel title="Spillover centrality" meta="Layer 7 · SSC by zone">
          {spillBars.length ? (
            <>
              <HBar data={spillBars} height={260} unit="" colorIndex={0} />
              <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                Source-strength + receiver-vulnerability centrality from the cross-excitation graph.{" "}
                <Link href="/layer7">View early-warning zones</Link>.
              </div>
            </>
          ) : (
            <Note warn>Spillover centrality export not found in outputs/.</Note>
          )}
        </Panel>
      </div>
    </>
  );
}
