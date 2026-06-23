import { tryLoadCsv } from "@/lib/csv";
import { countWhere, nums } from "@/lib/stats";
import { toNum, fmtNum } from "@/lib/format";
import { Kpi, PageHeader, Note, EmptyState } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { MapplsMap, type HotPoint } from "@/components/MapplsMap";


export const revalidate = 30;

export default function MapPage() {
  const hs = tryLoadCsv("layer2_hotspots.csv");

  const points: HotPoint[] = hs
    ? hs.rows
        .map((r) => ({
          name: r["junction"] ?? "",
          lat: toNum(r["latitude"]),
          lng: toNum(r["longitude"]),
          prob: toNum(r["hotspot_probability"]),
          z: toNum(r["z_score"]),
          count: toNum(r["raw_count"]),
          sig: ["true", "1"].includes((r["is_significant"] ?? "").toLowerCase()),
        }))
        .filter((p) => !Number.isNaN(p.lat) && !Number.isNaN(p.lng))
    : [];

  const sigCount = hs ? countWhere(hs.rows, (r) => ["true", "1"].includes((r["is_significant"] ?? "").toLowerCase())) : 0;
  const maxProb = points.length ? Math.max(...points.map((p) => p.prob)) : NaN;

  return (
    <>
      <PageHeader
        eyebrow="Tools"
        title="Hotspot Map"
        lede="The Layer 2 hotspot junctions plotted on a Bengaluru basemap. Each junction carries its real coordinates, bootstrap hotspot probability and Gi* significance from layer2_hotspots.csv. Markers are coloured by risk; click one for its statistics."
      />

      {points.length === 0 ? (
        <EmptyState message="layer2_hotspots.csv has no valid coordinates in outputs/." />
      ) : (
        <>
          <div className="grid grid-4" style={{ marginBottom: 24 }}>
            <Kpi label="Junctions mapped" value={fmtNum(points.length)} sub="with valid coordinates" />
            <Kpi label="Significant hotspots" value={fmtNum(sigCount)} sub="bootstrap is_significant" />
            <Kpi label="Peak hotspot prob." value={Number.isNaN(maxProb) ? "-" : fmtNum(maxProb)} sub="max bootstrap probability" />
            <Kpi label="Basemap" isText value="Leaflet" sub="CartoDB dark tiles · no API key" accent />
          </div>

          <div style={{ marginBottom: 24 }}>
            <MapplsMap points={points} />
          </div>

          <DataTable
            dataset="layer2_hotspots_geo"
            title="Hotspot junctions"
            subtitle="Coordinates, bootstrap probability and Gi* significance · same data as the map"
            searchPlaceholder="Filter by junction…"
          />

          <div style={{ marginTop: 20 }}>
            <Note>
              Coordinates come from <span className="mono">layer2_hotspots.csv</span>. The map uses
              Leaflet with CartoDB dark tiles — no API key required.
            </Note>
          </div>
        </>
      )}
    </>
  );
}
