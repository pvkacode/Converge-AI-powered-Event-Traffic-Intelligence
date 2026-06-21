import { tryLoadCsv } from "@/lib/csv";
import { countWhere, nums } from "@/lib/stats";
import { toNum, fmtNum } from "@/lib/format";
import { Kpi, PageHeader, Panel, Note } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { BurdenExplorer, type ObiRow, type HotRow } from "@/components/BurdenExplorer";

export const dynamic = "force-dynamic";

export default function Layer2Page() {
  const hs = tryLoadCsv("frontend/hotspot_rankings.csv");
  const ob = tryLoadCsv("frontend/operational_burden.csv");

  const total = hs?.rows.length ?? 0;
  const giHot = hs
    ? countWhere(hs.rows, (r) =>
        ["gi_star_h1", "gi_star_h2", "gi_star_h3", "gi_star_h5"].some((c) => toNum(r[c]) > 0)
      )
    : 0;
  const spsPos = hs ? countWhere(hs.rows, (r) => toNum(r["sps"]) > 0) : 0;
  const obiMax = ob ? Math.max(...nums(ob.rows, "operational_burden_index")) : NaN;

  return (
    <>
      <PageHeader
        eyebrow="Layer 2 · Predict"
        title="Spatial Intelligence"
        lede="Where do disruptions concentrate? Layer 2 scores every junction for spatial persistence (SPS), neighbourhood hotspot intensity (NHI), and Getis-Ord Gi* clustering across multiple time horizons, then folds severity, persistence, self-excitation and duration risk into a single Operational Burden Index. The ranking below is the prioritised list of junctions, with no external map required."
      />

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Junctions ranked" value={fmtNum(total)} sub="in hotspot catalog" />
        <Kpi label="Positive Gi* clustering" value={fmtNum(giHot)} sub="any horizon, Gi* > 0" />
        <Kpi label="Positive persistence" value={fmtNum(spsPos)} sub="SPS > 0" />
        <Kpi label="Peak burden index" value={Number.isNaN(obiMax) ? "-" : fmtNum(obiMax)} sub="max OBI (0-1)" />
      </div>

      {ob && hs && (
        <div style={{ marginBottom: 24 }}>
          <Panel title="Operational burden explorer" meta="Adjust Top-N and the Gi* threshold to re-slice the ranking">
            <BurdenExplorer
              obi={ob.rows as unknown as ObiRow[]}
              hot={hs.rows as unknown as HotRow[]}
            />
          </Panel>
        </div>
      )}

      <div className="stack gap-6">
        <DataTable
          dataset="hotspot_rankings"
          title="Hotspot rankings"
          subtitle="SPS, NHI and Gi* by junction · click any header to sort"
          searchPlaceholder="Filter by junction…"
        />
        <DataTable
          dataset="operational_burden"
          title="Operational burden index"
          subtitle="Component breakdown per junction"
          searchPlaceholder="Filter by junction…"
        />
      </div>

      <div style={{ marginTop: 20 }}>
        <Note>
          Gi* values here sit below the conventional 1.96 significance band for most junctions, so
          ranking leans on the composite burden index rather than raw clustering significance. The
          numbers are shown exactly as exported.
        </Note>
      </div>
    </>
  );
}
