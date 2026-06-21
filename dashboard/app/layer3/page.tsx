import { tryLoadCsv } from "@/lib/csv";
import { countWhere, topBy } from "@/lib/stats";
import { toNum, fmtNum } from "@/lib/format";
import { Kpi, PageHeader, Panel, Note } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { HBar } from "@/components/charts";

export const dynamic = "force-dynamic";

export default function Layer3Page() {
  const rs = tryLoadCsv("frontend/risk_scores.csv");
  const cf = tryLoadCsv("frontend/corridor_fragility.csv");

  const scored = rs?.rows.length ?? 0;
  // tiers derived in the same way the API augments them (quartile/tail bins)
  const scores = rs ? rs.rows.map((r) => toNum(r["survival_risk_score"])).filter((n) => !Number.isNaN(n)).sort((a, b) => a - b) : [];
  const p80 = scores.length ? scores[Math.floor(scores.length * 0.8)] : NaN;
  const highPlus = rs ? countWhere(rs.rows, (r) => toNum(r["survival_risk_score"]) >= p80) : 0;
  const corridors = cf?.rows.length ?? 0;
  const fragileBars = cf
    ? topBy(cf.rows, "corridor", "fragility_log", 12).filter((d) => d.name !== "Non-corridor")
    : [];
  const topFragile = fragileBars[0];

  return (
    <>
      <PageHeader
        eyebrow="Layer 3 · Fuse"
        title="Resource Optimization"
        lede="Which incidents and corridors deserve scarce officers, barricades and tow units? Layer 3 fuses Layer 1 duration risk with Layer 2 spatial burden into a per-event disruption-impact / survival-risk score, and runs a Hawkes branching analysis per corridor to flag where a single incident is most likely to cascade into many."
      />

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Events scored" value={fmtNum(scored)} sub="survival-risk rows" />
        <Kpi label="High / critical tier" value={fmtNum(highPlus)} sub="top 20% of risk scores" />
        <Kpi label="Corridors modelled" value={fmtNum(corridors)} sub="Hawkes fragility rows" />
        <Kpi
          label="Most fragile corridor"
          isText
          accent
          value={topFragile?.name ?? "-"}
          sub={topFragile ? `log-fragility ${fmtNum(topFragile.value)}` : undefined}
        />
      </div>

      {fragileBars.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <Panel title="Corridor fragility" meta="Top corridors by log-fragility (Hawkes cascade potential)">
            <HBar data={fragileBars} height={320} colorIndex={2} />
            <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
              Fragility combines the Hawkes branching ratio and current intensity: higher means a
              disruption here is more likely to trigger follow-on incidents.
            </div>
          </Panel>
        </div>
      )}

      <div style={{ marginBottom: 20 }}>
        <Note>
          The <span className="mono">Risk tier</span> column is derived in the dashboard by binning
          each event&apos;s <span className="mono">survival_risk_score</span> on its own empirical
          distribution (Low &lt; P50, Moderate &lt; P80, High &lt; P95, Critical otherwise). The
          underlying score comes straight from <span className="mono">risk_scores.csv</span>; the tier
          label is a presentation aid, not a backend column.
        </Note>
      </div>

      <div className="stack gap-6">
        <DataTable
          dataset="risk_scores"
          title="Disruption-impact / survival-risk scores"
          subtitle="Per-event score with derived risk tier · 8,007 rows, paginated"
          searchPlaceholder="Filter by cause, corridor or junction…"
        />
        <DataTable
          dataset="corridor_fragility"
          title="Corridor fragility"
          subtitle="Hawkes parameters and branching ratio per corridor"
          searchPlaceholder="Filter by corridor…"
        />
      </div>
    </>
  );
}
