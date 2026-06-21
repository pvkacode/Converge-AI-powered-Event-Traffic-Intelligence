import { tryLoadCsv } from "@/lib/csv";
import { toNum, fmtNum, fmtCompact } from "@/lib/format";
import { Kpi, PageHeader, Panel, Note, EmptyState } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { GroupedBar, LineSeries } from "@/components/charts";

export const dynamic = "force-dynamic";

export default function Layer5Page() {
  const metrics = tryLoadCsv("layer5_optimization_metrics.csv");
  const prepost = tryLoadCsv("layer5_pre_post_cvar_comparison.csv");
  const cvar = tryLoadCsv("layer5_cvar_summary.csv");
  const pareto = tryLoadCsv("layer5_pareto_front.csv");

  const m: Record<string, number> = {};
  metrics?.rows.forEach((r) => {
    m[r["metric"]] = toNum(r["value"]);
  });

  const ppData = prepost
    ? prepost.rows.map((r) => ({
        name: r["scope"].replace(/_/g, " "),
        baseline: toNum(r["baseline_cvar"]),
        optimized: toNum(r["optimized_cvar"]),
      }))
    : [];

  // The "all sites" CVaR frontier across confidence level alpha (rows with no tier).
  const frontier = cvar
    ? cvar.rows
        .filter((r) => (r["service_tier"] ?? "").trim() === "")
        .map((r) => ({ alpha: toNum(r["alpha"]), cvar: toNum(r["cvar"]), var: toNum(r["var"]) }))
        .filter((d) => !Number.isNaN(d.alpha))
        .sort((a, b) => a.alpha - b.alpha)
    : [];

  const paretoHasData = !!pareto && pareto.rows.length > 0;

  return (
    <>
      <PageHeader
        eyebrow="Layer 5 · Optimize"
        title="Robust Optimization"
        lede="Layer 5 allocates officers, barricades, tow trucks and QRU units under uncertainty. Instead of optimising the average outcome, it minimises Conditional Value-at-Risk (CVaR) over a scenario ensemble, so the plan is robust to the worst tail of disruption days, subject to chance constraints on critical sites."
      />

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Active sites" value={fmtNum(m["n_active_sites"] ?? NaN)} sub={`${fmtNum(m["n_scenarios"] ?? NaN)} scenarios`} />
        <Kpi
          label="Expected delay cut"
          value={Number.isNaN(m["expected_delay_reduction_pct"]) ? "-" : `${fmtNum(m["expected_delay_reduction_pct"])}%`}
          sub="optimised vs raw"
          accent
        />
        <Kpi
          label="Officers deployed"
          value={fmtNum(m["total_officers_deployed"] ?? NaN)}
          sub={`${fmtNum(m["total_barricades_deployed"] ?? NaN)} barricades · ${fmtNum(m["total_tow_trucks_deployed"] ?? NaN)} tow`}
        />
        <Kpi label="Critical sites" value={fmtNum(m["critical_sites"] ?? NaN)} sub={`chance-constraint ${Number.isNaN(m["chance_constraint_satisfaction_critical"]) ? "-" : fmtNum(m["chance_constraint_satisfaction_critical"])}`} />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 24, alignItems: "start" }}>
        <Panel title="CVaR before vs after" meta="Tail risk reduction by service tier (alpha = 0.9)">
          {ppData.length ? (
            <>
              <GroupedBar
                data={ppData}
                keys={[
                  { key: "baseline", label: "Baseline CVaR", colorIndex: 3 },
                  { key: "optimized", label: "Optimised CVaR", colorIndex: 0 },
                ]}
                height={300}
              />
              <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
                Robust allocation reduces all-sites tail risk by{" "}
                {fmtNum(toNum(prepost!.rows[0]?.["percentage_reduction"]))}% (CVaR{" "}
                {fmtCompact(toNum(prepost!.rows[0]?.["baseline_cvar"]))} →{" "}
                {fmtCompact(toNum(prepost!.rows[0]?.["optimized_cvar"]))} delay-minutes).
              </div>
            </>
          ) : (
            <EmptyState message="layer5_pre_post_cvar_comparison.csv not found." />
          )}
        </Panel>

        <Panel title="Risk frontier" meta="CVaR vs confidence level alpha">
          {paretoHasData ? (
            <EmptyState title="Pareto front" message="Rendering the exported Pareto front." />
          ) : frontier.length ? (
            <>
              <LineSeries
                data={frontier}
                xKey="alpha"
                series={[
                  { key: "cvar", label: "CVaR", colorIndex: 0 },
                  { key: "var", label: "VaR", colorIndex: 1 },
                ]}
                height={300}
                xLabel="Confidence level (alpha)"
              />
              <Note warn>
                <span className="mono">layer5_pareto_front.csv</span> is present but contains only a
                header row (no points), so the cost/CVaR Pareto curve cannot be drawn. Shown instead is
                the real CVaR-vs-alpha risk frontier from{" "}
                <span className="mono">layer5_cvar_summary.csv</span>: tail risk rises sharply as you
                demand robustness against rarer worst-case days.
              </Note>
            </>
          ) : (
            <EmptyState message="No Pareto or CVaR-frontier data available in outputs/." />
          )}
        </Panel>
      </div>

      <div className="stack gap-6">
        <DataTable
          dataset="layer5_frontend_export"
          title="Per-site allocation"
          subtitle="Officers, barricades, tow, QRU, diversion and CVaR contribution per site"
          searchPlaceholder="Filter by event id, cause or tier…"
        />
        <DataTable
          dataset="layer5_cvar_summary"
          title="CVaR summary"
          subtitle="CVaR / VaR and scenario delays by alpha and service tier"
          searchPlaceholder="Filter by service tier…"
        />
      </div>
    </>
  );
}
