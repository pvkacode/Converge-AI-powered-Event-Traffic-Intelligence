"use client";
import { useMemo, useState } from "react";
import { VBar } from "./charts";
import { Note } from "./ui";
import { fmtNum, fmtMinutes, titleCaseValue, toNum } from "@/lib/format";

export interface DurRow {
  event_cause: string;
  corridor: string;
  n: string;
  p50_min: string;
  p80_min: string;
  p95_min: string;
}

export function DurationExplorer({ rows }: { rows: DurRow[] }) {
  const causes = useMemo(
    () => Array.from(new Set(rows.map((r) => r.event_cause))).sort(),
    [rows]
  );
  const [cause, setCause] = useState(
    causes.includes("vehicle_breakdown") ? "vehicle_breakdown" : causes[0] ?? ""
  );

  const corridorsForCause = useMemo(
    () => rows.filter((r) => r.event_cause === cause).map((r) => r.corridor).sort(),
    [rows, cause]
  );
  const [corridor, setCorridor] = useState(corridorsForCause[0] ?? "");

  // keep corridor valid when cause changes
  const activeCorridor = corridorsForCause.includes(corridor) ? corridor : corridorsForCause[0] ?? "";

  const row = useMemo(
    () => rows.find((r) => r.event_cause === cause && r.corridor === activeCorridor),
    [rows, cause, activeCorridor]
  );

  const chartData = row
    ? [
        { q: "P50", minutes: toNum(row.p50_min), __color: "var(--viz-1)" },
        { q: "P80", minutes: toNum(row.p80_min), __color: "var(--viz-2)" },
        { q: "P95", minutes: toNum(row.p95_min), __color: "var(--viz-3)" },
      ]
    : [];
  const censored = row && toNum(row.p50_min) > 100000;

  return (
    <div className="grid" style={{ gridTemplateColumns: "280px 1fr", gap: 20, alignItems: "start" }}>
      <div className="stack gap-4">
        <label className="stack gap-2">
          <span className="kpi-label">Incident cause</span>
          <select
            className="select"
            value={cause}
            onChange={(e) => {
              setCause(e.target.value);
              const first = rows.filter((r) => r.event_cause === e.target.value).map((r) => r.corridor).sort()[0];
              if (first) setCorridor(first);
            }}
          >
            {causes.map((c) => <option key={c} value={c}>{titleCaseValue(c)}</option>)}
          </select>
        </label>
        <label className="stack gap-2">
          <span className="kpi-label">Corridor</span>
          <select className="select" value={activeCorridor} onChange={(e) => setCorridor(e.target.value)}>
            {corridorsForCause.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        {row && (
          <div className="stack gap-2" style={{ marginTop: 4 }}>
            <div className="metric-line"><span className="ml-k">Sample incidents</span><span className="ml-v">{fmtNum(row.n)}</span></div>
            <div className="metric-line"><span className="ml-k">P50</span><span className="ml-v">{fmtMinutes(row.p50_min)}</span></div>
            <div className="metric-line"><span className="ml-k">P80</span><span className="ml-v">{fmtMinutes(row.p80_min)}</span></div>
            <div className="metric-line"><span className="ml-k">P95</span><span className="ml-v">{fmtMinutes(row.p95_min)}</span></div>
          </div>
        )}
      </div>

      <div>
        {row ? (
          <>
            <VBar data={chartData} xKey="q" yKey="minutes" height={260} unit=" min" />
            {censored && (
              <div style={{ marginTop: 10 }}>
                <Note warn>
                  These quantiles collapse to one large value: a right-censored tail in the raw log
                  for this pairing. Layer 4.5 guards it before any allocation.
                </Note>
              </div>
            )}
          </>
        ) : (
          <Note warn>
            No duration lookup row for {titleCaseValue(cause)} on {activeCorridor || "this corridor"}.
            In the live pipeline this triggers the Layer 1 cause-only fallback, then Layer 4 retrieval.
          </Note>
        )}
      </div>
    </div>
  );
}
