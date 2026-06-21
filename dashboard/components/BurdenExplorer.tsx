"use client";
import { useMemo, useState } from "react";
import { HBar } from "./charts";
import { fmtNum, toNum } from "@/lib/format";

export interface ObiRow { junction: string; operational_burden_index: string }
export interface HotRow { junction: string; gi_star_h1: string }

export function BurdenExplorer({ obi, hot }: { obi: ObiRow[]; hot: HotRow[] }) {
  const [topN, setTopN] = useState(12);
  const [giThresh, setGiThresh] = useState(0);

  const ranked = useMemo(
    () =>
      obi
        .map((r) => ({ name: r.junction, value: toNum(r.operational_burden_index) }))
        .filter((d) => !Number.isNaN(d.value))
        .sort((a, b) => b.value - a.value),
    [obi]
  );
  const top = ranked.slice(0, topN);

  const giCount = useMemo(
    () => hot.filter((r) => toNum(r.gi_star_h1) >= giThresh).length,
    [hot, giThresh]
  );
  const giMax = useMemo(
    () => Math.max(...hot.map((r) => toNum(r.gi_star_h1)).filter((n) => !Number.isNaN(n)), 0),
    [hot]
  );

  return (
    <div className="grid" style={{ gridTemplateColumns: "260px 1fr", gap: 20, alignItems: "start" }}>
      <div className="stack gap-6">
        <label className="stack gap-2">
          <span className="kpi-label">Top-N junctions · {topN}</span>
          <input className="range" type="range" min={5} max={30} value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
        </label>
        <div>
          <label className="stack gap-2">
            <span className="kpi-label">Gi* threshold · {giThresh.toFixed(2)}</span>
            <input
              className="range"
              type="range"
              min={-0.2}
              max={Number(giMax.toFixed(2))}
              step={0.01}
              value={giThresh}
              onChange={(e) => setGiThresh(Number(e.target.value))}
            />
          </label>
          <div className="kpi" style={{ marginTop: 10, minHeight: 92 }}>
            <span className="kpi-label">Junctions above threshold</span>
            <span className="kpi-value" style={{ fontSize: 26 }}>{fmtNum(giCount)}</span>
            <span className="kpi-sub">of {hot.length} ranked</span>
          </div>
        </div>
      </div>
      <div>
        <HBar data={top} height={Math.max(220, topN * 22)} colorIndex={1} />
        <div className="dim" style={{ fontSize: 12, marginTop: 8 }}>
          Drag Top-N to change how many junctions the bar chart shows; drag the Gi* slider to count
          how many junctions exceed a clustering-significance threshold.
        </div>
      </div>
    </div>
  );
}
