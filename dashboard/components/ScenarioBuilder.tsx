"use client";
import { useMemo, useState } from "react";
import { ArrowRight, Lightning } from "@phosphor-icons/react";
import { fmtNum, fmtMinutes, titleCaseValue } from "@/lib/format";
import { Badge, MetricLine, Note } from "./ui";

export interface DurationCell { p50: number; p80: number; p95: number; n: number }
export interface RiskCell { count: number; max: number; median: number; tier: string }
export interface FragCell { branching_ratio: number; fragility_log: number; current_intensity: number }
export interface PlannedCell {
  officers: number; barricades: number; tow: number; supervisors: number; qru: number;
  confidence_band: string; duration_p50: number;
}
export interface SpillCell { ssc: number; half_life: number; s_source: number; v_receiver: number }

export interface ScenarioData {
  causes: string[];
  corridors: string[];
  zones: string[];
  duration: Record<string, DurationCell>;
  risk: Record<string, RiskCell>;
  frag: Record<string, FragCell>;
  planned: Record<string, PlannedCell>;
  spillover: Record<string, SpillCell>;
}

function tierVariant(tier: string) {
  const t = tier.toLowerCase();
  if (t === "critical") return "critical" as const;
  if (t === "high") return "warning" as const;
  if (t === "moderate") return "accent" as const;
  return "ok" as const;
}

export function ScenarioBuilder({ data }: { data: ScenarioData }) {
  const [cause, setCause] = useState(
    data.causes.includes("vehicle_breakdown") ? "vehicle_breakdown" : data.causes[0] ?? ""
  );
  const [corridor, setCorridor] = useState(
    data.corridors.includes("Mysore Road") ? "Mysore Road" : data.corridors[0] ?? ""
  );
  const [zone, setZone] = useState(data.zones[0] ?? "");

  const key = `${cause}|${corridor}`;
  const dur = data.duration[key];
  const risk = data.risk[key];
  const frag = data.frag[corridor];
  const plan = data.planned[key];
  const spill = data.spillover[zone];

  const censored = dur && dur.p50 > 100000;

  const anyMatch = useMemo(() => !!(dur || risk || frag || plan), [dur, risk, frag, plan]);

  return (
    <div className="grid" style={{ gridTemplateColumns: "320px 1fr", gap: 24, alignItems: "start" }}>
      {/* ---- inputs ---- */}
      <div className="panel" style={{ padding: 18, position: "sticky", top: 80 }}>
        <div className="row gap-2" style={{ marginBottom: 14 }}>
          <Lightning size={18} weight="fill" className="kpi-accent" />
          <h2 className="section-title">Build a scenario</h2>
        </div>
        <div className="stack gap-4">
          <label className="stack gap-2">
            <span className="kpi-label">Incident cause</span>
            <select className="select" value={cause} onChange={(e) => setCause(e.target.value)}>
              {data.causes.map((c) => (
                <option key={c} value={c}>{titleCaseValue(c)}</option>
              ))}
            </select>
          </label>
          <label className="stack gap-2">
            <span className="kpi-label">Corridor</span>
            <select className="select" value={corridor} onChange={(e) => setCorridor(e.target.value)}>
              {data.corridors.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <label className="stack gap-2">
            <span className="kpi-label">Spillover zone</span>
            <select className="select" value={zone} onChange={(e) => setZone(e.target.value)}>
              {data.zones.map((z) => (
                <option key={z} value={z}>{z}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="dim" style={{ fontSize: 12, marginTop: 16, lineHeight: 1.5 }}>
          The output is assembled live from the precomputed pipeline exports for this exact
          combination. No models are re-run.
        </div>
      </div>

      {/* ---- assembled output ---- */}
      <div className="stack gap-6">
        <div className="row gap-2 wrap" style={{ fontSize: 13 }}>
          <Badge variant="neutral">{titleCaseValue(cause)}</Badge>
          <ArrowRight size={14} className="dim" />
          <Badge variant="neutral">{corridor}</Badge>
          <ArrowRight size={14} className="dim" />
          <Badge variant="accent">{zone}</Badge>
        </div>

        {!anyMatch && (
          <Note warn>
            No exported rows match {titleCaseValue(cause)} on {corridor}. This pairing was not present
            in the historical incident set, so the pipeline has no duration or risk estimate for it.
            Try another cause or corridor.
          </Note>
        )}

        {/* Layer 1 */}
        <div className="panel">
          <div className="panel-head">
            <div className="stack" style={{ gap: 2 }}>
              <h3 className="section-title">Layer 1 · Expected duration</h3>
              <span className="section-meta">survival model quantiles</span>
            </div>
            <Badge variant="muted">L1</Badge>
          </div>
          <div className="panel-body">
            {dur ? (
              <>
                <div className="grid grid-3" style={{ gap: 12 }}>
                  <div className="kpi" style={{ minHeight: 90 }}>
                    <span className="kpi-label">P50 typical</span>
                    <span className="kpi-value" style={{ fontSize: 22 }}>{fmtMinutes(dur.p50)}</span>
                  </div>
                  <div className="kpi" style={{ minHeight: 90 }}>
                    <span className="kpi-label">P80 planning</span>
                    <span className="kpi-value" style={{ fontSize: 22 }}>{fmtMinutes(dur.p80)}</span>
                  </div>
                  <div className="kpi" style={{ minHeight: 90 }}>
                    <span className="kpi-label">P95 worst case</span>
                    <span className="kpi-value" style={{ fontSize: 22 }}>{fmtMinutes(dur.p95)}</span>
                  </div>
                </div>
                <div className="dim" style={{ fontSize: 12, marginTop: 10 }}>
                  Based on {fmtNum(dur.n)} historical incidents of this type on this corridor.
                </div>
                {censored && (
                  <div style={{ marginTop: 10 }}>
                    <Note warn>
                      This cell is right-censored in the raw log (quantiles collapse to one large
                      value). Layer 4.5 guards it down to an operational duration before any allocation.
                    </Note>
                  </div>
                )}
              </>
            ) : (
              <span className="dim">No duration lookup for this cause and corridor.</span>
            )}
          </div>
        </div>

        {/* Layer 3 */}
        <div className="panel">
          <div className="panel-head">
            <div className="stack" style={{ gap: 2 }}>
              <h3 className="section-title">Layer 3 · Risk & corridor fragility</h3>
              <span className="section-meta">disruption impact and cascade potential</span>
            </div>
            <Badge variant="muted">L3</Badge>
          </div>
          <div className="panel-body grid grid-2" style={{ gap: 20 }}>
            <div>
              <div className="kpi-label" style={{ marginBottom: 8 }}>Survival-risk score</div>
              {risk ? (
                <>
                  <div className="row between" style={{ marginBottom: 8 }}>
                    <span className="kpi-value" style={{ fontSize: 26 }}>{fmtNum(risk.max)}</span>
                    <Badge variant={tierVariant(risk.tier)}>{risk.tier}</Badge>
                  </div>
                  <MetricLine k="Median score" v={fmtNum(risk.median)} />
                  <MetricLine k="Matching events" v={fmtNum(risk.count)} />
                </>
              ) : (
                <span className="dim">No risk rows for this pairing.</span>
              )}
            </div>
            <div>
              <div className="kpi-label" style={{ marginBottom: 8 }}>Corridor fragility</div>
              {frag ? (
                <>
                  <MetricLine k="Branching ratio" v={fmtNum(frag.branching_ratio)} />
                  <MetricLine k="Current intensity" v={fmtNum(frag.current_intensity)} />
                  <MetricLine k="Log-fragility" v={fmtNum(frag.fragility_log)} />
                </>
              ) : (
                <span className="dim">No fragility row for this corridor.</span>
              )}
            </div>
          </div>
        </div>

        {/* Layer 4 */}
        {plan && (
          <div className="panel">
            <div className="panel-head">
              <div className="stack" style={{ gap: 2 }}>
                <h3 className="section-title">Layer 4 · Recommended deployment</h3>
                <span className="section-meta">from retrieved precedents</span>
              </div>
              <Badge variant={plan.confidence_band.toLowerCase().includes("med") ? "warning" : "muted"}>
                {plan.confidence_band || "n/a"} confidence
              </Badge>
            </div>
            <div className="panel-body">
              <div className="grid grid-5" style={{ gap: 10 }}>
                {[
                  ["Officers", plan.officers],
                  ["Barricades", plan.barricades],
                  ["Tow units", plan.tow],
                  ["Supervisors", plan.supervisors],
                  ["QRU", plan.qru],
                ].map(([label, val]) => (
                  <div key={label as string} className="kpi" style={{ minHeight: 84 }}>
                    <span className="kpi-label">{label as string}</span>
                    <span className="kpi-value" style={{ fontSize: 24 }}>{fmtNum(val as number)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Layer 7 */}
        <div className="panel">
          <div className="panel-head">
            <div className="stack" style={{ gap: 2 }}>
              <h3 className="section-title">Layer 7 · Zone spillover</h3>
              <span className="section-meta">cross-excitation for {zone}</span>
            </div>
            <Badge variant="muted">L7</Badge>
          </div>
          <div className="panel-body">
            {spill ? (
              <div className="grid grid-4" style={{ gap: 12 }}>
                <MetricLine k="SSC centrality" v={fmtNum(spill.ssc)} />
                <MetricLine k="Source strength" v={fmtNum(spill.s_source)} />
                <MetricLine k="Receiver vuln." v={fmtNum(spill.v_receiver)} />
                <MetricLine k="Half-life (h)" v={fmtNum(spill.half_life)} />
              </div>
            ) : (
              <span className="dim">No spillover data for this zone.</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
