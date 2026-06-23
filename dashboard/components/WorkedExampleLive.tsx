"use client";
import nextDynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { Lightning, ArrowClockwise } from "@phosphor-icons/react";
import {
  fetchOptions,
  fetchHealth,
  runWorkedExample,
  type Options,
  type ScenarioInput,
  type WorkedExampleResult,
  type LayerSection,
  type Provenance,
} from "@/lib/api";
import type { WxMapData } from "@/lib/map-junctions";
import { fmtNum, fmtMinutes, titleCaseValue } from "@/lib/format";
import { Badge, MetricLine } from "./ui";
import { MapPlaceholder } from "@/components/maps/map-ui";
import { BackendWakeNotice } from "@/components/BackendWakeNotice";
import { WorkedExampleExecutiveSummary } from "@/components/WorkedExampleExecutiveSummary";
import { useSlowLoading } from "@/hooks/useSlowLoading";

const WorkedExampleMap = nextDynamic(() => import("@/components/maps/WorkedExampleMap"), {
  ssr: false,
  loading: () => <MapPlaceholder height={300} message="Loading map…" />,
});

const DEFAULT_INPUT: ScenarioInput = {
  cause: "public_event",
  corridor: "Mysore Road",
  hour_local: 9,
  dow_local: 0,
  requires_road_closure: false,
  priority: "High",
};

const EXAMPLE_HINT =
  "Try: public_event × Mysore Road for a fully traced example (L3 + L4 non-zero).";

function layer3AllZero(section: LayerSection): boolean {
  if (section.insufficient_evidence === true) return true;
  const o = section.officers as number | undefined;
  const b = section.barricades as number | undefined;
  const t = section.tow as number | undefined;
  return (o ?? 0) === 0 && (b ?? 0) === 0 && (t ?? 0) === 0;
}

function layer4Insufficient(section: LayerSection): boolean {
  if (section.insufficient_evidence === true) return true;
  if (section.applicable === false) return true;
  const rec = section.recommended as Record<string, unknown> | undefined;
  const o = rec?.officers as number | undefined;
  const ess = section.evidence_weight as number | undefined;
  return (o ?? 0) === 0 && (ess ?? 0) === 0;
}

function InsufficientEvidence({
  layer,
  corridor,
}: {
  layer: "L3" | "L4";
  corridor?: string;
}) {
  return (
    <div className="empty" style={{ padding: "14px 0 4px" }}>
      <div className="empty-title" style={{ fontSize: 13.5 }}>
        Insufficient evidence for this combination
      </div>
      <p className="dim" style={{ fontSize: 12, margin: "6px 0 0", lineHeight: 1.5, maxWidth: "52ch" }}>
        {layer === "L4"
          ? "Layer 4 retrieval applies to planned events only (procession, protest, public_event, vip_movement). Unplanned causes are covered by Layers 1–3."
          : `No historical analogs with enough confidence — Layer 3 rule-based estimates unavailable${corridor ? ` for ${corridor}` : ""}.`}
      </p>
    </div>
  );
}

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

function provBadge(p: Provenance) {
  if (p === "live") return <Badge variant="ok" dot>Live inference</Badge>;
  if (p === "fallback") return <Badge variant="warning" dot>Fallback</Badge>;
  return <Badge variant="neutral">Precomputed</Badge>;
}

function Rows({ rows }: { rows: [string, unknown][] }) {
  const visible = rows.filter(([, v]) => v !== null && v !== undefined && v !== "" && v !== "nan");
  if (!visible.length) return <span className="dim">No matching data for this input.</span>;
  return (
    <>
      {visible.map(([k, v]) => (
        <MetricLine key={k} k={k} v={typeof v === "number" ? fmtNum(v) : String(v)} />
      ))}
    </>
  );
}

function PipeItem({
  idx,
  title,
  layerTag,
  section,
  children,
}: {
  idx: string;
  title: string;
  layerTag: string;
  section: LayerSection;
  children: React.ReactNode;
}) {
  const prov = section.provenance;
  return (
    <div className="pipe-item">
      <div className={`pipe-node ${prov === "live" ? "live" : prov === "fallback" ? "fallback" : ""}`}>{idx}</div>
      <div className="panel">
        <div className="panel-head">
          <div className="stack" style={{ gap: 2 }}>
            <h3 className="section-title">{title}</h3>
            <span className="section-meta">{layerTag}</span>
          </div>
          {provBadge(prov)}
        </div>
        <div className="panel-body">
          {children}
          {section.note ? (
            <div className="dim" style={{ fontSize: 12, marginTop: 10, lineHeight: 1.5 }}>{String(section.note)}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function WorkedExampleLive({ mapData }: { mapData: WxMapData | null }) {
  const [options, setOptions] = useState<Options | null>(null);
  const [input, setInput] = useState<ScenarioInput>(DEFAULT_INPUT);
  const [result, setResult] = useState<WorkedExampleResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [offline, setOffline] = useState(false);
  const [live, setLive] = useState<{ ok: boolean; reason: string } | null>(null);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bootstrapping = options == null && !offline;
  const waiting = loading || bootstrapping;
  const slow = useSlowLoading(waiting);

  const run = useCallback(async (inp: ScenarioInput) => {
    setLoading(true);
    setOffline(false);
    try {
      const r = await runWorkedExample(inp);
      setResult(r);
    } catch {
      setOffline(true);
    } finally {
      setLoading(false);
    }
  }, []);

  // initial load: options + health + first run
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [opt, h] = await Promise.all([fetchOptions(), fetchHealth().catch(() => null)]);
        if (cancelled) return;
        setOptions(opt);
        if (h) setLive({ ok: h.layer1_live, reason: h.layer1_reason });
        await run(DEFAULT_INPUT);
      } catch {
        if (!cancelled) setOffline(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [run]);

  // debounced re-run on input change
  const update = (patch: Partial<ScenarioInput>) => {
    const next = { ...input, ...patch };
    setInput(next);
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => run(next), 350);
  };

  const causes = options?.causes ?? [DEFAULT_INPUT.cause];
  const corridors = options?.corridors ?? [DEFAULT_INPUT.corridor];

  return (
    <div className="grid" style={{ gridTemplateColumns: "320px 1fr", gap: 24, alignItems: "start" }}>
      {/* ---------- input form ---------- */}
      <div className="panel" style={{ padding: 18, position: "sticky", top: 80 }}>
        <div className="row gap-2 between" style={{ marginBottom: 14 }}>
          <div className="row gap-2">
            <Lightning size={18} weight="fill" className="kpi-accent" />
            <h2 className="section-title">Live inference</h2>
          </div>
          {loading && <ArrowClockwise size={15} className="dim spin" />}
        </div>

        <div className="stack gap-4">
          <label className="stack gap-2">
            <span className="kpi-label">Incident cause</span>
            <select className="select" value={input.cause} onChange={(e) => update({ cause: e.target.value })}>
              {causes.map((c) => <option key={c} value={c}>{titleCaseValue(c)}</option>)}
            </select>
          </label>
          <label className="stack gap-2">
            <span className="kpi-label">Corridor</span>
            <select className="select" value={input.corridor} onChange={(e) => update({ corridor: e.target.value })}>
              {corridors.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label className="stack gap-2">
            <span className="kpi-label">Hour of day · {String(input.hour_local).padStart(2, "0")}:00</span>
            <input
              className="range"
              type="range"
              min={0}
              max={23}
              value={input.hour_local}
              onChange={(e) => update({ hour_local: Number(e.target.value) })}
            />
          </label>
          <label className="stack gap-2">
            <span className="kpi-label">Day of week</span>
            <select className="select" value={input.dow_local} onChange={(e) => update({ dow_local: Number(e.target.value) })}>
              {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
          </label>
          <label className="stack gap-2">
            <span className="kpi-label">Priority</span>
            <select className="select" value={input.priority} onChange={(e) => update({ priority: e.target.value })}>
              {(options?.priorities ?? ["High", "Low", "Unknown"]).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
          <div className="toggle-row">
            <span className="kpi-label" style={{ margin: 0 }}>Requires road closure</span>
            <input
              type="checkbox"
              checked={input.requires_road_closure}
              onChange={(e) => update({ requires_road_closure: e.target.checked })}
            />
          </div>
          <button className="btn btn-accent" onClick={() => run(input)} disabled={loading}>
            {loading ? "Running pipeline…" : "Run pipeline"}
          </button>
          {waiting && slow ? (
            <BackendWakeNotice slow compact />
          ) : null}
        </div>

        <p className="dim" style={{ fontSize: 11.5, marginTop: 12, lineHeight: 1.45 }}>
          {EXAMPLE_HINT}
        </p>

        <div style={{ marginTop: 16, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
          <div className="row gap-2 between">
            <span className="kpi-label" style={{ margin: 0 }}>Layer 1 engine</span>
            {live == null ? (
              <span className="dim">checking…</span>
            ) : live.ok ? (
              <Badge variant="ok" dot>Live</Badge>
            ) : (
              <Badge variant="warning">Fallback</Badge>
            )}
          </div>
          {live && !live.ok && (
            <div className="dim" style={{ fontSize: 11.5, marginTop: 6, lineHeight: 1.5 }}>
              Survival models need the pipeline venv + data/events_clean.parquet. Without them, Layer 1
              is served from the precomputed lookup. Reason: <span className="mono">{live.reason}</span>
            </div>
          )}
        </div>
      </div>

      {/* ---------- result pipeline ---------- */}
      <div>
        {offline ? (
          <div className="panel">
            <div className="panel-body">
              <BackendWakeNotice offline onRetry={() => run(input)} />
            </div>
          </div>
        ) : !result ? (
          <div className="panel">
            <div className="panel-body" style={{ padding: slow ? 18 : 36 }}>
              {slow ? (
                <BackendWakeNotice slow />
              ) : (
                <div className="empty">
                  <div className="empty-title">Loading pipeline…</div>
                  <p className="dim" style={{ fontSize: 13, marginTop: 8 }}>
                    Connecting to the inference API and running all seven layers.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <>
            <WorkedExampleExecutiveSummary result={result} />

            <div className="pipe" id="wx-layer-trace">
              <PipeItem idx="L1" title="Duration" layerTag="Layer 1 · survival quantiles" section={result.layer1_duration}>
                <div className="grid grid-3" style={{ gap: 10 }}>
                  {(["p50", "p80", "p95"] as const).map((q) => (
                    <div key={q} className="kpi" style={{ minHeight: 76 }}>
                      <span className="kpi-label">{q.toUpperCase()}</span>
                      <span className="kpi-value" style={{ fontSize: 20 }}>
                        {result.layer1_duration[q] != null ? fmtMinutes(result.layer1_duration[q] as number) : "n/a"}
                      </span>
                    </div>
                  ))}
                </div>
                {result.layer1_duration.tail_trustworthy === false ? (
                  <p className="dim" style={{ fontSize: 12, marginTop: 10, lineHeight: 1.45 }}>
                    {String(result.layer1_duration.tail_note || "P80/P95 are censored KM estimates — not used for planning.")}
                    {result.recommendation.duration_plan?.minutes != null ? (
                      <> Planning uses{" "}
                        <strong>{fmtMinutes(result.recommendation.duration_plan.minutes)}</strong>
                        {" "}({result.recommendation.duration_plan.quantile?.toUpperCase()} ·{" "}
                        {result.recommendation.duration_plan.source === "layer45_guarded"
                          ? "Layer 4.5 guarded"
                          : "Layer 1"}
                        ).
                      </>
                    ) : null}
                  </p>
                ) : null}
                <div style={{ marginTop: 10 }}>
                  <Rows rows={[["Source", result.layer1_duration.source], ["Sample n", result.layer1_duration.n], ["Confidence", result.layer1_duration.confidence]]} />
                </div>
                {mapData ? (
                  <WorkedExampleMap mapData={mapData} input={input} />
                ) : (
                  <MapPlaceholder height={300} message="Map unavailable — check outputs/frontend/ exports" />
                )}
              </PipeItem>

              <PipeItem idx="L2" title="Spatial" layerTag="Layer 2 · hotspot match" section={result.layer2_spatial}>
                <Rows rows={[
                  ["Matched hotspot", result.layer2_spatial.matched_hotspot],
                  ["Operational burden (OBI)", result.layer2_spatial.OBI],
                  ["Gi* (h1)", result.layer2_spatial.gi_significance],
                  ["SPS", result.layer2_spatial.sps],
                ]} />
              </PipeItem>

              <PipeItem idx="L3" title="Resources" layerTag="Layer 3 · risk + allocation" section={result.layer3_resources}>
                <div className="row gap-3 wrap" style={{ marginBottom: 8 }}>
                  {result.layer3_resources.risk_tier ? (
                    <Badge variant={result.layer3_resources.risk_tier === "Critical" ? "critical" : result.layer3_resources.risk_tier === "High" ? "warning" : result.layer3_resources.risk_tier === "Moderate" ? "accent" : "ok"}>
                      {String(result.layer3_resources.risk_tier)} tier
                    </Badge>
                  ) : null}
                </div>
                {layer3AllZero(result.layer3_resources) ? (
                  <InsufficientEvidence layer="L3" corridor={input.corridor} />
                ) : (
                  <Rows rows={[
                    ["Disruption impact (DIS)", result.layer3_resources.dis],
                    ["Officers", result.layer3_resources.officers],
                    ["Barricades", result.layer3_resources.barricades],
                    ["Tow units", result.layer3_resources.tow],
                    ["Corridor branching ratio", (result.layer3_resources.fragility as Record<string, unknown> | undefined)?.branching_ratio],
                  ]} />
                )}
              </PipeItem>

              <PipeItem idx="L4" title="Event memory" layerTag="Layer 4 · retrieved precedent" section={result.layer4_event}>
                {layer4Insufficient(result.layer4_event) ? (
                  <InsufficientEvidence layer="L4" corridor={input.corridor} />
                ) : (
                  <Rows rows={[
                    ["Confidence tier", result.layer4_event.confidence_tier],
                    ["Institutional memory (IMS)", result.layer4_event.IMS],
                    ["Evidence weight (ESS)", result.layer4_event.evidence_weight],
                    ["Recommended officers", (result.layer4_event.recommended as Record<string, unknown> | undefined)?.officers],
                  ]} />
                )}
              </PipeItem>

              <PipeItem idx="4.5" title="Predictive fusion" layerTag="Layer 4.5 · guarded quantiles" section={result.layer45_fusion}>
                <Rows rows={[
                  ["Guarded P50", (result.layer45_fusion.duration_quantiles as Record<string, number> | undefined)?.p50],
                  ["Guarded P80", (result.layer45_fusion.duration_quantiles as Record<string, number> | undefined)?.p80],
                  ["Tail-risk prob", result.layer45_fusion.tail_risk_prob],
                  ["Novelty flag", result.layer45_fusion.novelty_flag === true ? "yes" : result.layer45_fusion.novelty_flag === false ? "no" : null],
                  ["Drift flag", result.layer45_fusion.drift_flag === true ? "yes" : result.layer45_fusion.drift_flag === false ? "no" : null],
                ]} />
              </PipeItem>

              <PipeItem idx="L5" title="Robust allocation" layerTag="Layer 5 · CVaR optimization" section={result.layer5_optimization}>
                <Rows rows={[
                  ["Service tier", (result.layer5_optimization.allocation as Record<string, unknown> | undefined)?.service_tier],
                  ["Officers", (result.layer5_optimization.allocation as Record<string, unknown> | undefined)?.officers],
                  ["CVaR before", result.layer5_optimization.cvar_before],
                  ["CVaR after", result.layer5_optimization.cvar_after],
                ]} />
              </PipeItem>

              <PipeItem idx="L6" title="Adaptive learning" layerTag="Layer 6 · monitoring" section={result.layer6_learning}>
                <div className="row gap-3 wrap" style={{ marginBottom: 8 }}>
                  {result.layer6_learning.relevant_health_status ? (
                    <Badge variant={String(result.layer6_learning.relevant_health_status).includes("CRIT") ? "critical" : "ok"} dot>
                      {String(result.layer6_learning.relevant_health_status)}
                    </Badge>
                  ) : null}
                </div>
                <Rows rows={[
                  ["Drift test", (result.layer6_learning.relevant_drift_signal as Record<string, unknown> | undefined)?.test],
                  ["Drift variable", (result.layer6_learning.relevant_drift_signal as Record<string, unknown> | undefined)?.variable],
                ]} />
              </PipeItem>

              <PipeItem idx="L7" title="Cross-zone spillover" layerTag="Layer 7 · early warning" section={result.layer7_spillover}>
                <Rows rows={[
                  ["Highest-risk zone", result.layer7_spillover.zone],
                  ["Expected Risk Index", result.layer7_spillover.eri],
                  ["Spillover centrality", result.layer7_spillover.spillover_centrality],
                  ["Persistence class", result.layer7_spillover.early_warning],
                ]} />
              </PipeItem>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
