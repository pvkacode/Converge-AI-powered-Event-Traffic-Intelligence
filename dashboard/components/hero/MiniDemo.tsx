"use client";

import Link from "next/link";
import { useState, useCallback } from "react";
import { Lightning, CircleNotch } from "@phosphor-icons/react";
import { runWorkedExample, type WorkedExampleResult, type ScenarioInput } from "@/lib/api";
import { fmtMinutes } from "@/lib/format";
import { PlanConfidenceBadge } from "@/components/PlanConfidenceBadge";
import { CAUSES, CORRIDORS, DOW_MAP } from "./constants";
import { useTypewriter } from "./hooks";

const DEFAULT_INPUT = {
  cause: "vehicle_breakdown",
  corridor: "Mysore Road",
  priority: "High",
  requires_road_closure: false,
  hour_local: 9,
  dow_local: 2,
};

function tierPillClass(tier: string | undefined) {
  const t = (tier ?? "").toLowerCase();
  if (t === "critical") return "hero-pill hero-pill-critical";
  if (t === "high") return "hero-pill hero-pill-high";
  if (t === "moderate") return "hero-pill hero-pill-moderate";
  return "hero-pill hero-pill-low";
}

function extractOutputs(result: WorkedExampleResult) {
  const l1 = result.layer1_duration;
  const l2 = result.layer2_spatial;
  const l3 = result.layer3_resources;
  const plan = result.recommendation.duration_plan;

  const p80 =
    plan?.minutes ??
    (l1.p80 as number | undefined) ??
    (l1.p50 as number | undefined);

  const quantile = plan?.quantile?.toUpperCase() ?? "P80";

  const officers = (l3.officers as number | undefined) ?? result.recommendation.officer_plan?.count ?? 0;
  const barricades = (l3.barricades as number | undefined) ?? 0;
  const tow = (l3.tow as number | undefined) ?? 0;

  const junction = (l2.matched_hotspot as string | undefined) ?? "—";
  const obi = l2.OBI as number | undefined;
  const gi = (l2.gi_significance as number | undefined) ?? (l2.gi_max as number | undefined);
  const isHotspot = gi != null && gi > 0;

  const riskTier = (l3.risk_tier as string | undefined) ?? "—";
  const headline = result.recommendation.headline ?? "";

  return {
    p80, quantile, officers, barricades, tow, junction, obi, isHotspot, riskTier, headline,
    provenance: result.provenance,
    layer3: l3,
    layer4: result.layer4_event,
  };
}

export function MiniDemo() {
  const [cause, setCause] = useState(DEFAULT_INPUT.cause);
  const [corridor, setCorridor] = useState(DEFAULT_INPUT.corridor);
  const [priority, setPriority] = useState(DEFAULT_INPUT.priority);
  const [closure, setClosure] = useState(DEFAULT_INPUT.requires_road_closure);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReturnType<typeof extractOutputs> | null>(null);
  const [revealStep, setRevealStep] = useState(0);

  const typedHeadline = useTypewriter(
    result?.headline ?? "",
    20,
    revealStep >= 4 && !!result?.headline
  );

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setRevealStep(0);

    const body: ScenarioInput = {
      cause,
      corridor,
      hour_local: DEFAULT_INPUT.hour_local,
      dow_local: DOW_MAP.Wednesday,
      requires_road_closure: closure,
      priority,
    };

    try {
      const res = await runWorkedExample(body);
      const out = extractOutputs(res);
      setResult(out);
      setRevealStep(1);
      window.setTimeout(() => setRevealStep(2), 80);
      window.setTimeout(() => setRevealStep(3), 160);
      window.setTimeout(() => setRevealStep(4), 240);
    } catch {
      setError("Inference API unavailable. Start the FastAPI service or set NEXT_PUBLIC_API_URL.");
    } finally {
      setLoading(false);
    }
  }, [cause, corridor, priority, closure]);

  return (
    <div className="hero-demo-wrap" id="hero-demo">
      <div className={`hero-demo-card${error ? " is-error" : ""}`}>
        <div className="hero-demo-header">
          <Lightning size={28} weight="fill" color="var(--hero-gold)" />
          <div>
            <h2 className="hero-section-title" style={{ fontSize: 24 }}>
              Live Inference
            </h2>
            <p className="hero-section-sub" style={{ marginTop: 6, fontSize: 14 }}>
              Run a real scenario through the pipeline. Powered by the same API as the full Worked
              Example.
            </p>
          </div>
        </div>

        <div className="hero-demo-grid">
          <div className="hero-demo-form stack gap-4">
            <div className="hero-field">
              <label htmlFor="hero-cause">Incident cause</label>
              <select id="hero-cause" value={cause} onChange={(e) => setCause(e.target.value)}>
                {CAUSES.map((c) => (
                  <option key={c} value={c}>
                    {c.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </div>
            <div className="hero-field">
              <label htmlFor="hero-corridor">Corridor</label>
              <select id="hero-corridor" value={corridor} onChange={(e) => setCorridor(e.target.value)}>
                {CORRIDORS.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div className="hero-field">
              <label htmlFor="hero-priority">Priority</label>
              <select id="hero-priority" value={priority} onChange={(e) => setPriority(e.target.value)}>
                <option value="High">High</option>
                <option value="Low">Low</option>
                <option value="Unknown">Unknown</option>
              </select>
            </div>
            <div className="hero-field">
              <label>Requires road closure</label>
              <div className="hero-toggle">
                <input
                  type="checkbox"
                  checked={closure}
                  onChange={(e) => setClosure(e.target.checked)}
                  id="hero-closure"
                />
                <label htmlFor="hero-closure" style={{ textTransform: "none", letterSpacing: 0, fontSize: 14, color: "var(--hero-muted)" }}>
                  {closure ? "Yes" : "No"}
                </label>
              </div>
            </div>
            <button type="button" className="hero-btn hero-btn-primary" style={{ width: "100%", justifyContent: "center" }} onClick={run} disabled={loading}>
              {loading ? <CircleNotch size={18} className="spin" weight="bold" /> : <Lightning size={18} weight="fill" />}
              Run Pipeline
            </button>
            {loading ? (
              <p className="dim" style={{ fontSize: 12, textAlign: "center", margin: 0 }}>
                Running 7 layers…
              </p>
            ) : null}
            {error ? (
              <p style={{ color: "var(--critical)", fontSize: 13, margin: 0 }}>{error}</p>
            ) : null}
          </div>

          <div className="hero-output">
            {!result && !loading && (
              <div className="hero-output-placeholder">Run the pipeline to see live results</div>
            )}

            {loading && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <div className="hero-shimmer" style={{ height: 48 }} />
                <div className="hero-shimmer" style={{ height: 80 }} />
                <div className="hero-shimmer" style={{ height: 40 }} />
                <div className="hero-shimmer" style={{ height: 24 }} />
              </div>
            )}

            {result && !loading && (
              <>
                <div className={`hero-out-field${revealStep >= 1 ? " is-visible" : ""}`}>
                  <p className="hero-out-label">Duration estimate</p>
                  <div className="hero-out-p80">{fmtMinutes(result.p80)}</div>
                  <p className="dim" style={{ fontSize: 12, margin: "4px 0 0" }}>
                    {result.quantile} planning horizon
                  </p>
                </div>

                <div className={`hero-out-field${revealStep >= 2 ? " is-visible" : ""}`}>
                  <p className="hero-out-label">Resource recommendation</p>
                  <div className="hero-resource-row">
                    <div className="hero-resource-mini">
                      <div className="hero-resource-num">{Math.round(result.officers)}</div>
                      <div className="dim" style={{ fontSize: 11 }}>Officers</div>
                    </div>
                    <div className="hero-resource-mini">
                      <div className="hero-resource-num">{Math.round(result.barricades)}</div>
                      <div className="dim" style={{ fontSize: 11 }}>Barricades</div>
                    </div>
                    <div className="hero-resource-mini">
                      <div className="hero-resource-num">{Math.round(result.tow)}</div>
                      <div className="dim" style={{ fontSize: 11 }}>Tow units</div>
                    </div>
                  </div>
                </div>

                <div className={`hero-out-field${revealStep >= 3 ? " is-visible" : ""}`}>
                  <p className="hero-out-label">Spatial status</p>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span className={result.isHotspot ? "hero-pill hero-pill-hot" : "hero-pill hero-pill-clear"}>
                      {result.isHotspot ? "Hotspot" : "Clear"}
                    </span>
                    <span style={{ fontSize: 14 }}>{result.junction}</span>
                  </div>
                  {result.obi != null && (
                    <div style={{ marginTop: 10, maxWidth: 220 }}>
                      <span className="dim" style={{ fontSize: 11 }}>
                        OBI {result.obi.toFixed(2)}
                      </span>
                      <div className="hero-obi-bar">
                        <div className="hero-obi-fill" style={{ width: `${Math.min(100, result.obi * 100)}%` }} />
                      </div>
                    </div>
                  )}
                  <div style={{ marginTop: 12 }}>
                    <span className={tierPillClass(result.riskTier)}>{result.riskTier}</span>
                  </div>
                </div>

                <div className={`hero-out-field${revealStep >= 4 ? " is-visible" : ""}`} style={{ marginTop: 20 }}>
                  <div className="row gap-2 between">
                    <p className="hero-out-label" style={{ margin: 0 }}>Synthesised recommendation</p>
                    <PlanConfidenceBadge
                      input={{ provenance: result.provenance, layer3: result.layer3, layer4: result.layer4 }}
                    />
                  </div>
                  <p style={{ fontSize: 15, fontStyle: "italic", lineHeight: 1.55, margin: "8px 0 0", color: "var(--hero-ink)" }}>
                    {typedHeadline || result.headline}
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      <p className="hero-demo-link">
        <Link href="/worked-example">Full inference trace with all 7 layers →</Link>
      </p>

      <style jsx>{`
        .spin {
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @media (prefers-reduced-motion: reduce) {
          .spin { animation: none; }
        }
      `}</style>
    </div>
  );
}
