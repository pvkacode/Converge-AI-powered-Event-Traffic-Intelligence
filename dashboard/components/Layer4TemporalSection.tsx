"use client";

import { useMemo } from "react";
import { Clock } from "@phosphor-icons/react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Panel } from "@/components/ui";
import { useVizColors } from "@/components/charts";

export type TemporalMetadata = {
  half_life_days?: number;
  lambda_decay?: number;
  layer6_half_life_days?: number;
  n_rank_changes?: number;
  pct_recency_matters?: number;
  mean_rank_change?: number;
  mean_phi_t_top1?: number;
  oldest_top1_days?: number;
  n_retrievals_processed?: number;
};

function DecayCurve({
  lambdaDecay,
  halfLife,
  layer6HalfLife,
}: {
  lambdaDecay: number;
  halfLife: number;
  layer6HalfLife: number;
}) {
  const c = useVizColors();
  const data = useMemo(
    () =>
      Array.from({ length: 91 }, (_, t) => ({
        days: t,
        weight: Math.exp(-lambdaDecay * t),
      })),
    [lambdaDecay]
  );

  return (
    <div>
      <div className="dim" style={{ fontSize: 11, marginBottom: 8 }}>
        Decay curve
      </div>
      <div className="chart-wrap" style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke={c.grid} vertical={false} />
            <XAxis
              dataKey="days"
              stroke={c.ink3}
              tick={{ fontSize: 10 }}
              label={{ value: "Δt (days)", position: "insideBottom", offset: -2, fill: c.ink3, fontSize: 10 }}
            />
            <YAxis
              domain={[0, 1]}
              stroke={c.ink3}
              tick={{ fontSize: 10 }}
              label={{ value: "φ_time", angle: -90, position: "insideLeft", fill: c.ink3, fontSize: 10 }}
            />
            <Tooltip
              formatter={(v: number) => [v.toFixed(3), "weight"]}
              labelFormatter={(x) => `At ${x} days`}
            />
            <ReferenceLine
              x={halfLife}
              stroke="#E8A53D"
              strokeDasharray="4 4"
              label={{ value: "half-life", position: "top", fill: "#E8A53D", fontSize: 10 }}
            />
            <ReferenceLine
              x={layer6HalfLife}
              stroke="#9CA3AF"
              strokeDasharray="4 4"
              label={{ value: "L6 ref", position: "insideTopRight", fill: "#9CA3AF", fontSize: 10 }}
            />
            <Line type="monotone" dataKey="weight" stroke="#4ECDC4" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="dim" style={{ fontSize: 11, marginTop: 8, marginBottom: 0 }}>
        Precedent from {halfLife.toFixed(0)} days ago receives 0.5× the weight of one from today
      </p>
    </div>
  );
}

function StatRow({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div className="mono" style={{ fontSize: 22, fontWeight: 600, lineHeight: 1.2 }}>
        {value}
      </div>
      <div style={{ fontSize: 13, marginTop: 4 }}>{label}</div>
      {sub ? <div className="dim" style={{ fontSize: 11, marginTop: 2 }}>{sub}</div> : null}
    </div>
  );
}

export function Layer4TemporalSection({
  metadata,
}: {
  metadata: TemporalMetadata | null;
}) {
  if (!metadata?.lambda_decay) {
    return (
      <Panel title="Retrieval Recency · Temporal Decay Weighting">
        <p className="dim" style={{ fontSize: 13 }}>
          Temporal decay outputs not found — run layer4_planned_event_retrieval.py
        </p>
      </Panel>
    );
  }

  const halfLife = metadata.half_life_days ?? 30;
  const lambdaDecay = metadata.lambda_decay ?? Math.log(2) / 30;
  const layer6HalfLife = metadata.layer6_half_life_days ?? 30;
  const nRankChanges = metadata.n_rank_changes ?? 0;
  const pctRecency = metadata.pct_recency_matters ?? 0;
  const meanRankChange = metadata.mean_rank_change ?? 0;
  const meanPhiTop1 = metadata.mean_phi_t_top1 ?? 0;
  const oldestTop1 = metadata.oldest_top1_days ?? 0;

  return (
    <div className="stack gap-6" style={{ marginTop: 24 }}>
      <Panel
        title={
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <Clock size={18} color="#4ECDC4" weight="bold" />
            Retrieval Recency · Temporal Decay Weighting
          </span>
        }
        meta="φ_time(Δt) = exp(−λ·Δt) with λ derived from duration autocorrelation. Same exponential family as Layer 6 forgetting and Layer 7 Hawkes decay."
      >
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
          <span className="mono" style={{ fontSize: 11, padding: "4px 8px", borderRadius: 6, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            λ = {lambdaDecay.toFixed(6)}
          </span>
          <span className="mono" style={{ fontSize: 11, padding: "4px 8px", borderRadius: 6, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            Half-life = {halfLife.toFixed(1)} days
          </span>
          <span className="mono" style={{ fontSize: 11, padding: "4px 8px", borderRadius: 6, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            Layer 6 half-life = {layer6HalfLife.toFixed(1)} days for comparison
          </span>
        </div>

        <div className="grid grid-3" style={{ alignItems: "start", gap: 20 }}>
          <DecayCurve
            lambdaDecay={lambdaDecay}
            halfLife={halfLife}
            layer6HalfLife={layer6HalfLife}
          />

          <div>
            <StatRow
              value={String(nRankChanges)}
              label="retrievals re-ranked by recency"
              sub={`${pctRecency.toFixed(1)}% of all retrievals affected`}
            />
            <StatRow
              value={`${meanRankChange.toFixed(2)} positions`}
              label="mean rank change magnitude"
            />
            <StatRow
              value={meanPhiTop1.toFixed(3)}
              label="mean φ_time of top-1 match"
              sub="how recent the best match typically is"
            />
            <StatRow
              value={`${oldestTop1.toFixed(0)} days`}
              label="oldest event that became top-1 after temporal weighting"
            />
          </div>

          <div
            style={{
              padding: 16,
              borderRadius: 10,
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 12 }}>Why this formula?</div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.7 }}>
              <li>Layer 6 Bayesian forgetting: w_i = exp(−λ·Δt_i) [30-day half-life]</li>
              <li>Layer 7 Hawkes decay: α·exp(−β(t−tᵢ)) [spillover excitation]</li>
              <li>Layer 4 retrieval decay: φ_time = exp(−λ·Δt) [this addition]</li>
            </ul>
            <p style={{ color: "#4ECDC4", fontSize: 12, marginTop: 12, marginBottom: 8 }}>
              Same mathematical family applied consistently across three layers.
            </p>
            <p className="dim" style={{ fontSize: 11, lineHeight: 1.5, margin: 0 }}>
              Layer 6 drift analysis found CRITICAL mean-shift z=3.21 between Nov–Feb and Mar–Apr,
              motivating recency weighting in retrieval.
            </p>
          </div>
        </div>

        <div
          style={{
            marginTop: 20,
            padding: "12px 14px",
            borderRadius: 8,
            background: "rgba(0,0,0,0.2)",
            border: "1px solid var(--border)",
            fontSize: 11,
            lineHeight: 1.6,
            color: "var(--ink-2)",
          }}
        >
          Temporal decay is applied as a post-scoring weight on existing Gower retrieval — it does
          not re-run the retrieval itself. Half-life of {halfLife.toFixed(1)} days derived from lag
          at which daily-average duration autocorrelation drops below 0.5. If autocorrelation was not
          computable, 30-day default used (matching Layer 6). This addition has not been validated
          against held-out retrieval accuracy.
        </div>
      </Panel>
    </div>
  );
}
