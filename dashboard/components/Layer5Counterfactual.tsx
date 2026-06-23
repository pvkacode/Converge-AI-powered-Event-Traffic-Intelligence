"use client";

import { useMemo, useState } from "react";
import { ArrowRight } from "@phosphor-icons/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Panel } from "@/components/ui";
import { useVizColors } from "@/components/charts";
import type { Row } from "@/lib/csv";
import { toNum } from "@/lib/format";

const SCENARIO_ORDER = [
  "baseline",
  "+2 officers",
  "+5 officers",
  "+3 barricades",
  "+1 tow unit",
  "+1 QRU",
  "diversion activated",
  "full surge (+5p +5b +1t)",
];

const SCENARIO_RESOURCE_UNITS: Record<string, number> = {
  baseline: 0,
  "+2 officers": 2,
  "+5 officers": 5,
  "+3 barricades": 3,
  "+1 tow unit": 1,
  "+1 QRU": 1,
  "diversion activated": 2,
  "full surge (+5p +5b +1t)": 11,
};

function barColor(pct: number, label: string): string {
  if (label === "baseline") return "#374151";
  if (pct > 10) return "#4ECDC4";
  if (pct > 5) return "#E8A53D";
  return "#374151";
}

type CfRow = {
  site_id: string;
  scenario_label: string;
  marginal_delay_reduction_pct: number;
  absolute_reduction_min: number;
  cost_per_pct_reduction: number;
  e_current: number;
  e_counterfactual: number;
  delta_effectiveness: number;
};

type BestRow = {
  site_id: string;
  service_tier: string;
  e_current: number;
  best_intervention_label: string;
  best_delta_effectiveness: number;
  best_absolute_reduction_min: number;
  best_cost_per_pct: number;
};

type CityRow = {
  scenario_label: string;
  pct_of_reducible_delay_citywide: number;
  total_absolute_reduction_min_citywide: number;
  n_sites_improved: number;
};

function parseCounterfactual(rows: Row[]): CfRow[] {
  return rows
    .map((r) => ({
      site_id: r.site_id ?? "",
      scenario_label: r.scenario_label ?? "",
      marginal_delay_reduction_pct: toNum(r.marginal_delay_reduction_pct),
      absolute_reduction_min: toNum(r.absolute_reduction_min),
      cost_per_pct_reduction: toNum(r.cost_per_pct_reduction),
      e_current: toNum(r.e_current),
      e_counterfactual: toNum(r.e_counterfactual),
      delta_effectiveness: toNum(r.delta_effectiveness),
    }))
    .filter((r) => r.site_id && r.scenario_label);
}

function parseBest(rows: Row[]): BestRow[] {
  return rows
    .map((r) => ({
      site_id: r.site_id ?? "",
      service_tier: r.service_tier ?? "",
      e_current: toNum(r.e_current),
      best_intervention_label: r.best_intervention_label ?? "",
      best_delta_effectiveness: toNum(r.best_delta_effectiveness),
      best_absolute_reduction_min: toNum(r.best_absolute_reduction_min),
      best_cost_per_pct: toNum(r.best_cost_per_pct),
    }))
    .filter((r) => r.site_id);
}

function parseCity(rows: Row[]): CityRow[] {
  return rows
    .map((r) => ({
      scenario_label: r.scenario_label ?? "",
      pct_of_reducible_delay_citywide: toNum(r.pct_of_reducible_delay_citywide),
      total_absolute_reduction_min_citywide: toNum(r.total_absolute_reduction_min_citywide),
      n_sites_improved: toNum(r.n_sites_improved),
    }))
    .filter((r) => r.scenario_label);
}

function CfTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: CfRow & { name: string; value: number } }[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const cost = Number.isFinite(d.cost_per_pct_reduction) ? d.cost_per_pct_reduction : Infinity;
  return (
    <div
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "10px 12px",
        fontSize: 12,
        lineHeight: 1.5,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.scenario_label}</div>
      <div>Marginal reduction: {d.marginal_delay_reduction_pct.toFixed(1)}%</div>
      <div>Absolute: {d.absolute_reduction_min.toFixed(0)} min saved</div>
      <div>
        Cost efficiency:{" "}
        {Number.isFinite(cost) ? `${cost.toFixed(1)} units per %` : "—"}
      </div>
    </div>
  );
}

export function Layer5Counterfactual({
  counterfactualRows,
  bestRows,
  cityRows,
}: {
  counterfactualRows: Row[];
  bestRows: Row[];
  cityRows: Row[];
}) {
  const c = useVizColors();
  const cf = useMemo(() => parseCounterfactual(counterfactualRows), [counterfactualRows]);
  const best = useMemo(() => parseBest(bestRows), [bestRows]);
  const city = useMemo(() => parseCity(cityRows), [cityRows]);

  const defaultSite = useMemo(() => {
    if (!best.length) return "";
    return [...best].sort((a, b) => b.e_current - a.e_current)[0]?.site_id ?? best[0].site_id;
  }, [best]);

  const [selectedSite, setSelectedSite] = useState(defaultSite);

  const siteIds = useMemo(
    () => [...new Set(best.map((r) => r.site_id))].sort(),
    [best]
  );

  const activeSite = selectedSite || defaultSite;
  const siteBest = best.find((r) => r.site_id === activeSite);

  const chartData = useMemo(() => {
    const siteScenarios = cf.filter((r) => r.site_id === activeSite);
    const order = new Map(SCENARIO_ORDER.map((l, i) => [l, i]));
    return [...siteScenarios]
      .sort((a, b) => (order.get(a.scenario_label) ?? 99) - (order.get(b.scenario_label) ?? 99))
      .map((r) => ({
        ...r,
        name: r.scenario_label,
        value: r.marginal_delay_reduction_pct,
      }));
  }, [cf, activeSite]);

  const maxPct = useMemo(
    () => Math.max(...chartData.map((d) => d.value), 1),
    [chartData]
  );

  const cityOrdered = useMemo(() => {
    const order = new Map(SCENARIO_ORDER.map((l, i) => [l, i]));
    return [...city].sort(
      (a, b) => (order.get(a.scenario_label) ?? 99) - (order.get(b.scenario_label) ?? 99)
    );
  }, [city]);

  const maxCityPct = useMemo(
    () =>
      Math.max(
        ...cityOrdered.map((r) => r.pct_of_reducible_delay_citywide),
        1
      ),
    [cityOrdered]
  );

  const bestOverall = useMemo(() => {
    let label = "baseline";
    let bestEff = -1;
    for (const row of cityOrdered) {
      const units = SCENARIO_RESOURCE_UNITS[row.scenario_label] ?? 0;
      if (units <= 0 || row.scenario_label === "baseline") continue;
      const eff = row.pct_of_reducible_delay_citywide / units;
      if (eff > bestEff) {
        bestEff = eff;
        label = row.scenario_label;
      }
    }
    return label;
  }, [cityOrdered]);

  if (!cf.length || !best.length) {
    return (
      <Panel title="What If? · Counterfactual Explorer">
        <p className="dim" style={{ fontSize: 13 }}>
          Counterfactual outputs not found — run layer5_robust_optimization.py
        </p>
      </Panel>
    );
  }

  return (
    <div className="stack gap-6" style={{ marginTop: 24 }}>
      <Panel title="What If? · Counterfactual Explorer" meta="Marginal impact of discrete resource additions per site">
        <div className="grid grid-2" style={{ alignItems: "start", gap: 24 }}>
          <div>
            <label className="dim" style={{ fontSize: 11, letterSpacing: "0.06em", display: "block", marginBottom: 8 }}>
              SELECT SITE TO ANALYSE
            </label>
            <select
              value={activeSite}
              onChange={(e) => setSelectedSite(e.target.value)}
              style={{
                width: "100%",
                marginBottom: 16,
                padding: "8px 10px",
                borderRadius: 6,
                border: "1px solid var(--border)",
                background: "var(--surface-2)",
                color: "var(--ink)",
                fontSize: 13,
              }}
            >
              {siteIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>

            <div className="chart-wrap" style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
                  <CartesianGrid horizontal={false} stroke={c.grid} />
                  <XAxis
                    type="number"
                    domain={[0, Math.ceil(maxPct * 1.1)]}
                    stroke={c.ink3}
                    tick={{ fontSize: 11 }}
                    unit="%"
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={160}
                    stroke={c.ink3}
                    tick={{ fontSize: 10, fill: c.ink }}
                    interval={0}
                  />
                  <Tooltip content={<CfTooltip />} cursor={{ fill: c.grid }} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
                    {chartData.map((d, i) => (
                      <Cell key={i} fill={barColor(d.value, d.scenario_label)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div
            style={{
              padding: 20,
              borderRadius: 10,
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
            }}
          >
            {siteBest ? (
              <>
                <div
                  style={{
                    color: "#E8A53D",
                    fontSize: 20,
                    fontWeight: 700,
                    marginBottom: 16,
                    lineHeight: 1.3,
                  }}
                >
                  {siteBest.best_intervention_label}
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12, fontSize: 14 }}>
                  <span>Effectiveness: {siteBest.e_current.toFixed(3)}</span>
                  <ArrowRight size={16} color="#4ECDC4" weight="bold" />
                  <span>
                    {(siteBest.e_current + siteBest.best_delta_effectiveness).toFixed(3)}
                  </span>
                </div>

                <div style={{ color: "#4ECDC4", fontSize: 14, marginBottom: 10 }}>
                  Delay reduction: +{(siteBest.best_delta_effectiveness * 100).toFixed(1)}%
                </div>

                <div style={{ color: "var(--ink)", fontSize: 14, marginBottom: 16 }}>
                  Minutes saved: ~{siteBest.best_absolute_reduction_min.toFixed(0)} min
                </div>

                <p className="dim" style={{ fontSize: 11, lineHeight: 1.5, margin: 0 }}>
                  Reduction estimate uses E(u) = 1-exp(-Σγᵢuᵢ) with fixed effectiveness
                  coefficients (γ not learned from deployment data). Treat as directional, not
                  precise.
                </p>
              </>
            ) : (
              <p className="dim" style={{ fontSize: 13 }}>No best-intervention data for this site.</p>
            )}
          </div>
        </div>
      </Panel>

      <Panel
        title="City-Wide Impact · Counterfactual Analysis"
        meta="If every site received this additional intervention…"
      >
        <div className="stack gap-3">
          {cityOrdered.map((row) => {
            const pct = row.pct_of_reducible_delay_citywide;
            const widthPct = (pct / maxCityPct) * 100;
            const fill = barColor(pct, row.scenario_label);
            return (
              <div
                key={row.scenario_label}
                style={{ display: "grid", gridTemplateColumns: "200px 1fr auto", gap: 12, alignItems: "center" }}
              >
                <span style={{ fontSize: 12 }}>{row.scenario_label}</span>
                <div
                  style={{
                    height: 18,
                    borderRadius: 4,
                    background: "rgba(255,255,255,0.06)",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      width: `${widthPct}%`,
                      height: "100%",
                      background: fill,
                      borderRadius: 4,
                      minWidth: pct > 0 ? 2 : 0,
                    }}
                  />
                </div>
                <div style={{ textAlign: "right", minWidth: 120 }}>
                  <span className="mono" style={{ fontSize: 12 }}>
                    {pct.toFixed(1)}%
                  </span>
                  <div className="dim" style={{ fontSize: 10 }}>
                    ({row.n_sites_improved} sites improved)
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div
          style={{
            marginTop: 20,
            padding: "14px 16px",
            borderRadius: 8,
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
          }}
        >
          <span style={{ color: "#E8A53D", fontWeight: 700 }}>
            Most cost-effective single intervention: {bestOverall}
          </span>
        </div>
      </Panel>

      <p className="dim" style={{ fontSize: 11, lineHeight: 1.6, margin: 0 }}>
        ΔRisk = (E_with - E_without) applied to mean scenario duration per site. Baseline
        effectiveness from existing MILP allocation. No MILP re-solve performed — counterfactuals
        evaluate the closed-form effectiveness function only. Resource caps (p≤12, b≤20, t≤4,
        q≤3) enforced per site.
      </p>
    </div>
  );
}
