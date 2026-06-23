"use client";

import { Graph, TreeStructure, Broadcast } from "@phosphor-icons/react";
import { Panel, Note } from "@/components/ui";
import { GroupedHBar } from "@/components/charts";
import type { Row } from "@/lib/csv";
import { toNum } from "@/lib/format";

const ROLE_STYLES: Record<string, { bg: string; color: string; label: string }> = {
  critical_hub: { bg: "#B0413E", color: "#FFFFFF", label: "CRITICAL HUB" },
  sink: { bg: "#E8A53D", color: "#1A1813", label: "SINK" },
  relay: { bg: "#1F4D47", color: "#FFFFFF", label: "RELAY" },
  influential: { bg: "#6B4FA0", color: "#FFFFFF", label: "INFLUENTIAL" },
  isolated: { bg: "#374151", color: "#FFFFFF", label: "ISOLATED" },
  moderate: { bg: "#1F2937", color: "#FFFFFF", label: "MODERATE" },
};

function RoleBadge({ role }: { role: string }) {
  const style = ROLE_STYLES[role] ?? ROLE_STYLES.moderate;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: 999,
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: "0.04em",
        background: style.bg,
        color: style.color,
      }}
    >
      {style.label}
    </span>
  );
}

export function Layer7GraphCentrality({ rows }: { rows: Row[] }) {
  if (!rows.length) {
    return (
      <Panel title="Propagation Hub Rankings · Graph Centrality Analysis">
        <p className="dim" style={{ fontSize: 13 }}>
          layer7_graph_centrality.csv not found — run layer7_cross_zone_hawkes.py
        </p>
      </Panel>
    );
  }

  const parsed = [...rows]
    .map((r) => ({
      zone: r.zone ?? "",
      hub_score: toNum(r.hub_score),
      role: (r.propagation_role ?? "moderate").trim(),
      pagerank: toNum(r.pagerank),
      pagerank_norm: toNum(r.pagerank_normalized),
      betweenness: toNum(r.betweenness_centrality),
      betweenness_norm: toNum(r.betweenness_normalized),
      eigenvector: toNum(r.eigenvector_centrality),
      eigenvector_norm: toNum(r.eigenvector_normalized),
    }))
    .filter((r) => r.zone)
    .sort((a, b) => b.hub_score - a.hub_score);

  const topHub = parsed[0];
  const topRelay = [...parsed].sort((a, b) => b.betweenness - a.betweenness)[0];
  const topReceiver = [...parsed].sort((a, b) => b.pagerank - a.pagerank)[0];

  const chartData = parsed.map((r) => ({
    name: r.zone,
    pagerank: r.pagerank_norm,
    betweenness: r.betweenness_norm,
    eigenvector: r.eigenvector_norm,
  }));

  const rawByZone = Object.fromEntries(
    parsed.map((r) => [
      r.zone,
      { pagerank: r.pagerank, betweenness: r.betweenness, eigenvector: r.eigenvector },
    ])
  );

  return (
    <div className="stack gap-6" style={{ marginTop: 24 }}>
      <Panel
        title="Propagation Hub Rankings · Graph Centrality Analysis"
        meta="PageRank · Betweenness · Eigenvector computed on cross-excitation graph α(u→v)"
      >
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Zone</th>
                <th>Hub score</th>
                <th>Role</th>
                <th className="num">PageRank</th>
                <th className="num">Betweenness</th>
                <th className="num">Eigenvector</th>
              </tr>
            </thead>
            <tbody>
              {parsed.map((r, i) => (
                <tr
                  key={r.zone}
                  style={i === 0 ? { borderLeft: "3px solid #E8A53D" } : undefined}
                >
                  <td>{r.zone}</td>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span className="mono">{r.hub_score.toFixed(3)}</span>
                      <div
                        style={{
                          width: 60,
                          height: 6,
                          borderRadius: 3,
                          background: "rgba(255,255,255,0.06)",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            width: `${Math.min(100, r.hub_score * 100)}%`,
                            height: "100%",
                            background: "#4ECDC4",
                            borderRadius: 3,
                          }}
                        />
                      </div>
                    </div>
                  </td>
                  <td>
                    <RoleBadge role={r.role} />
                  </td>
                  <td className="num">{r.pagerank.toFixed(4)}</td>
                  <td className="num">{r.betweenness.toFixed(4)}</td>
                  <td className="num">{r.eigenvector.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel
        title="Centrality Profile by Zone"
        meta="Normalized to [0,1] for comparison · raw values in table above"
      >
        <GroupedHBar
          data={chartData}
          keys={[
            { key: "pagerank", label: "PageRank", color: "#4ECDC4" },
            { key: "betweenness", label: "Betweenness", color: "#E8A53D" },
            { key: "eigenvector", label: "Eigenvector", color: "#6B4FA0" },
          ]}
          colors={["#4ECDC4", "#E8A53D", "#6B4FA0"]}
          height={320}
          xDomain={[0, 1]}
          rawByZone={rawByZone}
        />
      </Panel>

      <div className="grid grid-3">
        <div className="kpi" style={{ position: "relative" }}>
          <Graph
            size={18}
            weight="duotone"
            style={{
              position: "absolute",
              top: 12,
              right: 12,
              color: topHub.role === "critical_hub" ? "#B0413E" : "#4ECDC4",
            }}
            aria-hidden
          />
          <div className="kpi-label">Top Propagation Hub</div>
          <div
            className="kpi-value text"
            style={{ color: topHub.role === "critical_hub" ? "#B0413E" : "#4ECDC4" }}
          >
            {topHub.zone}
          </div>
          <div className="kpi-sub">Hub Score: {topHub.hub_score.toFixed(3)}</div>
        </div>
        <div className="kpi" style={{ position: "relative" }}>
          <TreeStructure
            size={18}
            weight="duotone"
            style={{ position: "absolute", top: 12, right: 12, color: "#E8A53D" }}
            aria-hidden
          />
          <div className="kpi-label">Critical Relay Zone</div>
          <div className="kpi-value text" style={{ color: "#E8A53D" }}>
            {topRelay.zone}
          </div>
          <div className="kpi-sub">BC: {topRelay.betweenness.toFixed(4)}</div>
        </div>
        <div className="kpi" style={{ position: "relative" }}>
          <Broadcast
            size={18}
            weight="duotone"
            style={{ position: "absolute", top: 12, right: 12, color: "#4ECDC4" }}
            aria-hidden
          />
          <div className="kpi-label">Highest Risk Receiver</div>
          <div className="kpi-value text" style={{ color: "#4ECDC4" }}>
            {topReceiver.zone}
          </div>
          <div className="kpi-sub">PR: {topReceiver.pagerank.toFixed(4)}</div>
        </div>
      </div>

      <Note>
        <span style={{ display: "inline-flex", gap: 8, alignItems: "flex-start" }}>
          <Graph size={16} weight="duotone" style={{ flexShrink: 0, marginTop: 2, color: "var(--accent)" }} />
          <span>
            Graph centrality computed on the cross-excitation matrix α(u→v) from the marked Hawkes fit
            (Part A). No new model trained — PageRank, Betweenness, and Eigenvector Centrality are
            deterministic functions of the existing alpha matrix. Betweenness uses inverted alpha as path
            cost (stronger edges = shorter effective distance). Hub Score = 0.40·PR_norm + 0.35·BC_norm +
            0.25·EC_norm.
          </span>
        </span>
      </Note>
    </div>
  );
}
