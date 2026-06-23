"use client";

import { motion, useReducedMotion } from "motion/react";
import { toNum } from "@/lib/format";
import type { Row } from "@/lib/csv";

const CLASS_COLORS: Record<string, string> = {
  RESILIENT: "#28A745",
  MODERATE: "#E8A53D",
  VULNERABLE: "#FF8C00",
  CRITICAL: "#B0413E",
};

const itemVariants = {
  hidden: { opacity: 0, y: 12 },
  show: {
    opacity: 1,
    y: 0,
    transition: { type: "spring" as const, stiffness: 280, damping: 26 },
  },
};

const itemInstant = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0 } },
};

export function NriKpiCard({ row }: { row?: Row }) {
  const reduceMotion = useReducedMotion();
  const item = reduceMotion ? itemInstant : itemVariants;

  if (!row) {
    return (
      <motion.div variants={item}>
        <div className="kpi">
          <div className="kpi-label">Network resilience</div>
          <div className="kpi-value text">-</div>
          <div className="kpi-sub">Run network_resilience_index.py</div>
        </div>
      </motion.div>
    );
  }

  const nri = toNum(row.NRI);
  const nriClass = (row.NRI_class ?? "").trim().toUpperCase();
  const color = CLASS_COLORS[nriClass] ?? "var(--ink)";
  const rangeLow = toNum(row.NRI_range_low);
  const rangeHigh = toNum(row.NRI_range_high);
  const hNorm = toNum(row.H_normalized);
  const fNorm = toNum(row.F_normalized);
  const sNorm = toNum(row.S_normalized);

  return (
    <motion.div variants={item}>
      <div className="kpi">
        <div className="kpi-label">Network resilience</div>
        <div className="kpi-value mono" style={{ color }}>
          {Number.isNaN(nri) ? "-" : nri.toFixed(2)}
        </div>
        {nriClass ? (
          <span
            style={{
              display: "inline-block",
              marginTop: 6,
              padding: "2px 10px",
              borderRadius: 999,
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: "0.05em",
              background: color,
              color: "#FFFFFF",
            }}
          >
            {nriClass}
          </span>
        ) : null}
        {!Number.isNaN(rangeLow) && !Number.isNaN(rangeHigh) ? (
          <div
            className="dim"
            style={{ fontSize: 11, marginTop: 8, cursor: "help" }}
            title="Sensitivity under 5 weight variants (H/F/S weights varied ±0.10)"
          >
            Range: {rangeLow.toFixed(2)} – {rangeHigh.toFixed(2)}
          </div>
        ) : null}
        <div
          className="dim"
          style={{
            fontSize: 11,
            marginTop: 10,
            paddingTop: 8,
            borderTop: "1px solid var(--border)",
          }}
        >
          H={Number.isNaN(hNorm) ? "-" : hNorm.toFixed(2)} · F=
          {Number.isNaN(fNorm) ? "-" : fNorm.toFixed(2)} · S=
          {Number.isNaN(sNorm) ? "-" : sNorm.toFixed(2)}
        </div>
      </div>
    </motion.div>
  );
}
