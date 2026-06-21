"use client";
import { motion, useReducedMotion } from "motion/react";
import { Kpi } from "@/components/ui";
import { fmtNum, fmtMinutes } from "@/lib/format";

export interface HeroKpisData {
  hsTotal: number;
  giHot: number;
  safe: number;
  overall: string;
  criticalChecks: number;
  totalChecks: number;
  alertTotal: number;
  sevSummary: string;
  spillZone: string;
  spillCentrality: number;
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09 } },
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

export function HeroKpis({
  hsTotal,
  giHot,
  safe,
  overall,
  criticalChecks,
  totalChecks,
  alertTotal,
  sevSummary,
  spillZone,
  spillCentrality,
}: HeroKpisData) {
  const reduceMotion = useReducedMotion();
  const item = reduceMotion ? itemInstant : itemVariants;

  const healthColor = overall.toLowerCase().includes("crit")
    ? "var(--critical)"
    : overall.toLowerCase().includes("warn")
    ? "var(--warning)"
    : "var(--ok)";

  return (
    <motion.div
      className="grid grid-5"
      style={{ marginBottom: 24 }}
      variants={container}
      initial="hidden"
      animate="show"
    >
      <motion.div variants={item}>
        <Kpi
          label="Ranked hotspots"
          value={fmtNum(hsTotal)}
          sub={`${giHot} with positive Gi* clustering`}
        />
      </motion.div>

      <motion.div variants={item}>
        <Kpi
          label="Median safe duration"
          value={Number.isNaN(safe) ? "-" : fmtMinutes(safe)}
          sub="P50, Layer 4.5 sanitized"
        />
      </motion.div>

      <motion.div variants={item}>
        <Kpi
          label="Model health"
          isText
          value={<span style={{ color: healthColor }}>{overall}</span>}
          sub={`${criticalChecks} of ${totalChecks} checks critical`}
        />
      </motion.div>

      <motion.div variants={item}>
        <Kpi
          label="Active alerts"
          value={fmtNum(alertTotal)}
          sub={sevSummary || "none"}
        />
      </motion.div>

      <motion.div variants={item}>
        <Kpi
          label="Top spillover zone"
          isText
          accent
          value={spillZone || "-"}
          sub={
            !Number.isNaN(spillCentrality)
              ? `SSC centrality ${fmtNum(spillCentrality)}`
              : undefined
          }
        />
      </motion.div>
    </motion.div>
  );
}
