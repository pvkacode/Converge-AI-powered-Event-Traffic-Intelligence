// Shared severity vocabulary for Layer 6 monitoring tables (retrain triggers,
// active alerts). Kept separate from lib/badges.ts because that module's
// generic badgeVariant() maps "info" to a plain neutral badge - here we want
// info to read as deliberately muted, not "uncategorised".
import type { BadgeVariant } from "@/lib/badges";

export type Severity = "critical" | "moderate" | "info" | string;

const RANK: Record<string, number> = { critical: 0, moderate: 1, info: 2 };

// Lower rank sorts first. Unknown severities sort last, after "info".
export function severityRank(severity: string): number {
  return RANK[severity.trim().toLowerCase()] ?? 3;
}

export function severityVariant(severity: string): BadgeVariant {
  const s = severity.trim().toLowerCase();
  if (s === "critical") return "critical";
  if (s === "moderate") return "warning";
  if (s === "info") return "muted";
  return "neutral";
}
