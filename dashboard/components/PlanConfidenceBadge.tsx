import { computePlanConfidence, type PlanConfidenceInput } from "@/lib/plan-confidence";
import { Badge } from "./ui";

const VARIANT = {
  High: "ok",
  Moderate: "warning",
  Low: "critical",
  Unknown: "muted",
} as const;

// Presentational only — all the level/reasons logic lives in lib/plan-confidence.ts.
// Tooltip is a plain `title` attribute (no tooltip library exists in this project).
export function PlanConfidenceBadge({ input }: { input: PlanConfidenceInput }) {
  const { level, reasons } = computePlanConfidence(input);
  const tooltip = `Plan confidence: ${level}\n${reasons.map((r) => `• ${r}`).join("\n")}`;

  return (
    <span title={tooltip}>
      <Badge variant={VARIANT[level]} dot={level !== "Unknown"}>
        Confidence: {level}
      </Badge>
    </span>
  );
}
