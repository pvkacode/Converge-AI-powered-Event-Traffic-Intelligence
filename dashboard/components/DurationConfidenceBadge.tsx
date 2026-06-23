// Presentational badge for Layer 1 duration estimates. Mutually exclusive
// with low-confidence display: a cause-level fallback always wins, since it
// is the stronger caveat (no corridor-specific signal at all).
import { Badge } from "./ui";

export type FallbackLevel = "cause" | "global";

const FALLBACK_TOOLTIP =
  "This corridor had insufficient resolved incidents (below MIN_GROUP_SIZE) for this cause, so a cause-level prior was used instead.";
const LOW_CONFIDENCE_TOOLTIP =
  "This corridor-specific estimate is based on a small sample of resolved incidents.";

export function DurationConfidenceBadge({
  isFallback,
  fallbackLevel = "cause",
  isLowConfidence = false,
}: {
  isFallback: boolean;
  fallbackLevel?: FallbackLevel;
  isLowConfidence?: boolean;
}) {
  if (isFallback) {
    const label =
      fallbackLevel === "global"
        ? "Global estimate — no cause history"
        : "Cause-level estimate — sparse corridor history";
    return (
      <span title={FALLBACK_TOOLTIP}>
        <Badge variant="muted">{label}</Badge>
      </span>
    );
  }

  if (isLowConfidence) {
    return (
      <span title={LOW_CONFIDENCE_TOOLTIP}>
        <Badge variant="muted">Low-confidence estimate</Badge>
      </span>
    );
  }

  return null;
}
