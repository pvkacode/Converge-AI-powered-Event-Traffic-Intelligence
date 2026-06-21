// Client-safe mapping from categorical cell values to palette-aware badge
// variants. All variants are defined as CSS classes in globals.css so the
// status colours stay harmonious with the warm palette in both themes.

export type BadgeVariant =
  | "critical"
  | "warning"
  | "ok"
  | "accent"
  | "neutral"
  | "muted";

// Normalise a value for matching.
function norm(v: string): string {
  return v.trim().toLowerCase();
}

// Map a value (optionally with column context) to a variant.
export function badgeVariant(value: string, column?: string): BadgeVariant {
  const v = norm(value);
  const col = column ? norm(column) : "";

  // Explicit status / severity vocabularies seen across the pipeline outputs.
  if (["critical", "emergency", "severe", "high_impact", "retrain_trigger", "escalating"].includes(v))
    return "critical";
  if (["warning", "warn", "moderate", "elevated", "medium", "watch", "drifting"].includes(v))
    return "warning";
  if (["ok", "healthy", "good", "stable", "normal", "low", "calm", "none", "pass", "nominal"].includes(v))
    return "ok";

  if (["strong", "high", "high_confidence"].includes(v)) return "accent";
  if (["weak"].includes(v)) return "warning";

  // Confidence bands.
  if (col.includes("confidence")) {
    if (v.includes("high") || v === "strong") return "accent";
    if (v.includes("med") || v === "moderate") return "warning";
    if (v.includes("low") || v === "weak") return "muted";
  }

  // Risk tiers (derived column).
  if (col.includes("tier") || col.includes("risk")) {
    if (v === "critical") return "critical";
    if (v === "high") return "warning";
    if (v === "moderate") return "accent";
    if (v === "low") return "ok";
  }

  // Layer 4 abstain decision (raw CSV is 0/1; UI may also show labels).
  if (col.includes("abstain")) {
    if (["1", "true", "yes", "abstained"].includes(v)) return "warning";
    if (["0", "false", "no", "recommend"].includes(v)) return "ok";
  }

  // Boolean-ish flags.
  if (["true", "1", "yes", "flagged", "activated"].includes(v)) return "warning";
  if (["false", "0", "no"].includes(v)) return "muted";

  return "neutral";
}

// Human label for overall health status used on KPI / monitoring surfaces.
export function healthVariant(status: string): BadgeVariant {
  const s = norm(status);
  if (s.includes("critical")) return "critical";
  if (s.includes("warn")) return "warning";
  if (s.includes("ok") || s.includes("healthy")) return "ok";
  return "neutral";
}
