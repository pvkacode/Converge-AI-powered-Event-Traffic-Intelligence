// Server-only derived-column augmenters. These add transparent, clearly-labelled
// derived fields to a parsed dataset. They never invent values: derived tiers are
// computed from the real distribution of an existing column and labelled as such
// in the UI.
import "server-only";
import type { ParsedFile, Row } from "../csv";
import { toNum } from "../format";

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

// Adds `risk_tier` to risk_scores by binning survival_risk_score on its own
// empirical quartiles/tail. Tiers: Low (<p50), Moderate (<p80), High (<p95),
// Critical (>=p95). The UI states explicitly that this tier is derived.
function riskTier(parsed: ParsedFile): ParsedFile {
  const col = "survival_risk_score";
  if (!parsed.columns.includes(col)) return parsed;
  const vals = parsed.rows
    .map((r) => toNum(r[col]))
    .filter((n) => !Number.isNaN(n))
    .sort((a, b) => a - b);
  const p50 = quantile(vals, 0.5);
  const p80 = quantile(vals, 0.8);
  const p95 = quantile(vals, 0.95);
  const rows: Row[] = parsed.rows.map((r) => {
    const v = toNum(r[col]);
    let tier = "Low";
    if (!Number.isNaN(v)) {
      if (v >= p95) tier = "Critical";
      else if (v >= p80) tier = "High";
      else if (v >= p50) tier = "Moderate";
    }
    return { ...r, risk_tier: tier };
  });
  const columns = parsed.columns.includes("risk_tier")
    ? parsed.columns
    : [...parsed.columns, "risk_tier"];
  return { columns, rows };
}

const AUGMENTERS: Record<string, (p: ParsedFile) => ParsedFile> = {
  risk_tier: riskTier,
};

// Cache augmented results keyed by augmenter id + a content signature (row count
// is a cheap proxy; the underlying loadCsv is already mtime-cached).
const augCache = new Map<string, { sig: number; data: ParsedFile }>();

export function applyAugment(
  id: string | undefined,
  parsed: ParsedFile
): ParsedFile {
  if (!id) return parsed;
  const fn = AUGMENTERS[id];
  if (!fn) return parsed;
  const sig = parsed.rows.length;
  const cached = augCache.get(id);
  if (cached && cached.sig === sig) return cached.data;
  const data = fn(parsed);
  augCache.set(id, { sig, data });
  return data;
}
