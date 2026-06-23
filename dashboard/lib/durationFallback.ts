// Display-only helper for Layer 1's duration explorer. The pipeline
// (src/layer1_survival.py) only ever writes a (cause, corridor) pair into
// duration_lookup.csv once it has cleared MIN_GROUP_SIZE resolved incidents.
// Sparse pairs are silently dropped from that file and, at serve time,
// lookup_expected_duration() falls back to the cause-only quantiles in
// layer1_survival_fallback.csv instead. Neither CSV carries a flag saying
// "this value is a fallback" - we derive that here, purely for display,
// by checking whether a (cause, corridor) row exists in the corridor table.
import { toNum } from "@/lib/format";
import type { DurRow } from "@/components/DurationExplorer";

export interface FallbackRow {
  event_cause: string;
  n: string;
  p50_min: string;
  p80_min: string;
  p95_min: string;
}

export interface DurRowWithFallback extends DurRow {
  isFallback: boolean;
  // Below this corridor-specific sample size, a true (non-fallback) estimate
  // is still thin enough to flag as low-confidence. 2x MIN_GROUP_SIZE (15 in
  // src/layer1_survival.py) is a reasonable display-only cutoff.
  isLowConfidence: boolean;
}

const LOW_CONFIDENCE_N_THRESHOLD = 30;

// All corridors the lookup table has ever resolved for any cause - the set a
// user can sensibly pick from in the explorer, regardless of which cause is
// selected (the live pipeline accepts any cause/corridor pair at request time).
export function allCorridors(corridorRows: DurRow[]): string[] {
  return Array.from(new Set(corridorRows.map((r) => r.corridor))).sort();
}

// Every cause that has either a resolved corridor row or a cause-only fallback.
export function allCauses(corridorRows: DurRow[], fallbackRows: FallbackRow[]): string[] {
  const causes = new Set<string>();
  corridorRows.forEach((r) => causes.add(r.event_cause));
  fallbackRows.forEach((r) => causes.add(r.event_cause));
  return Array.from(causes).sort();
}

// Resolve what the explorer should show for a given (cause, corridor) pick:
// the real corridor-specific row if one exists, otherwise the cause-level
// fallback (re-shaped to look like a DurRow), otherwise null.
export function resolveDurationRow(
  cause: string,
  corridor: string,
  corridorRows: DurRow[],
  fallbackRows: FallbackRow[]
): DurRowWithFallback | null {
  const exact = corridorRows.find((r) => r.event_cause === cause && r.corridor === corridor);
  if (exact) {
    return {
      ...exact,
      isFallback: false,
      isLowConfidence: toNum(exact.n) < LOW_CONFIDENCE_N_THRESHOLD,
    };
  }

  const fallback = fallbackRows.find((r) => r.event_cause === cause);
  if (fallback) {
    return {
      event_cause: cause,
      corridor,
      n: fallback.n,
      p50_min: fallback.p50_min,
      p80_min: fallback.p80_min,
      p95_min: fallback.p95_min,
      isFallback: true,
      isLowConfidence: false,
    };
  }

  return null;
}
