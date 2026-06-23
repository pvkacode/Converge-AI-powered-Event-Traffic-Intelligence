// Server-only counts from data/events_clean.csv (pipeline cleaned batch).
import "server-only";
import fs from "node:fs";
import path from "node:path";
import Papa from "papaparse";
import { outputsDir } from "./csv";

export interface EventsCleanStats {
  total: number;
  closedWithoutTimestamp: number;
  truePlanned: number;
}

let cached: { mtimeMs: number; stats: EventsCleanStats } | null = null;

function dataPath(): string | null {
  const envDir = process.env.DATA_DIR?.trim();
  if (envDir) {
    const p = path.join(envDir, "events_clean.csv");
    if (fs.existsSync(p)) return p;
  }
  const candidates = [
    path.join(process.cwd(), "..", "data", "events_clean.csv"),
    path.join(process.cwd(), "data", "events_clean.csv"),
    path.join(process.cwd(), "..", "..", "data", "events_clean.csv"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function truthy(v: string | undefined): boolean {
  return ["true", "1", "yes"].includes(String(v ?? "").trim().toLowerCase());
}

// Committed extract from data/events_clean.csv (outputs/frontend/kpi-summary.json),
// generated once and checked into the repo so the deployed app — which never has
// access to the gitignored data/ directory — doesn't silently fall back to the
// hardcoded literal below. See KPI_SOURCE.md for provenance.
function loadKpiSummaryFromJson(): EventsCleanStats | null {
  try {
    const abs = path.join(outputsDir(), "frontend", "kpi-summary.json");
    if (!fs.existsSync(abs)) return null;
    const parsed = JSON.parse(fs.readFileSync(abs, "utf8")) as Partial<EventsCleanStats>;
    if (
      typeof parsed.total === "number" &&
      typeof parsed.closedWithoutTimestamp === "number" &&
      typeof parsed.truePlanned === "number"
    ) {
      return {
        total: parsed.total,
        closedWithoutTimestamp: parsed.closedWithoutTimestamp,
        truePlanned: parsed.truePlanned,
      };
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Counts behind the overview KPIs, in priority order:
 *   1. Live recompute from data/events_clean.csv (most accurate, dev-only —
 *      data/ is gitignored and not deployed).
 *   2. Committed outputs/frontend/kpi-summary.json extract (deployed source of truth).
 *   3. Hardcoded literal (last-ditch fallback so this never renders blank).
 */
export function loadEventsCleanStats(): EventsCleanStats {
  const fallback: EventsCleanStats = {
    total: 8173,
    closedWithoutTimestamp: 3526,
    truePlanned: 191,
  };

  const abs = dataPath();
  if (!abs) return loadKpiSummaryFromJson() ?? fallback;

  try {
    const stat = fs.statSync(abs);
    if (cached && cached.mtimeMs === stat.mtimeMs) return cached.stats;

    const text = fs.readFileSync(abs, "utf8");
    const parsed = Papa.parse<Record<string, string>>(text, {
      header: true,
      skipEmptyLines: "greedy",
      dynamicTyping: false,
    });
    const rows = (parsed.data ?? []).filter((r) => r && Object.keys(r).length > 0);

    let closedWithoutTimestamp = 0;
    let truePlanned = 0;
    for (const row of rows) {
      const status = String(row.status ?? "").trim().toLowerCase();
      if (status === "closed" && truthy(row.is_censored)) closedWithoutTimestamp += 1;
      if (truthy(row.is_true_planned_event)) truePlanned += 1;
    }

    const stats: EventsCleanStats = {
      total: rows.length,
      closedWithoutTimestamp,
      truePlanned,
    };
    cached = { mtimeMs: stat.mtimeMs, stats };
    return stats;
  } catch {
    return loadKpiSummaryFromJson() ?? fallback;
  }
}
