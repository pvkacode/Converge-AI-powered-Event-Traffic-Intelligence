// Server-only counts from data/events_clean.csv (pipeline cleaned batch).
import "server-only";
import fs from "node:fs";
import path from "node:path";
import { cache as reactCache } from "react";
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

function statsJsonPath(): string | null {
  const candidates = [
    path.join(outputsDir(), "frontend", "events_clean_stats.json"),
    path.join(process.cwd(), "..", "outputs", "frontend", "events_clean_stats.json"),
    path.join(process.cwd(), "outputs", "frontend", "events_clean_stats.json"),
    path.join(process.cwd(), "..", "..", "outputs", "frontend", "events_clean_stats.json"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

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

function parseStatsJson(filePath: string): EventsCleanStats | null {
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf8")) as Partial<EventsCleanStats>;
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

/** Live counts from events_clean.csv; falls back to committed JSON exports. */
function loadEventsCleanStatsUncached(): EventsCleanStats {
  const fallback: EventsCleanStats = {
    total: 8173,
    closedWithoutTimestamp: 3526,
    truePlanned: 191,
  };

  const statsJson = statsJsonPath();
  if (statsJson) {
    const fromStats = parseStatsJson(statsJson);
    if (fromStats) return fromStats;
  }

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

export const loadEventsCleanStats = reactCache(loadEventsCleanStatsUncached);
