// Server-only CSV loader with mtime-aware in-memory caching.
// Reads the real pipeline outputs from the sibling `outputs/` directory.
// This module never writes to disk and never touches anything under src/ or data/.
import "server-only";
import fs from "node:fs";
import path from "node:path";
import { cache as reactCache } from "react";
import Papa from "papaparse";

export type Row = Record<string, string>;

export interface ParsedFile {
  columns: string[];
  rows: Row[];
}

interface CacheEntry {
  mtimeMs: number;
  data: ParsedFile;
}

// Resolve the outputs directory robustly whether the app runs from the
// dashboard/ folder (cwd = dashboard) or from the repo root.
let cachedOutputsDir: string | null = null;
export function outputsDir(): string {
  if (cachedOutputsDir) return cachedOutputsDir;
  const envDir = process.env.OUTPUTS_DIR?.trim();
  if (envDir && fs.existsSync(path.join(envDir, "frontend"))) {
    cachedOutputsDir = path.resolve(envDir);
    return cachedOutputsDir;
  }
  const candidates = [
    path.join(process.cwd(), "outputs"),
    path.join(process.cwd(), "..", "outputs"),
    path.join(process.cwd(), "..", "..", "outputs"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, "frontend"))) {
      cachedOutputsDir = c;
      return c;
    }
  }
  cachedOutputsDir = candidates[1];
  return cachedOutputsDir;
}

const csvStore = new Map<string, CacheEntry>();

export function fileExists(relPath: string): boolean {
  return fs.existsSync(path.join(outputsDir(), relPath));
}

function parseCsvText(text: string): ParsedFile {
  const parsed = Papa.parse<Row>(text, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false,
  });
  const columns = (parsed.meta.fields ?? []).filter((f) => f != null && f !== "");
  const rows = (parsed.data as Row[]).filter((r) => r && Object.keys(r).length > 0);
  return { columns, rows };
}

function loadCsvUncached(relPath: string): ParsedFile {
  const abs = path.join(outputsDir(), relPath);
  const stat = fs.statSync(abs);
  const hit = csvStore.get(abs);
  if (hit && hit.mtimeMs === stat.mtimeMs) return hit.data;

  const data = parseCsvText(fs.readFileSync(abs, "utf8"));
  csvStore.set(abs, { mtimeMs: stat.mtimeMs, data });
  return data;
}

// Dedupe reads within a single render; csvStore retains parsed data across navigations.
export const loadCsv = reactCache(loadCsvUncached);

export function tryLoadCsv(relPath: string): ParsedFile | null {
  if (!fileExists(relPath)) return null;
  try {
    return loadCsv(relPath);
  } catch {
    return null;
  }
}

const textStore = new Map<string, { mtimeMs: number; text: string }>();

export function tryReadText(relPath: string): string | null {
  const abs = path.join(outputsDir(), relPath);
  if (!fs.existsSync(abs)) return null;
  try {
    const stat = fs.statSync(abs);
    const hit = textStore.get(abs);
    if (hit && hit.mtimeMs === stat.mtimeMs) return hit.text;
    const text = fs.readFileSync(abs, "utf8");
    textStore.set(abs, { mtimeMs: stat.mtimeMs, text });
    return text;
  } catch {
    return null;
  }
}
