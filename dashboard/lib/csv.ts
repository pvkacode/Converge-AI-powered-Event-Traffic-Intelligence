// Server-only CSV loader with mtime-aware in-memory caching.
// Reads the real pipeline outputs from the sibling `outputs/` directory.
// This module never writes to disk and never touches anything under src/ or data/.
import "server-only";
import fs from "node:fs";
import path from "node:path";
import Papa from "papaparse";

export type Row = Record<string, string>;

export interface ParsedFile {
  columns: string[];
  rows: Row[];
}

// Resolve the outputs directory robustly whether the app runs from the
// dashboard/ folder (cwd = dashboard) or from the repo root.
let cachedOutputsDir: string | null = null;
export function outputsDir(): string {
  if (cachedOutputsDir) return cachedOutputsDir;
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
  // Fall back to the most likely location even if not found, so callers get a
  // clear ENOENT pointing at the expected path rather than a vague error.
  cachedOutputsDir = candidates[1];
  return cachedOutputsDir;
}

interface CacheEntry {
  mtimeMs: number;
  data: ParsedFile;
}
const cache = new Map<string, CacheEntry>();

export function fileExists(relPath: string): boolean {
  return fs.existsSync(path.join(outputsDir(), relPath));
}

// Load + parse a CSV by its path relative to outputs/. Cached by mtime.
export function loadCsv(relPath: string): ParsedFile {
  const abs = path.join(outputsDir(), relPath);
  const stat = fs.statSync(abs); // throws ENOENT if missing - callers guard with fileExists
  const cached = cache.get(abs);
  if (cached && cached.mtimeMs === stat.mtimeMs) return cached.data;

  const text = fs.readFileSync(abs, "utf8");
  const parsed = Papa.parse<Row>(text, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false, // keep strings; numeric coercion handled at sort/format time
  });
  const columns = (parsed.meta.fields ?? []).filter((f) => f != null && f !== "");
  const rows = (parsed.data as Row[]).filter((r) => r && Object.keys(r).length > 0);
  const data: ParsedFile = { columns, rows };
  cache.set(abs, { mtimeMs: stat.mtimeMs, data });
  return data;
}

// Safe loader: returns null instead of throwing when the file is absent.
export function tryLoadCsv(relPath: string): ParsedFile | null {
  if (!fileExists(relPath)) return null;
  try {
    return loadCsv(relPath);
  } catch {
    return null;
  }
}

// Read a small text artifact (e.g. *_summary.txt) if present.
export function tryReadText(relPath: string): string | null {
  const abs = path.join(outputsDir(), relPath);
  if (!fs.existsSync(abs)) return null;
  try {
    return fs.readFileSync(abs, "utf8");
  } catch {
    return null;
  }
}
