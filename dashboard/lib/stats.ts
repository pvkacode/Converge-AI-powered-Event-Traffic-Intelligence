// Pure numeric helpers used by server pages to compute KPIs and chart series
// from the parsed CSV rows. No I/O.
import { toNum } from "./format";
import type { Row } from "./csv";

export function nums(rows: Row[], col: string): number[] {
  return rows.map((r) => toNum(r[col])).filter((n) => !Number.isNaN(n));
}

export function median(arr: number[]): number {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function quantile(arr: number[], q: number): number {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return s[base + 1] !== undefined ? s[base] + rest * (s[base + 1] - s[base]) : s[base];
}

export function mean(arr: number[]): number {
  if (!arr.length) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function countWhere(rows: Row[], pred: (r: Row) => boolean): number {
  let c = 0;
  for (const r of rows) if (pred(r)) c++;
  return c;
}

export function valueCounts(rows: Row[], col: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const v = (r[col] ?? "").trim();
    if (v === "") continue;
    out[v] = (out[v] ?? 0) + 1;
  }
  return out;
}

// Top N rows by a numeric column, returned as {name, value}.
export function topBy(
  rows: Row[],
  nameCol: string,
  valCol: string,
  n: number,
  dir: "asc" | "desc" = "desc"
): { name: string; value: number }[] {
  const mapped = rows
    .map((r) => ({ name: String(r[nameCol] ?? ""), value: toNum(r[valCol]) }))
    .filter((d) => !Number.isNaN(d.value));
  mapped.sort((a, b) => (dir === "desc" ? b.value - a.value : a.value - b.value));
  return mapped.slice(0, n);
}
