// Pure formatting helpers shared by server and client. No side effects.

// Parse a cell into a number, or NaN if it is not numeric.
export function toNum(v: unknown): number {
  if (v == null) return NaN;
  const s = String(v).trim();
  if (s === "" || s.toLowerCase() === "nan" || s.toLowerCase() === "none") return NaN;
  const n = Number(s);
  return Number.isFinite(n) ? n : NaN;
}

export function isNumeric(v: unknown): boolean {
  return !Number.isNaN(toNum(v));
}

// Smart display formatting for a numeric cell. Keeps integers clean, gives
// small magnitudes more precision, and groups thousands for large values.
export function fmtNum(v: unknown): string {
  const n = toNum(v);
  if (Number.isNaN(n)) return v == null ? "" : String(v);
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (Number.isInteger(n) && abs < 1e7) return n.toLocaleString("en-US");
  if (abs >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (abs >= 1) return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
  if (abs >= 0.01) return n.toFixed(3);
  return n.toExponential(2);
}

// Compact metric formatting for KPI tiles (e.g. 13.5k, 1.9M).
export function fmtCompact(v: unknown): string {
  const n = toNum(v);
  if (Number.isNaN(n)) return v == null ? "" : String(v);
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

// Render a raw duration in minutes as a readable label.
export function fmtMinutes(v: unknown): string {
  const n = toNum(v);
  if (Number.isNaN(n)) return v == null ? "" : String(v);
  if (n < 90) return `${n.toFixed(0)} min`;
  const h = n / 60;
  if (h < 48) return `${h.toFixed(1)} h`;
  const d = h / 24;
  return `${d.toFixed(1)} d`;
}

// Turn a snake_case / camelCase column key into a human label.
export function humanize(key: string): string {
  if (!key) return "";
  const overrides: Record<string, string> = {
    p50_min: "P50 (min)",
    p80_min: "P80 (min)",
    p95_min: "P95 (min)",
    n: "Samples",
    sps: "SPS",
    nhi: "NHI",
    gi_star_h1: "Gi* h1",
    gi_star_h2: "Gi* h2",
    gi_star_h3: "Gi* h3",
    gi_star_h5: "Gi* h5",
    obi: "OBI",
    operational_burden_index: "OBI",
    cvar: "CVaR",
    cvar_90: "CVaR 90",
    eri: "ERI",
    ssc_centrality: "SSC centrality",
    ssc_norm: "SSC (norm)",
    auc: "AUC",
    ece: "ECE",
    rmse: "RMSE",
    mae: "MAE",
    var_level: "VaR level",
    id: "ID",
    event_id: "Event ID",
    alert_id: "Alert ID",
    eri_3h: "ERI 3h",
    eri_6h: "ERI 6h",
    eri_9h: "ERI 9h",
  };
  const lower = key.toLowerCase();
  if (overrides[lower]) return overrides[lower];
  return key
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bP50\b/i, "P50")
    .replace(/\bCi\b/g, "CI")
    .replace(/\bUtc\b/g, "UTC")
    .replace(/\bProb\b/g, "Prob");
}

export function titleCaseValue(v: string): string {
  return v
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
