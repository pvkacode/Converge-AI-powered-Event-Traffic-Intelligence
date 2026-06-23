export function formatIncidents(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString("en-US") : "—";
}

export function formatHotspotsRatio(sig: number, total: number): string {
  if (!Number.isFinite(sig) || !total) return "—";
  return `${sig} / ${total}`;
}
