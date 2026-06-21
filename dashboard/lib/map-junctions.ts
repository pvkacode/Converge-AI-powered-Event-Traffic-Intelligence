// Server-only junction map data built from pipeline CSV exports (outputs/).
import "server-only";
import { tryLoadCsv } from "@/lib/csv";
import { toNum } from "@/lib/format";

export interface JunctionMapPoint {
  junction: string;
  lat: number;
  lng: number;
  weighted_intensity: number;
  z_score: number;
  p_sim: number;
  is_significant: boolean;
  obi_rank: number | null;
}

export interface JunctionMapData {
  points: JunctionMapPoint[];
  stats: { significant: number; total: number };
}

export interface WxDiversionRow {
  junction: string;
  route_label: string;
  diversion_path: string;
  diversion_corridor: string;
}

export interface WxMapData {
  points: JunctionMapPoint[];
  corridorJunction: Record<string, string>;
  diversions: WxDiversionRow[];
  coordsByJunction: Record<string, { lat: number; lng: number }>;
}

function parseSignificant(v: string | undefined): boolean {
  return ["true", "1", "yes"].includes(String(v ?? "").trim().toLowerCase());
}

function buildObiRankMap(): Map<string, number> {
  const ob = tryLoadCsv("frontend/operational_burden.csv");
  const rank = new Map<string, number>();
  if (!ob) return rank;
  const sorted = ob.rows
    .map((r) => ({
      junction: String(r.junction ?? "").trim(),
      obi: toNum(r.operational_burden_index),
    }))
    .filter((r) => r.junction && !Number.isNaN(r.obi))
    .sort((a, b) => b.obi - a.obi);
  sorted.forEach((r, i) => rank.set(r.junction, i + 1));
  return rank;
}

/** Geo + Gi* from layer2_hotspots.csv; OBI rank from operational_burden.csv */
export function buildJunctionMapData(): JunctionMapData | null {
  const geo = tryLoadCsv("layer2_hotspots.csv");
  if (!geo) return null;

  const obiRank = buildObiRankMap();
  const points: JunctionMapPoint[] = [];

  for (const row of geo.rows) {
    const lat = toNum(row.latitude);
    const lng = toNum(row.longitude);
    if (Number.isNaN(lat) || Number.isNaN(lng)) continue;
    const junction = String(row.junction ?? "").trim();
    if (!junction) continue;
    points.push({
      junction,
      lat,
      lng,
      weighted_intensity: toNum(row.weighted_intensity),
      z_score: toNum(row.z_score),
      p_sim: toNum(row.p_sim),
      is_significant: parseSignificant(row.is_significant),
      obi_rank: obiRank.get(junction) ?? null,
    });
  }

  if (!points.length) return null;
  return {
    points,
    stats: {
      significant: points.filter((p) => p.is_significant).length,
      total: points.length,
    },
  };
}

function buildCorridorJunctionMap(): Record<string, string> {
  const rs = tryLoadCsv("frontend/risk_scores.csv");
  if (!rs) return {};
  const best: Record<string, { junction: string; score: number }> = {};
  for (const row of rs.rows) {
    const corridor = String(row.corridor ?? "").trim();
    const junction = String(row.junction ?? "").trim();
    if (!corridor || !junction) continue;
    const score = toNum(row.survival_risk_score);
    if (Number.isNaN(score)) continue;
    if (!best[corridor] || score > best[corridor].score) {
      best[corridor] = { junction, score };
    }
  }
  return Object.fromEntries(Object.entries(best).map(([c, v]) => [c, v.junction]));
}

export function buildWxMapData(): WxMapData | null {
  const base = buildJunctionMapData();
  if (!base) return null;

  const coordsByJunction: Record<string, { lat: number; lng: number }> = {};
  for (const p of base.points) {
    coordsByJunction[p.junction] = { lat: p.lat, lng: p.lng };
  }

  const diversions: WxDiversionRow[] = [];
  const div = tryLoadCsv("layer3_diversion_recommendations.csv");
  if (div) {
    for (const row of div.rows) {
      const junction = String(row.junction ?? "").trim();
      if (!junction) continue;
      diversions.push({
        junction,
        route_label: String(row.route_label ?? row.recommendation_label ?? "").trim(),
        diversion_path: String(row.diversion_path ?? "").trim(),
        diversion_corridor: String(row.diversion_corridor ?? "").trim(),
      });
    }
  }

  return {
    points: base.points,
    corridorJunction: buildCorridorJunctionMap(),
    diversions,
    coordsByJunction,
  };
}

export function topJunctionsByObi(data: JunctionMapData, n: number): JunctionMapPoint[] {
  return [...data.points]
    .filter((p) => p.obi_rank != null)
    .sort((a, b) => (a.obi_rank ?? 999) - (b.obi_rank ?? 999))
    .slice(0, n);
}
