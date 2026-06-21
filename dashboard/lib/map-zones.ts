// Server-only zone spillover map data from outputs/.
import "server-only";
import { tryLoadCsv } from "@/lib/csv";
import { toNum } from "@/lib/format";

export interface ZoneMapCircle {
  zone: string;
  center: [number, number];
  radius: number;
  ssc: number;
  half_life_hours: number;
}

export const ZONE_GEOMETRY: Record<string, { center: [number, number]; radius: number }> = {
  "Central Zone 1": { center: [12.9716, 77.5946], radius: 3000 },
  "Central Zone 2": { center: [12.9352, 77.6245], radius: 2800 },
  "North Zone 1": { center: [13.0827, 77.5877], radius: 3500 },
  "North Zone 2": { center: [13.0358, 77.597], radius: 3000 },
  "East Zone 1": { center: [12.9784, 77.6408], radius: 2800 },
  "East Zone 2": { center: [12.9539, 77.724], radius: 3200 },
  "South Zone 1": { center: [12.8742, 77.6135], radius: 3000 },
  "South Zone 2": { center: [12.8406, 77.6802], radius: 2800 },
  "West Zone 1": { center: [12.9698, 77.512], radius: 3200 },
  "West Zone 2": { center: [12.9166, 77.4849], radius: 2800 },
};

export function buildZoneMapData(): ZoneMapCircle[] | null {
  const spill = tryLoadCsv("layer7_spillover_centrality.csv");
  if (!spill) return null;

  const sscByZone = new Map<string, { ssc: number; half_life_hours: number }>();
  for (const row of spill.rows) {
    const zone = String(row.zone ?? "").trim();
    if (!zone) continue;
    sscByZone.set(zone, {
      ssc: toNum(row.SSC_centrality),
      half_life_hours: toNum(row.half_life_hours),
    });
  }

  const circles: ZoneMapCircle[] = [];
  for (const [zone, geom] of Object.entries(ZONE_GEOMETRY)) {
    const meta = sscByZone.get(zone);
    circles.push({
      zone,
      center: geom.center,
      radius: geom.radius,
      ssc: meta?.ssc ?? 0,
      half_life_hours: meta?.half_life_hours ?? 0.53,
    });
  }
  return circles.length ? circles : null;
}
