"use client";

import nextDynamic from "next/dynamic";
import { MapPlaceholder } from "./map-ui";
import type { JunctionMapPoint } from "@/lib/map-junctions";
import type { ZoneMapCircle, ZoneEdge } from "@/lib/map-zones";

const L2HotspotMapInner = nextDynamic(() => import("./L2HotspotMap"), {
  ssr: false,
  loading: () => <MapPlaceholder height={480} message="Loading map…" />,
});

const OverviewMiniMapInner = nextDynamic(() => import("./OverviewMiniMap"), {
  ssr: false,
  loading: () => <MapPlaceholder height={260} message="Loading map…" />,
});

const SpilloverZoneMapInner = nextDynamic(() => import("./SpilloverZoneMap"), {
  ssr: false,
  loading: () => <MapPlaceholder height={400} message="Loading map…" />,
});

export function L2HotspotMap(props: {
  points: JunctionMapPoint[];
  stats: { significant: number; total: number };
}) {
  return <L2HotspotMapInner {...props} />;
}

export function OverviewMiniMap(props: {
  points: JunctionMapPoint[];
  stats: { significant: number; total: number };
}) {
  return <OverviewMiniMapInner {...props} />;
}

export function SpilloverZoneMap(props: { zones: ZoneMapCircle[]; edges?: ZoneEdge[] }) {
  return <SpilloverZoneMapInner {...props} />;
}
