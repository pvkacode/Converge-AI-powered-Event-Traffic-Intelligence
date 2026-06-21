// Map: Leaflet + CartoDB dark tiles. Data from outputs/frontend/ only.
// No external API calls beyond map tile loading.
"use client";

import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, Circle, Polyline, Tooltip } from "react-leaflet";
import type { ZoneMapCircle, ZoneEdge } from "@/lib/map-zones";
import { MAP_BORDER, BENGALURU_CENTER } from "./map-ui";

// Neutral slate-blue — distinct from the semantic red/amber/teal on circles.
const EDGE_COLOR = "#94A3B8";

function zoneStyle(ssc: number) {
  if (ssc >= 3.0) {
    return { color: "#DC2626", fillColor: "#DC2626", fillOpacity: 0.3, weight: 2 };
  }
  if (ssc >= 2.0) {
    return { color: "#D97706", fillColor: "#D97706", fillOpacity: 0.25, weight: 2 };
  }
  return { color: "#0D9488", fillColor: "#0D9488", fillOpacity: 0.2, weight: 2 };
}

function edgeStyle(alpha: number, maxAlpha: number) {
  const norm = maxAlpha > 0 ? alpha / maxAlpha : 0;
  return {
    color: EDGE_COLOR,
    weight: Math.max(1, norm * 5),
    opacity: 0.25 + norm * 0.6,
  };
}

export default function SpilloverZoneMap({
  zones,
  edges = [],
}: {
  zones: ZoneMapCircle[];
  edges?: ZoneEdge[];
}) {
  if (!zones.length) {
    return (
      <div style={{ height: 400, width: "100%", ...MAP_BORDER, background: "#1E293B", display: "flex", alignItems: "center", justifyContent: "center", color: "#64748B" }}>
        Map unavailable — check outputs/frontend/ exports
      </div>
    );
  }

  const maxAlpha = edges.reduce((m, e) => Math.max(m, e.alpha), 0);

  return (
    <div>
      <MapContainer
        center={BENGALURU_CENTER}
        zoom={11}
        style={{ height: "400px", width: "100%", ...MAP_BORDER }}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />
        {edges.map((e, i) => (
          <Polyline
            key={i}
            positions={[e.from, e.to]}
            pathOptions={edgeStyle(e.alpha, maxAlpha)}
          >
            <Tooltip sticky>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: "#1E293B", minWidth: 200 }}>
                <b>{e.source} → {e.dest}</b>
                <br />
                alpha = {e.alpha.toFixed(3)}
                <br />
                95% CI [{e.ci_lower.toFixed(3)}, {e.ci_upper.toFixed(3)}]
                <br />
                half-life = {e.half_life_hours.toFixed(2)}h
              </div>
            </Tooltip>
          </Polyline>
        ))}
        {zones.map((z) => (
          <Circle
            key={z.zone}
            center={z.center}
            radius={z.radius}
            pathOptions={zoneStyle(z.ssc)}
          >
            <Tooltip>
              <div style={{ fontFamily: "monospace", fontSize: 12, color: "#1E293B" }}>
                <b>{z.zone}</b>
                <br />
                SSC: {z.ssc.toFixed(2)}
                <br />
                <i>
                  Excitation half-life: {z.half_life_hours.toFixed(2)} hrs (~
                  {Math.round(z.half_life_hours * 60)} min)
                </i>
              </div>
            </Tooltip>
          </Circle>
        ))}
      </MapContainer>
      <p style={{ color: "#64748B", fontSize: 12, fontStyle: "italic", marginTop: 8, marginBottom: 0 }}>
        Zone boundaries are approximate catchment areas, not administrative boundaries.
        {edges.length > 0 && ` Lines show ${edges.length} statistically significant spillover pairs (CI lower > 0); line weight and opacity scale to alpha coefficient.`}
        {" "}LRT stat=534.9, df=34, p≈2.5×10⁻⁹¹
      </p>
    </div>
  );
}
