// Map: Leaflet + CartoDB dark tiles. Data from outputs/frontend/ only.
// No external API calls beyond map tile loading.
"use client";

import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import type { JunctionMapPoint } from "@/lib/map-junctions";
import { MAP_BORDER, BENGALURU_CENTER } from "./map-ui";

function markerStyle(p: JunctionMapPoint) {
  if (!p.is_significant) {
    return { radius: 4, pathOptions: { color: "#475569", fillColor: "#475569", fillOpacity: 0.5, weight: 1 } };
  }
  if (p.obi_rank != null && p.obi_rank <= 10) {
    return { radius: 10, pathOptions: { color: "#DC2626", fillColor: "#DC2626", fillOpacity: 0.85, weight: 2 } };
  }
  return { radius: 7, pathOptions: { color: "#D97706", fillColor: "#D97706", fillOpacity: 0.75, weight: 2 } };
}

export default function L2HotspotMap({
  points,
  stats,
}: {
  points: JunctionMapPoint[];
  stats: { significant: number; total: number };
}) {
  if (!points.length) {
    return (
      <div style={{ height: 480, width: "100%", ...MAP_BORDER, background: "#1E293B", display: "flex", alignItems: "center", justifyContent: "center", color: "#64748B" }}>
        Map unavailable — check outputs/frontend/ exports
      </div>
    );
  }

  return (
    <div>
      <MapContainer
        center={BENGALURU_CENTER}
        zoom={11}
        style={{ height: "480px", width: "100%", ...MAP_BORDER }}
        scrollWheelZoom={false}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />
        {points.map((p) => {
          const { radius, pathOptions } = markerStyle(p);
          return (
            <CircleMarker key={p.junction} center={[p.lat, p.lng]} radius={radius} pathOptions={pathOptions}>
              <Popup>
                <div style={{ fontFamily: "monospace", fontSize: "12px", minWidth: "180px", color: "#1E293B" }}>
                  <b>{p.junction}</b>
                  <br />
                  OBI rank: {p.obi_rank ?? "—"}
                  <br />
                  Gi* z-score: {Number.isFinite(p.z_score) ? p.z_score.toFixed(2) : "—"}
                  <br />
                  p_sim: {Number.isFinite(p.p_sim) ? p.p_sim.toFixed(4) : "—"}
                  <br />
                  Intensity: {Number.isFinite(p.weighted_intensity) ? p.weighted_intensity.toFixed(1) : "—"}
                  <br />
                  <i>{p.is_significant ? "Significant hotspot" : "Not significant"}</i>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>
      <div style={{ marginTop: 10, fontSize: 12, color: "var(--ink-3)", lineHeight: 1.6 }}>
        <span style={{ marginRight: 14 }}>● Red = Top 10 OBI hotspot</span>
        <span style={{ marginRight: 14 }}>● Amber = Significant (p_sim &lt; 0.05)</span>
        <span style={{ marginRight: 14 }}>● Grey = Not significant</span>
        <br />
        {stats.significant} statistically significant junctions / {stats.total} total
      </div>
    </div>
  );
}
