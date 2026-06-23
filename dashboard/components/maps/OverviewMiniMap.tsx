// Map: Leaflet + CartoDB dark tiles. Data from outputs/frontend/ only.
// No external API calls beyond map tile loading.
"use client";

import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, Marker } from "react-leaflet";
import L from "leaflet";
import type { JunctionMapPoint } from "@/lib/map-junctions";
import { MAP_BORDER, BENGALURU_CENTER } from "./map-ui";

const pulseIcon = L.divIcon({
  html: `<div style="
    width:12px;height:12px;
    background:#DC2626;
    border-radius:50%;
    border:2px solid white;
    box-shadow: 0 0 0 0 rgba(220,38,38,0.7);
    animation: pulse-ring 2s ease-out infinite;
  "/>`,
  className: "",
  iconAnchor: [6, 6],
});

export default function OverviewMiniMap({
  points,
  stats,
}: {
  points: JunctionMapPoint[];
  stats: { significant: number; total: number };
}) {
  if (!points.length) {
    return (
      <div style={{ height: 260, width: "100%", ...MAP_BORDER, background: "#1E293B", display: "flex", alignItems: "center", justifyContent: "center", color: "#64748B" }}>
        Map unavailable — check outputs/frontend/ exports
      </div>
    );
  }

  return (
    <div>
      <style>{`
        @keyframes pulse-ring {
          0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.7); }
          70%  { box-shadow: 0 0 0 10px rgba(220,38,38,0); }
          100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
        }
      `}</style>
      <MapContainer
        center={BENGALURU_CENTER}
        zoom={11}
        style={{ height: "260px", width: "100%", ...MAP_BORDER }}
        scrollWheelZoom={false}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />
        {points.map((p) => (
          <Marker key={p.junction} position={[p.lat, p.lng]} icon={pulseIcon} />
        ))}
      </MapContainer>
      <p style={{ color: "#64748B", fontSize: 12, marginTop: 8, marginBottom: 0 }}>
        {stats.significant} significant · {stats.total} junctions · Gi* permutation p_sim &lt; 0.05 · read live from outputs/frontend/
      </p>
    </div>
  );
}
