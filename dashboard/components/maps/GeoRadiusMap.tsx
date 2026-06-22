"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, Circle, CircleMarker, Popup, Marker, useMap } from "react-leaflet";
import { MAP_BORDER } from "./map-ui";

export interface GeoRadiusMatch {
  match_event_id: string;
  match_junction: string;
  match_lat: number;
  match_lon: number;
  match_cause: string;
  match_duration_min: number;
  dist_km: number;
  gower_sim: number;
  phi_weight: number;
  s_final: number;
  within_2km: boolean;
  is_relevant: boolean;
}

export interface GeoRadiusMapProps {
  queryJunction: string;
  queryLat: number;
  queryLon: number;
  matches: GeoRadiusMatch[];
  geoRadius2kmCount: number;
  nearestKm: number | null;
  sigmaKm: number;
  selectedMatchId?: string | null;
}

const TEAL = "#4ECDC4";
const GOLD = "#E8A53D";

function markerColor(sFinal: number) {
  if (sFinal >= 0.6) return TEAL;
  if (sFinal >= 0.4) return GOLD;
  if (sFinal >= 0.2) return "#9CA3AF";
  return "#4B5563";
}

function offsetEast(lat: number, lon: number, km: number): [number, number] {
  const dLon = km / (111.32 * Math.cos((lat * Math.PI) / 180));
  return [lat, lon + dLon];
}

function RingLabel({ center, radiusKm, label, color }: { center: [number, number]; radiusKm: number; label: string; color: string }) {
  const pos = offsetEast(center[0], center[1], radiusKm);
  const icon = L.divIcon({
    className: "",
    html: `<span style="font-size:10px;color:${color};font-family:var(--font-mono,monospace);white-space:nowrap;text-shadow:0 1px 2px rgba(0,0,0,0.8);">${label}</span>`,
    iconSize: [32, 14],
    iconAnchor: [0, 7],
  });
  return <Marker position={pos} icon={icon} interactive={false} />;
}

function PulseRing({ center }: { center: [number, number] }) {
  const map = useMap();
  const circleRef = useRef<L.Circle | null>(null);
  const rafRef = useRef<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const circle = L.circle(center, {
      radius: 2000,
      color: GOLD,
      fillColor: GOLD,
      fillOpacity: 0,
      opacity: 0.5,
      weight: 1.5,
      interactive: false,
    });
    circle.addTo(map);
    circleRef.current = circle;

    const animate = () => {
      const start = performance.now();
      const duration = 1500;
      const run = (now: number) => {
        const t = Math.min(1, (now - start) / duration);
        const radius = 2000 + t * 400;
        const opacity = 0.5 * (1 - t);
        circle.setRadius(radius);
        circle.setStyle({ opacity, fillOpacity: opacity * 0.08 });
        if (t < 1) {
          rafRef.current = requestAnimationFrame(run);
        }
      };
      rafRef.current = requestAnimationFrame(run);
    };

    animate();
    intervalRef.current = setInterval(animate, 2000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      map.removeLayer(circle);
    };
  }, [map, center]);

  return null;
}

export default function GeoRadiusMap({
  queryJunction,
  queryLat,
  queryLon,
  matches,
  selectedMatchId,
}: GeoRadiusMapProps) {
  const center: [number, number] = [queryLat, queryLon];

  return (
    <div style={{ position: "relative", background: "#1A1A2E", borderRadius: 8 }}>
      <MapContainer
        center={center}
        zoom={13}
        style={{ height: 320, width: "100%", ...MAP_BORDER, background: "#1A1A2E" }}
        scrollWheelZoom={false}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />

        <Circle
          center={center}
          radius={1000}
          pathOptions={{ color: TEAL, fillColor: TEAL, fillOpacity: 0.04, opacity: 0.6, weight: 1.5 }}
        />
        <Circle
          center={center}
          radius={2000}
          pathOptions={{ color: GOLD, fillColor: GOLD, fillOpacity: 0.02, opacity: 0.4, weight: 1.5, dashArray: "4 4" }}
        />
        <Circle
          center={center}
          radius={4000}
          pathOptions={{ color: "#FFFFFF", fillColor: "transparent", fillOpacity: 0, opacity: 0.12, weight: 1, dashArray: "2 6" }}
        />

        <PulseRing center={center} />

        <RingLabel center={center} radiusKm={1} label="1km" color={TEAL} />
        <RingLabel center={center} radiusKm={2} label="2km" color={GOLD} />
        <RingLabel center={center} radiusKm={4} label="4km" color="rgba(255,255,255,0.55)" />

        <CircleMarker
          center={center}
          radius={10}
          pathOptions={{ color: TEAL, fillColor: TEAL, fillOpacity: 0.9, weight: 2 }}
        >
          <Popup>
            <div style={{ fontFamily: "monospace", fontSize: 12 }}>
              <b>{queryJunction}</b>
              <br />
              Query location
            </div>
          </Popup>
        </CircleMarker>

        {matches.map((m) => {
          const selected = selectedMatchId === m.match_event_id;
          const color = markerColor(m.s_final);
          return (
            <CircleMarker
              key={m.match_event_id}
              center={[m.match_lat, m.match_lon]}
              radius={selected ? 9 : m.within_2km ? 6 : 5}
              pathOptions={{
                color: selected ? "#FFFFFF" : color,
                fillColor: color,
                fillOpacity: m.is_relevant ? 0.85 : 0.5,
                weight: selected ? 3 : m.is_relevant ? 2 : 1,
              }}
            >
              <Popup>
                <div style={{ fontFamily: "monospace", fontSize: 12, minWidth: 180 }}>
                  <b>Junction:</b> {m.match_junction}
                  <br />
                  <b>Distance:</b> {m.dist_km.toFixed(1)} km
                  <br />
                  <b>Cause:</b> {m.match_cause}
                  <br />
                  <b>Duration:</b> {Math.round(m.match_duration_min)} min
                  <br />
                  <b>Geo-weighted similarity:</b> {m.s_final.toFixed(3)}
                  <br />
                  (Gower: {m.gower_sim.toFixed(3)} × φ: {m.phi_weight.toFixed(3)})
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {matches.length === 0 && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
            color: "#9CA3AF",
            fontSize: 13,
            textAlign: "center",
            padding: 24,
          }}
        >
          No retrieved precedents with coordinates
        </div>
      )}

      <div
        style={{
          position: "absolute",
          right: 8,
          bottom: 8,
          background: "rgba(15,20,40,0.9)",
          padding: "8px 12px",
          borderRadius: 6,
          border: "1px solid rgba(255,255,255,0.1)",
          fontSize: 10,
          color: "#fff",
          zIndex: 1000,
          lineHeight: 1.6,
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 4 }}>Similarity</div>
        <div><span style={{ color: TEAL }}>●</span> High (≥0.6)</div>
        <div><span style={{ color: GOLD }}>●</span> Medium (≥0.4)</div>
        <div><span style={{ color: "#9CA3AF" }}>●</span> Low (&lt;0.4)</div>
        <div style={{ borderTop: "1px solid rgba(255,255,255,0.12)", margin: "6px 0" }} />
        <div><span style={{ color: GOLD }}>┄</span> 2km radius</div>
      </div>
    </div>
  );
}
