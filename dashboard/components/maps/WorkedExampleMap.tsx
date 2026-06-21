// Map: Leaflet + CartoDB dark tiles. Data from outputs/frontend/ only.
// No external API calls beyond map tile loading.
"use client";

import "leaflet/dist/leaflet.css";
import { useEffect, useMemo } from "react";
import { MapContainer, TileLayer, Marker, CircleMarker, Polyline, Tooltip, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import type { JunctionMapPoint, WxMapData } from "@/lib/map-junctions";
import type { ScenarioInput } from "@/lib/api";
import { MAP_BORDER, BENGALURU_CENTER } from "./map-ui";

function FlyTo({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.flyTo(center, zoom, { duration: 0.8 });
  }, [center, zoom, map]);
  return null;
}

const redIcon = L.divIcon({
  html: `<div style="width:14px;height:14px;background:#DC2626;border-radius:50%;border:3px solid white;box-shadow:0 0 6px #DC2626"/>`,
  className: "",
  iconAnchor: [7, 7],
});

function resolveIncident(mapData: WxMapData, input: ScenarioInput) {
  const corridorKey = Object.keys(mapData.corridorJunction).find(
    (c) => c.toLowerCase() === input.corridor.toLowerCase()
  );
  const junction = corridorKey ? mapData.corridorJunction[corridorKey] : undefined;
  const coords = junction ? mapData.coordsByJunction[junction] : undefined;
  return { junction, coords, corridorKey: corridorKey ?? input.corridor };
}

function diversionTarget(mapData: WxMapData, junction: string | undefined) {
  if (!junction) return null;
  const row = mapData.diversions.find(
    (d) => d.junction === junction && /route\s*a/i.test(d.route_label)
  ) ?? mapData.diversions.find((d) => d.junction === junction);
  if (!row) return null;
  const pathParts = row.diversion_path.split("|").map((s) => s.trim()).filter(Boolean);
  const targetName = pathParts[pathParts.length - 1] || row.diversion_corridor;
  const targetCoords = mapData.coordsByJunction[targetName];
  if (!targetCoords) return null;
  return { targetName, targetCoords, label: row.route_label || "Route A" };
}

export default function WorkedExampleMap({
  mapData,
  input,
}: {
  mapData: WxMapData;
  input: ScenarioInput;
}) {
  const incident = useMemo(() => resolveIncident(mapData, input), [mapData, input]);
  const diversion = useMemo(
    () => diversionTarget(mapData, incident.junction),
    [mapData, incident.junction]
  );

  const nearby = useMemo(() => {
    if (!incident.coords) return [];
    const { lat, lng } = incident.coords;
    return mapData.points.filter(
      (p) =>
        p.is_significant &&
        p.junction !== incident.junction &&
        Math.abs(p.lat - lat) <= 0.02 &&
        Math.abs(p.lng - lng) <= 0.02
    );
  }, [mapData.points, incident.coords, incident.junction]);

  const mapCenter: [number, number] = incident.coords
    ? [incident.coords.lat, incident.coords.lng]
    : BENGALURU_CENTER;

  return (
    <div style={{ position: "relative", marginTop: 16 }}>
      <MapContainer
        center={mapCenter}
        zoom={13}
        style={{ height: "300px", width: "100%", ...MAP_BORDER }}
        scrollWheelZoom={false}
        attributionControl={false}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />
        {incident.coords ? (
          <>
            <FlyTo center={[incident.coords.lat, incident.coords.lng]} zoom={13} />
            <Marker position={[incident.coords.lat, incident.coords.lng]} icon={redIcon}>
              <Popup>
                Incident: {input.cause} · {input.corridor} · {input.hour_local}:00
              </Popup>
            </Marker>
            {nearby.map((p) => (
              <CircleMarker
                key={p.junction}
                center={[p.lat, p.lng]}
                radius={6}
                pathOptions={{ color: "#D97706", fillColor: "#D97706", fillOpacity: 0.75, weight: 2 }}
              />
            ))}
            {diversion ? (
              <Polyline
                positions={[
                  [incident.coords.lat, incident.coords.lng],
                  [diversion.targetCoords.lat, diversion.targetCoords.lng],
                ]}
                pathOptions={{ color: "#059669", weight: 4, opacity: 0.85, dashArray: "8 4" }}
              >
                <Tooltip sticky>Recommended diversion: {diversion.label}</Tooltip>
              </Polyline>
            ) : null}
          </>
        ) : null}
      </MapContainer>
      {!incident.coords ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(15,17,26,0.85)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#94A3B8",
            fontSize: 13,
            borderRadius: 8,
          }}
        >
          Coordinates not available for this corridor
        </div>
      ) : null}
    </div>
  );
}
