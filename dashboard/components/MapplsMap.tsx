"use client";
import { useEffect, useRef, useState } from "react";
import { fmtNum } from "@/lib/format";

export interface HotPoint {
  name: string;
  lat: number;
  lng: number;
  prob: number;
  z: number;
  count: number;
  sig: boolean;
}

const CENTER: [number, number] = [12.9745, 77.5926];

function colorFor(p: HotPoint): string {
  if (p.prob >= 0.8 || p.sig) return "#9B2C20";
  if (p.prob >= 0.5) return "#8A6308";
  return "#0F6E66";
}

function injectLeafletCss(): void {
  if (document.getElementById("leaflet-css")) return;
  const link = document.createElement("link");
  link.id = "leaflet-css";
  link.rel = "stylesheet";
  link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
  document.head.appendChild(link);
}

function resetLeafletContainer(el: HTMLElement): void {
  if ((el as HTMLElement & { _leaflet_id?: number })._leaflet_id != null) {
    delete (el as HTMLElement & { _leaflet_id?: number })._leaflet_id;
  }
  el.innerHTML = "";
  el.removeAttribute("tabindex");
}

export function MapplsMap({ points }: { points: HotPoint[]; envKey?: string }) {
  const [sigOnly, setSigOnly] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.CircleMarker[]>([]);

  const visible = sigOnly ? points.filter((p) => p.sig) : points;

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const L = (await import("leaflet")).default;
        if (cancelled || !containerRef.current) return;

        injectLeafletCss();

        delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
        L.Icon.Default.mergeOptions({
          iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
          iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
          shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
        });

        const el = containerRef.current;
        if (mapRef.current) {
          mapRef.current.remove();
          mapRef.current = null;
        }
        resetLeafletContainer(el);

        const map = L.map(el, {
          center: CENTER,
          zoom: 11,
          zoomControl: true,
          attributionControl: true,
        });

        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
          attribution:
            '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
          subdomains: "abcd",
          maxZoom: 19,
        }).addTo(map);

        mapRef.current = map;
        if (cancelled) {
          map.remove();
          mapRef.current = null;
          return;
        }

        // Leaflet needs a layout pass after the loading overlay is removed.
        requestAnimationFrame(() => {
          if (cancelled || !mapRef.current) return;
          mapRef.current.invalidateSize();
          setReady(true);
          setError(null);
        });
      } catch (err) {
        if (!cancelled) {
          setReady(false);
          setError(err instanceof Error ? err.message : "Map failed to load");
        }
      }
    })();

    return () => {
      cancelled = true;
      setReady(false);
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];
      mapRef.current?.remove();
      mapRef.current = null;
      if (containerRef.current) resetLeafletContainer(containerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!ready || !mapRef.current) return;

    let cancelled = false;

    (async () => {
      const L = (await import("leaflet")).default;
      if (cancelled || !mapRef.current) return;

      const map = mapRef.current;
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];

      const pts = sigOnly ? points.filter((p) => p.sig) : points;

      for (const p of pts) {
        const color = colorFor(p);
        const marker = L.circleMarker([p.lat, p.lng], {
          radius: p.prob >= 0.8 || p.sig ? 9 : p.prob >= 0.5 ? 6 : 4,
          color,
          fillColor: color,
          fillOpacity: p.sig ? 0.85 : 0.55,
          weight: 2,
        });

        marker.bindPopup(`
          <div style="font-family:monospace;min-width:180px;font-size:12px">
            <div style="font-weight:700;margin-bottom:6px;font-size:13px">
              ${p.name}
            </div>
            <div style="color:#334155;line-height:1.8">
              Hotspot prob: <b>${fmtNum(p.prob)}</b><br/>
              Gi* z-score: <b>${fmtNum(p.z)}</b><br/>
              Incidents: <b>${fmtNum(p.count)}</b><br/>
              <span style="color:${p.sig ? "#059669" : "#94A3B8"}">
                ${p.sig ? "✓ Significant" : "Not significant"}
              </span>
            </div>
          </div>
        `);

        marker.addTo(map);
        markersRef.current.push(marker);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [ready, points, sigOnly]);

  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <div className="panel-head">
        <div className="stack" style={{ gap: 2 }}>
          <h2 className="section-title">Hotspot map</h2>
          <span className="section-meta">
            {visible.length} junctions · Leaflet + CartoDB dark tiles
          </span>
        </div>
        <label className="row gap-2" style={{ fontSize: 13, cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={sigOnly}
            onChange={(e) => setSigOnly(e.target.checked)}
          />
          Significant only
        </label>
      </div>

      <div style={{ position: "relative" }}>
        <div
          ref={containerRef}
          style={{
            width: "100%",
            height: 560,
            background: "var(--surface-inset)",
            zIndex: 0,
          }}
        />
        {!ready && !error && (
          <div className="empty" style={{ position: "absolute", inset: 0, zIndex: 1 }}>
            <div className="empty-title">Loading map…</div>
          </div>
        )}
        {error ? (
          <div className="empty" style={{ position: "absolute", inset: 0, zIndex: 1 }}>
            <div className="empty-title">Map failed to load</div>
            <p className="dim" style={{ fontSize: 12, marginTop: 8 }}>
              {error}
            </p>
          </div>
        ) : null}
      </div>

      <div className="panel-body" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="legend">
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: "#9B2C20" }} />
            Significant / high prob
          </span>
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: "#8A6308" }} />
            Moderate
          </span>
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: "#0F6E66" }} />
            Lower / not significant
          </span>
        </div>
      </div>
    </div>
  );
}
