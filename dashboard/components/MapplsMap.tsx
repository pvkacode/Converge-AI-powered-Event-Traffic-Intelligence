"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { MapPin, Key, WarningCircle } from "@phosphor-icons/react";
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

declare global {
  interface Window {
    mappls?: any;
  }
}

const CENTER: [number, number] = [12.9745, 77.5926]; // Bengaluru

function colorFor(p: HotPoint): string {
  if (p.prob >= 0.8 || p.sig) return "#9B2C20"; // critical
  if (p.prob >= 0.5) return "#8A6308"; // warning
  return "#0F6E66"; // accent
}

export function MapplsMap({ points, envKey }: { points: HotPoint[]; envKey?: string }) {
  const [apiKey, setApiKey] = useState(envKey ?? "");
  const [keyInput, setKeyInput] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    envKey ? "loading" : "idle"
  );
  const [sigOnly, setSigOnly] = useState(false);
  const mapRef = useRef<any>(null);
  const markersRef = useRef<any[]>([]);
  const containerId = "mappls-map";

  const visible = sigOnly ? points.filter((p) => p.sig) : points;

  const renderMarkers = useCallback(() => {
    const mappls = window.mappls;
    const map = mapRef.current;
    if (!mappls || !map) return;
    // clear previous
    markersRef.current.forEach((m) => {
      try { m.remove?.(); } catch { /* noop */ }
    });
    markersRef.current = [];
    const pts = sigOnly ? points.filter((p) => p.sig) : points;
    for (const p of pts) {
      const color = colorFor(p);
      try {
        const marker = new mappls.Marker({
          map,
          position: { lat: p.lat, lng: p.lng },
          html: `<div style="width:13px;height:13px;border-radius:999px;background:${color};border:2px solid var(--surface);box-shadow:0 0 0 1px ${color}"></div>`,
          popupHtml: `<div style="font-family:var(--font-sans);min-width:170px">
            <div style="font-weight:600;color:var(--ink);margin-bottom:4px">${p.name}</div>
            <div style="font-size:12px;color:var(--ink-2);font-family:var(--font-mono)">
              hotspot prob ${fmtNum(p.prob)}<br/>Gi* z ${fmtNum(p.z)}<br/>incidents ${fmtNum(p.count)}<br/>${p.sig ? "significant" : "not significant"}
            </div></div>`,
        });
        markersRef.current.push(marker);
      } catch { /* marker API variance; skip silently */ }
    }
  }, [points, sigOnly]);

  const initMap = useCallback(() => {
    const mappls = window.mappls;
    if (!mappls || !mappls.Map) {
      setStatus("error");
      return;
    }
    try {
      const map = new mappls.Map(containerId, { center: CENTER, zoom: 11, zoomControl: true });
      mapRef.current = map;
      const onReady = () => {
        setStatus("ready");
        renderMarkers();
      };
      if (typeof map.on === "function") map.on("load", onReady);
      else setTimeout(onReady, 800);
    } catch {
      setStatus("error");
    }
  }, [renderMarkers]);

  // load the SDK once a key is available
  useEffect(() => {
    if (!apiKey) return;
    if (window.mappls && window.mappls.Map) {
      setStatus("loading");
      initMap();
      return;
    }
    setStatus("loading");
    const existing = document.getElementById("mappls-sdk");
    if (existing) {
      existing.addEventListener("load", initMap);
      return () => existing.removeEventListener("load", initMap);
    }
    const script = document.createElement("script");
    script.id = "mappls-sdk";
    script.src = `https://apis.mappls.com/advancedmaps/api/${apiKey}/map_sdk?layer=vector&v=3.0`;
    script.async = true;
    script.onload = initMap;
    script.onerror = () => setStatus("error");
    document.body.appendChild(script);
    return () => {
      script.onload = null;
      script.onerror = null;
    };
  }, [apiKey, initMap]);

  // re-render markers when the significance filter changes
  useEffect(() => {
    if (status === "ready") renderMarkers();
  }, [sigOnly, status, renderMarkers]);

  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <div className="panel-head">
        <div className="stack" style={{ gap: 2 }}>
          <h2 className="section-title">Hotspot map</h2>
          <span className="section-meta">{visible.length} junctions · MapmyIndia (Mappls)</span>
        </div>
        <label className="row gap-2" style={{ fontSize: 13, cursor: "pointer" }}>
          <input type="checkbox" checked={sigOnly} onChange={(e) => setSigOnly(e.target.checked)} />
          Significant only
        </label>
      </div>

      {status === "ready" || status === "loading" ? (
        <div style={{ position: "relative" }}>
          <div id={containerId} style={{ width: "100%", height: 560, background: "var(--surface-inset)" }} />
          {status === "loading" && (
            <div className="empty" style={{ position: "absolute", inset: 0 }}>
              <div className="empty-title">Loading map…</div>
            </div>
          )}
        </div>
      ) : status === "error" ? (
        <div className="panel-body">
          <div className="empty">
            <WarningCircle size={26} className="empty-icon" />
            <div className="empty-title">Map failed to load</div>
            <div style={{ maxWidth: 460 }}>
              The Mappls SDK could not initialise with that key. Check the token is a valid Mappls
              (MapmyIndia) <span className="mono">map_sdk</span> key with the right domain restrictions,
              then try again.
            </div>
            <button className="btn btn-sm" style={{ marginTop: 10 }} onClick={() => { setApiKey(""); setStatus("idle"); }}>
              Enter a different key
            </button>
          </div>
        </div>
      ) : (
        <div className="panel-body">
          <div className="empty" style={{ paddingTop: 28, paddingBottom: 16 }}>
            <Key size={26} className="empty-icon" />
            <div className="empty-title">Add a Mappls API key to render the map</div>
            <div style={{ maxWidth: 480 }}>
              This view uses the MapmyIndia (Mappls) vector SDK. Paste a Mappls{" "}
              <span className="mono">map_sdk</span> token below, or set{" "}
              <span className="mono">NEXT_PUBLIC_MAPPLS_KEY</span> in{" "}
              <span className="mono">dashboard/.env.local</span> and restart.
            </div>
            <div className="row gap-2" style={{ marginTop: 14, width: "100%", maxWidth: 480, justifyContent: "center" }}>
              <input
                className="input"
                placeholder="Mappls map_sdk token"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                style={{ maxWidth: 320 }}
              />
              <button
                className="btn btn-accent"
                disabled={!keyInput.trim()}
                onClick={() => setApiKey(keyInput.trim())}
              >
                Load map
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="panel-body" style={{ borderTop: "1px solid var(--border)" }}>
        <div className="legend">
          <span className="legend-item"><span className="legend-swatch" style={{ background: "#9B2C20" }} /> High prob or significant</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: "#8A6308" }} /> Moderate</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: "#0F6E66" }} /> Lower</span>
        </div>
      </div>
    </div>
  );
}
