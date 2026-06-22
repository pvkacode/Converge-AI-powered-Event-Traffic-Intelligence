"use client";

import { useMemo, useState } from "react";
import nextDynamic from "next/dynamic";
import { MapPin, Target, Pulse, Info } from "@phosphor-icons/react";
import { Panel, Note } from "@/components/ui";
import { MapPlaceholder } from "@/components/maps/map-ui";
import type { GeoRadiusMatch } from "@/components/maps/GeoRadiusMap";
import type { Row } from "@/lib/csv";
import { toNum } from "@/lib/format";

const GeoRadiusMap = nextDynamic(() => import("@/components/maps/GeoRadiusMap"), {
  ssr: false,
  loading: () => <MapPlaceholder height={320} message="Loading geo-radius map…" />,
});

function markerBarColor(sFinal: number) {
  if (sFinal >= 0.6) return "#4ECDC4";
  if (sFinal >= 0.4) return "#E8A53D";
  if (sFinal >= 0.2) return "#9CA3AF";
  return "#4B5563";
}

function distColor(km: number) {
  if (km <= 1.0) return "#4ECDC4";
  if (km <= 2.0) return "#E8A53D";
  return "#6B7280";
}

export function Layer4GeoSection({
  diagnostics,
  matches,
}: {
  diagnostics: Row[];
  matches: Row[];
}) {
  const eventOptions = useMemo(() => {
    const ids = new Set<string>();
    for (const r of matches) {
      const id = (r.query_event_id ?? "").trim();
      if (id) ids.add(id);
    }
    if (ids.size === 0) {
      for (const r of diagnostics) {
        const id = (r.query_event_id ?? "").trim();
        if (id) ids.add(id);
      }
    }
    const scored = [...ids].map((id) => {
      const diag = diagnostics.find((d) => d.query_event_id === id);
      const count = toNum(diag?.geo_radius_2km_count ?? "0");
      const junction = matches.find((m) => m.query_event_id === id)?.query_junction ?? id;
      return { id, junction, count: Number.isNaN(count) ? 0 : count };
    });
    scored.sort((a, b) => b.count - a.count || a.junction.localeCompare(b.junction));
    return scored;
  }, [diagnostics, matches]);

  const [selectedId, setSelectedId] = useState(() => eventOptions[0]?.id ?? "");
  const [selectedMatchId, setSelectedMatchId] = useState<string | null>(null);

  const selectedDiag = diagnostics.find((d) => d.query_event_id === selectedId);
  const selectedMatches = useMemo(() => {
    return matches
      .filter((m) => m.query_event_id === selectedId)
      .sort((a, b) => toNum(a.match_rank) - toNum(b.match_rank))
      .map(
        (m): GeoRadiusMatch => ({
          match_event_id: m.match_event_id ?? "",
          match_junction: m.match_junction ?? "",
          match_lat: toNum(m.match_lat),
          match_lon: toNum(m.match_lon),
          match_cause: m.match_cause ?? "",
          match_duration_min: toNum(m.match_duration_min),
          dist_km: toNum(m.dist_km),
          gower_sim: toNum(m.gower_sim),
          phi_weight: toNum(m.phi_weight),
          s_final: toNum(m.s_final),
          within_2km: ["1", "true", "yes"].includes((m.within_2km ?? "").toLowerCase()),
          is_relevant: ["1", "true", "yes"].includes((m.is_relevant ?? "").toLowerCase()),
        })
      )
      .filter((m) => Number.isFinite(m.match_lat) && Number.isFinite(m.match_lon));
  }, [matches, selectedId]);

  const queryRow = matches.find((m) => m.query_event_id === selectedId);
  const queryLat = toNum(queryRow?.query_lat);
  const queryLon = toNum(queryRow?.query_lon);
  const queryJunction = queryRow?.query_junction ?? selectedId;

  const geoCount = toNum(selectedDiag?.geo_radius_2km_count ?? "0");
  const nearestKm = toNum(selectedDiag?.geo_radius_nearest_km ?? "");
  const sigmaKm = toNum(selectedDiag?.geo_sigma_km ?? matches[0]?.geo_sigma_km ?? "");

  const relevantWithin2km = selectedMatches.filter((m) => m.is_relevant);

  if (!eventOptions.length) {
    return (
      <Panel title="Spatial Retrieval · Geo-Radius Precedents" meta="Geo-radius enrichment not available">
        <EmptyState />
      </Panel>
    );
  }

  return (
    <div className="stack gap-6" style={{ marginBottom: 24 }}>
      <Panel
        title={
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            <MapPin size={18} weight="duotone" style={{ color: "var(--accent)" }} />
            Spatial Retrieval · Geo-Radius Precedents
          </span>
        }
        action={
          Number.isFinite(sigmaKm) ? (
            <span className="badge badge-ok" style={{ fontFamily: "var(--font-mono)", fontSize: 11 }}>
              σ = {sigmaKm.toFixed(2)} km bandwidth
            </span>
          ) : null
        }
        meta="Geo-weighted similarity s_final = s_gower · φ(d) · τ — spatial decay applied to existing Gower retrieval. Coordinates from Layer 2 hotspot CSV."
      >
        <div className="stack gap-4">
          <label className="stack" style={{ gap: 6 }}>
            <span className="kpi-label">SELECT EVENT TO INSPECT</span>
            <select
              className="input"
              value={selectedId}
              onChange={(e) => {
                setSelectedId(e.target.value);
                setSelectedMatchId(null);
              }}
            >
              {eventOptions.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.junction} · {opt.id}
                </option>
              ))}
            </select>
          </label>

          <div className="flex wrap gap-4" style={{ fontSize: 13, color: "var(--ink-2)" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <MapPin size={16} style={{ color: "var(--accent)" }} />
              {Number.isNaN(geoCount) ? "—" : geoCount} precedents within 2km
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <Target size={16} style={{ color: "var(--accent)" }} />
              {Number.isFinite(nearestKm) ? `${nearestKm.toFixed(2)} km` : "—"} nearest match
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <Pulse size={16} style={{ color: "var(--accent)" }} />
              σ = {Number.isFinite(sigmaKm) ? `${sigmaKm.toFixed(2)} km` : "—"} (data-derived bandwidth)
            </span>
          </div>

          {Number.isFinite(queryLat) && Number.isFinite(queryLon) ? (
            <GeoRadiusMap
              queryJunction={queryJunction}
              queryLat={queryLat}
              queryLon={queryLon}
              matches={selectedMatches}
              geoRadius2kmCount={Number.isNaN(geoCount) ? 0 : geoCount}
              nearestKm={Number.isFinite(nearestKm) ? nearestKm : null}
              sigmaKm={Number.isFinite(sigmaKm) ? sigmaKm : 0}
              selectedMatchId={selectedMatchId}
            />
          ) : (
            <MapPlaceholder height={320} message="Junction coordinates unavailable for this event" />
          )}

          <Note>
            <span style={{ display: "inline-flex", gap: 8, alignItems: "flex-start" }}>
              <Info size={16} weight="duotone" style={{ flexShrink: 0, marginTop: 2, color: "var(--accent)" }} />
              <span>
                Spatial decay φ(d) = exp(−d²/2σ²) re-weights existing Gower retrieval scores using junction
                coordinates from Layer 2. No new model is trained — this makes explicit the spatial structure
                already implicit in corridor matching. σ derived from median pairwise junction distance within
                corridors.
              </span>
            </span>
          </Note>
        </div>
      </Panel>

      <Panel
        title={
          <span className="mono" style={{ color: "var(--accent)", fontSize: 13, letterSpacing: "0.04em" }}>
            Matched precedents · {relevantWithin2km.length} within 2km of {queryJunction}
          </span>
        }
      >
        {selectedMatches.length === 0 ? (
          <div style={{ textAlign: "center", padding: "32px 16px", color: "var(--ink-3)" }}>
            <div>No precedents retrieved for this junction</div>
            <div style={{ fontSize: 12, marginTop: 8 }}>
              Retrieval may have abstained (low confidence) or junction coordinates unavailable.
            </div>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Junction</th>
                  <th>Distance</th>
                  <th>Cause</th>
                  <th>Duration</th>
                  <th>Geo sim</th>
                  <th>Match</th>
                </tr>
              </thead>
              <tbody>
                {selectedMatches.map((m) => {
                  const active = selectedMatchId === m.match_event_id;
                  return (
                    <tr
                      key={m.match_event_id}
                      onClick={() => setSelectedMatchId(m.match_event_id)}
                      style={{
                        cursor: "pointer",
                        background: m.is_relevant ? "rgba(255,255,255,0.04)" : undefined,
                        outline: active ? "1px solid var(--accent-line)" : undefined,
                      }}
                    >
                      <td title={m.match_junction}>
                        {m.match_junction.length > 28 ? `${m.match_junction.slice(0, 28)}…` : m.match_junction}
                      </td>
                      <td style={{ color: distColor(m.dist_km) }}>
                        {m.dist_km.toFixed(1)} km
                        {m.within_2km && (
                          <span style={{ marginLeft: 6, color: "#E8A53D" }} aria-hidden>
                            ●
                          </span>
                        )}
                        {!m.within_2km && (
                          <span className="dim" style={{ marginLeft: 8, fontSize: 11 }}>
                            outside 2km
                          </span>
                        )}
                      </td>
                      <td>{m.match_cause}</td>
                      <td>{Math.round(m.match_duration_min)} min</td>
                      <td title={`Gower: ${m.gower_sim.toFixed(3)} × φ(d): ${m.phi_weight.toFixed(3)} × trust = ${m.s_final.toFixed(3)}`}>
                        {m.s_final.toFixed(3)}
                      </td>
                      <td>
                        <div
                          style={{
                            width: 80,
                            height: 6,
                            borderRadius: 3,
                            background: "rgba(255,255,255,0.06)",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${Math.min(100, m.s_final * 100)}%`,
                              height: "100%",
                              background: markerBarColor(m.s_final),
                              borderRadius: 3,
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}

function EmptyState() {
  return (
    <div style={{ textAlign: "center", padding: "32px 16px", color: "var(--ink-3)" }}>
      <div>Geo-radius data not available</div>
      <div style={{ fontSize: 12, marginTop: 8 }}>
        Run layer4_planned_event_retrieval.py to generate layer4_geo_radius_matches.csv
      </div>
    </div>
  );
}
