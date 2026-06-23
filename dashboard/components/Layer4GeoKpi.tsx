"use client";

import { MapPin } from "@phosphor-icons/react";
import { fmtNum } from "@/lib/format";

export function Layer4GeoKpi({
  geoUnavailable,
  totalGeoCount,
  meanGeoCount,
  nearestOverall,
}: {
  geoUnavailable: boolean;
  totalGeoCount: number;
  meanGeoCount: number;
  nearestOverall: number;
}) {
  return (
    <div className="kpi" style={{ position: "relative" }}>
      <MapPin
        size={18}
        weight="duotone"
        style={{ position: "absolute", top: 12, right: 12, color: "var(--accent)" }}
        aria-hidden
      />
      <div className="kpi-label">Spatial precedents</div>
      {geoUnavailable ? (
        <>
          <div className="kpi-value text">Corridor-level only</div>
          <div className="kpi-sub">Junction coordinates unavailable</div>
        </>
      ) : (
        <>
          <div className="kpi-value">{fmtNum(totalGeoCount)}</div>
          <div className="kpi-sub">
            {Number.isNaN(meanGeoCount) ? "—" : meanGeoCount.toFixed(1)} avg per event within 2km
          </div>
          <div className="kpi-sub">
            nearest: {Number.isNaN(nearestOverall) ? "—" : `${nearestOverall.toFixed(2)} km`}
          </div>
        </>
      )}
    </div>
  );
}
