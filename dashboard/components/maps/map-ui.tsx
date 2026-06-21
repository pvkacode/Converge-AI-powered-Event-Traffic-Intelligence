import type { CSSProperties, ReactNode } from "react";

export const MAP_BORDER: CSSProperties = {
  borderRadius: "8px",
  border: "1px solid #334155",
};

export const BENGALURU_CENTER: [number, number] = [12.9716, 77.5946];

export function MapPlaceholder({
  height,
  message,
  children,
}: {
  height: number | string;
  message: string;
  children?: ReactNode;
}) {
  return (
    <div
      style={{
        height,
        width: "100%",
        ...MAP_BORDER,
        background: "#1E293B",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "#64748B",
        fontSize: 14,
        position: "relative",
      }}
    >
      {message}
      {children}
    </div>
  );
}

export function DarkTileLayer() {
  // Re-exported pattern: import in client map files directly from react-leaflet
  return null;
}
