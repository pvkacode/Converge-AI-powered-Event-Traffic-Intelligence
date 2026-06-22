import "./container-scroll.css";

const BAR_HEIGHTS = [55, 75, 60, 50, 85, 65, 72, 55, 90, 70, 80, 65, 85, 72];

function MapPinIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="dm-icon">
      <path
        d="M12 21s6-5.2 6-10a6 6 0 1 0-12 0c0 4.8 6 10 6 10Z"
        stroke="currentColor"
        strokeWidth="2"
      />
      <circle cx="12" cy="11" r="2" fill="currentColor" />
    </svg>
  );
}

function TargetIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="dm-icon">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
      <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2" />
      <circle cx="12" cy="12" r="1.5" fill="currentColor" />
    </svg>
  );
}

function TrendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="dm-icon">
      <path d="M4 16l5-5 4 4 7-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M16 6h4v4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function BoltIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden className="dm-icon">
      <path
        d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const STAT_CARDS = [
  { icon: <MapPinIcon />, value: "8,173", label: "real incidents" },
  { icon: <TargetIcon />, value: "80 / 294", label: "hotspots ranked" },
  { icon: <TrendIcon />, value: "49.81%", label: "CVaR reduction" },
  { icon: <BoltIcon />, value: "7", label: "critical triggers" },
] as const;

export default function DashboardMockup() {
  return (
    <div
      style={{
        height: "100%",
        width: "100%",
        background: "#FAFAF0",
        borderRadius: "1rem",
        overflow: "hidden",
        fontFamily: "var(--font-mono)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          height: 36,
          background: "#F5F5E8",
          borderBottom: "1px solid rgba(0,0,0,0.08)",
          display: "flex",
          alignItems: "center",
          gap: 12,
          paddingLeft: 16,
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {["#FF5F57", "#FFBD2E", "#28C840"].map((c) => (
            <span
              key={c}
              style={{
                width: 12,
                height: 12,
                borderRadius: "50%",
                background: c,
                display: "inline-block",
              }}
            />
          ))}
        </div>
        <span style={{ fontSize: 12, color: "#6B7B6B" }}>converge / overview</span>
      </div>

      <div className="dm-stats">
        {STAT_CARDS.map((card) => (
          <div
            key={card.label}
            style={{
              background: "#fff",
              border: "1px solid rgba(0,0,0,0.08)",
              borderRadius: 10,
              padding: 16,
            }}
          >
            <div style={{ marginBottom: 8 }}>{card.icon}</div>
            <div
              style={{
                fontSize: 24,
                fontWeight: 700,
                fontFamily: "var(--font-mono)",
                color: "#1A1A1A",
                lineHeight: 1.1,
              }}
            >
              {card.value}
            </div>
            <div style={{ fontSize: 12, color: "#6B7280", marginTop: 4 }}>{card.label}</div>
          </div>
        ))}
      </div>

      <div
        style={{
          padding: "0 16px 16px",
          flex: 1,
          display: "flex",
          alignItems: "flex-end",
          gap: 8,
          minHeight: 160,
        }}
      >
        {BAR_HEIGHTS.map((h, i) => (
          <div
            key={i}
            className="dm-bar"
            style={{
              flex: 1,
              height: `${Math.max(20, (h / 100) * 160)}px`,
              minHeight: 20,
            }}
          />
        ))}
      </div>
    </div>
  );
}
