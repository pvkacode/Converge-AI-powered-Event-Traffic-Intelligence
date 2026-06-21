// Presentational primitives usable from both server and client components
// (no hooks, no browser APIs).
import { badgeVariant, type BadgeVariant } from "@/lib/badges";
import { titleCaseValue } from "@/lib/format";

export function Badge({
  children,
  variant,
  dot = false,
}: {
  children: React.ReactNode;
  variant: BadgeVariant;
  dot?: boolean;
}) {
  return (
    <span className={`badge badge-${variant}`}>
      {dot && <span className={`status-dot dot-${variant === "muted" || variant === "neutral" ? "accent" : variant}`} />}
      {children}
    </span>
  );
}

function abstainLabel(value: string): string | null {
  const v = value.trim().toLowerCase();
  if (["1", "true", "yes"].includes(v)) return "Abstained";
  if (["0", "false", "no"].includes(v)) return "Recommend";
  return null;
}

// Auto-variant badge from a raw cell value (used by tables and inline).
export function ValueBadge({ value, column }: { value: string; column?: string }) {
  if (value == null || value === "") return <span className="dim">-</span>;
  const col = column?.toLowerCase() ?? "";
  const label = col.includes("abstain") ? abstainLabel(value) : null;
  const variant = badgeVariant(label ?? value, column);
  const display = label ?? titleCaseValue(value);
  return (
    <Badge variant={variant} dot={col.includes("abstain") && label === "Recommend"}>
      {display}
    </Badge>
  );
}

export function Kpi({
  label,
  value,
  sub,
  accent = false,
  isText = false,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  accent?: boolean;
  isText?: boolean;
}) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value${isText ? " text" : ""}${accent ? " kpi-accent" : ""}`}>{value}</div>
      {sub != null && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  lede,
  children,
}: {
  eyebrow: string;
  title: string;
  lede?: string;
  children?: React.ReactNode;
}) {
  return (
    <header style={{ marginBottom: 24 }}>
      <div className="page-eyebrow">{eyebrow}</div>
      <h1 className="page-title">{title}</h1>
      {lede && <p className="page-lede">{lede}</p>}
      {children}
    </header>
  );
}

export function Panel({
  title,
  meta,
  action,
  children,
  bodyPad = true,
}: {
  title?: React.ReactNode;
  meta?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  bodyPad?: boolean;
}) {
  return (
    <section className="panel">
      {(title || action) && (
        <div className="panel-head">
          <div className="stack" style={{ gap: 2 }}>
            {title && <h2 className="section-title">{title}</h2>}
            {meta && <span className="section-meta">{meta}</span>}
          </div>
          {action}
        </div>
      )}
      <div className={bodyPad ? "panel-body" : ""}>{children}</div>
    </section>
  );
}

export function EmptyState({
  title = "Data not available",
  message,
}: {
  title?: string;
  message?: string;
}) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {message && <div style={{ maxWidth: 420 }}>{message}</div>}
    </div>
  );
}

export function Note({
  children,
  warn = false,
}: {
  children: React.ReactNode;
  warn?: boolean;
}) {
  return <div className={`note${warn ? " warn" : ""}`}>{children}</div>;
}

export function MetricLine({ k, v }: { k: React.ReactNode; v: React.ReactNode }) {
  return (
    <div className="metric-line">
      <span className="ml-k">{k}</span>
      <span className="ml-v">{v}</span>
    </div>
  );
}
