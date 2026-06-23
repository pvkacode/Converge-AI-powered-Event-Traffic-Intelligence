import { tryLoadCsv } from "@/lib/csv";
import { valueCounts, countWhere } from "@/lib/stats";
import { fmtNum } from "@/lib/format";
import { Kpi, PageHeader, Panel, Badge, Note } from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { VBar } from "@/components/charts";
import { healthVariant } from "@/lib/badges";
import { Layer6TriggerPanel } from "@/components/Layer6TriggerPanel";


export const revalidate = 30;

export default function Layer6Page() {
  const health = tryLoadCsv("layer6_model_health_summary.csv");
  const alerts = tryLoadCsv("layer6_active_alerts.csv");
  const drift = tryLoadCsv("layer6_drift_report.csv");
  const triggers = tryLoadCsv("layer6_retrain_triggers.csv");

  const overall =
    health?.rows.map((r) => r["overall_health"]).find((v) => v && v.trim() !== "") ?? "UNKNOWN";
  const statusCounts = health ? valueCounts(health.rows, "status") : {};
  const totalChecks = health?.rows.length ?? 0;
  const criticalChecks = statusCounts["critical"] ?? 0;
  const warnChecks = statusCounts["warning"] ?? 0;

  const alertTotal = alerts?.rows.length ?? 0;
  const sev = alerts ? valueCounts(alerts.rows, "severity") : {};

  const driftAlerting = drift
    ? countWhere(drift.rows, (r) => ["true", "1", "yes"].includes((r["alert"] ?? "").toLowerCase()))
    : 0;

  const statusColorFor = (s: string) =>
    s === "critical" ? "var(--critical)" : s === "warning" ? "var(--warning)" : "var(--ok)";
  const statusBars = Object.entries(statusCounts).map(([k, v]) => ({
    status: k,
    count: v,
    __color: statusColorFor(k),
  }));

  const variant = healthVariant(overall);
  const bannerColor =
    variant === "critical" ? "var(--critical)" : variant === "warning" ? "var(--warning)" : "var(--ok)";
  const bannerBg =
    variant === "critical" ? "var(--critical-bg)" : variant === "warning" ? "var(--warning-bg)" : "var(--ok-bg)";

  return (
    <>
      <PageHeader
        eyebrow="Layer 6 · Learn"
        title="Adaptive Learning"
        lede="Layer 6 is the monitoring and self-correction layer. It runs Bayesian model-averaging diagnostics, posterior predictive checks, calibration and drift tests over the operational feedback log, then raises alerts and recalibration triggers when the deployed models start to disagree with reality."
      />

      <div
        className="panel"
        style={{
          marginBottom: 24,
          borderColor: bannerColor,
          background: bannerBg,
          padding: "20px 22px",
        }}
      >
        <div className="row between wrap gap-4">
          <div className="stack gap-2">
            <span className="kpi-label">Overall model health</span>
            <div className="row gap-3">
              <span style={{ fontSize: 34, fontWeight: 700, letterSpacing: "-0.03em", color: bannerColor, fontFamily: "var(--font-mono)" }}>
                {overall}
              </span>
              <Badge variant={variant} dot>
                live status
              </Badge>
            </div>
            <span className="muted" style={{ fontSize: 13 }}>
              {criticalChecks} of {totalChecks} health checks are critical and {warnChecks} are in
              warning. Overall status escalates to the worst component.
            </span>
          </div>
          <div className="row gap-6">
            <div className="stack" style={{ alignItems: "flex-end" }}>
              <span className="kpi-value" style={{ color: "var(--critical)" }}>{fmtNum(criticalChecks)}</span>
              <span className="kpi-label">critical</span>
            </div>
            <div className="stack" style={{ alignItems: "flex-end" }}>
              <span className="kpi-value" style={{ color: "var(--warning)" }}>{fmtNum(warnChecks)}</span>
              <span className="kpi-label">warning</span>
            </div>
            <div className="stack" style={{ alignItems: "flex-end" }}>
              <span className="kpi-value" style={{ color: "var(--ok)" }}>{fmtNum(statusCounts["healthy"] ?? 0)}</span>
              <span className="kpi-label">healthy</span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Kpi label="Health checks" value={fmtNum(totalChecks)} sub="metric-group rows" />
        <Kpi label="Active alerts" value={fmtNum(alertTotal)} sub={Object.entries(sev).map(([k, v]) => `${v} ${k}`).join(", ") || "none"} />
        <Kpi label="Drift tests alerting" value={fmtNum(driftAlerting)} sub={`${drift?.rows.length ?? 0} drift tests run`} />
        <Kpi
          label="Critical alerts"
          value={fmtNum(sev["critical"] ?? 0)}
          sub="severity = critical"
        />
      </div>

      {statusBars.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <Panel title="Health checks by status" meta="Component-level model health">
            <VBar data={statusBars} xKey="status" yKey="count" height={220} />
          </Panel>
        </div>
      )}

      <div className="stack gap-6">
        <DataTable
          dataset="layer6_model_health"
          title="Model health checks"
          subtitle="Holdout vs feedback value, relative change and status per metric"
          searchPlaceholder="Filter by metric or group…"
        />
        <Layer6TriggerPanel
          triggers={(triggers?.rows ?? []) as Record<string, string>[]}
          alerts={alerts ? (alerts.rows as Record<string, string>[]) : null}
        />
        <DataTable
          dataset="layer6_drift_report"
          title="Drift report"
          subtitle="Page-Hinkley / PSI drift tests with retrain urgency"
          searchPlaceholder="Filter by test or variable…"
        />
      </div>

      <div style={{ marginTop: 20 }}>
        <Note warn>
          Overall health reads <span className="mono">CRITICAL</span> while most individual checks are
          healthy, because the status aggregates to the worst component (here the duration drift and
          recalibration triggers). This is the monitoring layer doing its job, surfaced exactly as
          exported.
        </Note>
      </div>
    </>
  );
}
