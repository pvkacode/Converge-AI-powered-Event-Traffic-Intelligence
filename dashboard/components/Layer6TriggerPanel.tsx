"use client";
import { useState } from "react";
import { Panel } from "./ui";
import { SeverityEventTable, type Row } from "./SeverityEventTable";

interface Props {
  triggers: Row[];
  // null when layer6_active_alerts.csv is absent entirely (vs. present-but-empty,
  // which still gets a tab so the empty state is visible).
  alerts: Row[] | null;
}

const TRIGGER_COLUMNS = [
  { key: "severity", label: "Severity" },
  { key: "trigger_id", label: "Trigger" },
  { key: "source_module", label: "Source" },
  { key: "score", label: "Score" },
  { key: "threshold", label: "Threshold" },
  { key: "affected_layer", label: "Layer" },
  { key: "recommendation", label: "Recommendation" },
];

const TRIGGER_DETAIL = [
  { key: "signal_type", label: "Signal type" },
  { key: "variable", label: "Variable" },
  { key: "test", label: "Test" },
  { key: "action", label: "Recommended action" },
  { key: "generated_at", label: "Generated at" },
];

const ALERT_COLUMNS = [
  { key: "severity", label: "Severity" },
  { key: "alert_id", label: "Alert" },
  { key: "source", label: "Source" },
  { key: "affected_layer", label: "Layer" },
  { key: "description", label: "Description" },
];

const ALERT_DETAIL = [{ key: "generated_at", label: "Generated at" }];

export function Layer6TriggerPanel({ triggers, alerts }: Props) {
  const hasAlertsFile = alerts !== null;
  const [tab, setTab] = useState<"triggers" | "alerts">("triggers");

  const counts = { critical: 0, moderate: 0, info: 0 };
  for (const r of triggers) {
    const s = (r.severity ?? "").trim().toLowerCase();
    if (s in counts) counts[s as keyof typeof counts] += 1;
  }
  const summary = `${counts.critical} critical · ${counts.moderate} moderate · ${counts.info} info`;

  return (
    <Panel
      title="Retrain triggers"
      meta={
        tab === "triggers"
          ? `${triggers.length} triggers this batch — ${summary}`
          : `${alerts?.length ?? 0} active alerts`
      }
      action={
        hasAlertsFile ? (
          <div className="row gap-2">
            <button
              type="button"
              className={`btn btn-sm${tab === "triggers" ? " btn-accent" : ""}`}
              onClick={() => setTab("triggers")}
            >
              Retrain triggers
            </button>
            <button
              type="button"
              className={`btn btn-sm${tab === "alerts" ? " btn-accent" : ""}`}
              onClick={() => setTab("alerts")}
            >
              Active alerts
            </button>
          </div>
        ) : undefined
      }
      bodyPad={false}
    >
      {tab === "triggers" ? (
        <SeverityEventTable
          rows={triggers}
          idField="trigger_id"
          scoreField="score"
          columns={TRIGGER_COLUMNS}
          detailFields={TRIGGER_DETAIL}
          searchFields={["trigger_id", "source_module", "variable", "recommendation", "action"]}
          searchPlaceholder="Filter by trigger, source or recommendation…"
          emptyMessage="No retrain triggers in this batch, or layer6_retrain_triggers.csv is missing from outputs/."
        />
      ) : (
        <SeverityEventTable
          rows={alerts ?? []}
          idField="alert_id"
          columns={ALERT_COLUMNS}
          detailFields={ALERT_DETAIL}
          searchFields={["alert_id", "source", "description"]}
          searchPlaceholder="Filter by source, layer or description…"
          emptyMessage="No active alerts, or layer6_active_alerts.csv is missing from outputs/."
        />
      )}
    </Panel>
  );
}
