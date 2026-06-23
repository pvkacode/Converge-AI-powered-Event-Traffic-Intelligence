"use client";

import { Clock, Users, MapPin, CaretDown } from "@phosphor-icons/react";
import type { WorkedExampleResult } from "@/lib/api";
import { fmtMinutes, titleCaseValue } from "@/lib/format";
import { Badge } from "@/components/ui";

const SOURCE_LABEL: Record<string, string> = {
  layer1: "Layer 1 KM",
  layer45_guarded: "Layer 4.5 guarded",
  layer3: "Layer 3",
  layer4: "L4 retrieval",
  layer5: "Layer 5 MILP",
};

function tierVariant(tier: string | undefined) {
  const t = (tier ?? "").toLowerCase();
  if (t === "critical") return "critical" as const;
  if (t === "high") return "warning" as const;
  if (t === "moderate") return "accent" as const;
  return "ok" as const;
}

function pickOfficers(result: WorkedExampleResult): { count: number; source: string } | null {
  const fromRec = result.recommendation.officer_plan;
  if (fromRec && fromRec.count > 0) {
    return { count: fromRec.count, source: fromRec.source ?? "layer3" };
  }
  const l3 = result.layer3_resources.officers as number | undefined;
  if (l3 != null && l3 > 0) return { count: l3, source: "layer3" };
  const l4 = (result.layer4_event.recommended as Record<string, unknown> | undefined)?.officers as
    | number
    | undefined;
  if (l4 != null && l4 > 0) return { count: l4, source: "layer4" };
  const l5 = (result.layer5_optimization.allocation as Record<string, unknown> | undefined)
    ?.officers as number | undefined;
  if (l5 != null && l5 > 0) return { count: l5, source: "layer5" };
  return null;
}

export function WorkedExampleExecutiveSummary({ result }: { result: WorkedExampleResult }) {
  const { recommendation, layer2_spatial, layer3_resources, layer7_spillover } = result;
  const plan = recommendation.duration_plan;
  const officers = pickOfficers(result);
  const riskTier = layer3_resources.risk_tier as string | undefined;
  const hotspot = layer2_spatial.matched_hotspot as string | undefined;
  const spillZone = layer7_spillover.zone as string | undefined;

  const barricades = (layer3_resources.barricades as number | undefined) ?? 0;
  const tow = (layer3_resources.tow as number | undefined) ?? 0;

  const hasAnswer =
    recommendation.headline &&
    recommendation.headline !== "Insufficient matching data for a synthesised recommendation.";

  return (
    <section className="wx-exec-summary" aria-label="Operational recommendation">
      <div className="wx-exec-summary-head">
        <div>
          <p className="wx-exec-eyebrow">Operational recommendation</p>
          <p className="wx-exec-scenario">
            {titleCaseValue(result.input.cause)} · {result.input.corridor}
          </p>
        </div>
        {riskTier ? <Badge variant={tierVariant(riskTier)}>{riskTier} risk</Badge> : null}
      </div>

      <p className="wx-exec-headline">
        {hasAnswer ? recommendation.headline : "Insufficient evidence for a synthesised plan on this input."}
      </p>

      <div className="wx-exec-stats">
        <div className="wx-exec-stat">
          <div className="wx-exec-stat-icon" aria-hidden>
            <Clock size={20} weight="duotone" />
          </div>
          <div className="wx-exec-stat-value">
            {plan?.minutes != null ? fmtMinutes(plan.minutes) : "—"}
          </div>
          <div className="wx-exec-stat-label">Planning duration</div>
          <div className="wx-exec-stat-sub">
            {plan?.quantile ? plan.quantile.toUpperCase() : "—"}
            {plan?.source ? ` · ${SOURCE_LABEL[plan.source] ?? plan.source}` : ""}
          </div>
        </div>

        <div className="wx-exec-stat">
          <div className="wx-exec-stat-icon" aria-hidden>
            <Users size={20} weight="duotone" />
          </div>
          <div className="wx-exec-stat-value">
            {officers ? Math.round(officers.count) : "—"}
          </div>
          <div className="wx-exec-stat-label">Officers</div>
          <div className="wx-exec-stat-sub">
            {officers
              ? SOURCE_LABEL[officers.source] ?? officers.source
              : "No allocation on this corridor"}
            {barricades > 0 || tow > 0
              ? ` · ${barricades > 0 ? `${Math.round(barricades)} barricades` : ""}${barricades > 0 && tow > 0 ? ", " : ""}${tow > 0 ? `${Math.round(tow)} tow` : ""}`
              : ""}
          </div>
        </div>

        <div className="wx-exec-stat">
          <div className="wx-exec-stat-icon" aria-hidden>
            <MapPin size={20} weight="duotone" />
          </div>
          <div className="wx-exec-stat-value wx-exec-stat-value-text">
            {hotspot ?? spillZone ?? "—"}
          </div>
          <div className="wx-exec-stat-label">
            {hotspot ? "Hotspot junction" : spillZone ? "Spillover watch" : "Spatial"}
          </div>
          <div className="wx-exec-stat-sub">
            {spillZone && hotspot ? `Also watch ${spillZone}` : ""}
            {!spillZone && !hotspot ? "No hotspot match" : ""}
          </div>
        </div>
      </div>

      {plan?.note ? (
        <p className="wx-exec-note">{plan.note}</p>
      ) : null}

      <a href="#wx-layer-trace" className="wx-exec-trace-link">
        <CaretDown size={14} weight="bold" aria-hidden />
        Full 7-layer technical trace below
      </a>
    </section>
  );
}
