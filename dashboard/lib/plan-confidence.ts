// Composite "plan confidence" signal built only from fields the worked-example
// API response actually returns: the per-layer provenance map, Layer 3/4's
// insufficient_evidence flags, Layer 4's abstain flag, and Layer 4's
// evidence_weight (effective sample size). No invented fields — if the
// response is missing one of these, that rule simply doesn't fire.
import type { LayerSection, Provenance } from "./api";

export type PlanConfidenceLevel = "High" | "Moderate" | "Low" | "Unknown";

export interface PlanConfidenceInput {
  provenance?: Record<string, Provenance>;
  layer3?: LayerSection;
  layer4?: LayerSection;
}

export interface PlanConfidenceResult {
  level: PlanConfidenceLevel;
  reasons: string[];
}

// Below this ESS, Layer 4's retrieved precedent set is thin even when it
// didn't abstain outright — few matching historical events to draw on.
const ESS_THIN_THRESHOLD = 30;
// Two or more layers served from "fallback" (live engine unavailable, not
// merely a precomputed lookup) means multiple parts of the pipeline degraded
// for this request, not just one.
const MULTI_FALLBACK_THRESHOLD = 2;

const RANK: Record<PlanConfidenceLevel, number> = { High: 2, Moderate: 1, Low: 0, Unknown: -1 };

function downgrade(level: PlanConfidenceLevel, candidate: PlanConfidenceLevel): PlanConfidenceLevel {
  return RANK[candidate] < RANK[level] ? candidate : level;
}

function l4Abstained(layer4?: LayerSection): boolean {
  if (!layer4) return false;
  const raw = String(layer4.abstain ?? "").trim().toLowerCase();
  return raw === "true" || layer4.insufficient_evidence === true;
}

// Layer 4 retrieval only applies to planned-event causes; for other causes
// `applicable === false` is expected and shouldn't read as weak evidence.
function l4Applicable(layer4?: LayerSection): boolean {
  return layer4?.applicable !== false;
}

export function computePlanConfidence(input: PlanConfidenceInput): PlanConfidenceResult {
  const { provenance, layer3, layer4 } = input;
  if (!provenance) {
    return { level: "Unknown", reasons: ["No provenance data on this response."] };
  }

  const reasons: string[] = [];
  let level: PlanConfidenceLevel = "High";

  const fallbackLayers = Object.entries(provenance).filter(([, p]) => p === "fallback");
  if (fallbackLayers.length >= MULTI_FALLBACK_THRESHOLD) {
    level = downgrade(level, "Low");
    reasons.push(
      `${fallbackLayers.length} layers on fallback (${fallbackLayers.map(([k]) => k).join(", ")}) — multiple parts of the live pipeline were unavailable.`,
    );
  } else if (fallbackLayers.length === 1) {
    level = downgrade(level, "Moderate");
    reasons.push(`${fallbackLayers[0][0]} served from fallback, not its preferred live/precomputed path.`);
  }

  if (layer3?.insufficient_evidence === true) {
    level = downgrade(level, "Moderate");
    reasons.push("L3 abstained — no rule-based resource estimate for this corridor; using upstream priors.");
  }

  if (l4Applicable(layer4)) {
    if (l4Abstained(layer4)) {
      level = downgrade(level, "Low");
      reasons.push("L4 abstained — no retrieved precedent for this cause/corridor combination.");
    } else if (layer4?.evidence_weight != null) {
      const ess = Number(layer4.evidence_weight);
      if (Number.isFinite(ess)) {
        if (ess < ESS_THIN_THRESHOLD) {
          level = downgrade(level, "Moderate");
          reasons.push(`L4 evidence weight (ESS) = ${ess.toFixed(0)} — thin precedent set.`);
        } else {
          reasons.push(`L4 evidence weight (ESS) = ${ess.toFixed(0)} — healthy.`);
        }
      }
    }
  }

  if (reasons.length === 0) {
    reasons.push("All layers ran on their preferred path with healthy evidence.");
  }

  return { level, reasons };
}

/*
Example reachability (for manual sanity-checking, not a test framework):

High:     computePlanConfidence({ provenance: { l1: "live", l4: "live" },
            layer4: { provenance: "live", abstain: "False", evidence_weight: 142 } })
          -> { level: "High", reasons: ["L4 evidence weight (ESS) = 142 — healthy."] }

Moderate: computePlanConfidence({ provenance: { l1: "live", l4: "precomputed_lookup" },
            layer4: { provenance: "precomputed_lookup", abstain: "False", evidence_weight: 12 } })
          -> { level: "Moderate", reasons: ["L4 evidence weight (ESS) = 12 — thin precedent set."] }

Low:      computePlanConfidence({ provenance: { l1: "live", l4: "fallback" },
            layer4: { provenance: "fallback", abstain: "True" } })
          -> { level: "Low", reasons: [... , "L4 abstained — ..."] }

Unknown:  computePlanConfidence({})
          -> { level: "Unknown", reasons: ["No provenance data on this response."] }
*/
