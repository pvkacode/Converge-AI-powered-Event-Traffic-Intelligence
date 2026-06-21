// Client helper for the separate FastAPI inference service (api/main.py).
// Base URL is configurable; defaults to the local uvicorn port.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export type Provenance = "live" | "precomputed_lookup" | "fallback";

export interface Options {
  causes: string[];
  corridors: string[];
  zones: string[];
  days: string[];
  priorities: string[];
  layer1_live: boolean;
  suggested_examples?: { cause: string; corridor: string; label: string }[];
}

export interface ScenarioInput {
  cause: string;
  corridor: string;
  hour_local: number;
  dow_local: number;
  requires_road_closure: boolean;
  priority: string;
}

// The worked-example response is intentionally loose-typed per section; each
// section is a record with a `provenance` field plus layer-specific keys.
export interface LayerSection {
  provenance: Provenance;
  note?: string;
  [k: string]: unknown;
}

export interface WorkedExampleResult {
  input: ScenarioInput;
  layer1_duration: LayerSection;
  layer2_spatial: LayerSection;
  layer3_resources: LayerSection;
  layer4_event: LayerSection;
  layer45_fusion: LayerSection;
  layer5_optimization: LayerSection;
  layer6_learning: LayerSection;
  layer7_spillover: LayerSection;
  recommendation: {
    headline: string;
    scenario: ScenarioInput;
    duration_plan?: {
      quantile: string;
      minutes: number;
      source: string;
      note?: string;
    } | null;
    officer_plan?: { count: number; source: string } | null;
  };
  provenance: Record<string, Provenance>;
}

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json() as Promise<T>;
}

export function fetchOptions(): Promise<Options> {
  return getJSON<Options>("/api/options");
}

export function runWorkedExample(input: ScenarioInput): Promise<WorkedExampleResult> {
  return getJSON<WorkedExampleResult>("/api/worked-example", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function fetchHealth(): Promise<{ layer1_live: boolean; layer1_reason: string }> {
  return getJSON("/health");
}
