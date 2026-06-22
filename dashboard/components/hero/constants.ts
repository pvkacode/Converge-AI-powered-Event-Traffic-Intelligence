export const LAYER_COLORS = {
  L1: "#2E5EAA",
  L2: "#1B7A6B",
  L3: "#B8841F",
  L4: "#6B4FA0",
  "L4.5": "#A0526B",
  L5: "#C0392B",
  L6: "#2E7D5B",
  L7: "#1B2A56",
} as const;

export const PIPELINE_LAYERS = [
  {
    id: "L1",
    title: "Duration Intelligence",
    techniques: "KM · Cox · RSF",
    metric: "RSF C-index: 0.70",
    href: "/layer1",
  },
  {
    id: "L2",
    title: "Spatial Intelligence",
    techniques: "Gi* · OBI · Hawkes",
    metric: "80 / 294 junctions",
    href: "/layer2",
  },
  {
    id: "L3",
    title: "Resource Optimization",
    techniques: "LP · Fragility · Diversion",
    metric: "21/22 corridors: Hawkes > Poisson",
    href: "/layer3",
  },
  {
    id: "L4",
    title: "Event Intelligence",
    techniques: "Retrieval · IMS · Abstain",
    metric: "MAE: 40.9 min",
    href: "/layer4",
  },
  {
    id: "L4.5",
    title: "Predictive Fusion",
    techniques: "CatBoost · Conformal",
    metric: "ECE: 7.14e-19",
    href: "/layer45",
  },
  {
    id: "L5",
    title: "Robust Optimization",
    techniques: "CVaR MILP · Chance constraints",
    metric: "49.81% CVaR reduction",
    href: "/layer5",
  },
  {
    id: "L7",
    title: "Cross-Zone Spillover",
    techniques: "Multivariate Hawkes · ERI",
    metric: "p ≈ 2.5 × 10⁻⁹¹",
    href: "/layer7",
  },
] as const;

const L6_LAYER = {
  id: "L6",
  title: "Adaptive Learning",
  techniques: "Bayesian · Drift · ESS",
  metric: "7 critical retrain triggers",
  href: "/layer6",
} as const;

/** Sticky top nav — all eight model layers (L6 sits in the feedback loop on the diagram) */
export const ALL_LAYERS = [
  PIPELINE_LAYERS[0],
  PIPELINE_LAYERS[1],
  PIPELINE_LAYERS[2],
  PIPELINE_LAYERS[3],
  PIPELINE_LAYERS[4],
  PIPELINE_LAYERS[5],
  L6_LAYER,
  PIPELINE_LAYERS[6],
] as const;

export const CORRIDORS = [
  "Mysore Road",
  "Old Airport Road",
  "Bellary Road 1",
  "Bellary Road 2",
  "Hosur Road",
  "Bannerghata Road",
  "ORR East 2",
  "CBD 2",
  "Silk Board",
  "Hebbal",
] as const;

export const CAUSES = [
  "vehicle_breakdown",
  "public_event",
  "water_logging",
  "accident",
  "construction",
  "procession",
] as const;

export const DOW_MAP: Record<string, number> = {
  Monday: 0,
  Tuesday: 1,
  Wednesday: 2,
  Thursday: 3,
  Friday: 4,
  Saturday: 5,
  Sunday: 6,
};
