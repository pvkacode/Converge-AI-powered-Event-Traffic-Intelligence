// Single source of truth for the sidebar navigation and breadcrumb titles.
export interface NavItem {
  href: string;
  label: string;
  idx: string; // mono index shown in the rail
  step?: string; // pipeline stage name (for the overview flow)
  short: string; // breadcrumb / topbar title
}

export const NAV: NavItem[] = [
  { href: "/", label: "Overview", idx: "00", short: "Overview" },
  { href: "/layer1", label: "Duration Intelligence", idx: "L1", step: "Measure", short: "Layer 1" },
  { href: "/layer2", label: "Spatial Intelligence", idx: "L2", step: "Predict", short: "Layer 2" },
  { href: "/layer3", label: "Resource Optimization", idx: "L3", step: "Fuse", short: "Layer 3" },
  { href: "/layer4", label: "Event Intelligence", idx: "L4", step: "Retrieve", short: "Layer 4" },
  { href: "/layer45", label: "Predictive Fusion", idx: "4.5", step: "Fuse", short: "Layer 4.5" },
  { href: "/layer5", label: "Robust Optimization", idx: "L5", step: "Optimize", short: "Layer 5" },
  { href: "/layer6", label: "Adaptive Learning", idx: "L6", step: "Learn", short: "Layer 6" },
  { href: "/layer7", label: "Cross-Zone Spillover", idx: "L7", step: "Spillover", short: "Layer 7" },
  { href: "/map", label: "Hotspot Map", idx: "MAP", short: "Hotspot Map" },
  { href: "/worked-example", label: "Worked Example", idx: "WX", short: "Worked Example" },
  { href: "/methodology", label: "Methodology & Honesty", idx: "MH", short: "Methodology" },
];

export function navTitle(pathname: string): string {
  const exact = NAV.find((n) => n.href === pathname);
  if (exact) return exact.label;
  return "Converge";
}
