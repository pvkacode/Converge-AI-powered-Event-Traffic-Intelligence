"use client";
import Link from "next/link";
import { ArrowRight } from "@phosphor-icons/react";

interface Node {
  step: string;
  name: string;
  layer: string;
  desc: string;
  href: string;
}

const NODES: Node[] = [
  { step: "01 Measure", name: "Duration", layer: "Layer 1", desc: "Survival models estimate how long a disruption lasts (P50/P80/P95).", href: "/layer1" },
  { step: "02 Predict", name: "Spatial", layer: "Layer 2", desc: "Hotspot ranking and Gi* clustering across junctions.", href: "/layer2" },
  { step: "03 Retrieve", name: "Events", layer: "Layer 4", desc: "Case-based retrieval of similar past events for recommendations.", href: "/layer4" },
  { step: "04 Fuse", name: "Predictive Fusion", layer: "Layer 4.5", desc: "Calibrated, sanity-guarded state vector with novelty + drift flags.", href: "/layer45" },
  { step: "05 Optimize", name: "Allocation", layer: "Layer 3 · 5", desc: "Resource allocation and CVaR-robust deployment under uncertainty.", href: "/layer5" },
  { step: "06 Learn", name: "Adaptive", layer: "Layer 6", desc: "Bayesian monitoring of model health, drift, and recalibration.", href: "/layer6" },
  { step: "07 Spillover", name: "Cross-Zone", layer: "Layer 7", desc: "Hawkes spillover, expected-risk index, early-warning zones.", href: "/layer7" },
];

export function FlowDiagram() {
  return (
    <div className="flow">
      {NODES.map((n, i) => (
        <div key={n.href} style={{ display: "contents" }}>
          <Link href={n.href} className="flow-node">
            <span className="fn-step">{n.step}</span>
            <span className="fn-name">{n.name}</span>
            <span className="fn-layer">{n.layer}</span>
            <span className="fn-desc">{n.desc}</span>
          </Link>
          {i < NODES.length - 1 && (
            <span className="flow-arrow" aria-hidden>
              <ArrowRight size={16} weight="bold" />
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
