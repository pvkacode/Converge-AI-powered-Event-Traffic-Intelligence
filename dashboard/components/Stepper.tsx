"use client";
import { useState } from "react";
import { CaretRight } from "@phosphor-icons/react";

export interface Step {
  tag: string;
  title: string;
  summary: string;
  content: React.ReactNode;
}

export function Stepper({ steps }: { steps: Step[] }) {
  const [active, setActive] = useState(0);
  return (
    <div className="grid" style={{ gridTemplateColumns: "260px 1fr", gap: 24, alignItems: "start" }}>
      <div className="step-rail" role="tablist" aria-label="Worked example steps">
        {steps.map((s, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={i === active}
            className={`step${i === active ? " active" : ""}`}
            onClick={() => setActive(i)}
            style={{
              textAlign: "left",
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "inherit",
              font: "inherit",
            }}
          >
            <span className="step-num">{i + 1}</span>
            <span className="stack gap-2" style={{ paddingTop: 2 }}>
              <span className="mono dim" style={{ fontSize: 11 }}>{s.tag}</span>
              <span style={{ fontWeight: 620, fontSize: 14, letterSpacing: "-0.01em" }}>{s.title}</span>
              <span className="muted" style={{ fontSize: 12.5, lineHeight: 1.45 }}>{s.summary}</span>
              {i === active && (
                <span className="row gap-2 kpi-accent" style={{ fontSize: 12, fontWeight: 600 }}>
                  Viewing <CaretRight size={12} weight="bold" />
                </span>
              )}
            </span>
          </button>
        ))}
      </div>
      <div className="stack gap-6">{steps[active].content}</div>
    </div>
  );
}
