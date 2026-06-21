import { PageHeader, Note } from "@/components/ui";
import { WorkedExampleLive } from "@/components/WorkedExampleLive";

export const dynamic = "force-dynamic";

export default function WorkedExamplePage() {
  return (
    <>
      <PageHeader
        eyebrow="Worked Example"
        title="Live pipeline trace"
        lede="Set an incident scenario and run it through the pipeline. The request flows down Layers 1 to 7 and accumulates into one operational recommendation. Each panel shows a provenance badge: Live means a real model function recomputed that layer this request; Precomputed means it was served from the existing exports keyed by your input; Fallback means the live engine was unavailable and the precomputed value was used instead."
      >
        <div style={{ marginTop: 14 }}>
          <Note>
            Layer 1 runs the real Kaplan-Meier survival functions from{" "}
            <span className="mono">src/layer1_survival.py</span> when the pipeline environment and{" "}
            <span className="mono">data/events_clean.parquet</span> are present. Layers 2 to 7 are
            served from the precomputed <span className="mono">outputs/</span> exports, keyed by your
            input, and labelled as such. The provenance badges are honest, never faked.
          </Note>
        </div>
      </PageHeader>

      <WorkedExampleLive />
    </>
  );
}
