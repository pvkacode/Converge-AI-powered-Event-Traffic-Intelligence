import { PageHeader, Note } from "@/components/ui";
import { buildWxMapData } from "@/lib/map-junctions";
import { WorkedExampleLive } from "@/components/WorkedExampleLive";
import { getBackendHostLabel, isHostedBackend } from "@/lib/backend-notice";

import { PAGE_REVALIDATE_SECONDS } from "@/lib/page-config";

export const revalidate = PAGE_REVALIDATE_SECONDS;

export default function WorkedExamplePage() {
  const wxMapData = buildWxMapData();
  const hosted = isHostedBackend();

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
          {hosted ? (
            <Note warn>
              Live inference is served from our Render backend ({getBackendHostLabel()}). After idle
              time the server sleeps — the first pipeline run may take <strong>30–60 seconds</strong>.
              If loading stalls, use the wake-up link in the notice that appears, then retry.
            </Note>
          ) : null}
        </div>
      </PageHeader>

      <WorkedExampleLive mapData={wxMapData} />
    </>
  );
}
