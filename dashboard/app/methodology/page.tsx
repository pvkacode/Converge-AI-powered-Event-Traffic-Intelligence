import { Kpi, PageHeader, Panel, Note } from "@/components/ui";

const NEGATIVE_RESULTS = [
  {
    tried: "Stacked survival ensemble (RSF + CatBoost + Cox)",
    happened: "C-index unchanged vs RSF alone",
    action: "Kept RSF as the operational model",
  },
  {
    tried: "Fallback blending in duration quality gate",
    happened: "RMSE got worse: 6834 → 6038 after removal",
    action: "Blending removed; reliability score logged only",
  },
  {
    tried: "Gi* z > 1.96 significance cutoff",
    happened: "Found 0 significant hotspots; p_sim found 80",
    action: "Switched to permutation test; documented as methodological finding",
  },
  {
    tried: "BMA across all 4 model families",
    happened: "3 of 4 families have only 1 model each",
    action: "BMA is operational for duration family only; rest are diagnostic",
  },
  {
    tried: "Gamma resource effectiveness updates",
    happened: "All 4 resources fully saturated → confidence = 0",
    action: "No gamma update applied; flagged as heuristic not causal",
  },
] as const;

const LIMITATIONS = [
  "Cox PH concordance: 0.56 (barely above random 0.50). Cause × corridor matters more than priority or time-of-day. Cox is kept for interpretable hazard ratios, not predictive power.",
  "Layer 6 feedback is simulated. ASTraM is a historical batch. We designate Nov–Feb as \"prior\" and Mar–Apr as \"feedback.\" In production this would run on a daily cadence against live incidents.",
  "Layer 7 holdout is 18 days. Zone labels only exist for 10 weeks (Nov 10–Jan 19). The LRT p≈10⁻⁹¹ reflects whether spillover exists (n=3,389 events) — not the quality of the 18-day forecast evaluation.",
  "BMA is mostly diagnostic. 3 of 4 model families have one model each. No within-family comparison is possible for calibration, retrieval, or surrogate families.",
  "Resource effectiveness coefficients (γ_p, γ_b, γ_t, γ_q) are assumed, not learned. No deployment outcome data exists to calibrate them.",
  "South Zone 2 has the weakest Layer 7 calibration (KS p=0.021, AUC at 9h=0.651 vs 0.770–0.852 for other zones). Its CI is widened ×1.5 in ERI outputs.",
] as const;

const BUGS_FIXED = [
  {
    bug: "Gi* asymptotic z-score cutoff",
    fix: (
      <>
        Switched to permutation-based <span className="mono">p_sim</span> as primary significance test.
        The textbook z &gt; 1.96 threshold found 0 significant hotspots on this sample.{" "}
        <span className="mono">p_sim</span> found 80.
      </>
    ),
  },
  {
    bug: "ORR East 2 looked like an outlier",
    fix: (
      <>
        Confirmed as real: recurring metro station construction, logged ~daily as separate incidents,
        each lasting ~12 hours.
      </>
    ),
  },
  {
    bug: "Fallback duration blending degraded accuracy",
    fix: (
      <>
        Removed after validation. Metrics after removal: RMSE 6038 (vs 6834 with blending), MAE 2362
        (vs 2560).
      </>
    ),
  },
] as const;

export default function MethodologyPage() {
  return (
    <>
      <PageHeader
        eyebrow="Converge · ASTraM"
        title="Methodology & Honesty"
        lede="What we tried, what failed, what we kept anyway, and what the data actually supports. This page is static documentation — not computed from exports — so judges and operators can see where the pipeline is strong and where we were explicit about limits."
      />

      <div className="stack gap-6">
        <Panel title="Negative results" meta="Documented honestly — these are features, not bugs.">
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">What we tried</th>
                  <th scope="col">What happened</th>
                  <th scope="col">What we did</th>
                </tr>
              </thead>
              <tbody>
                {NEGATIVE_RESULTS.map((row) => (
                  <tr key={row.tried}>
                    <td>{row.tried}</td>
                    <td>{row.happened}</td>
                    <td>{row.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Honest scope" meta="Known limitations we disclose in every review.">
          <div className="stack gap-3">
            {LIMITATIONS.map((text) => (
              <Note key={text.slice(0, 40)}>{text}</Note>
            ))}
          </div>
        </Panel>

        <Panel title="What we found in the data" meta="Data quality findings from the cleaning pass.">
          <div className="grid grid-3">
            <Kpi
              label="Closed without timestamp"
              value="~3,500"
              sub="rows marked status=closed with no end timestamp — status and timestamp fields maintained inconsistently"
            />
            <Kpi
              label="True planned events"
              value="191 / 8,173"
              sub="2.3% of total — too sparse to train a forecaster; retrieval used instead"
            />
            <Kpi
              label="Censored rows"
              value="4,523+"
              sub="no end timestamp at all — why survival analysis, not regression"
            />
          </div>
        </Panel>

        <Panel title="Bugs we found and fixed" meta="Self-caught issues before external review.">
          <div className="grid grid-2" style={{ gap: 16, alignItems: "start" }}>
            {BUGS_FIXED.map((item) => (
              <div key={item.bug} className="panel" style={{ padding: 16 }}>
                <div className="kpi-label" style={{ marginBottom: 6 }}>
                  Bug
                </div>
                <p style={{ margin: "0 0 12px", fontSize: 14, fontWeight: 500 }}>{item.bug}</p>
                <div className="kpi-label" style={{ marginBottom: 6 }}>
                  Fix
                </div>
                <p className="muted" style={{ margin: 0, fontSize: 13.5, lineHeight: 1.55 }}>
                  {item.fix}
                </p>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}
