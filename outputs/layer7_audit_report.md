# Layer 7 Pre-Expansion Audit Report

**Scope:** M1–M6 (Operational State, Active Site View + Override, Explainability + Counterfactual, Dashboard + API, Decision Confidence + Governance, Digital Twin).
**Type:** Read-only audit. No source/output was modified, regenerated, or rerun. All evidence read from existing `outputs/`.
**Date:** 2026-06-20.

Artifacts produced (all new, additive): `layer7_audit_inventory.csv`, `layer7_dataflow_audit.csv`, `layer7_operational_state_audit.csv`, `layer7_alert_audit.csv`, `layer7_active_site_audit.csv`, `layer7_explainability_audit.csv`, `layer7_override_audit.csv`, `layer7_dashboard_audit.csv`, `layer7_api_audit_review.csv`, `layer7_decision_confidence_audit.csv`, `layer7_governance_audit.csv`, `layer7_digital_twin_audit.csv`, `layer7_cross_layer_audit.csv`, `layer7_audit_findings.csv`, this report.

---

## 1. Executive Summary

Layer 7 is **architecturally sound and operationally safe**: it is strictly additive, never mutates Layers 1–6 (SHA-verified across all milestones), confines writes to the `layer7_` namespace, is fully deterministic/reproducible, and keeps every action human-in-the-loop. The engineering quality (validation harnesses, append-only logs, degrade-if-absent API, namespace guards) is strong.

The weaknesses are **almost entirely in the scoring/aggregation mathematics**, not in plumbing or safety. The single most important issue is that the **Operational Risk Score is dominated ~94% by one input (`tail_risk_prob_z`)**, which makes ORS — and everything derived from it (active ranking, explanations, twin, SVI) — effectively a one-signal score despite using six nominal weights. Several other scores have calibration problems: the Alert Severity Score's **implementation diverges from its own documented formula and exceeds its stated [0,1] bound**, P1 alerts are inflated to 31.5%, the **absolute-OIS high-impact band is mathematically unreachable** (so a governance KPI and an operator-queue band are permanently empty/zero), and the **Twin Confidence Score double-counts** robustness and uncertainty.

None of these are data-integrity or safety defects. They are signal-quality defects that would be **inherited and amplified** by the planned Sensor Fusion / Kalman / Control-Policy / GNN work, which consumes these scores as state. They should be patched before that work begins.

**Total findings: 21 — Critical 1, High 4, Medium 10, Low 6.**

## 2. Strongest Components

- **M4 Dashboard + API** — clean schema/manifest consistency, real HTTP-verified read-only endpoints, backward-compatible extension in M5, no endpoint drift (catalog == OpenAPI == routes).
- **M2 Override Engine** — append-only audit trail, unique IDs, 7 enforced safety rules, advisory-only (no auto-execution).
- **Cross-layer discipline** — verified zero upstream mutation, namespace-confined writes, deterministic reproducibility everywhere.
- **M3 Explainability faithfulness** — contribution shares sum to 1 and are computed directly from the ORS logit (faithful); they actually *expose* the ORS dominance problem rather than hide it.

## 3. Weakest Components

- **M1 Operational State (ORS)** — single-feature dominance (F-001, Critical); scope mixes 3,498 events with 50 active sites (F-006); dead `critical_alert_indicator` term (F-015).
- **M1 Alert Prioritization** — formula/spec divergence + unbounded ASS (F-002, High); P1 inflation (F-003, High); near-constant recency (F-010).
- **M5/M6 confidence & impact scoring** — DCS compression (F-009); tercile-only tiering (F-008); TCS double-counting (F-005, High); unreachable high-impact band (F-004, High).
- **M6 Digital Twin** — deterministic, tautological scenario ranking (F-011).

## 4. Critical Findings

- **F-001 (Critical) — ORS single-feature dominance.** `tail_risk_prob_z` accounts for 94% of mean absolute contribution; fragility 1%, OBI 4%, drift 0.8%, novelty 0.2%, critical-alert 0%. The z-scores are consumed without re-standardization, so the highest-variance input swamps the nominal weights. ORS is effectively `f(tail_risk_prob_z)`. **This is the root cause behind several downstream findings and is the top priority to fix before any state-estimation/GNN work.**

## 5. High Priority Findings

- **F-002 (High) — ASS formula divergence + unbounded.** Implemented `base×corroboration×recency` (multiplicative); the documented `layer7_math_specification.md` §5.2 is an additive model with `clip[0,1]`. Observed ASS up to 1.262 (9 alerts > 1.0). Spec ≠ code; score not bounded as claimed.
- **F-003 (High) — P1 inflation.** 31.5% of alerts (23/73) are P1; any corroborated critical clears the 0.85 cut, so P1 saturates and loses triage value.
- **F-004 (High) — Dead high-impact override band.** `absolute_ois` maxes at 0.29 (anchored to large L5 ranges) but governance `high_impact_override_rate`, the operator-queue priority-4 band, and M3 `impact_level` High/Critical all gate on `>= 0.50` → permanently 0/empty.
- **F-005 (High) — TCS double counting.** `TCS = 0.4·DCS + 0.3·robustness + 0.3·(1−uncertainty)` while `DCS = (1−uncertainty)·robustness·reliability` — robustness and uncertainty are counted both inside DCS and again standalone; TCS is not an independent confidence axis.

## 6. Recommended Patch Order

1. **F-001** — re-standardize/winsorize ORS z-inputs so weights are meaningful (unblocks active ranking, explanations, twin, SVI, and any future estimator).
2. **F-005** — redefine TCS from independent signals (or drop the DCS term).
3. **F-004** — recalibrate absolute-OIS thresholds (or anchors) so the [0,1] range and high-impact band are usable.
4. **F-002 / F-003** — reconcile ASS spec vs code, bound the score, and calibrate priority cut-points on the realized distribution.
5. **F-009 / F-008** — fix DCS compression (avoid raw triple-product) and add absolute thresholds alongside rank tiers.
6. **F-012 / F-013 / F-014** — add override approval-transition enforcement, fix the alert-ack denominator, and propagate L6/L4.5 integrity/sanity flags into a quality gate.
7. Low-severity items (F-016…F-021) — opportunistic.

## 7. Should Sensor Fusion proceed now?

**Not yet — patch F-001 and F-005 first.** The M7 SVI scaffold is fine as readiness, but real sensor fusion fuses observations into operational *state*. If that state (ORS / confidence) is a one-feature score that double-counts confidence terms, the fused sensors are diluted into a degenerate signal. Fix F-001 (and ideally F-005, F-004) so there is a well-conditioned multi-signal state to fuse into. After that, Sensor Fusion can proceed.

## 8. Should GNN implementation proceed now?

**No — STOP for now.** Two reasons: (a) a GNN is a *new ML model*, which contradicts the standing "no new ML models" constraint and the frozen-layer governance — it requires explicit approval and a leakage/retraining review; (b) a GNN over the operational state is pointless while ORS is single-feature-dominated (F-001) and tiers are pure terciles (F-008) — there is little independent graph signal to learn from. Defer GNN until F-001/F-008 are fixed and new-ML approval is granted.

## 9. Should any existing Layer 7 module be patched before new work begins?

**Yes.** Patch the M1 scoring (F-001 ORS, F-002/F-003 ASS) and the M5/M6 confidence/impact scoring (F-004, F-005, F-008, F-009) before building Sensor Fusion / State Estimation / Kalman / Control Policy / GNN, because those modules will consume these scores as state/inputs and inherit the defects. The plumbing, governance, API, and integrity layers do **not** require patching.

---

## Final Recommendation

**PROCEED WITH PATCHES.**

- **Total findings:** 21 (Critical 1, High 4, Medium 10, Low 6).
- **Findings by severity:** Critical = F-001; High = F-002, F-003, F-004, F-005; Medium = F-006, F-007, F-008, F-009, F-010, F-011, F-012, F-013, F-014, F-015; Low = F-016, F-017, F-018, F-019, F-020, F-021.
- **Modules requiring patches before new work:** M1 `operational_state` (ORS), M1 `alert_prioritization` (ASS), M5 `decision_confidence` + `governance`, M6 `twin_confidence` + (reframe) `scenario_simulator`/`plan_comparison`, M2 `override_engine` (approval transitions), and a cross-layer L6/L4.5 integrity-flag gate.
- **No blocking safety/integrity defect found** — Layer 7 is safe to keep running as-is; the patches are about score *quality*, which must precede Sensor Fusion and especially GNN work.
