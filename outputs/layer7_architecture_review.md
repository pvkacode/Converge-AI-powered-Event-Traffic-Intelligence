# Layer 7 — Phase 3: Architecture Review

Layer 7 = **Operational Intelligence**. It consumes frozen L1–L6 outputs and produces operator-facing state, alerts, override accounting, explanations, an API surface, and (future) simulation. It introduces **no new predictive ML model** and never retrains or mutates upstream.

## 3.1 Component classification

| # | Component | Classification | Rationale |
|---|-----------|----------------|-----------|
| 1 | **Sensor Fusion Engine** | **Future-only** | No live sensor feed exists; ASTraM is a historical batch. Build a stub interface that today fuses only the L4.5 JOSV ("the sensors"), so the seam exists for future real sensors. Not on MVP critical path. |
| 2 | **Operational State Engine** | **Mandatory / MVP** | The spine. Joins L4.5 JOSV + L5 allocation + L3 context + L6 health into one per-site operational state table. Everything else depends on it. |
| 3 | **Alerting Engine** | **Mandatory / MVP** | Highest-value, lowest-risk: normalizes L6 `active_alerts` + `retrain_triggers` + L5 `chance_constraint_violations` into one deduplicated, severity-ranked feed. |
| 4 | **Human Override Engine** | **MVP (lightweight)** | Human-in-the-loop is core to the mission. MVP = an append-only override ledger + impact accounting against L5 plan (no live write-back to optimizer). Full closed-loop = research-grade. |
| 5 | **Explainability Engine** | **MVP** | Assembles "why" narratives per recommendation/alert by tracing existing L6 posteriors, L5 robustness, L4.5 SHAP/guard reasons. Pure aggregation — no new model. |
| 6 | **API Layer** | **Optional (MVP-thin)** | A read-only file-backed service (FastAPI) exposing the L7 outputs. MVP can be a thin layer over CSVs; full REST/auth is post-hackathon. |
| 7 | **Digital Twin Simulator** | **Research-grade / future** | A what-if engine re-running L5-style scenario math under operator-perturbed inputs **without** re-solving the frozen optimizer. High value, high effort; not MVP. |

## 3.2 Recommended build tiers

- **Tier 0 — Hackathon MVP (must-have):** Operational State Engine → Alerting Engine → Explainability Engine → Exports (+ thin read-only API).
- **Tier 1 — MVP+:** Human Override Engine (ledger + impact accounting), Dashboard backend exports for all 6 pages.
- **Tier 2 — Research-grade:** Digital Twin Simulator (deterministic re-evaluation of CVaR/delay under overrides, reusing L5 formulas read-only).
- **Tier 3 — Future:** Sensor Fusion Engine with real telemetry adapters; closed-loop override write-back; live streaming API.

## 3.3 Architectural principles (carried from L1–L6)
1. **Pure consumption.** Read CSV/JSON/parquet only; never import upstream module internals that trigger recomputation.
2. **Separate namespace.** Write only to `outputs/layer7_*` and `outputs/frontend/layer7_*`.
3. **Idempotent & deterministic.** Re-running Layer 7 over the same inputs yields identical outputs (except timestamps), like `frontend_exports.py`.
4. **Degrade gracefully.** Any missing/empty/stale upstream file produces a logged warning + a coverage flag, never a crash (mirror `_safe_read`).
5. **No new mathematics that competes with L1–L6.** L7 scores are deterministic compositions (weighted sums, normalizations, rule logic) of existing outputs — auditable, not learned.

## 3.4 Is the proposed architecture still appropriate?
**Yes, with two refinements:**
- Demote **Sensor Fusion** and **Digital Twin** from MVP to future/research tiers — there is no live sensor stream and the optimizer is frozen, so these are seams, not deliverables.
- Promote a **shared read-only loader** (`layer7_feedback_store`-style) as an implicit 8th component — every engine needs the same defensive ingestion, so it should be built first and once.

## 3.5 What Layer 7 explicitly is NOT (guardrails)
- Not a new predictor, not a retrainer, not a replacement for L4.5/L5/L6.
- Does not recompute CVaR/MILP/posteriors from scratch — it reads their results. (The Digital Twin *re-evaluates* using published formulas on operator inputs, but never re-solves the frozen MILP or refits models.)
