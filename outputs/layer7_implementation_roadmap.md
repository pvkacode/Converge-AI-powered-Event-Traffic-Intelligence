# Layer 7 — Phase 11: Implementation Roadmap

Branch: `layer7-operations` (already checked out). Additive-only. Each commit is independently runnable and leaves the pipeline green (`validate_consistency.py` + `layer7_validate.py` pass).

## 11.1 Implementation order & commits

| Commit | Scope | New files | Depends on | Runtime |
|--------|-------|-----------|------------|---------|
| **C1** | Config + read-only loader + ingestion manifest | `layer7_config.py`, `layer7_feedback_store.py` | — | <2 s |
| **C2** | Operational State Engine + ORS | `layer7_operational_state.py` | C1 | <5 s |
| **C3** | Alerting Engine + ASS (normalize/dedupe/rank) | `layer7_alerting.py` | C1 | <2 s |
| **C4** | Explainability Engine | `layer7_explainability.py` | C2, C3 | <5 s |
| **C5** | Override Engine + OIS (ledger + impact) | `layer7_override_engine.py` | C1 | <2 s |
| **C6** | Sensor Fusion stub (seam, JOSV passthrough) | `layer7_sensor_fusion.py` | C1 | <2 s |
| **C7** | Exports → `outputs/frontend/layer7_*` | `layer7_exports.py` | C2–C5 | <2 s |
| **C8** | Orchestrator + Validator | `layer7_operations.py`, `layer7_validate.py` | C1–C7 | <30 s |
| **C9** | Digital Twin (Tier 2, what-if re-eval) | `layer7_digital_twin.py` | C1, C2 | <10 s |
| **C10** | Read-only API (Tier 1, optional dep) | `layer7_api.py` | C7 | service |
| **C11** | Tests + docs | `tests/layer7/*`, `docs/LAYER7.md` | all | — |

## 11.2 Milestones

- **M1 — Operational MVP (C1–C4, C7–C8):** State + Alerts + Explainability + Exports + Orchestrator + Validator. This alone satisfies the core mission (Operational Intelligence, Monitoring, Alerting, Explainability, Dashboard backend) and is the hackathon-demoable deliverable.
- **M2 — Human-in-the-loop (C5, C6):** Override Engine + Sensor Fusion seam.
- **M3 — API + Simulation (C9, C10):** Digital Twin + read-only API.
- **M4 — Hardening (C11):** full test suite, golden tests, docs, optional CI hook.

## 11.3 Estimated end-to-end runtime
Full `layer7_operations.main()` over committed `outputs/`: **< 60 s** (no model inference, no MILP, no retraining). Each engine individually < 10 s.

## 11.4 Dependency posture
- Core (C1–C9): **no new packages** — pandas/numpy/scipy/networkx already in `requirements.txt`.
- API (C10): optional `fastapi`+`uvicorn`, degrade-if-absent; add to `requirements.txt` only with explicit approval.

## 11.5 Definition of done (per commit)
1. New files only; `git status` shows zero modifications to existing `src/` or `outputs/` (except newly added `layer7_*`).
2. `layer7_validate.py` namespace + schema checks pass.
3. `validate_consistency.py` still passes (clean parquet untouched).
4. Compatibility test: SHA-256 of all non-`layer7_` files unchanged after run.
5. Idempotent re-run (diff only in `generated_at`).

## 11.6 Suggested commit messages
```
C1  Layer 7: config + read-only feedback store + ingestion manifest (additive)
C2  Layer 7: Operational State Engine + Operational Risk Score
C3  Layer 7: Alerting Engine + unified Alert Severity Score
C4  Layer 7: Explainability Engine (provenance traces, no new model)
C5  Layer 7: Human Override Engine + Override Impact Score (append-only ledger)
C6  Layer 7: Sensor Fusion seam (inverse-variance, JOSV passthrough)
C7  Layer 7: dashboard exports → outputs/frontend/layer7_*
C8  Layer 7: orchestrator + read-only validator
C9  Layer 7: Digital Twin what-if re-evaluation (reuses L5 formulas, no MILP re-solve)
C10 Layer 7: read-only file-backed API (optional fastapi dep)
C11 Layer 7: tests + docs/LAYER7.md
```

## 11.7 Risk-ordered sequencing rationale
Build the **loader first** (everything depends on safe ingestion), then the **two highest-value/lowest-risk engines** (State, Alerting), then Explainability (binds them), then the human-in-the-loop and future seams, then API/Twin. Validator and namespace guard land with the orchestrator so the freeze guarantee is enforced from the first integrated run.
