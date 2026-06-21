# Converge / ASTraM inference API

A thin, **separate** FastAPI service that powers the live Worked Example. It does not
modify, re-run, or import-for-side-effect any pipeline script. It only:

- imports the import-safe `src/layer1_survival.py` and calls its real functions, and
- reads the precomputed `outputs/` CSVs (read-only) for every other layer.

## Run

```bash
cd api
python -m pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

The dashboard (Next.js, port 3000) calls this service at `http://127.0.0.1:8000` by
default. Override with `NEXT_PUBLIC_API_URL` in `dashboard/.env.local` if needed.

Start order does not matter; the dashboard shows a clear "start the inference API"
state if the service is down, and the rest of the read-only dashboard keeps working.

## Endpoints

- `GET  /health` - status + whether the live Layer 1 engine initialised.
- `GET  /api/options` - distinct causes, corridors, zones for the form dropdowns.
- `POST /api/worked-example` - body `{ cause, corridor, hour_local, dow_local,
  requires_road_closure, priority }`; returns one section per layer plus a
  `provenance` map and a synthesised recommendation.

## Live vs precomputed, per layer (honest)

| Layer | Mode | Why |
|------|------|-----|
| **1 Duration** | **LIVE** when the pipeline venv + `data/events_clean.parquet` are present | `src/layer1_survival.py` is import-safe (functions only, `__main__`-guarded). The API builds Kaplan-Meier strata once at startup, then calls `lookup_expected_duration(cause, corridor, km_table, km_fallback, quantile)` per request. If lifelines/pyarrow or the parquet are missing it falls back to `duration_lookup.csv` and labels the layer `fallback`. |
| 2 Spatial | precomputed | `layer2_hotspots.py` writes outputs at import (no `__main__` guard). Served from `hotspot_rankings.csv` + `operational_burden.csv`, keyed by the corridor's highest-burden junction. |
| 3 Resources | precomputed | `layer3_resource_optimization.py` and `layer3_corridor_fragility.py` run the full pipeline and write `outputs/` at import. Served from `risk_scores.csv`, `layer3_full_dashboard.csv`, `corridor_fragility.csv`. |
| 4 Event | precomputed | `layer4_*` modules execute and write at import. Served from `planned_event_recommendations.csv`. |
| 4.5 Fusion | precomputed | `run_deployment_inference` is batch-only, needs CatBoost artifacts, and writes to `outputs/`. Served from `layer45_operational_state_vector_normalized.csv`. |
| 5 Optimization | precomputed | MILP solve takes minutes; far too slow for a live request. Served from `layer5_frontend_export.csv` (nearest precomputed allocation), clearly labelled. |
| 6 Learning | precomputed | Monitoring is computed over the feedback log offline. Served from `layer6_model_health_summary.csv` + `layer6_drift_report.csv`. |
| 7 Spillover | precomputed | Multivariate Hawkes fit is batch/slow. Served from `layer7_expected_risk_index.csv` (the requested hour selects the top-ERI zone), `layer7_spillover_centrality.csv`, `layer7_top_k_early_warning.csv`. |

Only Layer 1 can genuinely run live in this codebase; everything else is import-unsafe
(writes outputs) or batch-only/too-slow, so it is served from the existing exports and
labelled `precomputed_lookup`. The provenance flags in the API response are the source of
truth shown by the UI.
