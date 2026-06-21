# ASTraM / Converge — Deployment Audit

**Date:** 2026-06-16  
**Repo:** monorepo at `converge/` (GitHub: `Converge-AI-powered-Event-Traffic-Intelligence`)

---

## Executive summary

| Service | Stack | Deploy target | Data dependency |
|---------|-------|---------------|-----------------|
| **Dashboard** | Next.js 15, React 19, Leaflet, Recharts | **Vercel** | `outputs/` CSVs (~58 MB, 257 tracked files) via Node `fs` |
| **Inference API** | FastAPI, uvicorn | **Render** | Same `outputs/` + optional `src/` + `data/` for live Layer 1 |

Recommended hackathon architecture:

```
Users → Vercel (dashboard/) → browser fetch → Render (api/) → outputs/
         └─ server-side CSV reads ─────────────────────────→ outputs/
```

---

## 1. Frontend

| Item | Value |
|------|-------|
| Framework | Next.js 15 App Router |
| Location | `dashboard/` |
| Entry | `dashboard/app/layout.tsx`, `dashboard/app/page.tsx` |
| Dev | `cd dashboard && npm run dev` → `localhost:3000` |
| Build | `npm run build` (runs `prebuild` → syncs `outputs/` on Vercel) |

**How data loads**

- **Most pages:** Server Components read CSVs via `dashboard/lib/csv.ts` (`fs` + PapaParse).
- **DataTables:** Client → `GET /api/dataset` (Next.js route) → full CSV in memory, paginated.
- **Worked Example only:** Client → FastAPI at `NEXT_PUBLIC_API_URL` (`dashboard/lib/api.ts`).

**Environment variables**

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | **Yes in prod** | `http://127.0.0.1:8000` | FastAPI base URL for Worked Example |
| `OUTPUTS_DIR` | No | auto-detect `outputs/` | Override CSV root |
| `DATA_DIR` | No | `../data/` | `events_clean.csv` for overview KPIs |
| `PORT` | No | `3000` | Dev/start port |

---

## 2. Backend

| Item | Value |
|------|-------|
| Framework | FastAPI |
| Location | `api/main.py` |
| Requirements | `api/requirements.txt` (thin: fastapi, uvicorn, pandas, numpy) |
| Dev | `cd api && pip install -r requirements.txt && uvicorn main:app --port 8000` |
| Docs | `http://127.0.0.1:8000/docs` |

**Endpoints**

- `GET /health`
- `GET /api/options`
- `POST /api/worked-example`

**Environment variables**

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `CORS_ORIGINS` | Prod recommended | `*` | Comma-separated Vercel URL(s) |
| `OUTPUTS_DIR` | No | `$ASTRAM_ROOT/outputs` | CSV root |
| `ASTRAM_ROOT` | No | repo root | Monorepo root |
| `DATA_DIR` | No | `$ASTRAM_ROOT/data` | Parquet for live Layer 1 |
| `PORT` | Render sets | `8000` locally | uvicorn bind port |

**Live Layer 1 (optional on Render)**

Requires full pipeline deps (`requirements.txt` root) + `data/events_clean.parquet` (gitignored). Without them, API falls back to `duration_lookup.csv` and labels provenance `fallback`. Fine for demo.

---

## 3. Model & artifact files

| Location | Size | Needed at runtime? |
|----------|------|-------------------|
| `outputs/layer45_model_artifacts/` | ~12 MB | **No** (dashboard + API use CSVs) |
| `outputs/layer7_model_artifacts/` | ~3 MB | **No** |
| `outputs/*.pkl`, `*.cbm`, `*.gpickle` | ~15 MB total | **No** for read-only serving |
| `outputs/*.csv`, `frontend/*.csv` | ~58 MB total | **Yes** |

Pipeline re-runs need root `requirements.txt` (CatBoost, lifelines, faiss, etc.).

---

## 4. Deployment blockers (and mitigations)

| # | Blocker | Severity | Mitigation |
|---|---------|----------|------------|
| 1 | Vercel bundles only `dashboard/` — `../outputs` not included | **Critical** | `dashboard/scripts/sync-outputs.mjs` copies on `VERCEL=1` during `prebuild` |
| 2 | Browser calls `127.0.0.1:8000` without env | **Critical** | Set `NEXT_PUBLIC_API_URL` in Vercel to Render URL |
| 3 | `data/events_clean.csv` gitignored | Low | KPI cards use documented fallbacks |
| 4 | Full CSV loaded in memory for tables | Medium | OK for hackathon (~58 MB); watch layer45/layer7 tables |
| 5 | Two services to operate | Medium | `render.yaml` + Vercel root=`dashboard` |
| 6 | Render free tier cold starts | Low | Accept ~30s first request; show loading state |
| 7 | CORS `*` in dev | Low | Set `CORS_ORIGINS` on Render to Vercel domain |

---

## 5. Hardcoded paths audit

| Path | File | Status |
|------|------|--------|
| `http://127.0.0.1:8000` | `dashboard/lib/api.ts` | Default only; override via env |
| `outputs/` resolution | `dashboard/lib/csv.ts` | Supports `OUTPUTS_DIR` + `../outputs` |
| `ROOT/OUT/DATA` | `api/main.py` | Supports `ASTRAM_ROOT`, `OUTPUTS_DIR`, `DATA_DIR` |
| Port 3000 | `dashboard/package.json` | Uses `${PORT:-3000}` |

No absolute `/Users/...` paths in code.

---

## 6. Risk assessment

| Risk | Likelihood | Impact | Notes |
|------|------------|--------|-------|
| Vercel deploy size > 100 MB | Medium | Build fail | `outputs/` ~58 MB + node_modules; monitor first deploy |
| Worked Example broken if API down | High | Partial outage | Rest of dashboard still works (read-only CSV) |
| Stale outputs after pipeline run | Medium | Wrong numbers | Re-commit `outputs/` or re-deploy |
| Live Layer 1 unavailable on Render | High | Fallback only | Expected without parquet + lifelines |

---

## 7. Pre-deploy checklist

```bash
# Local smoke test
cd dashboard && npm run build && npm start
cd api && uvicorn main:app --port 8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:3000

# Verify outputs committed
git ls-files outputs | wc -l   # expect ~257
```

---

## 8. Related docs

- [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) — step-by-step Render API
- [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md) — step-by-step Vercel dashboard
- [dashboard/README.md](./dashboard/README.md) — local dev
- [api/README.md](./api/README.md) — API semantics

---

## 9. File changes made for deployment

| File | Change |
|------|--------|
| `api/main.py` | `CORS_ORIGINS`, `OUTPUTS_DIR`, `ASTRAM_ROOT`, `DATA_DIR` env support |
| `dashboard/lib/csv.ts` | `OUTPUTS_DIR` env |
| `dashboard/lib/events-clean.ts` | `DATA_DIR` env |
| `dashboard/scripts/sync-outputs.mjs` | Vercel outputs copy |
| `dashboard/vercel.json` | Vercel project hints |
| `render.yaml` | Render Blueprint |
| `dashboard/.env.example`, `api/.env.example` | Env templates |
