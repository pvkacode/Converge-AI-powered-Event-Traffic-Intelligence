# Vercel Deployment — ASTraM Dashboard

Deploy the **Next.js** dashboard (`dashboard/`) to Vercel. The FastAPI API runs separately on Render.

---

## Critical settings

| Setting | Value |
|---------|-------|
| **Root Directory** | `dashboard` |
| **Framework** | Next.js (auto-detected) |
| **Build Command** | `npm run build` (default) |
| **Output** | `.next` (default) |

`dashboard/vercel.json` is included for reference; Vercel auto-detects Next.js.

---

## Step-by-step

### 1. Push to GitHub

Ensure `outputs/` is committed (~58 MB, required for all layer pages).

```bash
git ls-files outputs | head
```

### 2. Import on Vercel

1. [vercel.com](https://vercel.com) → **Add New Project**.
2. Import your GitHub repo.
3. **Root Directory:** click **Edit** → set to `dashboard`.
4. **Framework Preset:** Next.js.

### 3. Environment variables

| Name | Value | Notes |
|------|-------|-------|
| `NEXT_PUBLIC_API_URL` | `https://astram-api.onrender.com` | Your Render API URL, **no trailing slash** |

Optional:

| Name | Value |
|------|-------|
| `OUTPUTS_DIR` | Only if you mount data elsewhere |
| `DATA_DIR` | Only if you ship `events_clean.csv` |

### 4. Deploy

Vercel sets `VERCEL=1` during build → `prebuild` runs `scripts/sync-outputs.mjs` → copies `../outputs` into `dashboard/outputs/` so serverless functions can read CSVs.

### 5. Post-deploy smoke test

- `/` — overview KPIs load
- `/layer1` — duration lookup table
- `/layer7` — spillover page
- `/worked-example` — dropdowns populate (API must be up)

---

## How outputs reach Vercel

Vercel only bundles files under the **Root Directory** (`dashboard/`). Sibling `../outputs` is **not** included in serverless bundles.

**Fix:** `dashboard/scripts/sync-outputs.mjs` copies `outputs/` → `dashboard/outputs/` when `VERCEL=1`.

Locally, dev uses `../outputs` directly (no copy). `dashboard/outputs/` is gitignored.

---

## Custom domain

After deploy:

1. Vercel → Project → **Domains** → add domain.
2. Update Render `CORS_ORIGINS` to include the new domain.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Pages load but all tables empty | Check build logs for `[sync-outputs]`; ensure `outputs/` in git |
| Worked Example: "API error" | Set `NEXT_PUBLIC_API_URL`; redeploy after env change |
| Build > size limit | Trim unused outputs or upgrade Vercel plan |
| `events_clean` KPIs show fallbacks | Expected — CSV is gitignored; optional upload via `DATA_DIR` |

---

## Local production preview

```bash
cd dashboard
export NEXT_PUBLIC_API_URL=https://your-api.onrender.com
SYNC_OUTPUTS=1 npm run build
npm start
```

---

## What does NOT go on Vercel

- FastAPI (`api/`) → Render
- Pipeline scripts (`src/`) → run locally or CI, not Vercel
- CatBoost / pickle models → not needed for read-only dashboard
