# Render Deployment — ASTraM Inference API

Deploy the **FastAPI** service (`api/main.py`) to Render. The dashboard stays on Vercel.

---

## What Render runs

```
Repo root (monorepo)
├── api/main.py          ← uvicorn entry
├── api/requirements.txt ← pip install
├── outputs/             ← CSVs (must be in git)
├── src/                 ← optional (live Layer 1)
└── data/                ← optional (parquet gitignored)
```

**Start command:**

```bash
cd api && uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Health check:** `GET /health`

---

## Option A — Blueprint (fastest)

1. Push monorepo to GitHub.
2. [render.com](https://render.com) → **New** → **Blueprint**.
3. Connect repo — Render reads `render.yaml` at repo root.
4. Set `CORS_ORIGINS` when prompted (your Vercel URL).
5. Deploy.

---

## Option B — Manual Web Service

1. **New Web Service** → connect GitHub repo.
2. Settings:

| Field | Value |
|-------|-------|
| **Name** | `astram-api` |
| **Region** | Singapore (or nearest) |
| **Branch** | `main` |
| **Root Directory** | *(leave blank — repo root)* |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r api/requirements.txt` |
| **Start Command** | `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT` |

3. **Environment variables:**

| Key | Value |
|-----|-------|
| `PYTHON_VERSION` | `3.11.9` |
| `CORS_ORIGINS` | `https://your-app.vercel.app` |
| `OUTPUTS_DIR` | `/opt/render/project/src/outputs` |
| `ASTRAM_ROOT` | `/opt/render/project/src` |

4. **Health Check Path:** `/health`
5. Deploy.

---

## Verify

```bash
curl https://astram-api.onrender.com/health
# {"status":"ok","layer1_live":false,...}

curl https://astram-api.onrender.com/api/options
```

Copy the service URL — you need it for Vercel `NEXT_PUBLIC_API_URL`.

---

## Optional: live Layer 1 on Render

Default install uses thin `api/requirements.txt` (no lifelines). For genuine live Kaplan–Meier:

1. Change build command:

   ```bash
   pip install -r requirements.txt
   ```

2. Upload `data/events_clean.parquet` via Render disk or build-time secret (not in git).
3. Set `DATA_DIR=/opt/render/project/src/data`.

For hackathon demos, **precomputed lookup is fine** — no extra setup.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `outputs_present: false` in `/health` | Ensure `outputs/` is committed and `OUTPUTS_DIR` points to it |
| CORS error from Vercel | Set `CORS_ORIGINS` to exact Vercel URL (no trailing slash) |
| 502 on cold start | Free tier sleeps; first request takes ~30s |
| Module not found | Start from `api/` or set `PYTHONPATH` |

---

## Cost

Render **free** tier: 750 hrs/month, spins down after inactivity. Fine for hackathon demos.
