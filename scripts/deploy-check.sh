#!/usr/bin/env bash
# Quick pre-deploy smoke check. Run from repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== ASTraM deploy check =="

echo -n "outputs/ tracked files: "
git ls-files outputs 2>/dev/null | wc -l | tr -d ' '

echo -n "outputs/ size: "
du -sh outputs 2>/dev/null | cut -f1

if [[ ! -d outputs/frontend ]]; then
  echo "FAIL: outputs/frontend missing"
  exit 1
fi

echo "OK: outputs/frontend exists"

if [[ ! -f api/main.py ]]; then
  echo "FAIL: api/main.py missing"
  exit 1
fi

if [[ ! -f dashboard/package.json ]]; then
  echo "FAIL: dashboard/package.json missing"
  exit 1
fi

echo ""
echo "API health (if running on :8000):"
curl -sf http://127.0.0.1:8000/health 2>/dev/null || echo "  (API not running — start with: cd api && uvicorn main:app --port 8000)"

echo ""
echo "Dashboard build test (optional — pass --build to run):"
if [[ "${1:-}" == "--build" ]]; then
  cd dashboard
  SYNC_OUTPUTS=1 npm run build
  echo "OK: dashboard build succeeded"
fi

echo ""
echo "Next steps:"
echo "  1. Render:  see RENDER_DEPLOYMENT.md"
echo "  2. Vercel:   see VERCEL_DEPLOYMENT.md  (Root Directory = dashboard)"
echo "  3. Set NEXT_PUBLIC_API_URL on Vercel to your Render URL"
echo "  4. Set CORS_ORIGINS on Render to your Vercel URL"
