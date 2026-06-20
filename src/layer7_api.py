"""
Layer 7 — M4: read-only Operational Dashboard API.

Thin, READ-ONLY HTTP layer over the M4 dashboard-backend exports. It serves data;
it never writes, never mutates any layer, and never recomputes a score.

Dependency posture: FastAPI/uvicorn are OPTIONAL (listed in requirements-layer7.txt).
The data-provider functions are pure and importable WITHOUT FastAPI, so the backend
and validation work even when the web framework is absent (degrade-if-absent).

Run the server (after installing optional deps and running the backend):
    pip install -r requirements-layer7.txt
    python src/layer7_dashboard_backend.py
    python src/layer7_api.py            # serves http://127.0.0.1:8000
"""

from __future__ import annotations

import json

import pandas as pd

from layer7_config import FRONT

try:  # optional web framework
    from fastapi import FastAPI
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover - optional dep
    _HAS_FASTAPI = False

# endpoint -> exported file
_PAGE_FILES = {
    "overview": "layer7_operations_overview.csv",
    "alerts": "layer7_active_alerts.csv",
    "recommendations": "layer7_resource_recommendations.csv",
    "overrides": "layer7_override_history.csv",
    "health": "layer7_model_health.csv",
    "counterfactuals": "layer7_counterfactuals.csv",
}


# ----------------------------------------------------------------- pure providers (no dep)
def _records(fname: str) -> list[dict]:
    path = FRONT / fname
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


def get_page(name: str) -> list[dict]:
    if name not in _PAGE_FILES:
        raise KeyError(name)
    return _records(_PAGE_FILES[name])


def get_manifest() -> dict:
    path = FRONT / "layer7_dashboard_manifest.json"
    if not path.exists():
        return {"error": "manifest not found; run layer7_dashboard_backend.py first"}
    return json.loads(path.read_text(encoding="utf-8"))


def available() -> bool:
    return _HAS_FASTAPI


# ----------------------------------------------------------------- app factory (needs dep)
def create_app():
    """Build the read-only FastAPI app. Raises if FastAPI is not installed."""
    if not _HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI not installed. Install optional deps: "
            "pip install -r requirements-layer7.txt"
        )
    app = FastAPI(
        title="ASTraM Layer 7 Operational Dashboard API",
        description="Read-only. Serves M4 dashboard-backend exports. No writes, no recompute.",
        version="M4",
    )

    @app.get("/")
    def root() -> dict:
        return {
            "service": "ASTraM Layer 7 Dashboard API (read-only)",
            "endpoints": ["/manifest"] + [f"/{p}" for p in _PAGE_FILES],
        }

    @app.get("/manifest")
    def manifest() -> dict:
        return get_manifest()

    @app.get("/overview")
    def overview() -> list[dict]:
        return get_page("overview")

    @app.get("/alerts")
    def alerts() -> list[dict]:
        return get_page("alerts")

    @app.get("/recommendations")
    def recommendations() -> list[dict]:
        return get_page("recommendations")

    @app.get("/overrides")
    def overrides() -> list[dict]:
        return get_page("overrides")

    @app.get("/health")
    def health() -> list[dict]:
        return get_page("health")

    @app.get("/counterfactuals")
    def counterfactuals() -> list[dict]:
        return get_page("counterfactuals")

    return app


def expected_routes() -> set[str]:
    return {"/", "/manifest", "/overview", "/alerts", "/recommendations",
            "/overrides", "/health", "/counterfactuals"}


if __name__ == "__main__":  # pragma: no cover
    if not _HAS_FASTAPI:
        print("[SKIP] FastAPI not installed. pip install -r requirements-layer7.txt")
    else:
        import uvicorn

        uvicorn.run(create_app(), host="127.0.0.1", port=8000)
