"""FastAPI backend for the Salary Benchmark interface."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root and backend dir to path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
_BACKEND_DIR = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import players, benchmark

app = FastAPI(
    title="Soccer Salary Benchmark API",
    version="1.0.0",
    description="Predict expected salary range for football players based on market value, age, position, and league.",
)

# CORS — allow local dev servers and any Cloud Run / custom domain.
# In production the frontend is served by nginx on the same origin, so CORS
# is only needed for local development. allow_origins=["*"] is safe here
# because there is no authentication (no cookies, no credentials).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(players.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")

logger = logging.getLogger(__name__)


@app.on_event("startup")
def warm_up_models():
    """Load the player pool + every model variant before serving traffic.

    Each AutoGluon predictor is otherwise lazy-loaded (and cached) on its
    first prediction, deserializing hundreds of MB from disk. Without this,
    the first request that happens to hit a given fallback variant pays that
    cold-load latency live — e.g. mid-demo. Cloud Run's startup probe should
    point at /api/health so no traffic reaches the instance until this
    (synchronous) startup event has finished.
    """
    from salary_benchmark.benchmark import _load_pool
    from salary_benchmark.model import fallback_available, predict_log_salary

    try:
        _load_pool()
    except Exception:
        logger.exception("Startup warm-up: failed to load player pool")

    for variant in ("full", "no_mv", "no_mv_no_pos", "no_mv_no_age"):
        if variant != "full" and not fallback_available(variant):
            continue
        try:
            predict_log_salary({}, variant=variant)
        except Exception:
            logger.exception("Startup warm-up: failed to load model variant %r", variant)


@app.get("/api/health")
def health():
    return {"status": "ok"}
