"""FastAPI backend for the Salary Benchmark interface."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
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
from fastapi.responses import JSONResponse

from routers import players, benchmark, meta
from schemas import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

# Warm-up outcome, written once by the lifespan handler and read by the
# health endpoints. Model states: "loaded", "failed", "not_trained".
WARMUP_STATE: dict = {"completed": False, "pool_loaded": False, "models": {}}

ALL_VARIANTS = ("full", "no_mv", "no_mv_no_pos", "no_mv_no_age")


def _warm_up() -> None:
    """Load the player pool + every model variant before serving traffic.

    Each AutoGluon predictor is otherwise lazy-loaded (and cached) on its
    first prediction, deserializing hundreds of MB from disk. Without this,
    the first request that happens to hit a given fallback variant pays that
    cold-load latency live — e.g. mid-demo. Failures are recorded in
    WARMUP_STATE so /api/health/ready can report them instead of silently
    serving a broken instance.
    """
    from salary_benchmark.benchmark import _load_pool
    from salary_benchmark.model import fallback_available, predict_log_salary

    try:
        _load_pool()
        WARMUP_STATE["pool_loaded"] = True
    except Exception:
        logger.exception("Startup warm-up: failed to load player pool")
        WARMUP_STATE["pool_loaded"] = False

    for variant in ALL_VARIANTS:
        if variant != "full" and not fallback_available(variant):
            WARMUP_STATE["models"][variant] = "not_trained"
            continue
        try:
            predict_log_salary({}, variant=variant)
            WARMUP_STATE["models"][variant] = "loaded"
        except Exception:
            logger.exception("Startup warm-up: failed to load model variant %r", variant)
            WARMUP_STATE["models"][variant] = "failed"

    # Pre-build SHAP explainers so the first "Why this estimate?" click is
    # ~0.2s instead of ~3s. Failures never block serving — explanations are
    # a bonus feature, and the endpoint will retry the build on demand.
    try:
        from salary_benchmark.explain import _get_explainer
        for variant, state in WARMUP_STATE["models"].items():
            if state == "loaded":
                _get_explainer(variant)
    except Exception:
        logger.exception("Startup warm-up: failed to pre-build SHAP explainers")

    WARMUP_STATE["completed"] = True


def _is_ready() -> bool:
    """Ready = warm-up finished, pool loaded, and no model failed to load.

    Untrained fallbacks don't block readiness — the engine refuses those
    players with a clear message — but a *failed* load (artifact present yet
    unloadable) means the instance would serve errors, so it is not ready.
    """
    return (
        WARMUP_STATE["completed"]
        and WARMUP_STATE["pool_loaded"]
        and WARMUP_STATE["models"].get("full") == "loaded"
        and all(state != "failed" for state in WARMUP_STATE["models"].values())
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up()
    yield

API_DESCRIPTION = """
Predict the expected salary range for football players based on market value,
age, position, and league, using AutoGluon models trained on Capology salary
data enriched with Transfermarkt market values.

## Quickstart

1. `GET /api/players/search?q=haaland` — find a player and note their `id`.
2. `POST /api/benchmark` with `{"player_id": <id>}` — get the expected salary
   range, percentile, OVERPAID/UNDERPAID/FAIRLY_PAID status, and comparables.

For hypothetical players, POST manual fields instead (`main_position` and
`age_months` are required; `market_value_current_eur` is optional — without it
the engine routes to a fallback model trained without market value).

## Authentication

The API itself has no authentication — there are no accounts, cookies, or
API keys, and CORS allows any origin.

On Cloud Run, access is controlled by IAM *in front of* the app: if the
service has the `allUsers` → `roles/run.invoker` binding it is public;
otherwise callers must send a Google identity token
(`Authorization: Bearer $(gcloud auth print-identity-token)`).

## Errors

All errors use the shape `{"detail": "<message>"}`:

| Status | Meaning |
|--------|---------|
| 400 | Invalid request the schema can't catch (e.g. manual benchmark without `age_months`) |
| 404 | Player not found (unknown `player_id` or no name match) |
| 422 | Request body/params failed schema validation |
| 500 | Unexpected server error (details logged server-side only) |
| 503 | Player pool not available (server misconfigured / data missing) |

## Interpretation notes

Salary predictions describe the *market*, which may embed biases (nationality
is a model feature). Treat OVERPAID/UNDERPAID as market descriptions, never as
normative advice for individual contract decisions.
"""

app = FastAPI(
    title="Soccer Salary Benchmark API",
    version="1.0.0",
    description=API_DESCRIPTION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "players", "description": "Search the player pool and fetch player details."},
        {"name": "benchmark", "description": "Run the salary benchmark for a real or hypothetical player."},
        {"name": "meta", "description": "Service health and metadata."},
    ],
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
app.include_router(meta.router, prefix="/api")


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["meta"],
    responses={503: {"model": ReadinessResponse, "description": "Warm-up incomplete or failed"}},
)
def health():
    """Legacy health check (used by existing probes). Same semantics as /api/health/ready with a minimal body."""
    if not _is_ready():
        return JSONResponse(status_code=503, content=_readiness_payload())
    return HealthResponse()


@app.get("/api/health/live", response_model=HealthResponse, tags=["meta"])
def health_live():
    """Liveness: the process is up and accepting connections. Never checks models or data."""
    return HealthResponse()


def _readiness_payload() -> dict:
    return {
        "status": "ready" if _is_ready() else "not_ready",
        "pool_loaded": WARMUP_STATE["pool_loaded"],
        "models": dict(WARMUP_STATE["models"]),
    }


@app.get(
    "/api/health/ready",
    response_model=ReadinessResponse,
    tags=["meta"],
    responses={503: {"model": ReadinessResponse, "description": "Warm-up incomplete or failed"}},
)
def health_ready():
    """Readiness: pool loaded and every trained model variant loadable.

    Returns 503 until warm-up finishes, or if the pool/full model failed to
    load, or if any trained fallback artifact failed to load. Untrained
    fallbacks are reported as "not_trained" and do not block readiness.
    Point Cloud Run's startup probe here (or at /api/health, same semantics).
    """
    payload = _readiness_payload()
    if payload["status"] != "ready":
        return JSONResponse(status_code=503, content=payload)
    return payload
