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
from schemas import HealthResponse

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


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness/startup check. Returns 200 once models are warmed up and the server is accepting traffic."""
    return HealthResponse()
