"""Benchmark endpoint — runs the salary benchmark for a player."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Ensure salary_benchmark and schemas are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schemas import BenchmarkRequest, BenchmarkResponse, ErrorResponse, ExplanationResponse
from salary_benchmark.benchmark import benchmark_by_id, benchmark_by_name, benchmark_player
from salary_benchmark.explain import explain_by_id, explain_by_name, explain_player

router = APIRouter(tags=["benchmark"])

logger = logging.getLogger(__name__)

_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "Invalid request (e.g. manual benchmark without age_months)",
    },
    404: {"model": ErrorResponse, "description": "Player not found"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
}


def _manual_player_dict(req: BenchmarkRequest) -> dict:
    """Build the engine player dict for a manual (custom) benchmark request.

    Raises HTTPException(400) when age is missing: age is the model's second
    most important feature — refusing is more honest than silently assuming
    a 25-year-old.
    """
    if req.age_months is None:
        raise HTTPException(
            status_code=400,
            detail="age_months is required for manual benchmarks",
        )
    import numpy as np
    mv = req.market_value_current_eur
    return {
        "player_name": "custom_player",
        "main_position": req.main_position,
        "competition_id": req.competition_id or "",
        "competition_country": req.competition_country or "",
        "age_months": req.age_months,
        # Market value optional: without it the engine routes to the
        # no-MV fallback model (if trained).
        "market_value_current_eur": mv,
        "log_market_value_current_eur": float(np.log1p(mv)) if mv is not None else None,
        "has_market_value": 1 if mv is not None else 0,
        "annual_fixed_eur": req.annual_fixed_eur,
    }


@router.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    responses=_ERROR_RESPONSES,
)
def run_benchmark(req: BenchmarkRequest):
    """Run salary benchmark for a player.

    Prefer player_id for UI calls because player names are not unique. Name
    lookup is kept for CLI/manual calls.
    """
    try:
        kwargs = {"range_width": req.range_width}
        if req.full_comparables:
            kwargs["n_comparables_display"] = None  # all comparables
        if req.player_id is not None:
            return benchmark_by_id(req.player_id, **kwargs)
        if req.player_name:
            return benchmark_by_name(req.player_name, **kwargs)
        if req.main_position:
            return benchmark_player(_manual_player_dict(req), **kwargs)
        raise HTTPException(status_code=400, detail="Provide player_name, player_id, or manual fields")
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except HTTPException:
        raise
    except Exception:
        # Log the full traceback server-side; never leak internals to clients.
        logger.exception("Benchmark request failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/benchmark/explain",
    response_model=ExplanationResponse,
    responses=_ERROR_RESPONSES,
)
def explain_benchmark(req: BenchmarkRequest):
    """Explain why the model predicts this player's salary (SHAP).

    Accepts the same identification as /benchmark (player_id, player_name,
    or manual fields) and returns per-feature contributions. Contributions
    are additive in log-salary space; `pct_effect` expresses each one as a
    multiplicative percentage on the salary estimate. `predicted_salary_eur`
    is the raw model estimate — the benchmark's calibrated median may differ
    slightly. Warm requests take ~0.2s; the first request per model variant
    builds the explainer (~2-3s).
    """
    try:
        if req.player_id is not None:
            return explain_by_id(req.player_id)
        if req.player_name:
            return explain_by_name(req.player_name)
        if req.main_position:
            return explain_player(_manual_player_dict(req))
        raise HTTPException(status_code=400, detail="Provide player_name, player_id, or manual fields")
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Explanation request failed")
        raise HTTPException(status_code=500, detail="Internal server error")
