"""Player search and detail endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schemas import BenchmarkOptions, PlayerSearchResult, PlayerDetail
from salary_benchmark.benchmark import POOL_PATH, _load_pool as _load_shared_pool

router = APIRouter(tags=["players"])


def _clean_record(record: dict) -> dict:
    cleaned = {}
    for key, value in record.items():
        try:
            if pd.isna(value):
                cleaned[key] = None
                continue
        except (TypeError, ValueError):
            pass
        cleaned[key] = value
    return cleaned


def _load_pool() -> pd.DataFrame:
    """Load the player pool via the benchmark engine's shared cache.

    Sharing one cache guarantees search results and benchmark computations
    always describe the same pool snapshot within a server process.
    """
    try:
        return _load_shared_pool()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Player pool not available. Build it with: "
                   "python build_model_features.py --mode pool "
                   "--input data_full.csv --output player_pool.csv",
        )


@router.get("/players/options", response_model=BenchmarkOptions)
def get_benchmark_options():
    """Distinct positions and competitions for the manual benchmark form."""
    pool = _load_pool()
    positions = sorted(pool["main_position"].dropna().unique().tolist())
    comp_cols = pool[["competition_id", "competition_country"]].copy()
    comp_cols["competition_name"] = pool.get("competition_name", pool["competition_id"])
    comps = (
        comp_cols
        .dropna(subset=["competition_id"])
        .groupby("competition_id", as_index=False)
        .agg(
            competition_name=("competition_name", "first"),
            competition_country=("competition_country", "first"),
            n_players=("competition_id", "size"),
        )
        .sort_values("n_players", ascending=False)
    )
    return {
        "positions": positions,
        "competitions": [
            {
                "id": row["competition_id"],
                "name": row["competition_name"] or row["competition_id"],
                "country": row["competition_country"],
            }
            for _, row in comps.iterrows()
        ],
    }


@router.get("/players/search", response_model=list[PlayerSearchResult])
def search_players(q: str = Query(..., min_length=2), limit: int = Query(20, ge=1, le=50)):
    """Search players by name (case-insensitive substring match)."""
    pool = _load_pool()
    mask = pool["player_name"].str.lower().str.contains(q.lower(), regex=False, na=False)
    results = pool[mask].head(limit)
    cols = [
        "id", "player_name", "main_position", "team_name",
        "competition_id", "competition_country", "nationality",
        "age_months", "market_value_current_eur", "annual_fixed_eur",
    ]
    return [_clean_record(row) for row in results[cols].to_dict(orient="records")]


@router.get("/players/{player_id}", response_model=PlayerDetail)
def get_player(player_id: int):
    """Get full player details by ID (row index)."""
    pool = _load_pool()
    if player_id < 0 or player_id >= len(pool):
        raise HTTPException(status_code=404, detail="Player not found")
    return _clean_record(pool.iloc[player_id].to_dict())
