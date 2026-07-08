"""Main benchmark entry point.

Given a player dict, player id, or player name from the pool, returns the full
salary benchmark output.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import salary_range
from .comparables import find_comparables, similarity_score
from .confidence import confidence_label, salary_status
from .model import fallback_available, predict_log_salary
from .monitoring import get_logger

POOL_PATH = Path(__file__).resolve().parents[1] / "player_pool.csv"

_POOL: pd.DataFrame | None = None


def _clean_value(value):
    """Convert pandas/numpy scalars and NaN values into JSON-safe Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _clean_record(record: dict) -> dict:
    return {key: _clean_value(value) for key, value in record.items()}


def _load_pool() -> pd.DataFrame:
    global _POOL
    if _POOL is None:
        if not POOL_PATH.exists():
            raise FileNotFoundError(
                f"Player pool not found at {POOL_PATH}. "
                f"Build it first: python build_model_features.py --mode pool "
                f"--input data_full.csv --output player_pool.csv"
            )
        _POOL = pd.read_csv(POOL_PATH)
        if "id" not in _POOL.columns:
            _POOL.insert(0, "id", _POOL.index)
    return _POOL


def select_model_variant(player: dict) -> str:
    """Route a player to the model variant its available features support.

    A cascade of fallback models covers players with missing key features:
      full          — market value present (position + age required)
      no_mv         — no market value
      no_mv_no_pos  — position unknown (market value, if any, unused)
      no_mv_no_age  — age unknown (market value, if any, unused)
    Players missing BOTH position and age are refused: too little signal
    for an estimate anyone should act on.

    Raises ValueError if no variant can serve the player or the required
    fallback artifact is not trained.
    """
    provided = set(k for k, v in player.items() if v is not None)
    has_mv = "market_value_current_eur" in provided
    has_pos = "main_position" in provided
    has_age = "age_months" in provided

    if not has_pos and not has_age:
        raise ValueError(
            "Player has neither main_position nor age_months — no model "
            "variant can produce a defensible estimate. Provide at least one."
        )
    if has_pos and has_age:
        variant = "full" if has_mv else "no_mv"
    elif not has_pos:
        variant = "no_mv_no_pos"
    else:
        variant = "no_mv_no_age"

    if variant != "full" and not fallback_available(variant):
        raise ValueError(
            f"Player is missing features required by the full model and the "
            f"'{variant}' fallback model is not available. Train it with: "
            f"python train_fallback_no_mv.py --variant {variant}"
        )
    return variant


def benchmark_player(player: dict,
                     range_width: str = "normal",
                     n_comparables_display: int | None = 10) -> dict:
    """Run the full salary benchmark for one player.

    Args:
        player: dict with player features. At least one of main_position /
            age_months is required; market_value_current_eur is optional
            (missing features route to the matching fallback model — see the
            variant cascade below). Optional: annual_fixed_eur (actual
            salary for status).
        range_width: "normal" (p25/p75) or "wide" (p10/p90).
        n_comparables_display: max comparable players to return (None = all).

    Returns:
        Full benchmark dict with JSON-safe values.
    """
    player = _clean_record(dict(player))
    _start_time = time.perf_counter()
    pool = _load_pool()

    variant = select_model_variant(player)
    log_pred = predict_log_salary(player, variant=variant)
    sal_range = salary_range(log_pred, width=range_width, variant=variant)
    low = sal_range["expected_salary_low_eur"]
    median = sal_range["expected_salary_median_eur"]
    high = sal_range["expected_salary_high_eur"]

    # --- Comparables ---
    pool_filtered = pool.copy()
    player_id = player.get("id")
    if player_id is not None and "id" in pool_filtered.columns:
        pool_filtered = pool_filtered[pool_filtered["id"] != player_id]
    elif "player_name" in player and "player_name" in pool_filtered.columns:
        pool_filtered = pool_filtered[pool_filtered["player_name"] != player["player_name"]]

    comparables, level_used = find_comparables(player, pool_filtered)

    scores = similarity_score(player, comparables)
    comparables = comparables.copy()
    comparables["similarity_score"] = scores.values
    comparables = comparables.sort_values("similarity_score", ascending=False)

    n_comp = len(comparables)
    avg_sim = float(comparables["similarity_score"].mean()) if n_comp > 0 else 0.0
    n_comparables_with_salary = (
        int(comparables["annual_fixed_eur"].notna().sum())
        if n_comp > 0 and "annual_fixed_eur" in comparables.columns else 0
    )

    # Warning when the range has no peer validation
    benchmark_warning = None
    if n_comp == 0:
        benchmark_warning = "No comparable players found — prediction is model-only with no peer validation"
    elif n_comparables_with_salary == 0:
        benchmark_warning = (
            "Comparable players found, but none has a known salary — "
            "the range is model-only, not peer-validated"
        )
    fallback_notes = {
        "no_mv": (
            "No market value available — estimate uses the fallback model "
            "trained without market value. Expect a wider, less precise range."
        ),
        "no_mv_no_pos": (
            "No position available — estimate uses the fallback model trained "
            "without market value or position. Expect a much wider, less "
            "precise range."
        ),
        "no_mv_no_age": (
            "No age available — estimate uses the fallback model trained "
            "without market value or age. Expect a much wider, less "
            "precise range."
        ),
    }
    if variant in fallback_notes:
        fallback_note = fallback_notes[variant]
        benchmark_warning = (
            f"{fallback_note} {benchmark_warning}" if benchmark_warning else fallback_note
        )

    # --- Confidence ---
    # Count comparables with known salary from same league (honesty check)
    player_league = player.get("competition_id")
    n_same_league_with_salary = 0
    if n_comp > 0 and "annual_fixed_eur" in comparables.columns and player_league:
        same_league_mask = comparables["competition_id"] == player_league
        n_same_league_with_salary = int(
            comparables.loc[same_league_mask, "annual_fixed_eur"].notna().sum()
        )

    rel_width = (high - low) / median if median > 0 else 999
    confidence = confidence_label(
        n_comp, avg_sim, rel_width,
        n_comparables_with_salary_same_league=n_same_league_with_salary,
    )
    # Fallback models are measurably weaker; a HIGH badge would overstate
    # what they deliver. The two-features-missing variants are weaker still,
    # so they are always LOW.
    if variant == "no_mv" and confidence == "high":
        confidence = "medium"
    elif variant in ("no_mv_no_pos", "no_mv_no_age"):
        confidence = "low"

    # --- Salary percentile within comparables ---
    actual_eur = player.get("annual_fixed_eur")
    percentile = None
    if actual_eur is not None and n_comp > 0 and "annual_fixed_eur" in comparables.columns:
        comp_salaries = comparables["annual_fixed_eur"].dropna()
        if len(comp_salaries) > 0:
            # Midrank for ties: a salary equal to every comparable ranks 50th,
            # not 0th (strict < would count ties as "everyone earns more").
            n_below = (comp_salaries < actual_eur).sum()
            n_equal = (comp_salaries == actual_eur).sum()
            percentile = int(round(
                (n_below + 0.5 * n_equal) / len(comp_salaries) * 100
            ))

    # --- Status ---
    status = salary_status(actual_eur, low, high)

    # --- Top comparable players for display ---
    # Prioritize comparables with known salary
    display_cols = [
        "id", "player_name", "main_position", "competition_id", "competition_country",
        "age_months", "market_value_current_eur", "annual_fixed_eur",
        "similarity_score",
    ]
    display_cols = [c for c in display_cols if c in comparables.columns]

    if n_comp > 0 and "annual_fixed_eur" in comparables.columns:
        with_salary = comparables[comparables["annual_fixed_eur"].notna()]
        without_salary = comparables[comparables["annual_fixed_eur"].isna()]
        # Show salary-known first, then fill remaining slots with no-salary
        ordered = pd.concat([with_salary, without_salary])
    else:
        ordered = comparables

    if n_comparables_display is not None:
        ordered = ordered.head(n_comparables_display)
    top_comparables = [
        _clean_record(record)
        for record in ordered[display_cols].to_dict(orient="records")
    ]

    result = {
        "player_id": player.get("id"),
        "player_name": player.get("player_name", "unknown"),
        "main_position": player.get("main_position"),
        "competition_id": player.get("competition_id"),
        "competition_country": player.get("competition_country"),
        "age_months": player.get("age_months"),
        "market_value_current_eur": player.get("market_value_current_eur"),
        # Salary range
        "expected_salary_low_eur": round(low),
        "expected_salary_median_eur": round(median),
        "expected_salary_high_eur": round(high),
        # Actual salary
        "actual_salary_eur": round(actual_eur) if actual_eur is not None else None,
        "salary_percentile": percentile,
        "salary_status": status.upper(),
        # Confidence
        "benchmark_confidence": confidence.upper(),
        "benchmark_n_comparables": n_comp,
        "benchmark_n_comparables_with_salary": n_comparables_with_salary,
        "benchmark_avg_similarity": round(avg_sim, 3),
        "comparable_level_used": level_used,
        "range_width_used": range_width,
        "model_used": variant,
        # Warning (if any)
        "benchmark_warning": benchmark_warning,
        # Comparables
        "comparable_players": top_comparables,
    }
    clean_result = _clean_record(result)

    # --- Monitoring: log prediction ---
    try:
        elapsed_ms = (time.perf_counter() - _start_time) * 1000
        get_logger().log(player, clean_result, elapsed_ms)
    except Exception:
        pass  # Monitoring should never break the benchmark

    return clean_result


def benchmark_by_id(player_id: int, **kwargs) -> dict:
    """Look up a player by row id in the pool and run benchmark."""
    pool = _load_pool()
    if player_id < 0 or player_id >= len(pool):
        raise ValueError(f"Player id {player_id} not found in player_pool.csv")
    player = pool.iloc[player_id].to_dict()
    return benchmark_player(player, **kwargs)


def resolve_player_by_name(player_name: str) -> dict:
    """Find the best pool match for a player name (exact, then substring).

    Ambiguous names resolve to the highest market value match. Raises
    ValueError when nothing matches.
    """
    pool = _load_pool()
    matches = pool[pool["player_name"].str.lower() == player_name.lower()]
    if matches.empty:
        matches = pool[pool["player_name"].str.lower().str.contains(
            player_name.lower(), regex=False
        )]
    if matches.empty:
        raise ValueError(f"Player '{player_name}' not found in player_pool.csv")
    if len(matches) > 1:
        matches = matches.sort_values("market_value_current_eur", ascending=False)
    return matches.iloc[0].to_dict()


def benchmark_by_name(player_name: str, **kwargs) -> dict:
    """Look up a player by name in the pool and run benchmark.

    The web interface should prefer benchmark_by_id because player names are
    not unique. This name lookup is kept for CLI convenience.
    """
    return benchmark_player(resolve_player_by_name(player_name), **kwargs)
