"""Comparable player logic — 3 levels + similarity score (0-1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Broad position groups for level-3 fallback
POSITION_GROUPS = {
    "Centre-Forward":       "attacker",
    "Left Winger":          "attacker",
    "Right Winger":         "attacker",
    "Second Striker":       "attacker",
    "Attacking Midfield":   "midfielder",
    "Central Midfield":     "midfielder",
    "Defensive Midfield":   "midfielder",
    "Left Midfield":        "midfielder",
    "Right Midfield":       "midfielder",
    "Centre-Back":          "defender",
    "Left-Back":            "defender",
    "Right-Back":           "defender",
    "Goalkeeper":           "goalkeeper",
}

# Similarity weights (must sum to 1.0)
W_MARKET_VALUE = 0.35
W_LEAGUE       = 0.25
W_POSITION     = 0.20
W_AGE          = 0.12
W_CONTRACT     = 0.08

MIN_COMPARABLES = 5


def _market_value_sim(mv_target: float, mv_pool: pd.Series) -> pd.Series:
    """Similarity based on log ratio of market values (0-1)."""
    if mv_target <= 0:
        return pd.Series(0.5, index=mv_pool.index)
    log_ratio = np.abs(np.log(mv_pool.clip(lower=1) / max(mv_target, 1)))
    return np.exp(-log_ratio)  # 1.0 when equal, decays smoothly


def _age_sim(age_target: float | None, age_pool: pd.Series) -> pd.Series:
    """Similarity based on age difference in months (0-1).

    Unknown age — on either side — scores a neutral 0.5 so missing data is
    neither rewarded nor punished.
    """
    if age_target is None or (isinstance(age_target, float) and np.isnan(age_target)):
        return pd.Series(0.5, index=age_pool.index)
    diff = np.abs(age_pool - age_target)
    return np.exp(-diff / 24).fillna(0.5)  # half-life ~24 months


def _position_sim(pos_target: str | None, pos_pool: pd.Series) -> pd.Series:
    """1.0 exact match, 0.5 same known group, 0.0 otherwise.

    An unknown target position scores a neutral 0.5 against everyone.
    Positions outside POSITION_GROUPS fall into "other", which is not a real
    group — two different unknown positions must not score the 0.5 group tier.
    """
    if not isinstance(pos_target, str) or pos_target == "":
        return pd.Series(0.5, index=pos_pool.index)
    exact = (pos_pool == pos_target).astype(float)
    group_target = POSITION_GROUPS.get(pos_target, "other")
    if group_target == "other":
        return exact
    same_group = pos_pool.map(lambda p: POSITION_GROUPS.get(p, "other") == group_target).astype(float)
    return exact + 0.5 * (same_group - exact)


def _league_sim(comp_target: str, country_target: str,
                comp_pool: pd.Series, country_pool: pd.Series) -> pd.Series:
    """1.0 same competition, 0.6 same country, 0.2 otherwise."""
    same_comp    = comp_pool == comp_target
    same_country = country_pool == country_target
    return pd.Series(
        np.where(same_comp, 1.0, np.where(same_country, 0.6, 0.2)),
        index=comp_pool.index,
    )


def _contract_sim(months_target: float | None, months_pool: pd.Series) -> pd.Series:
    """Similarity based on contract length/remaining months (0-1).

    Unknown months — on either the target or the pool side — score a neutral
    0.5 so missing data is neither rewarded nor punished.
    """
    if months_target is None or np.isnan(months_target):
        return pd.Series(0.5, index=months_pool.index)
    diff = np.abs(months_pool - months_target)
    return np.exp(-diff / 12).fillna(0.5)


def similarity_score(target: dict, pool: pd.DataFrame) -> pd.Series:
    """Compute normalized similarity score (0-1) for each player in pool."""
    # `or 0` also maps an explicit None (unknown market value) to the neutral
    # 0.5 branch of _market_value_sim instead of crashing on a comparison.
    mv_sim  = _market_value_sim(
        target.get("market_value_current_eur") or 0,
        pool.get("market_value_current_eur", pd.Series(0, index=pool.index)),
    )
    # No default fill for age/position: an absent value routes to the neutral
    # 0.5 branch of the similarity function instead of pretending a value.
    age_sim = _age_sim(
        target.get("age_months"),
        pool.get("age_months", pd.Series(np.nan, index=pool.index)),
    )
    pos_sim = _position_sim(
        target.get("main_position"),
        pool.get("main_position", pd.Series("", index=pool.index)),
    )
    league_sim = _league_sim(
        target.get("competition_id", ""),
        target.get("competition_country", ""),
        pool.get("competition_id", pd.Series("", index=pool.index)),
        pool.get("competition_country", pd.Series("", index=pool.index)),
    )
    contract_sim = _contract_sim(
        target.get("contract_months_remaining", target.get("contract_length_months")),
        pool.get(
            "contract_months_remaining",
            pool.get("contract_length_months", pd.Series(np.nan, index=pool.index)),
        ),
    )

    score = (
        W_MARKET_VALUE * mv_sim
        + W_LEAGUE      * league_sim
        + W_POSITION    * pos_sim
        + W_AGE         * age_sim
        + W_CONTRACT    * contract_sim
    )
    return score.clip(0, 1)


def find_comparables(target: dict, pool: pd.DataFrame,
                     n_min: int = MIN_COMPARABLES) -> tuple[pd.DataFrame, int]:
    """Find comparable players using 3-level fallback.

    Any filter whose target-side value is unknown is skipped at every level
    (a NaN market value / age / position would otherwise match nobody):
    - no market value → skip the market-value band
    - no age → skip the age band
    - no position → skip the position filter

    Returns:
        (comparables_df, level_used)  where level_used is 1, 2, or 3.
    """
    _all = pd.Series(True, index=pool.index)

    mv_raw = target.get("market_value_current_eur")
    mv_known = mv_raw is not None and not np.isnan(mv_raw) and mv_raw > 0
    mv = mv_raw if mv_known else 0

    age_raw = target.get("age_months")
    age_known = age_raw is not None and not (isinstance(age_raw, float) and np.isnan(age_raw))
    age = age_raw if age_known else 0

    pos_raw = target.get("main_position")
    pos_known = isinstance(pos_raw, str) and pos_raw != ""
    pos = pos_raw if pos_known else ""
    group = POSITION_GROUPS.get(pos, "other")

    comp_id = target.get("competition_id", "")
    country = target.get("competition_country", "")

    def mv_band(low_factor: float, high_factor: float) -> pd.Series:
        if not mv_known:
            return _all
        return pool["market_value_current_eur"].between(mv * low_factor, mv * high_factor)

    def age_band(delta: float) -> pd.Series:
        if not age_known:
            return _all
        return pool["age_months"].between(age - delta, age + delta)

    def pos_exact() -> pd.Series:
        if not pos_known:
            return _all
        return pool["main_position"] == pos

    def pos_group() -> pd.Series:
        if not pos_known:
            return _all
        return pool["main_position"].map(lambda p: POSITION_GROUPS.get(p, "other") == group)

    # Level 1 — strict
    mask1 = (
        pos_exact() &
        (pool["competition_id"] == comp_id) &
        age_band(24) &
        mv_band(0.5, 2.0)
    )
    if mask1.sum() >= n_min:
        return pool[mask1].copy(), 1

    # Level 2 — relaxed
    mask2 = (
        pos_exact() &
        (pool["competition_country"] == country) &
        age_band(36) &
        mv_band(0.33, 3.0)
    )
    if mask2.sum() >= n_min:
        return pool[mask2].copy(), 2

    # Level 3 — broad
    mask3 = (
        pos_group() &
        age_band(60) &
        mv_band(0.25, 4.0)
    )
    return pool[mask3].copy(), 3
