"""Tests for salary_benchmark/comparables.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from salary_benchmark.comparables import (
    _age_sim,
    _contract_sim,
    _league_sim,
    _market_value_sim,
    _position_sim,
    find_comparables,
    similarity_score,
)


# --- _market_value_sim ---

class TestMarketValueSim:
    def test_equal_values(self):
        pool = pd.Series([100_000_000])
        result = _market_value_sim(100_000_000, pool)
        assert abs(result.iloc[0] - 1.0) < 0.01

    def test_double_value(self):
        pool = pd.Series([200_000_000])
        result = _market_value_sim(100_000_000, pool)
        # log(2) ≈ 0.693, exp(-0.693) ≈ 0.5
        assert 0.4 < result.iloc[0] < 0.6

    def test_far_value(self):
        pool = pd.Series([1_000_000])
        result = _market_value_sim(100_000_000, pool)
        assert result.iloc[0] < 0.05

    def test_zero_target(self):
        pool = pd.Series([50_000_000, 100_000_000])
        result = _market_value_sim(0, pool)
        assert (result == 0.5).all()

    def test_multiple_values(self):
        pool = pd.Series([100_000_000, 50_000_000, 200_000_000])
        result = _market_value_sim(100_000_000, pool)
        # First is exact, others are further
        assert result.iloc[0] > result.iloc[1]
        assert result.iloc[0] > result.iloc[2]


# --- _age_sim ---

class TestAgeSim:
    def test_same_age(self):
        pool = pd.Series([300])
        result = _age_sim(300, pool)
        assert abs(result.iloc[0] - 1.0) < 0.01

    def test_24_months_diff(self):
        pool = pd.Series([324])
        result = _age_sim(300, pool)
        # exp(-24/24) = exp(-1) ≈ 0.368
        assert abs(result.iloc[0] - np.exp(-1)) < 0.01

    def test_large_diff(self):
        pool = pd.Series([500])
        result = _age_sim(300, pool)
        assert result.iloc[0] < 0.05


# --- _position_sim ---

class TestPositionSim:
    def test_exact_match(self):
        pool = pd.Series(["Centre-Forward"])
        result = _position_sim("Centre-Forward", pool)
        assert result.iloc[0] == 1.0

    def test_same_group(self):
        # Left Winger and Centre-Forward are both "attacker"
        pool = pd.Series(["Left Winger"])
        result = _position_sim("Centre-Forward", pool)
        assert result.iloc[0] == 0.5

    def test_different_group(self):
        pool = pd.Series(["Goalkeeper"])
        result = _position_sim("Centre-Forward", pool)
        assert result.iloc[0] == 0.0

    def test_mixed(self):
        pool = pd.Series(["Centre-Forward", "Left Winger", "Goalkeeper"])
        result = _position_sim("Centre-Forward", pool)
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 0.5
        assert result.iloc[2] == 0.0

    def test_find_comparables_without_market_value_skips_mv_band(self, sample_pool_df):
        """A target with unknown market value must still find comparables."""
        target = {
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": 300,
            "market_value_current_eur": None,
        }
        comps, level = find_comparables(target, sample_pool_df)
        assert len(comps) > 0

    def test_two_different_unknown_positions_score_zero(self):
        """Unknown positions share the 'other' bucket but are not a real group."""
        pool = pd.Series(["Sweeper"])
        result = _position_sim("Libero", pool)
        assert result.iloc[0] == 0.0

    def test_same_unknown_position_exact_match(self):
        pool = pd.Series(["Sweeper"])
        result = _position_sim("Sweeper", pool)
        assert result.iloc[0] == 1.0

    def test_missing_target_position_is_neutral(self):
        """A target with no position scores neutral 0.5 against everyone."""
        pool = pd.Series(["Centre-Forward", "Goalkeeper"])
        for missing in (None, "", float("nan")):
            result = _position_sim(missing, pool)
            assert (result == 0.5).all()

    def test_missing_target_age_is_neutral(self):
        """A target with no age scores neutral 0.5 against everyone."""
        pool = pd.Series([250.0, 400.0])
        for missing in (None, float("nan")):
            result = _age_sim(missing, pool)
            assert (result == 0.5).all()

    def test_find_comparables_without_position_skips_position_filter(self, sample_pool_df):
        """A target with unknown position must still find comparables."""
        target = {
            "main_position": None,
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": 300,
            "market_value_current_eur": 100_000_000,
        }
        comps, level = find_comparables(target, sample_pool_df)
        assert len(comps) > 0

    def test_find_comparables_without_age_skips_age_band(self, sample_pool_df):
        """A target with unknown age must still find comparables."""
        target = {
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": None,
            "market_value_current_eur": 100_000_000,
        }
        comps, level = find_comparables(target, sample_pool_df)
        assert len(comps) > 0


# --- _league_sim ---

class TestLeagueSim:
    def test_same_competition(self):
        comp_pool = pd.Series(["ES1"])
        country_pool = pd.Series(["Spain"])
        result = _league_sim("ES1", "Spain", comp_pool, country_pool)
        assert result.iloc[0] == 1.0

    def test_same_country_different_comp(self):
        comp_pool = pd.Series(["ES2"])
        country_pool = pd.Series(["Spain"])
        result = _league_sim("ES1", "Spain", comp_pool, country_pool)
        assert abs(result.iloc[0] - 0.6) < 0.01

    def test_different_country(self):
        comp_pool = pd.Series(["GB1"])
        country_pool = pd.Series(["England"])
        result = _league_sim("ES1", "Spain", comp_pool, country_pool)
        assert abs(result.iloc[0] - 0.2) < 0.01


# --- _contract_sim ---

class TestContractSim:
    def test_same_months(self):
        pool = pd.Series([48.0])
        result = _contract_sim(48.0, pool)
        assert abs(result.iloc[0] - 1.0) < 0.01

    def test_12_months_diff(self):
        pool = pd.Series([60.0])
        result = _contract_sim(48.0, pool)
        # exp(-12/12) = exp(-1) ≈ 0.368
        assert abs(result.iloc[0] - np.exp(-1)) < 0.01

    def test_nan_target(self):
        pool = pd.Series([48.0, 60.0])
        result = _contract_sim(None, pool)
        assert (result == 0.5).all()

    def test_nan_target_float(self):
        pool = pd.Series([48.0, 60.0])
        result = _contract_sim(float("nan"), pool)
        assert (result == 0.5).all()

    def test_nan_in_pool(self):
        pool = pd.Series([48.0, np.nan])
        result = _contract_sim(48.0, pool)
        # Known identical contract scores 1.0; unknown contract scores neutral 0.5
        assert abs(result.iloc[0] - 1.0) < 0.01
        assert abs(result.iloc[1] - 0.5) < 0.01


# --- similarity_score ---

class TestSimilarityScore:
    def test_returns_series(self, sample_pool_df):
        target = {
            "market_value_current_eur": 150_000_000,
            "age_months": 300,
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "contract_months_remaining": 48,
        }
        result = similarity_score(target, sample_pool_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_pool_df)

    def test_values_between_0_and_1(self, sample_pool_df):
        target = {
            "market_value_current_eur": 150_000_000,
            "age_months": 300,
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "contract_months_remaining": 48,
        }
        result = similarity_score(target, sample_pool_df)
        assert (result >= 0).all()
        assert (result <= 1).all()


# --- find_comparables ---

class TestFindComparables:
    def _make_pool(self, n, pos="Centre-Forward", comp_id="ES1", country="Spain",
                   age=300, mv=100_000_000):
        """Helper to create a pool matching specific criteria."""
        return pd.DataFrame({
            "main_position": [pos] * n,
            "competition_id": [comp_id] * n,
            "competition_country": [country] * n,
            "age_months": [age + i for i in range(n)],
            "market_value_current_eur": [mv] * n,
        })

    def test_level_1(self):
        # Strict match: same position, same comp, age within 24, mv within 0.5-2x
        pool = self._make_pool(10, pos="Centre-Forward", comp_id="ES1", age=300, mv=100_000_000)
        target = {
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": 300,
            "market_value_current_eur": 100_000_000,
        }
        result, level = find_comparables(target, pool, n_min=5)
        assert level == 1
        assert len(result) >= 5

    def test_level_2(self):
        # Same position, same country (not comp), relaxed age/mv
        pool = self._make_pool(10, pos="Centre-Forward", comp_id="ES2", country="Spain", age=300, mv=80_000_000)
        target = {
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": 300,
            "market_value_current_eur": 100_000_000,
        }
        result, level = find_comparables(target, pool, n_min=5)
        assert level == 2
        assert len(result) >= 5

    def test_level_3(self):
        # Only same position group, very different everything else
        pool = self._make_pool(10, pos="Left Winger", comp_id="FR1", country="France", age=300, mv=80_000_000)
        target = {
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": 300,
            "market_value_current_eur": 100_000_000,
        }
        result, level = find_comparables(target, pool, n_min=5)
        assert level == 3

    def test_empty_pool(self):
        pool = pd.DataFrame({
            "main_position": pd.Series([], dtype=str),
            "competition_id": pd.Series([], dtype=str),
            "competition_country": pd.Series([], dtype=str),
            "age_months": pd.Series([], dtype=float),
            "market_value_current_eur": pd.Series([], dtype=float),
        })
        target = {
            "main_position": "Centre-Forward",
            "competition_id": "ES1",
            "competition_country": "Spain",
            "age_months": 300,
            "market_value_current_eur": 100_000_000,
        }
        result, level = find_comparables(target, pool, n_min=5)
        assert level == 3
        assert len(result) == 0
