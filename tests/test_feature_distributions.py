"""Feature distribution tests — statistical sanity checks.

These catch data drift, corruption, or pipeline bugs by verifying that
feature values fall within expected ranges and distributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
DATA_FULL = DATA_DIR / "data_full.csv"
DATA_TEST = DATA_DIR / "data_test.csv"
PLAYER_POOL = DATA_DIR / "player_pool.csv"
CALIBRATION = DATA_DIR / "models" / "calibration.json"
RESULTS = DATA_DIR / "models" / "autogluon_results.json"

pytestmark = pytest.mark.skipif(
    not DATA_TEST.exists(), reason="Data files not present"
)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(DATA_TEST)


@pytest.fixture(scope="module")
def pool():
    return pd.read_csv(PLAYER_POOL)


@pytest.fixture(scope="module")
def full():
    return pd.read_csv(DATA_FULL)


class TestAgeDistribution:
    def test_age_in_reasonable_range(self, test_df):
        """All ages should be between 15 and 45 years (180-540 months)."""
        ages = test_df["age_months"].dropna()
        assert ages.min() >= 150, f"Min age too low: {ages.min():.0f} months"
        assert ages.max() <= 600, f"Max age too high: {ages.max():.0f} months"

    def test_median_age_reasonable(self, test_df):
        """Median age should be around 23-27 years (276-324 months)."""
        median = test_df["age_months"].median()
        assert 240 <= median <= 360, f"Median age unusual: {median:.0f} months"


class TestMarketValueDistribution:
    def test_market_value_positive_when_present(self, pool):
        """No negative market values."""
        mv = pool["market_value_current_eur"].dropna()
        assert (mv >= 0).all(), f"Found {(mv < 0).sum()} negative market values"

    def test_max_market_value_reasonable(self, pool):
        """Max MV should be ≤ €300M (no data corruption)."""
        mv = pool["market_value_current_eur"].dropna()
        assert mv.max() <= 300_000_000, f"Max MV too high: €{mv.max():,.0f}"

    def test_log_market_value_not_negative_infinity(self, test_df):
        """log1p should never produce -inf."""
        log_mv = test_df["log_market_value_current_eur"].dropna()
        assert not np.isinf(log_mv).any(), "Found -inf in log_market_value"


class TestSalaryDistribution:
    def test_salary_positive_when_present(self, pool):
        """No negative salaries."""
        sal = pool["annual_fixed_eur"].dropna()
        assert (sal >= 0).all(), f"Found {(sal < 0).sum()} negative salaries"

    def test_salary_max_reasonable(self, pool):
        """Max salary should be ≤ €50M (no data corruption)."""
        sal = pool["annual_fixed_eur"].dropna()
        assert sal.max() <= 50_000_000, f"Max salary too high: €{sal.max():,.0f}"

    def test_log_salary_not_negative_infinity(self, test_df):
        """log1p should never produce -inf."""
        log_sal = test_df["log_annual_fixed_eur"].dropna()
        assert not np.isinf(log_sal).any(), "Found -inf in log_annual_fixed_eur"

    def test_salary_count_minimum(self, pool):
        """We need at least 3000 salary data points for meaningful training."""
        count = pool["annual_fixed_eur"].notna().sum()
        assert count >= 3000, f"Only {count} salary data points"


class TestContractDistribution:
    def test_contract_length_reasonable(self, test_df):
        """Contract length should be 0-120 months (0-10 years)."""
        cl = test_df["contract_length_months"].dropna()
        assert cl.min() >= 0, f"Negative contract length: {cl.min()}"
        assert cl.max() <= 120, f"Contract > 10 years: {cl.max()}"

    def test_contract_remaining_not_absurd(self, test_df):
        """Remaining months should be between -24 and 120."""
        cr = test_df["contract_months_remaining"].dropna()
        assert cr.min() >= -36, f"Contract remaining too negative: {cr.min():.0f}"
        assert cr.max() <= 120, f"Contract remaining > 10 years: {cr.max():.0f}"


class TestCategoricalDistribution:
    def test_season_start_year_is_2025(self, test_df):
        """All rows should be season 2025 (single-season dataset)."""
        years = test_df["season_start_year"].dropna().unique()
        assert list(years) == [2025], f"Unexpected seasons: {years}"

    def test_competition_country_in_expected_set(self, full):
        """Only the 5 expected countries."""
        expected = {"Italy", "Spain", "Germany", "England", "France"}
        actual = set(full["competition_country"].dropna().unique())
        unexpected = actual - expected
        assert unexpected == set(), f"Unexpected countries: {unexpected}"

    def test_position_values_in_known_set(self, full):
        """All positions should be from the known Transfermarkt set."""
        known = {
            "Centre-Forward", "Left Winger", "Right Winger", "Second Striker",
            "Attacking Midfield", "Central Midfield", "Defensive Midfield",
            "Left Midfield", "Right Midfield",
            "Centre-Back", "Left-Back", "Right-Back",
            "Goalkeeper",
            # Less common but valid
            "Midfielder", "Striker", "Defender",
        }
        actual = set(full["main_position"].dropna().unique())
        unexpected = actual - known
        assert unexpected == set(), f"Unexpected positions: {unexpected}"

    def test_binary_flags_are_0_or_1(self, test_df):
        """Binary indicator columns should only contain 0 and 1."""
        for col in ["has_contract_dates", "has_market_value", "has_release_clause"]:
            values = test_df[col].dropna().unique()
            assert set(values).issubset({0, 1}), f"{col} has values: {set(values)}"


class TestCalibrationQuality:
    @pytest.fixture
    def calibration(self):
        if not CALIBRATION.exists():
            pytest.skip("calibration.json not present")
        return json.loads(CALIBRATION.read_text())

    def test_residuals_roughly_symmetric(self, calibration):
        """Skew should be small — mean close to 0."""
        assert abs(calibration["residual_mean"]) < 0.1

    def test_rmse_reasonable(self, calibration):
        """RMSE should be < 1.0 on log scale (otherwise model is useless)."""
        assert calibration["rmse"] < 1.0

    def test_std_reasonable(self, calibration):
        """Std should be between 0.2 and 1.0 (not too tight, not too wide)."""
        assert 0.2 < calibration["residual_std"] < 1.0


class TestModelQuality:
    def test_model_r2_above_threshold(self):
        """Saved model R² should be > 0.70."""
        if not RESULTS.exists():
            pytest.skip("autogluon_results.json not present")
        results = json.loads(RESULTS.read_text())
        assert results["r2"] > 0.70, f"Model R² too low: {results['r2']:.4f}"

    def test_model_rmse_below_threshold(self):
        """Saved model RMSE should be < 0.60 on log scale."""
        if not RESULTS.exists():
            pytest.skip("autogluon_results.json not present")
        results = json.loads(RESULTS.read_text())
        assert results["rmse"] < 0.60, f"Model RMSE too high: {results['rmse']:.4f}"
