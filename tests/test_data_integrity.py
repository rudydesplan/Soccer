"""Data integrity tests — cross-file consistency checks.

These verify that data_full.csv, data_test.csv, and player_pool.csv are
consistent with each other and with the model/calibration artifacts.
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

pytestmark = pytest.mark.skipif(
    not DATA_FULL.exists(), reason="Data files not present (CI without data)"
)


@pytest.fixture(scope="module")
def full():
    return pd.read_csv(DATA_FULL)


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(DATA_TEST)


@pytest.fixture(scope="module")
def pool():
    return pd.read_csv(PLAYER_POOL)


class TestRowCountConsistency:
    def test_all_files_same_row_count(self, full, test_df, pool):
        assert len(full) == len(test_df) == len(pool)

    def test_expected_row_count(self, full):
        assert len(full) == 19_476


class TestColumnConsistency:
    def test_data_test_features_subset_of_pool(self, test_df, pool):
        """Every column in data_test must exist in player_pool."""
        missing = set(test_df.columns) - set(pool.columns)
        assert missing == set(), f"Columns in data_test but not in pool: {missing}"

    def test_pool_has_display_columns(self, pool):
        """player_pool must have columns needed for UI display."""
        required = {"player_name", "team_name", "main_position", "competition_id",
                    "competition_country", "age_months", "market_value_current_eur",
                    "annual_fixed_eur"}
        missing = required - set(pool.columns)
        assert missing == set(), f"Missing display columns: {missing}"

    def test_data_full_has_enrichment_columns(self, full):
        """data_full must have the Capology enrichment columns."""
        enrichment = {"capology_url", "status", "league", "annual_fixed_eur",
                      "annual_bonus_eur", "annual_total_eur", "salary_currency",
                      "gross_contract", "signed_date", "expiration_date",
                      "release_clause_eur", "market_value_current_eur"}
        missing = enrichment - set(full.columns)
        assert missing == set(), f"Missing enrichment columns: {missing}"


class TestSalaryConsistency:
    def test_salary_count_consistent_across_files(self, full, test_df, pool):
        """Same number of non-null salaries in all files."""
        full_count = full["annual_fixed_eur"].notna().sum()
        pool_count = pool["annual_fixed_eur"].notna().sum()
        test_count = test_df["log_annual_fixed_eur"].notna().sum()
        assert full_count == pool_count == test_count

    def test_salary_values_match_between_full_and_pool(self, full, pool):
        """Salary values in pool should match data_full."""
        full_sal = full["annual_fixed_eur"].dropna().sort_values().reset_index(drop=True)
        pool_sal = pool["annual_fixed_eur"].dropna().sort_values().reset_index(drop=True)
        pd.testing.assert_series_equal(full_sal, pool_sal, check_names=False)


class TestModelFeatureAlignment:
    def test_model_features_match_data_test(self, test_df):
        """Model FEATURES list should match data_test columns minus target."""
        from salary_benchmark.model import FEATURES
        test_features = set(test_df.columns) - {"log_annual_fixed_eur"}
        model_features = set(FEATURES)
        # Model features should be a subset of test features
        missing_in_test = model_features - test_features
        assert missing_in_test == set(), f"Model expects features not in data_test: {missing_in_test}"


class TestCalibrationConsistency:
    @pytest.fixture
    def calibration(self):
        if not CALIBRATION.exists():
            pytest.skip("calibration.json not present")
        return json.loads(CALIBRATION.read_text())

    def test_calibration_sample_count_matches_salary_count(self, calibration, full):
        """calibration.json n_samples should equal number of players with salary."""
        expected = int(full["annual_fixed_eur"].notna().sum())
        # Allow calibration to be from a previous data version (within 5% tolerance)
        actual = calibration["n_samples"]
        assert abs(actual - expected) / expected < 0.05, (
            f"calibration n_samples={actual} vs data salary count={expected}. "
            f"Rebuild calibration: python -m salary_benchmark.calibration"
        )

    def test_calibration_percentiles_ordered(self, calibration):
        """p10 < p25 < p50 < p75 < p90."""
        assert calibration["residual_p10"] < calibration["residual_p25"]
        assert calibration["residual_p25"] < calibration["residual_p50"]
        assert calibration["residual_p50"] < calibration["residual_p75"]
        assert calibration["residual_p75"] < calibration["residual_p90"]

    def test_calibration_mean_near_zero(self, calibration):
        """Residual mean should be close to 0 (unbiased model)."""
        assert abs(calibration["residual_mean"]) < 0.1


class TestNoDataBugs:
    def test_no_raw_player_urls(self, full):
        """No /player/... raw paths should remain (the old bug)."""
        urls = full["capology_url"].dropna().astype(str)
        raw_paths = urls[urls.str.startswith("/player/")]
        assert len(raw_paths) == 0, f"Found {len(raw_paths)} raw /player/ URLs"

    def test_all_capology_urls_are_full(self, full):
        """All non-null capology_url values should start with https://."""
        urls = full["capology_url"].dropna().astype(str)
        bad = urls[~urls.str.startswith("https://")]
        assert len(bad) == 0, f"Found {len(bad)} non-https capology URLs"
