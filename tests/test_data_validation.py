"""Tests for input data validation and cleaning decisions.

Verifies how the feature engineering handles edge cases:
- Out-of-range values
- Empty/null fields
- Expired contracts
- Duplicate players
- Missing enrichment data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_model_features import add_numeric_features, build_model_features, MODEL_DROP_COLUMNS


def _make_df(**overrides):
    """Create a minimal valid input DataFrame with optional overrides."""
    base = {
        "player_id": [1],
        "player_name": ["Test Player"],
        "position": ["ATT"],
        "main_position": ["Centre-Forward"],
        "birth_date": ["2000-01-15"],
        "nationality": ["Spain"],
        "team_id": [131],
        "team_name": ["FC Barcelona"],
        "competition_id": ["ES1"],
        "competition_name": ["LaLiga"],
        "competition_country": ["Spain"],
        "season": ["2025-2026"],
        "capology_url": [None],
        "status": [None],
        "league": [None],
        "annual_fixed_eur": [None],
        "annual_bonus_eur": [None],
        "annual_total_eur": [None],
        "salary_currency": [None],
        "gross_contract": [None],
        "signed_date": [None],
        "expiration_date": [None],
        "release_clause_eur": [None],
        "market_value_current_eur": [50_000_000],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestNegativeValues:
    """Negative monetary values should be clipped to 0 before log transform."""

    def test_negative_salary_clipped_to_zero(self):
        df = _make_df(annual_fixed_eur=[-1_000_000])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        # log1p(clip(lower=0)) → log1p(0) = 0
        assert result["log_annual_fixed_eur"].iloc[0] == 0.0

    def test_negative_market_value_clipped_to_zero(self):
        df = _make_df(market_value_current_eur=[-50_000_000])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["log_market_value_current_eur"].iloc[0] == 0.0

    def test_negative_release_clause_clipped_to_zero(self):
        df = _make_df(release_clause_eur=[-100])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["log_release_clause_eur"].iloc[0] == 0.0

    def test_zero_salary_produces_zero_log(self):
        df = _make_df(annual_fixed_eur=[0])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["log_annual_fixed_eur"].iloc[0] == 0.0


class TestMissingFields:
    """Missing/null fields produce NaN features, not crashes."""

    def test_missing_birth_date_produces_nan_age(self):
        df = _make_df(birth_date=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert np.isnan(result["age_months"].iloc[0])

    def test_missing_expiration_produces_nan_remaining(self):
        df = _make_df(expiration_date=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert np.isnan(result["contract_months_remaining"].iloc[0])

    def test_missing_gross_contract_produces_nan_length(self):
        df = _make_df(gross_contract=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert np.isnan(result["contract_length_months"].iloc[0])

    def test_missing_signed_date_produces_nan_recency(self):
        df = _make_df(signed_date=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert np.isnan(result["contract_recency_months"].iloc[0])

    def test_missing_market_value_produces_nan_log(self):
        df = _make_df(market_value_current_eur=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert np.isnan(result["log_market_value_current_eur"].iloc[0])

    def test_missing_market_value_sets_flag_to_zero(self):
        df = _make_df(market_value_current_eur=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["has_market_value"].iloc[0] == 0

    def test_missing_release_clause_sets_flag_to_zero(self):
        df = _make_df(release_clause_eur=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["has_release_clause"].iloc[0] == 0

    def test_empty_position_preserved_as_nan(self):
        df = _make_df(main_position=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert pd.isna(result["main_position"].iloc[0])

    def test_missing_salary_produces_nan_log(self):
        df = _make_df(annual_fixed_eur=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert np.isnan(result["log_annual_fixed_eur"].iloc[0])


class TestContractEdgeCases:
    """Edge cases for contract date calculations."""

    def test_future_signed_date_produces_negative_recency(self):
        """A contract signed after season start → negative recency (valid)."""
        df = _make_df(signed_date=["2025-09-01"], season=["2025-2026"])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        # Season start is 2025-07-01, signed 2025-09-01 → recency is negative
        assert result["contract_recency_months"].iloc[0] < 0

    def test_expired_contract_produces_negative_remaining(self):
        """A contract that already expired → negative remaining months."""
        df = _make_df(expiration_date=["2024-06-30"])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["contract_months_remaining"].iloc[0] < 0

    def test_expiration_today_produces_zero_remaining(self):
        df = _make_df(expiration_date=["2025-07-01"])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["contract_months_remaining"].iloc[0] == 0.0

    def test_has_contract_dates_both_present(self):
        df = _make_df(signed_date=["2023-01-01"], expiration_date=["2027-06-30"])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["has_contract_dates"].iloc[0] == 1

    def test_has_contract_dates_one_missing(self):
        df = _make_df(signed_date=["2023-01-01"], expiration_date=[None])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["has_contract_dates"].iloc[0] == 0


class TestDuplicateAndMultipleRows:
    """Same player appearing multiple times (different competitions)."""

    def test_duplicate_player_rows_preserved(self):
        """Same player in two competitions → both rows kept."""
        df = pd.DataFrame({
            "player_id": [1, 1],
            "player_name": ["Test", "Test"],
            "position": ["ATT", "ATT"],
            "main_position": ["Centre-Forward", "Centre-Forward"],
            "birth_date": ["2000-01-15", "2000-01-15"],
            "nationality": ["Spain", "Spain"],
            "team_id": [131, 131],
            "team_name": ["FC Barcelona", "FC Barcelona"],
            "competition_id": ["ES1", "CL"],
            "competition_name": ["LaLiga", "Champions League"],
            "competition_country": ["Spain", "Europe"],
            "season": ["2025-2026", "2025-2026"],
            "capology_url": [None, None],
            "status": [None, None],
            "league": [None, None],
            "annual_fixed_eur": [5_000_000, 5_000_000],
            "annual_bonus_eur": [None, None],
            "annual_total_eur": [None, None],
            "salary_currency": [None, None],
            "gross_contract": [None, None],
            "signed_date": [None, None],
            "expiration_date": [None, None],
            "release_clause_eur": [None, None],
            "market_value_current_eur": [50_000_000, 50_000_000],
        })
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert len(result) == 2

    def test_all_null_enrichment_row_preserved(self):
        """Player not in Capology → row kept with null salary columns."""
        df = _make_df(
            capology_url=[None],
            annual_fixed_eur=[None],
            annual_bonus_eur=[None],
            annual_total_eur=[None],
            status=[None],
            league=[None],
            gross_contract=[None],
            signed_date=[None],
            expiration_date=[None],
            release_clause_eur=[None],
        )
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert len(result) == 1
        assert result["player_name"].iloc[0] == "Test Player"
        assert np.isnan(result["log_annual_fixed_eur"].iloc[0])


class TestOutliers:
    """Salary outliers are NOT rejected — they are valid data points."""

    def test_very_high_salary_not_rejected(self):
        df = _make_df(annual_fixed_eur=[50_000_000])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["log_annual_fixed_eur"].iloc[0] == np.log1p(50_000_000)

    def test_very_low_salary_not_rejected(self):
        df = _make_df(annual_fixed_eur=[10_000])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["log_annual_fixed_eur"].iloc[0] == np.log1p(10_000)

    def test_very_high_market_value_not_rejected(self):
        df = _make_df(market_value_current_eur=[300_000_000])
        result = add_numeric_features(df, as_of_date="2025-07-01")
        assert result["log_market_value_current_eur"].iloc[0] == np.log1p(300_000_000)


class TestBuildModelFeaturesEndToEnd:
    """End-to-end test: CSV in → CSV out with validation."""

    def test_output_has_no_raw_monetary_columns(self, tmp_path):
        df = _make_df(annual_fixed_eur=[5_000_000], market_value_current_eur=[50_000_000])
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        df.to_csv(input_csv, index=False)

        result = build_model_features(input_csv, output_csv, drop_columns=MODEL_DROP_COLUMNS, as_of_date="2025-07-01")
        # Raw monetary columns should be dropped
        assert "annual_fixed_eur" not in result.columns
        assert "market_value_current_eur" not in result.columns
        # Log versions should exist
        assert "log_annual_fixed_eur" in result.columns
        assert "log_market_value_current_eur" in result.columns

    def test_non_numeric_salary_becomes_nan(self, tmp_path):
        """If salary field has non-numeric text, it becomes NaN."""
        df = _make_df(annual_fixed_eur=["not_a_number"])
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"
        df.to_csv(input_csv, index=False)

        result = build_model_features(input_csv, output_csv, drop_columns=[], as_of_date="2025-07-01")
        assert np.isnan(result["log_annual_fixed_eur"].iloc[0])
