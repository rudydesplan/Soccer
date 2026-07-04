"""Tests for build_model_features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_model_features import (
    MODEL_DROP_COLUMNS,
    add_numeric_features,
    build_model_features,
    build_player_pool,
    main,
    months_between,
    parse_contract_length_months,
    parse_drop_columns,
    parse_season_start,
    parse_season_start_year,
)


# --- parse_season_start_year ---

class TestParseSeasonStartYear:
    def test_normal_format(self):
        assert parse_season_start_year("2025-2026") == 2025

    def test_slash_format(self):
        assert parse_season_start_year("2025/26") == 2025

    def test_nan_input(self):
        assert np.isnan(parse_season_start_year(np.nan))

    def test_none_input(self):
        assert np.isnan(parse_season_start_year(None))

    def test_empty_string(self):
        assert np.isnan(parse_season_start_year(""))

    def test_no_digits(self):
        assert np.isnan(parse_season_start_year("no-year-here"))

    def test_numeric_input(self):
        # Edge: a plain number should still extract digits
        assert parse_season_start_year(2024) == 2024


# --- parse_season_start ---

class TestParseSeasonStart:
    def test_normal(self):
        result = parse_season_start("2025-2026")
        assert result == pd.Timestamp(2025, 7, 1)

    def test_returns_nat_for_nan(self):
        result = parse_season_start(np.nan)
        assert pd.isna(result)

    def test_returns_nat_for_empty(self):
        result = parse_season_start("")
        assert pd.isna(result)


# --- parse_contract_length_months ---

class TestParseContractLengthMonths:
    def test_four_years(self):
        assert parse_contract_length_months("4-yrs/€54.0M + 13.5M") == 48.0

    def test_three_years(self):
        assert parse_contract_length_months("3-yrs/€10.0M") == 36.0

    def test_one_year(self):
        assert parse_contract_length_months("1-yr/€2.0M") == 12.0

    def test_nan_input(self):
        assert np.isnan(parse_contract_length_months(np.nan))

    def test_none_input(self):
        assert np.isnan(parse_contract_length_months(None))

    def test_no_match(self):
        assert np.isnan(parse_contract_length_months("no contract info"))

    def test_empty_string(self):
        assert np.isnan(parse_contract_length_months(""))


# --- months_between ---

class TestMonthsBetween:
    def test_known_dates(self):
        start = pd.Series(["2020-01-01"])
        end = pd.Series(["2020-07-01"])
        result = months_between(start, end)
        assert result.iloc[0] == 6.0

    def test_one_year(self):
        start = pd.Series(["2020-01-01"])
        end = pd.Series(["2021-01-01"])
        result = months_between(start, end)
        assert result.iloc[0] == 12.0

    def test_nat_start(self):
        start = pd.Series([pd.NaT])
        end = pd.Series(["2021-01-01"])
        result = months_between(start, end)
        assert np.isnan(result.iloc[0])

    def test_nat_end(self):
        start = pd.Series(["2020-01-01"])
        end = pd.Series([pd.NaT])
        result = months_between(start, end)
        assert np.isnan(result.iloc[0])

    def test_multiple_rows(self):
        start = pd.Series(["2020-01-01", "2020-06-01"])
        end = pd.Series(["2020-04-01", "2020-12-01"])
        result = months_between(start, end)
        assert result.iloc[0] == 3.0
        assert result.iloc[1] == 6.0


# --- add_numeric_features ---

class TestAddNumericFeatures:
    def test_creates_expected_columns(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        expected_cols = [
            "season_start_year", "season_start_date", "age_months",
            "contract_length_months", "contract_months_remaining",
            "contract_recency_months", "has_contract_dates", "has_market_value",
            "log_market_value_current_eur", "has_release_clause",
            "log_release_clause_eur", "log_annual_fixed_eur",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_renames_player_id(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        assert "transfermarkt_player_id" in result.columns
        assert "player_id" not in result.columns

    def test_renames_team_id(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        assert "transfermarkt_team_id" in result.columns
        assert "team_id" not in result.columns

    def test_season_start_year_values(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        assert result["season_start_year"].iloc[0] == 2024
        assert result["season_start_year"].iloc[1] == 2023

    def test_contract_length_months_values(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        assert result["contract_length_months"].iloc[0] == 48.0
        assert result["contract_length_months"].iloc[1] == 36.0
        assert result["contract_length_months"].iloc[2] == 60.0

    def test_has_release_clause(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        assert result["has_release_clause"].iloc[0] == 1
        assert result["has_release_clause"].iloc[1] == 0

    def test_log_market_value(self, sample_input_csv):
        df = pd.read_csv(sample_input_csv)
        result = add_numeric_features(df, as_of_date="2025-01-01")
        expected = np.log1p(50_000_000)
        assert abs(result["log_market_value_current_eur"].iloc[0] - expected) < 0.01


# --- build_model_features ---

class TestBuildModelFeatures:
    def test_writes_output_file(self, sample_input_csv, tmp_path):
        output = tmp_path / "output.csv"
        build_model_features(sample_input_csv, output, drop_columns=MODEL_DROP_COLUMNS, as_of_date="2025-01-01")
        assert output.exists()

    def test_drops_specified_columns(self, sample_input_csv, tmp_path):
        output = tmp_path / "output.csv"
        result = build_model_features(sample_input_csv, output, drop_columns=["player_name", "league"], as_of_date="2025-01-01")
        assert "player_name" not in result.columns
        assert "league" not in result.columns

    def test_keeps_all_when_no_drop(self, sample_input_csv, tmp_path):
        output = tmp_path / "output.csv"
        result = build_model_features(sample_input_csv, output, drop_columns=[], as_of_date="2025-01-01")
        assert "player_name" in result.columns


# --- build_player_pool ---

class TestBuildPlayerPool:
    def test_writes_output(self, sample_input_csv, tmp_path):
        output = tmp_path / "pool.csv"
        build_player_pool(sample_input_csv, output, as_of_date="2025-01-01")
        assert output.exists()

    def test_drops_season_start_date(self, sample_input_csv, tmp_path):
        output = tmp_path / "pool.csv"
        result = build_player_pool(sample_input_csv, output, as_of_date="2025-01-01")
        assert "season_start_date" not in result.columns

    def test_keeps_source_columns(self, sample_input_csv, tmp_path):
        output = tmp_path / "pool.csv"
        result = build_player_pool(sample_input_csv, output, as_of_date="2025-01-01")
        # Pool mode keeps player_name and other source columns
        assert "player_name" in result.columns


# --- parse_drop_columns ---

class TestParseDropColumns:
    def test_none_returns_defaults(self):
        assert parse_drop_columns(None) == MODEL_DROP_COLUMNS

    def test_empty_string_returns_empty(self):
        assert parse_drop_columns("") == []

    def test_whitespace_only(self):
        assert parse_drop_columns("   ") == []

    def test_comma_separated(self):
        assert parse_drop_columns("col1, col2, col3") == ["col1", "col2", "col3"]

    def test_single_column(self):
        assert parse_drop_columns("player_name") == ["player_name"]


# --- main CLI ---

class TestMain:
    def test_model_mode(self, sample_input_csv, tmp_path, capsys):
        output = tmp_path / "out.csv"
        main(["--input", str(sample_input_csv), "--output", str(output), "--mode", "model", "--as-of-date", "2025-01-01"])
        assert output.exists()
        captured = capsys.readouterr()
        assert "Wrote" in captured.out

    def test_pool_mode(self, sample_input_csv, tmp_path, capsys):
        output = tmp_path / "pool.csv"
        main(["--input", str(sample_input_csv), "--output", str(output), "--mode", "pool", "--as-of-date", "2025-01-01"])
        assert output.exists()
        captured = capsys.readouterr()
        assert "Players with known salary" in captured.out

    def test_custom_drop_columns(self, sample_input_csv, tmp_path):
        output = tmp_path / "out.csv"
        main(["--input", str(sample_input_csv), "--output", str(output), "--drop-columns", "player_name,league", "--as-of-date", "2025-01-01"])
        result = pd.read_csv(output)
        assert "player_name" not in result.columns


# --- Error handling ---

class TestErrorHandling:
    def test_missing_input_file(self, tmp_path, capsys):
        output = tmp_path / "out.csv"
        with pytest.raises(SystemExit) as exc_info:
            main(["--input", "/nonexistent/file.csv", "--output", str(output)])
        assert exc_info.value.code == 1

    def test_invalid_as_of_date(self, sample_input_csv, tmp_path, capsys):
        output = tmp_path / "out.csv"
        with pytest.raises(SystemExit) as exc_info:
            main(["--input", str(sample_input_csv), "--output", str(output), "--as-of-date", "not-a-date"])
        assert exc_info.value.code == 1

    def test_missing_required_columns(self, tmp_path):
        """CSV missing 'season' column should raise ValueError."""
        csv_path = tmp_path / "bad.csv"
        # Missing 'season' and other required columns
        pd.DataFrame({"player_name": ["Alice"]}).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="missing required column"):
            build_model_features(csv_path, tmp_path / "out.csv")

    def test_add_numeric_features_validates_columns(self):
        """add_numeric_features raises on missing required columns."""
        df = pd.DataFrame({"player_name": ["Alice"], "age": [25]})
        with pytest.raises(ValueError, match="missing required column"):
            add_numeric_features(df)
