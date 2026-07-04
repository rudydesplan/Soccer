"""Tests for salary_benchmark/benchmark.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from salary_benchmark import benchmark as bm
from salary_benchmark.benchmark import (
    _clean_record,
    _clean_value,
    benchmark_by_id,
    benchmark_by_name,
    benchmark_player,
)


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset the module-level pool and calibration caches."""
    from salary_benchmark import calibration
    bm._POOL = None
    calibration._CAL_CACHE.clear()
    yield
    bm._POOL = None
    calibration._CAL_CACHE.clear()


# --- _clean_value ---

class TestCleanValue:
    def test_nan_becomes_none(self):
        assert _clean_value(float("nan")) is None

    def test_np_nan_becomes_none(self):
        assert _clean_value(np.nan) is None

    def test_none_stays_none(self):
        assert _clean_value(None) is None

    def test_np_int64_becomes_int(self):
        result = _clean_value(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_float64_becomes_float(self):
        result = _clean_value(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_string_unchanged(self):
        assert _clean_value("hello") == "hello"

    def test_regular_int_unchanged(self):
        assert _clean_value(42) == 42


# --- _clean_record ---

class TestCleanRecord:
    def test_mixed_types(self):
        record = {
            "name": "Test",
            "value": np.nan,
            "count": np.int64(5),
            "score": np.float64(0.8),
            "missing": None,
        }
        result = _clean_record(record)
        assert result["name"] == "Test"
        assert result["value"] is None
        assert result["count"] == 5
        assert isinstance(result["count"], int)
        assert result["score"] == 0.8
        assert result["missing"] is None


# --- benchmark_player ---

class TestBenchmarkPlayer:
    @pytest.fixture
    def mock_deps(self, sample_pool_df, mock_calibration):
        """Patch model + calibration for benchmark_player tests."""
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0) as mock_pred, \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            yield mock_pred

    def test_with_known_salary(self, mock_deps, target_player_dict):
        result = benchmark_player(target_player_dict)
        assert "expected_salary_low_eur" in result
        assert "expected_salary_median_eur" in result
        assert "expected_salary_high_eur" in result
        assert result["actual_salary_eur"] == 20_000_000
        assert result["salary_status"] in ("UNDERPAID", "FAIRLY_PAID", "OVERPAID")
        assert result["benchmark_confidence"] in ("HIGH", "MEDIUM", "LOW")
        assert isinstance(result["comparable_players"], list)

    def test_without_salary(self, mock_deps, target_player_dict):
        target_player_dict["annual_fixed_eur"] = None
        result = benchmark_player(target_player_dict)
        assert result["actual_salary_eur"] is None
        assert result["salary_status"] == "UNKNOWN"
        assert result["salary_percentile"] is None

    def test_nan_salary_becomes_none(self, mock_deps, target_player_dict):
        target_player_dict["annual_fixed_eur"] = float("nan")
        result = benchmark_player(target_player_dict)
        assert result["actual_salary_eur"] is None
        assert result["salary_status"] == "UNKNOWN"

    def test_result_keys(self, mock_deps, target_player_dict):
        result = benchmark_player(target_player_dict)
        expected_keys = [
            "player_id", "player_name", "main_position",
            "competition_id", "competition_country", "age_months",
            "market_value_current_eur",
            "expected_salary_low_eur", "expected_salary_median_eur", "expected_salary_high_eur",
            "actual_salary_eur", "salary_percentile", "salary_status",
            "benchmark_confidence", "benchmark_n_comparables",
            "benchmark_avg_similarity", "comparable_level_used",
            "range_width_used", "comparable_players",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_warning_when_no_comparable_has_salary(self, sample_pool_df, mock_calibration,
                                                   target_player_dict):
        """Comparables without any known salary must trigger the model-only warning."""
        pool = sample_pool_df.copy()
        pool["annual_fixed_eur"] = np.nan
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=pool):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            result = benchmark_player(target_player_dict)
        assert result["benchmark_n_comparables"] > 0
        assert result["benchmark_n_comparables_with_salary"] == 0
        assert result["benchmark_warning"] is not None
        assert "none has a known salary" in result["benchmark_warning"]
        assert result["benchmark_confidence"] == "LOW"

    def test_filters_by_name_when_no_id(self, mock_deps, sample_pool_df):
        """When target has no 'id', should filter pool by player_name."""
        player = {
            "player_name": "Lionel Messi",
            "main_position": "Right Winger",
            "competition_id": "MLS1",
            "competition_country": "United States",
            "age_months": 440,
            "market_value_current_eur": 20_000_000,
            "annual_fixed_eur": 15_000_000,
        }
        result = benchmark_player(player)
        # Messi should be excluded from his own comparables
        for comp in result["comparable_players"]:
            assert comp.get("player_name") != "Lionel Messi"


# --- fallback routing (no market value) ---

class TestFallbackRouting:
    @pytest.fixture
    def no_mv_player(self, target_player_dict):
        target_player_dict["market_value_current_eur"] = None
        target_player_dict["log_market_value_current_eur"] = None
        target_player_dict["has_market_value"] = 0
        return target_player_dict

    def _set_calibrations(self, mock_calibration):
        from salary_benchmark import calibration
        calibration._CAL_CACHE["full"] = mock_calibration
        # Fallback calibrations are wider, as expected from weaker models.
        wider = {
            **mock_calibration,
            "residual_p25": -0.6, "residual_p75": 0.6,
            "residual_p10": -1.1, "residual_p90": 1.1,
        }
        for variant in ("no_mv", "no_mv_no_pos", "no_mv_no_age"):
            calibration._CAL_CACHE[variant] = wider

    def test_routes_to_no_mv_variant(self, sample_pool_df, mock_calibration, no_mv_player):
        seen = {}

        def fake_predict(player, variant="full"):
            seen["variant"] = variant
            return 15.0

        with patch("salary_benchmark.benchmark.predict_log_salary", side_effect=fake_predict), \
             patch("salary_benchmark.benchmark.fallback_available", return_value=True), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            self._set_calibrations(mock_calibration)
            result = benchmark_player(no_mv_player)

        assert seen["variant"] == "no_mv"
        assert result["model_used"] == "no_mv"
        assert "fallback" in result["benchmark_warning"].lower()
        assert result["benchmark_confidence"] in ("MEDIUM", "LOW")

    def test_fallback_range_is_wider(self, sample_pool_df, mock_calibration,
                                     no_mv_player, target_player_dict):
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark.fallback_available", return_value=True), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            self._set_calibrations(mock_calibration)
            fallback_result = benchmark_player(dict(no_mv_player))
        width_fallback = (fallback_result["expected_salary_high_eur"]
                          - fallback_result["expected_salary_low_eur"])

        full_player = dict(no_mv_player)
        full_player["market_value_current_eur"] = 150_000_000
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            self._set_calibrations(mock_calibration)
            full_result = benchmark_player(full_player)
        width_full = (full_result["expected_salary_high_eur"]
                      - full_result["expected_salary_low_eur"])

        assert width_fallback > width_full

    def test_no_mv_without_fallback_model_raises(self, sample_pool_df, no_mv_player):
        with patch("salary_benchmark.benchmark.fallback_available", return_value=False), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            with pytest.raises(ValueError, match="fallback model is not available"):
                benchmark_player(no_mv_player)

    def test_full_model_used_when_mv_present(self, sample_pool_df, mock_calibration,
                                             target_player_dict):
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            self._set_calibrations(mock_calibration)
            result = benchmark_player(target_player_dict)
        assert result["model_used"] == "full"
        assert result["benchmark_warning"] is None

    def _run_with_variant_capture(self, player, sample_pool_df, mock_calibration):
        seen = {}

        def fake_predict(p, variant="full"):
            seen["variant"] = variant
            return 15.0

        with patch("salary_benchmark.benchmark.predict_log_salary", side_effect=fake_predict), \
             patch("salary_benchmark.benchmark.fallback_available", return_value=True), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            self._set_calibrations(mock_calibration)
            result = benchmark_player(player)
        return seen["variant"], result

    def test_routes_to_no_mv_no_pos_when_position_missing(self, sample_pool_df,
                                                          mock_calibration, no_mv_player):
        no_mv_player["main_position"] = None
        variant, result = self._run_with_variant_capture(
            no_mv_player, sample_pool_df, mock_calibration)
        assert variant == "no_mv_no_pos"
        assert result["model_used"] == "no_mv_no_pos"
        assert "position" in result["benchmark_warning"].lower()
        assert result["benchmark_confidence"] == "LOW"

    def test_routes_to_no_mv_no_age_when_age_missing(self, sample_pool_df,
                                                     mock_calibration, no_mv_player):
        no_mv_player["age_months"] = None
        variant, result = self._run_with_variant_capture(
            no_mv_player, sample_pool_df, mock_calibration)
        assert variant == "no_mv_no_age"
        assert result["model_used"] == "no_mv_no_age"
        assert "age" in result["benchmark_warning"].lower()
        assert result["benchmark_confidence"] == "LOW"

    def test_position_missing_with_mv_still_uses_no_pos_fallback(self, sample_pool_df,
                                                                 mock_calibration,
                                                                 target_player_dict):
        """A player WITH market value but no position uses no_mv_no_pos (MV unused)."""
        target_player_dict["main_position"] = None
        variant, result = self._run_with_variant_capture(
            target_player_dict, sample_pool_df, mock_calibration)
        assert variant == "no_mv_no_pos"

    def test_missing_position_and_age_refused(self, sample_pool_df, no_mv_player):
        no_mv_player["main_position"] = None
        no_mv_player["age_months"] = None
        with patch("salary_benchmark.benchmark.fallback_available", return_value=True), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            with pytest.raises(ValueError, match="neither main_position nor age_months"):
                benchmark_player(no_mv_player)


# --- benchmark_by_id ---

class TestBenchmarkById:
    def test_valid_id(self, sample_pool_df, mock_calibration):
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            result = benchmark_by_id(0)
            assert result["player_name"] == "Lionel Messi"

    def test_invalid_id_negative(self, sample_pool_df):
        with patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            with pytest.raises(ValueError, match="not found"):
                benchmark_by_id(-1)

    def test_invalid_id_too_large(self, sample_pool_df):
        with patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            with pytest.raises(ValueError, match="not found"):
                benchmark_by_id(9999)


# --- benchmark_by_name ---

class TestBenchmarkByName:
    def test_exact_match(self, sample_pool_df, mock_calibration):
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            result = benchmark_by_name("Lionel Messi")
            assert result["player_name"] == "Lionel Messi"

    def test_case_insensitive(self, sample_pool_df, mock_calibration):
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            result = benchmark_by_name("lionel messi")
            assert result["player_name"] == "Lionel Messi"

    def test_partial_match(self, sample_pool_df, mock_calibration):
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            result = benchmark_by_name("mbappe")
            assert "Mbappe" in result["player_name"]

    def test_not_found(self, sample_pool_df):
        with patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            with pytest.raises(ValueError, match="not found"):
                benchmark_by_name("Nonexistent Player XYZ")

    def test_multiple_matches_uses_highest_market_value(self, mock_calibration):
        """When multiple players match, the highest market_value is selected."""
        pool = pd.DataFrame({
            "id": [0, 1],
            "player_name": ["John Smith", "John Smith"],
            "main_position": ["Centre-Forward", "Centre-Forward"],
            "competition_id": ["GB1", "ES1"],
            "competition_country": ["England", "Spain"],
            "nationality": ["England", "England"],
            "age_months": [300, 280],
            "market_value_current_eur": [50_000_000, 100_000_000],
            "log_market_value_current_eur": [np.log1p(50_000_000), np.log1p(100_000_000)],
            "has_market_value": [1, 1],
            "annual_fixed_eur": [5_000_000, 10_000_000],
            "contract_length_months": [48, 60],
            "contract_months_remaining": [24, 48],
            "status": ["active", "active"],
            "season_start_year": [2024, 2024],
        })
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=pool):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            result = benchmark_by_name("John Smith")
            # Should pick the one with higher market value (id=1, mv=100M)
            assert result["market_value_current_eur"] == 100_000_000


# --- Error handling ---

class TestBenchmarkValidation:
    def test_missing_required_keys(self, sample_pool_df, mock_calibration):
        """benchmark_player raises on missing required keys."""
        with patch("salary_benchmark.benchmark.predict_log_salary", return_value=15.0), \
             patch("salary_benchmark.benchmark._load_pool", return_value=sample_pool_df):
            from salary_benchmark import calibration
            calibration._CAL_CACHE["full"] = mock_calibration
            # Missing main_position, market_value_current_eur, age_months
            with pytest.raises(ValueError, match="neither main_position nor age_months"):
                benchmark_player({"player_name": "Test"})

    def test_pool_file_not_found(self):
        """_load_pool raises FileNotFoundError when CSV doesn't exist."""
        from salary_benchmark.benchmark import _load_pool, POOL_PATH
        import salary_benchmark.benchmark as bm_mod
        bm_mod._POOL = None
        with patch.object(bm_mod, "POOL_PATH", Path("/nonexistent/pool.csv")):
            with pytest.raises(FileNotFoundError, match="Player pool not found"):
                _load_pool()
        bm_mod._POOL = None
