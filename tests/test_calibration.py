"""Tests for salary_benchmark/calibration.py."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from salary_benchmark import calibration
from salary_benchmark.calibration import load_calibration, salary_range


# --- salary_range ---

class TestSalaryRange:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Reset the module-level cache before each test."""
        calibration._CAL_CACHE.clear()
        yield
        calibration._CAL_CACHE.clear()

    def test_normal_width(self, mock_calibration):
        calibration._CAL_CACHE["full"] = mock_calibration
        result = salary_range(log_pred=14.0, width="normal")
        assert "expected_salary_low_eur" in result
        assert "expected_salary_median_eur" in result
        assert "expected_salary_high_eur" in result
        assert result["expected_salary_low_eur"] < result["expected_salary_median_eur"]
        assert result["expected_salary_median_eur"] < result["expected_salary_high_eur"]

    def test_wide_width(self, mock_calibration):
        calibration._CAL_CACHE["full"] = mock_calibration
        result = salary_range(log_pred=14.0, width="wide")
        # Wide uses p10/p90 which are wider than p25/p75
        normal_result = salary_range(log_pred=14.0, width="normal")
        assert result["expected_salary_low_eur"] < normal_result["expected_salary_low_eur"]
        assert result["expected_salary_high_eur"] > normal_result["expected_salary_high_eur"]

    def test_median_is_expm1_of_log_pred(self, mock_calibration):
        calibration._CAL_CACHE["full"] = mock_calibration
        log_pred = 15.0
        result = salary_range(log_pred=log_pred, width="normal")
        expected_median = float(np.expm1(log_pred))
        assert abs(result["expected_salary_median_eur"] - expected_median) < 1.0

    def test_uses_calibration_residuals(self, mock_calibration):
        calibration._CAL_CACHE["full"] = mock_calibration
        log_pred = 14.0
        result = salary_range(log_pred=log_pred, width="normal")
        # p25=-0.4, p75=0.4
        expected_low = float(np.expm1(14.0 + (-0.4)))
        expected_high = float(np.expm1(14.0 + 0.4))
        assert abs(result["expected_salary_low_eur"] - expected_low) < 1.0
        assert abs(result["expected_salary_high_eur"] - expected_high) < 1.0


# --- load_calibration ---

class TestLoadCalibration:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        calibration._CAL_CACHE.clear()
        yield
        calibration._CAL_CACHE.clear()

    def test_load_from_file(self, mock_calibration, tmp_path):
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text(json.dumps(mock_calibration))
        with patch.dict(calibration.CALIBRATION_PATHS, {"full": cal_file}):
            result = load_calibration()
            assert result["n_samples"] == 500
            assert result["residual_p25"] == -0.4

    def test_caching(self, mock_calibration, tmp_path):
        cal_file = tmp_path / "calibration.json"
        cal_file.write_text(json.dumps(mock_calibration))
        with patch.dict(calibration.CALIBRATION_PATHS, {"full": cal_file}):
            result1 = load_calibration()
            result2 = load_calibration()
            assert result1 is result2

    def test_raises_if_missing(self, mock_calibration, tmp_path):
        missing_file = tmp_path / "does_not_exist.json"
        with patch.dict(calibration.CALIBRATION_PATHS, {"full": missing_file}):
            with pytest.raises(FileNotFoundError, match="not found"):
                load_calibration()

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="Unknown calibration variant"):
            load_calibration("bogus")


class TestBuildCalibration:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        calibration._CAL_CACHE.clear()
        yield
        calibration._CAL_CACHE.clear()

    @pytest.mark.slow
    def test_build_calibration(self, tmp_path):
        """Test build_calibration produces valid calibration JSON (real AutoGluon, slow)."""
        pool_csv = tmp_path / "pool.csv"
        import pandas as pd
        n = 100
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "annual_fixed_eur": rng.uniform(500_000, 20_000_000, n),
            "main_position": rng.choice(["CF", "LW", "GK", "CB"], n),
            "nationality": ["Spain"] * n,
            "competition_id": rng.choice(["GB1", "ES1"], n),
            "competition_country": rng.choice(["England", "Spain"], n),
            "status": ["active"] * n,
            "season_start_year": [2025] * n,
            "age_months": rng.uniform(200, 400, n),
            "contract_length_months": rng.uniform(12, 60, n),
            "contract_months_remaining": rng.uniform(6, 48, n),
            "contract_recency_months": rng.uniform(1, 24, n),
            "has_contract_dates": [1] * n,
            "has_market_value": [1] * n,
            "log_market_value_current_eur": rng.uniform(14, 19, n),
            "has_release_clause": rng.choice([0, 1], n),
            "log_release_clause_eur": rng.uniform(0, 20, n),
        })
        df["log_annual_fixed_eur"] = np.log1p(df["annual_fixed_eur"])
        df.to_csv(pool_csv, index=False)

        out_path = tmp_path / "calibration.json"

        from salary_benchmark.calibration import build_calibration
        result = build_calibration(pool_path=pool_csv, out_path=out_path, n_folds=3)

        assert out_path.exists()
        assert "n_samples" in result
        assert result["n_samples"] == n
        assert "residual_p10" in result
        assert "residual_p90" in result
        assert "rmse" in result
        assert result["method"] == "out_of_fold_cv"
        assert result["residual_p10"] < result["residual_p90"]

    def test_build_calibration_empty_pool(self, tmp_path):
        """build_calibration should handle empty salary data gracefully."""
        pool_csv = tmp_path / "pool.csv"
        import pandas as pd
        df = pd.DataFrame({
            "annual_fixed_eur": [np.nan, np.nan],
            "main_position": ["CF", "CF"],
        })
        df.to_csv(pool_csv, index=False)

        from salary_benchmark.calibration import build_calibration
        # Should either raise or produce empty result
        with pytest.raises(Exception):
            build_calibration(pool_path=pool_csv, out_path=tmp_path / "cal.json", n_folds=2)

    def test_build_calibration_missing_pool(self, tmp_path):
        """build_calibration raises for missing pool file."""
        from salary_benchmark.calibration import build_calibration
        with pytest.raises(Exception):
            build_calibration(pool_path=tmp_path / "nonexistent.csv")

    def test_load_calibration_corrupt_json(self, tmp_path):
        """load_calibration raises for corrupt JSON."""
        cal_file = tmp_path / "bad.json"
        cal_file.write_text("{not valid json!!!")
        with patch.dict(calibration.CALIBRATION_PATHS, {"full": cal_file}):
            with pytest.raises(Exception):
                load_calibration()
