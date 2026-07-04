"""Tests for model_evaluation.py — evaluation metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from model_evaluation import (
    _salary_tier,
    _stratification_bins,
    bootstrap_ci,
    calibration_coverage,
    calibration_quality_label,
    segment_analysis,
)
import pandas as pd


class TestCalibrationQualityLabel:
    def test_on_target_is_good(self):
        assert calibration_quality_label(0.505) == "good"

    def test_optimistic_coverage_needs_adjustment(self):
        # In-sample-style over-coverage must not be labelled good
        assert calibration_quality_label(0.581) == "needs_adjustment"

    def test_under_coverage_needs_adjustment(self):
        assert calibration_quality_label(0.40) == "needs_adjustment"


class TestSalaryTier:
    def test_low_tier(self):
        assert _salary_tier(np.log1p(500_000)) == "<€1M"

    def test_mid_tier(self):
        assert _salary_tier(np.log1p(2_000_000)) == "€1-5M"

    def test_high_tier(self):
        assert _salary_tier(np.log1p(8_000_000)) == "€5-15M"

    def test_very_high_tier(self):
        assert _salary_tier(np.log1p(20_000_000)) == ">€15M"

    def test_zero(self):
        assert _salary_tier(np.log1p(0)) == "<€1M"


class TestBootstrapCI:
    def test_returns_expected_keys(self):
        result = bootstrap_ci([0.5, 0.6, 0.55, 0.52, 0.58])
        assert "mean" in result
        assert "std" in result
        assert "ci_low" in result
        assert "ci_high" in result

    def test_ci_contains_mean(self):
        values = [0.5, 0.6, 0.55, 0.52, 0.58]
        result = bootstrap_ci(values)
        assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    def test_narrow_ci_for_identical_values(self):
        values = [0.5] * 10
        result = bootstrap_ci(values)
        assert result["ci_high"] - result["ci_low"] < 0.01

    def test_wider_ci_for_variable_values(self):
        values = [0.1, 0.9, 0.2, 0.8, 0.5]
        result = bootstrap_ci(values)
        assert result["ci_high"] - result["ci_low"] > 0.1


class TestStratificationBins:
    def test_creates_bins(self):
        df = pd.DataFrame({
            "competition_country": ["England", "Spain", "England"],
            "log_annual_fixed_eur": [np.log1p(500_000), np.log1p(3_000_000), np.log1p(20_000_000)],
        })
        bins = _stratification_bins(df)
        assert len(bins) == 3
        assert "England" in bins[0]
        assert "Spain" in bins[1]

    def test_handles_missing_country(self):
        df = pd.DataFrame({
            "competition_country": [None, "Spain"],
            "log_annual_fixed_eur": [np.log1p(1_000_000), np.log1p(2_000_000)],
        })
        bins = _stratification_bins(df)
        assert "Unknown" in bins[0]
