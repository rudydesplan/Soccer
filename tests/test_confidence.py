"""Tests for salary_benchmark/confidence.py."""

from __future__ import annotations

import pytest

from salary_benchmark.confidence import confidence_label, salary_status


# --- confidence_label ---

class TestConfidenceLabel:
    def test_high_confidence(self):
        assert confidence_label(n_comparables=50, avg_similarity=0.75, relative_range_width=0.59) == "high"

    def test_high_confidence_above_threshold(self):
        assert confidence_label(n_comparables=100, avg_similarity=0.90, relative_range_width=0.30) == "high"

    def test_medium_confidence(self):
        assert confidence_label(n_comparables=20, avg_similarity=0.60, relative_range_width=0.99) == "medium"

    def test_medium_not_high(self):
        # Meets medium but not high (n_comparables < 50)
        assert confidence_label(n_comparables=30, avg_similarity=0.80, relative_range_width=0.50) == "medium"

    def test_low_confidence_few_comparables(self):
        assert confidence_label(n_comparables=5, avg_similarity=0.80, relative_range_width=0.30) == "low"

    def test_low_confidence_low_similarity(self):
        assert confidence_label(n_comparables=100, avg_similarity=0.50, relative_range_width=0.30) == "low"

    def test_low_confidence_wide_range(self):
        assert confidence_label(n_comparables=100, avg_similarity=0.80, relative_range_width=1.50) == "low"

    def test_boundary_high_exact(self):
        # Exact boundary values for high
        assert confidence_label(n_comparables=50, avg_similarity=0.75, relative_range_width=0.59) == "high"

    def test_boundary_high_fails_width(self):
        # Width exactly 0.60 → not < 0.60, so not high
        assert confidence_label(n_comparables=50, avg_similarity=0.75, relative_range_width=0.60) != "high"

    def test_boundary_medium_exact(self):
        assert confidence_label(n_comparables=20, avg_similarity=0.60, relative_range_width=0.99) == "medium"

    def test_boundary_medium_fails_width(self):
        # Width exactly 1.00 → not < 1.00, so not medium
        assert confidence_label(n_comparables=20, avg_similarity=0.60, relative_range_width=1.00) == "low"

    def test_boundary_medium_fails_similarity(self):
        assert confidence_label(n_comparables=20, avg_similarity=0.59, relative_range_width=0.80) == "low"

    def test_boundary_medium_fails_n(self):
        assert confidence_label(n_comparables=19, avg_similarity=0.60, relative_range_width=0.80) == "low"


# --- salary_status ---

class TestSalaryStatus:
    def test_underpaid(self):
        assert salary_status(actual_eur=5000, low_eur=10000, high_eur=20000) == "underpaid"

    def test_overpaid(self):
        assert salary_status(actual_eur=25000, low_eur=10000, high_eur=20000) == "overpaid"

    def test_fairly_paid(self):
        assert salary_status(actual_eur=15000, low_eur=10000, high_eur=20000) == "fairly_paid"

    def test_fairly_paid_at_low_boundary(self):
        assert salary_status(actual_eur=10000, low_eur=10000, high_eur=20000) == "fairly_paid"

    def test_fairly_paid_at_high_boundary(self):
        assert salary_status(actual_eur=20000, low_eur=10000, high_eur=20000) == "fairly_paid"

    def test_unknown_when_none(self):
        assert salary_status(actual_eur=None, low_eur=10000, high_eur=20000) == "unknown"
