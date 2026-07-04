"""Tests for salary_benchmark/monitoring.py — prediction logging and drift detection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from salary_benchmark.monitoring import (
    PredictionLogger,
    DriftDetector,
    MonitoringReporter,
)


class TestPredictionLogger:
    def test_logs_to_file(self, tmp_path):
        log_path = tmp_path / "predictions.jsonl"
        logger = PredictionLogger(log_path)
        logger.log(
            request={"player_id": 0},
            result={
                "player_id": 0,
                "player_name": "Test",
                "expected_salary_median_eur": 5_000_000,
                "actual_salary_eur": 4_000_000,
                "salary_status": "FAIRLY_PAID",
                "benchmark_confidence": "MEDIUM",
                "benchmark_n_comparables": 25,
                "comparable_level_used": 1,
                "market_value_current_eur": 50_000_000,
            },
            latency_ms=123.45,
        )
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["player_name"] == "Test"
        assert entry["latency_ms"] == 123.45
        assert entry["predicted_median_eur"] == 5_000_000

    def test_appends_multiple_entries(self, tmp_path):
        log_path = tmp_path / "predictions.jsonl"
        logger = PredictionLogger(log_path)
        for i in range(5):
            logger.log(
                request={},
                result={"player_name": f"P{i}", "expected_salary_median_eur": i * 1_000_000},
                latency_ms=float(i),
            )
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_creates_parent_directory(self, tmp_path):
        log_path = tmp_path / "deep" / "nested" / "log.jsonl"
        logger = PredictionLogger(log_path)
        logger.log(request={}, result={"expected_salary_median_eur": 1}, latency_ms=1.0)
        assert log_path.exists()


class TestDriftDetector:
    def test_psi_identical_distributions(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0, 1, 1000)
        psi = DriftDetector.psi(a, b)
        assert psi < 0.1  # No significant drift

    def test_psi_shifted_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(2, 1, 1000)  # Shifted by 2 std
        psi = DriftDetector.psi(a, b)
        assert psi > 0.25  # Significant drift

    def test_ks_test_same_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        result = DriftDetector.ks_test(a, b)
        assert result["significant_drift"] == False

    def test_ks_test_different_distribution(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(3, 1, 500)
        result = DriftDetector.ks_test(a, b)
        assert result["significant_drift"] == True

    def test_check_prediction_drift_insufficient_data(self):
        detector = DriftDetector()
        result = detector.check_prediction_drift([1.0, 2.0])
        assert result["status"] == "insufficient_data"

    def test_check_prediction_drift_checked(self):
        detector = DriftDetector()
        # Need at least 20 predictions for drift check
        rng = np.random.default_rng(42)
        predictions = list(rng.normal(14.0, 0.45, 100))
        result = detector.check_prediction_drift(predictions)
        assert result["status"] == "checked"
        assert "psi" in result
        assert "ks_test" in result


class TestMonitoringReporter:
    def test_no_data_report(self, tmp_path):
        reporter = MonitoringReporter(
            log_path=tmp_path / "empty.jsonl",
            report_path=tmp_path / "report.json",
        )
        report = reporter.generate_report()
        assert report["status"] == "no_data"

    def test_generates_report_from_logs(self, tmp_path):
        log_path = tmp_path / "predictions.jsonl"
        report_path = tmp_path / "report.json"

        # Write some fake logs
        entries = []
        for i in range(20):
            entries.append(json.dumps({
                "timestamp": f"2025-07-0{(i%9)+1}T12:00:00Z",
                "player_id": i,
                "player_name": f"Player {i}",
                "predicted_median_eur": 5_000_000 + i * 100_000,
                "actual_salary_eur": 4_500_000 + i * 50_000,
                "salary_status": "FAIRLY_PAID" if i % 3 == 0 else "OVERPAID",
                "confidence": "MEDIUM" if i % 2 == 0 else "LOW",
                "n_comparables": 20 + i,
                "comparable_level": 1,
                "latency_ms": 100 + i * 10,
                "market_value_eur": 50_000_000,
                "log_predicted": float(np.log1p(5_000_000 + i * 100_000)),
            }))
        log_path.write_text("\n".join(entries))

        reporter = MonitoringReporter(log_path=log_path, report_path=report_path)
        report = reporter.generate_report()

        assert report["n_predictions"] == 20
        assert "latency" in report
        assert report["latency"]["mean_ms"] > 0
        assert "prediction_distribution" in report
        assert "confidence_distribution" in report
        assert "status_distribution" in report
        assert "drift_detection" in report
        assert "alerts" in report
        assert report_path.exists()

    def test_alerts_on_high_latency(self, tmp_path):
        log_path = tmp_path / "slow.jsonl"
        report_path = tmp_path / "report.json"

        entries = [json.dumps({
            "timestamp": "2025-07-01T12:00:00Z",
            "latency_ms": 6000,  # Very slow
            "log_predicted": 15.0,
            "confidence": "LOW",
            "salary_status": "UNKNOWN",
        }) for _ in range(10)]
        log_path.write_text("\n".join(entries))

        reporter = MonitoringReporter(log_path=log_path, report_path=report_path)
        report = reporter.generate_report()
        assert "HIGH_LATENCY" in str(report["alerts"])
