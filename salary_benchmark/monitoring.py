"""Production prediction logging and drift detection scaffold.

This module provides:
- Request-level logging of every benchmark prediction (JSONL)
- Basic distribution drift detection (PSI + KS test)
- Latency tracking

LIMITATIONS (honest scope):
- Drift detection is TEMPORAL: it splits the recent log-scale predictions into
  a first-half reference window and a second-half current window and compares
  the two (PSI + KS test). It does NOT compare against the training prediction
  distribution — real drift detection would store training-time predictions
  and compare production predictions against those.
- This is a monitoring SKELETON suitable for a prototype. Production monitoring
  would use a proper observability stack (Prometheus, Grafana, MLflow, etc.).

Usage:
    # Monitoring runs automatically when integrated into benchmark.py
    # Manual report generation:
    python -m salary_benchmark.monitoring
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model import FEATURES as MODEL_FEATURES

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = _PROJECT_ROOT / "logs" / "benchmark_predictions.jsonl"
REPORT_PATH = _PROJECT_ROOT / "models" / "monitoring_report.json"


class PredictionLogger:
    """Logs every benchmark prediction to a JSONL file."""

    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, request: dict, result: dict, latency_ms: float):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "player_id": result.get("player_id"),
            "player_name": result.get("player_name"),
            "predicted_median_eur": result.get("expected_salary_median_eur"),
            "actual_salary_eur": result.get("actual_salary_eur"),
            "salary_status": result.get("salary_status"),
            "confidence": result.get("benchmark_confidence"),
            "n_comparables": result.get("benchmark_n_comparables"),
            "comparable_level": result.get("comparable_level_used"),
            "model_used": result.get("model_used"),
            "latency_ms": round(latency_ms, 2),
            "market_value_eur": result.get("market_value_current_eur"),
            "log_predicted": float(np.log1p(result["expected_salary_median_eur"])) if result.get("expected_salary_median_eur") else None,
            # Input features, so feature-level drift can be analysed later.
            "features": {k: request.get(k) for k in MODEL_FEATURES},
        }
        # Exclusive lock: concurrent writers (server + CLI + scripts) must not
        # interleave partial lines in the JSONL the drift detector reads.
        import fcntl

        with open(self.log_path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, default=str) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


class DriftDetector:
    """Detects temporal distribution drift within recent production predictions."""

    @staticmethod
    def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index — measures distribution shift.

        PSI < 0.1: no significant shift
        PSI 0.1-0.25: moderate shift
        PSI > 0.25: significant shift
        """
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            n_bins + 1,
        )
        expected_pcts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_pcts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Avoid division by zero
        expected_pcts = np.clip(expected_pcts, 0.001, None)
        actual_pcts = np.clip(actual_pcts, 0.001, None)

        return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))

    @staticmethod
    def ks_test(sample_a: np.ndarray, sample_b: np.ndarray) -> dict:
        """Kolmogorov-Smirnov test for distribution difference."""
        from scipy import stats
        stat, p_value = stats.ks_2samp(sample_a, sample_b)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_drift": p_value < 0.05,
        }

    def check_prediction_drift(self, recent_predictions: list[float]) -> dict:
        """Temporal drift check on the recent log-scale predictions.

        Splits the recent predictions into a first-half reference window and a
        second-half current window, then compares the two distributions with
        PSI and a KS test. We don't store training-time predictions, so this
        detects shifts WITHIN the production window, not train-vs-production
        drift (see module docstring).
        """
        if not recent_predictions:
            return {"status": "insufficient_data"}

        recent = np.array(recent_predictions)
        if len(recent) < 20:
            return {"status": "insufficient_data", "n_recent": len(recent)}

        # Split into reference (first 50%) and recent (last 50%) for temporal drift
        mid = len(recent) // 2
        reference = recent[:mid]
        current = recent[mid:]

        # Compute drift metrics
        psi_value = self.psi(reference, current)
        ks_result = self.ks_test(reference, current)

        return {
            "status": "checked",
            "n_recent": len(recent),
            "n_reference": len(reference),
            "n_current": len(current),
            "reference_mean": float(reference.mean()),
            "reference_std": float(reference.std()),
            "current_mean": float(current.mean()),
            "current_std": float(current.std()),
            "psi": psi_value,
            "psi_alert": psi_value > 0.25,
            "ks_test": ks_result,
        }


class MonitoringReporter:
    """Generates monitoring reports from prediction logs."""

    def __init__(self, log_path: Path = LOG_PATH, report_path: Path = REPORT_PATH):
        self.log_path = log_path
        self.report_path = report_path
        self.drift_detector = DriftDetector()

    def _load_logs(self, last_n: int | None = None) -> list[dict]:
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if last_n:
            entries = entries[-last_n:]
        return entries

    def generate_report(self, last_n: int = 1000) -> dict:
        """Generate a monitoring report from recent predictions."""
        entries = self._load_logs(last_n)
        if not entries:
            return {"status": "no_data", "message": "No prediction logs found"}

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_predictions": len(entries),
            "time_range": {
                "first": entries[0].get("timestamp"),
                "last": entries[-1].get("timestamp"),
            },
        }

        # Latency stats
        latencies = [e["latency_ms"] for e in entries if e.get("latency_ms") is not None]
        if latencies:
            report["latency"] = {
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "max_ms": float(np.max(latencies)),
            }

        # Prediction distribution
        predictions = [e["log_predicted"] for e in entries if e.get("log_predicted") is not None]
        if predictions:
            report["prediction_distribution"] = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            }

        # Confidence distribution
        confidences = [e.get("confidence") for e in entries if e.get("confidence")]
        if confidences:
            from collections import Counter
            conf_counts = Counter(confidences)
            total = len(confidences)
            report["confidence_distribution"] = {
                k: {"count": v, "pct": round(v / total * 100, 1)}
                for k, v in conf_counts.items()
            }

        # Status distribution
        statuses = [e.get("salary_status") for e in entries if e.get("salary_status")]
        if statuses:
            from collections import Counter
            status_counts = Counter(statuses)
            total = len(statuses)
            report["status_distribution"] = {
                k: {"count": v, "pct": round(v / total * 100, 1)}
                for k, v in status_counts.items()
            }

        # Drift detection
        if predictions:
            drift = self.drift_detector.check_prediction_drift(predictions)
            report["drift_detection"] = drift

        # Alerts
        alerts = []
        if report.get("latency", {}).get("p95_ms", 0) > 5000:
            alerts.append("HIGH_LATENCY: p95 > 5s")
        if report.get("drift_detection", {}).get("psi_alert"):
            alerts.append("PREDICTION_DRIFT: PSI > 0.25")
        if report.get("drift_detection", {}).get("ks_test", {}).get("significant_drift"):
            alerts.append("DISTRIBUTION_SHIFT: KS test significant")
        report["alerts"] = alerts

        # Save report
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2, default=str))
        return report


# Singleton logger for use in benchmark.py
_LOGGER: PredictionLogger | None = None


def get_logger() -> PredictionLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = PredictionLogger()
    return _LOGGER


if __name__ == "__main__":
    reporter = MonitoringReporter()
    report = reporter.generate_report()
    print(json.dumps(report, indent=2, default=str))
