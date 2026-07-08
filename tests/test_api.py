"""Tests for soccer-benchmark/backend API endpoints.

Uses FastAPI TestClient with mocked pool/model to avoid loading real data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Add the backend dir to sys.path so routers can be imported
_BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "soccer-benchmark" / "backend")
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture
def api_pool_df():
    """Small pool for API tests."""
    return pd.DataFrame({
        "id": [0, 1, 2],
        "player_name": ["Lionel Messi", "Kylian Mbappe", "Erling Haaland"],
        "main_position": ["Right Winger", "Centre-Forward", "Centre-Forward"],
        "team_name": ["Inter Miami", "Real Madrid", "Manchester City"],
        "competition_id": ["MLS1", "ES1", "GB1"],
        "competition_country": ["United States", "Spain", "England"],
        "nationality": ["Argentina", "France", "Norway"],
        "age_months": [440, 306, 290],
        "market_value_current_eur": [20_000_000, 180_000_000, 170_000_000],
        "annual_fixed_eur": [15_000_000, 25_000_000, 20_000_000],
        "log_market_value_current_eur": [np.log1p(20_000_000), np.log1p(180_000_000), np.log1p(170_000_000)],
        "has_market_value": [1, 1, 1],
        "contract_length_months": [24, 60, 60],
        "contract_months_remaining": [12, 48, 50],
        "status": ["active", "active", "active"],
        "season_start_year": [2024, 2024, 2024],
    })


@pytest.fixture
def client(api_pool_df):
    """FastAPI TestClient with mocked dependencies."""
    # Patch the pool loading in both routers.players and salary_benchmark.benchmark
    with patch("routers.players._load_pool", return_value=api_pool_df), \
         patch("routers.players.POOL_PATH", Path("/fake/path.csv")):
        # Import app AFTER patches are in place for the module-level state
        # But since _load_pool is a function called at request time, just patch it
        from main import app
        from starlette.testclient import TestClient
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_live_always_ok(self, client):
        response = client.get("/api/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_when_warmed_up(self, client):
        response = client.get("/api/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["pool_loaded"] is True
        assert data["models"]["full"] == "loaded"
        # Every variant is reported, whatever its state
        assert set(data["models"]) == {"full", "no_mv", "no_mv_no_pos", "no_mv_no_age"}
        assert all(state in ("loaded", "not_trained") for state in data["models"].values())

    def test_ready_503_when_pool_missing(self, client):
        import main
        original = main.WARMUP_STATE["pool_loaded"]
        main.WARMUP_STATE["pool_loaded"] = False
        try:
            response = client.get("/api/health/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"
            # Legacy /api/health mirrors readiness
            assert client.get("/api/health").status_code == 503
            # Liveness is unaffected
            assert client.get("/api/health/live").status_code == 200
        finally:
            main.WARMUP_STATE["pool_loaded"] = original

    def test_ready_503_when_model_failed(self, client):
        import main
        original = main.WARMUP_STATE["models"].get("no_mv")
        main.WARMUP_STATE["models"]["no_mv"] = "failed"
        try:
            response = client.get("/api/health/ready")
            assert response.status_code == 503
            assert response.json()["models"]["no_mv"] == "failed"
        finally:
            main.WARMUP_STATE["models"]["no_mv"] = original

    def test_untrained_fallback_does_not_block_readiness(self, client):
        import main
        original = main.WARMUP_STATE["models"].get("no_mv_no_age")
        main.WARMUP_STATE["models"]["no_mv_no_age"] = "not_trained"
        try:
            assert client.get("/api/health/ready").status_code == 200
        finally:
            main.WARMUP_STATE["models"]["no_mv_no_age"] = original


class TestOpenAPISchema:
    """The generated OpenAPI spec is part of the API contract — assert its shape."""

    EXPECTED_PATHS = {
        "/api/health",
        "/api/health/live",
        "/api/health/ready",
        "/api/players/options",
        "/api/players/search",
        "/api/players/{player_id}",
        "/api/benchmark",
        "/api/benchmark/explain",
        "/api/meta/model-card",
    }

    @pytest.fixture
    def spec(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        return response.json()

    def test_all_endpoints_present(self, spec):
        assert set(spec["paths"]) == self.EXPECTED_PATHS

    def test_metadata(self, spec):
        info = spec["info"]
        assert info["title"] == "Soccer Salary Benchmark API"
        assert info["version"] == "1.0.0"
        # The hand-written guide must survive in the description
        assert "Authentication" in info["description"]
        assert "Quickstart" in info["description"]

    def test_benchmark_error_responses_documented(self, spec):
        responses = spec["paths"]["/api/benchmark"]["post"]["responses"]
        assert {"200", "400", "404", "422", "500"} <= set(responses)

    def test_player_detail_error_responses_documented(self, spec):
        responses = spec["paths"]["/api/players/{player_id}"]["get"]["responses"]
        assert {"200", "404", "503"} <= set(responses)

    def test_search_pool_unavailable_documented(self, spec):
        responses = spec["paths"]["/api/players/search"]["get"]["responses"]
        assert "503" in responses

    def test_benchmark_request_has_examples(self, spec):
        schema = spec["components"]["schemas"]["BenchmarkRequest"]
        examples = schema.get("examples")
        assert examples, "BenchmarkRequest should expose request examples for Try it out"
        assert any("player_id" in ex for ex in examples)
        assert any("main_position" in ex for ex in examples)

    def test_health_has_response_model(self, spec):
        ref = spec["paths"]["/api/health"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert ref.get("$ref", "").endswith("HealthResponse")

    def test_error_schema_shape(self, spec):
        error_schema = spec["components"]["schemas"]["ErrorResponse"]
        assert "detail" in error_schema["properties"]

    def test_docs_pages_served(self, client):
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200


class TestPlayersSearch:
    def test_search_found(self, client):
        response = client.get("/api/players/search", params={"q": "messi"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["player_name"] == "Lionel Messi"

    def test_search_partial(self, client):
        response = client.get("/api/players/search", params={"q": "mb"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert "Mbappe" in data[0]["player_name"]

    def test_search_no_results(self, client):
        response = client.get("/api/players/search", params={"q": "zzzzz"})
        assert response.status_code == 200
        assert response.json() == []

    def test_search_short_query(self, client):
        response = client.get("/api/players/search", params={"q": "x"})
        assert response.status_code == 422

    def test_search_negative_limit_rejected(self, client):
        """Negative limit must be rejected — pandas head(-n) would dump the pool."""
        response = client.get("/api/players/search", params={"q": "messi", "limit": -1})
        assert response.status_code == 422

    def test_search_zero_limit_rejected(self, client):
        response = client.get("/api/players/search", params={"q": "messi", "limit": 0})
        assert response.status_code == 422


class TestPlayersDetail:
    def test_get_player_valid(self, client):
        response = client.get("/api/players/0")
        assert response.status_code == 200
        data = response.json()
        assert data["player_name"] == "Lionel Messi"

    def test_get_player_not_found(self, client):
        response = client.get("/api/players/99999")
        assert response.status_code == 404


class TestBenchmarkEndpoint:
    def _full_mock_result(self, **overrides):
        """Create a complete benchmark result that passes BenchmarkResponse validation."""
        base = {
            "player_id": 0,
            "player_name": "Test Player",
            "main_position": "Centre-Forward",
            "competition_id": "GB1",
            "competition_country": "England",
            "age_months": 300,
            "market_value_current_eur": 50_000_000,
            "expected_salary_low_eur": 5_000_000,
            "expected_salary_median_eur": 8_000_000,
            "expected_salary_high_eur": 12_000_000,
            "actual_salary_eur": 10_000_000,
            "salary_percentile": 60,
            "salary_status": "FAIRLY_PAID",
            "benchmark_confidence": "MEDIUM",
            "benchmark_n_comparables": 25,
            "benchmark_n_comparables_with_salary": 20,
            "benchmark_avg_similarity": 0.72,
            "comparable_level_used": 1,
            "range_width_used": "normal",
            "benchmark_warning": None,
            "comparable_players": [],
        }
        base.update(overrides)
        return base

    def test_benchmark_by_id(self, client, api_pool_df):
        mock_result = self._full_mock_result(player_name="Lionel Messi")
        with patch("routers.benchmark.benchmark_by_id", return_value=mock_result):
            response = client.post("/api/benchmark", json={"player_id": 0})
            assert response.status_code == 200
            data = response.json()
            assert data["player_name"] == "Lionel Messi"

    def test_benchmark_by_name(self, client):
        mock_result = self._full_mock_result(player_name="Kylian Mbappe")
        with patch("routers.benchmark.benchmark_by_name", return_value=mock_result):
            response = client.post("/api/benchmark", json={"player_name": "Kylian Mbappe"})
            assert response.status_code == 200
            data = response.json()
            assert data["player_name"] == "Kylian Mbappe"

    def test_options_endpoint(self, client):
        """The manual form's options endpoint returns positions and competitions."""
        response = client.get("/api/players/options")
        assert response.status_code == 200
        data = response.json()
        assert "Centre-Forward" in data["positions"]
        comp_ids = [c["id"] for c in data["competitions"]]
        assert set(comp_ids) == {"MLS1", "ES1", "GB1"}
        # Sorted by player count: Centre-Forward leagues have 1 each; order stable
        for comp in data["competitions"]:
            assert comp["name"]
            assert "country" in comp

    def test_benchmark_manual_fields(self, client):
        mock_result = self._full_mock_result(player_name="custom_player", salary_status="UNKNOWN", actual_salary_eur=None)
        with patch("routers.benchmark.benchmark_player", return_value=mock_result):
            response = client.post("/api/benchmark", json={
                "main_position": "Centre-Forward",
                "market_value_current_eur": 50_000_000,
                "competition_id": "GB1",
                "competition_country": "England",
                "age_months": 300,
            })
            assert response.status_code == 200
            data = response.json()
            assert data["player_name"] == "custom_player"

    def test_benchmark_missing_fields(self, client):
        """Empty request body should fail Pydantic validation (422)."""
        response = client.post("/api/benchmark", json={})
        assert response.status_code == 422

    def test_benchmark_manual_without_age_rejected(self, client):
        """Manual benchmark must not silently assume an age."""
        response = client.post("/api/benchmark", json={
            "main_position": "Centre-Forward",
            "market_value_current_eur": 50_000_000,
        })
        assert response.status_code == 400
        assert "age_months" in response.json()["detail"]

    def test_benchmark_not_found(self, client):
        with patch("routers.benchmark.benchmark_by_name", side_effect=ValueError("not found")):
            response = client.post("/api/benchmark", json={"player_name": "Nobody"})
            assert response.status_code == 404

    def test_benchmark_invalid_range_width(self, client):
        response = client.post("/api/benchmark", json={"player_id": 0, "range_width": "invalid"})
        assert response.status_code == 422

    def test_benchmark_service_unavailable(self, client):
        with patch("routers.benchmark.benchmark_by_id", side_effect=FileNotFoundError("pool not found")):
            response = client.post("/api/benchmark", json={"player_id": 0})
            assert response.status_code == 500


class TestExplainEndpoint:
    _MOCK_EXPLANATION = {
        "player_name": "Erling Haaland",
        "model_used": "full",
        "base_salary_eur": 366_718,
        "predicted_salary_eur": 20_027_641,
        "features": [
            {
                "feature": "log_market_value_current_eur",
                "label": "Market value",
                "value": "€200.0M",
                "shap_log": 2.753,
                "pct_effect": 1467.1,
            },
            {
                "feature": "age_months",
                "label": "Age",
                "value": "24.9 years",
                "shap_log": 0.289,
                "pct_effect": 33.5,
            },
        ],
    }

    def test_explain_by_id(self, client):
        with patch("routers.benchmark.explain_by_id", return_value=self._MOCK_EXPLANATION) as mock:
            response = client.post("/api/benchmark/explain", json={"player_id": 2})
            assert response.status_code == 200
            data = response.json()
            assert data["player_name"] == "Erling Haaland"
            assert data["features"][0]["label"] == "Market value"
            mock.assert_called_once_with(2)

    def test_explain_by_name(self, client):
        with patch("routers.benchmark.explain_by_name", return_value=self._MOCK_EXPLANATION) as mock:
            response = client.post("/api/benchmark/explain", json={"player_name": "Haaland"})
            assert response.status_code == 200
            mock.assert_called_once_with("Haaland")

    def test_explain_manual_player(self, client):
        with patch("routers.benchmark.explain_player", return_value=self._MOCK_EXPLANATION) as mock:
            response = client.post("/api/benchmark/explain", json={
                "main_position": "Centre-Forward",
                "age_months": 300,
            })
            assert response.status_code == 200
            player = mock.call_args[0][0]
            assert player["main_position"] == "Centre-Forward"
            assert player["age_months"] == 300

    def test_explain_manual_without_age_rejected(self, client):
        response = client.post("/api/benchmark/explain", json={
            "main_position": "Centre-Forward",
        })
        assert response.status_code == 400
        assert "age_months" in response.json()["detail"]

    def test_explain_not_found(self, client):
        with patch("routers.benchmark.explain_by_name", side_effect=ValueError("Player 'X' not found")):
            response = client.post("/api/benchmark/explain", json={"player_name": "Nobody"})
            assert response.status_code == 404

    def test_explain_empty_request_rejected(self, client):
        response = client.post("/api/benchmark/explain", json={})
        assert response.status_code == 422

    def test_explain_unexpected_error_is_500(self, client):
        with patch("routers.benchmark.explain_by_id", side_effect=RuntimeError("boom")):
            response = client.post("/api/benchmark/explain", json={"player_id": 0})
            assert response.status_code == 500
            assert response.json()["detail"] == "Internal server error"


class TestModelCardEndpoint:
    """GET /api/meta/model-card reads real training artifacts from models/."""

    @pytest.fixture
    def card(self, client):
        response = client.get("/api/meta/model-card")
        assert response.status_code == 200
        return response.json()

    def test_top_level_shape(self, card):
        assert set(card) >= {
            "model_name", "framework", "target", "variants",
            "metrics", "calibration", "top_features", "coverage",
        }
        assert card["framework"] == "AutoGluon"
        assert card["variants"]["full"] is True

    def test_metrics_sane(self, card):
        m = card["metrics"]
        assert 0 < m["r2"] <= 1
        assert m["n_train"] > 0 and m["n_test"] > 0
        assert 0 < m["median_ape_pct"] < 100
        assert 0 < m["within_50_pct"] <= 100
        # Grouped split is a hard requirement — a random split would leak
        # players between train and test and overstate accuracy.
        assert "grouped" in m["split"]

    def test_calibration_percentiles_ordered(self, card):
        cal = card["calibration"]
        assert cal["residual_p10"] < cal["residual_p25"] < cal["residual_p50"] \
            < cal["residual_p75"] < cal["residual_p90"]
        assert cal["n_folds"] >= 2
        assert cal["n_samples"] > 0

    def test_feature_importance_sorted_and_labeled(self, card):
        features = card["top_features"]
        assert len(features) > 0
        importances = [f["importance"] for f in features]
        assert importances == sorted(importances, reverse=True)
        for f in features:
            assert f["feature"] and f["label"]

    def test_coverage_counts_consistent(self, card):
        cov = card["coverage"]
        assert 0 < cov["n_with_salary"] <= cov["n_rows"]
        assert 0 < cov["n_players"] <= cov["n_rows"]
        assert len(cov["countries"]) > 0
        assert cov["n_leagues"] > 0
        assert len(cov["seasons"]) > 0

    def test_second_call_served_from_cache(self, client):
        first = client.get("/api/meta/model-card").json()
        second = client.get("/api/meta/model-card").json()
        assert first == second
