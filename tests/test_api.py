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
