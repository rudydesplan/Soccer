"""Shared fixtures for the Soccer test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_prediction_log(tmp_path, monkeypatch):
    """Redirect benchmark prediction logging to a per-test temp file.

    Without this, every benchmark_player() call made by the test suite appends
    synthetic entries to logs/benchmark_predictions.jsonl — the very file the
    drift monitoring reads — polluting production monitoring data.
    """
    from salary_benchmark import monitoring

    test_logger = monitoring.PredictionLogger(log_path=tmp_path / "predictions.jsonl")
    monkeypatch.setattr(monitoring, "_LOGGER", test_logger)
    # benchmark.py imported get_logger by name; patch its reference too.
    from salary_benchmark import benchmark as benchmark_module

    monkeypatch.setattr(benchmark_module, "get_logger", lambda: test_logger)
    yield


@pytest.fixture
def sample_pool_df():
    """Small player pool DataFrame with all columns needed by salary_benchmark."""
    data = {
        "id": list(range(10)),
        "player_name": [
            "Lionel Messi", "Kylian Mbappe", "Erling Haaland",
            "Vinicius Junior", "Jude Bellingham", "Bukayo Saka",
            "Pedri", "Phil Foden", "Florian Wirtz", "Jamal Musiala",
        ],
        "main_position": [
            "Right Winger", "Centre-Forward", "Centre-Forward",
            "Left Winger", "Attacking Midfield", "Right Winger",
            "Central Midfield", "Attacking Midfield", "Attacking Midfield", "Attacking Midfield",
        ],
        "competition_id": ["MLS1", "ES1", "GB1", "ES1", "ES1", "GB1", "ES1", "GB1", "L1", "L1"],
        "competition_country": [
            "United States", "Spain", "England",
            "Spain", "Spain", "England",
            "Spain", "England", "Germany", "Germany",
        ],
        "nationality": [
            "Argentina", "France", "Norway",
            "Brazil", "England", "England",
            "Spain", "England", "Germany", "Germany",
        ],
        "age_months": [440, 306, 290, 295, 260, 272, 262, 290, 264, 262],
        "market_value_current_eur": [
            20_000_000, 180_000_000, 170_000_000,
            150_000_000, 150_000_000, 140_000_000,
            100_000_000, 130_000_000, 130_000_000, 110_000_000,
        ],
        "log_market_value_current_eur": [
            np.log1p(20_000_000), np.log1p(180_000_000), np.log1p(170_000_000),
            np.log1p(150_000_000), np.log1p(150_000_000), np.log1p(140_000_000),
            np.log1p(100_000_000), np.log1p(130_000_000), np.log1p(130_000_000), np.log1p(110_000_000),
        ],
        "annual_fixed_eur": [
            15_000_000, 25_000_000, 20_000_000,
            10_000_000, 12_000_000, 8_000_000,
            5_000_000, 10_000_000, 6_000_000, 5_000_000,
        ],
        "contract_length_months": [24, 60, 60, 48, 72, 48, 48, 60, 60, 60],
        "contract_months_remaining": [12, 48, 50, 36, 60, 40, 36, 48, 48, 48],
        "contract_recency_months": [12, 12, 10, 12, 12, 8, 12, 12, 12, 12],
        "has_contract_dates": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "has_market_value": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "has_release_clause": [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        "log_release_clause_eur": [0, np.log1p(1_000_000_000), 0, np.log1p(1_000_000_000), np.log1p(1_000_000_000), 0, np.log1p(800_000_000), 0, 0, 0],
        "status": ["active"] * 10,
        "season_start_year": [2024] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def target_player_dict():
    """A sample target player dict for benchmark/comparable tests."""
    return {
        "id": 99,
        "player_name": "Test Player",
        "main_position": "Centre-Forward",
        "competition_id": "ES1",
        "competition_country": "Spain",
        "nationality": "Brazil",
        "age_months": 300,
        "market_value_current_eur": 150_000_000,
        "log_market_value_current_eur": np.log1p(150_000_000),
        "has_market_value": 1,
        "annual_fixed_eur": 20_000_000,
        "contract_length_months": 60,
        "contract_months_remaining": 48,
        "status": "active",
        "season_start_year": 2024,
    }


@pytest.fixture
def mock_calibration():
    """Mock calibration dict matching the shape of models/calibration.json."""
    return {
        "n_samples": 500,
        "residual_p10": -0.8,
        "residual_p25": -0.4,
        "residual_p50": 0.0,
        "residual_p75": 0.4,
        "residual_p90": 0.8,
        "residual_mean": 0.0,
        "residual_std": 0.5,
        "rmse": 0.5,
    }


@pytest.fixture
def sample_input_csv(tmp_path):
    """Write a small enriched CSV to tmp_path and return its path."""
    csv_path = tmp_path / "data_full.csv"
    data = {
        "player_id": [1, 2, 3],
        "player_name": ["Alice", "Bob", "Charlie"],
        "team_id": [10, 20, 30],
        "season": ["2024-2025", "2023-2024", "2024-2025"],
        "birth_date": ["1995-03-15", "1998-07-20", "2000-01-01"],
        "signed_date": ["2022-07-01", "2023-01-15", "2024-06-01"],
        "expiration_date": ["2026-06-30", "2027-06-30", "2028-06-30"],
        "gross_contract": ["4-yrs/€54.0M + 13.5M", "3-yrs/€10.0M", "5-yrs/€20.0M"],
        "main_position": ["Centre-Forward", "Left Winger", "Goalkeeper"],
        "nationality": ["France", "Brazil", "Germany"],
        "competition_id": ["GB1", "ES1", "L1"],
        "competition_country": ["England", "Spain", "Germany"],
        "market_value_current_eur": [50000000, 30000000, 10000000],
        "release_clause_eur": [100000000, np.nan, 20000000],
        "annual_fixed_eur": [5000000, 3000000, 1000000],
        "annual_bonus_eur": [1000000, 500000, 200000],
        "annual_total_eur": [6000000, 3500000, 1200000],
        "salary_currency": ["EUR", "EUR", "EUR"],
        "league": ["Premier League", "La Liga", "Bundesliga"],
        "capology_url": ["http://cap.com/1", "http://cap.com/2", "http://cap.com/3"],
        "competition_name": ["Premier League", "La Liga", "Bundesliga"],
        "team_name": ["Team A", "Team B", "Team C"],
        "position": ["CF", "LW", "GK"],
        "status": ["active", "active", "active"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path
