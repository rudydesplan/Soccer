"""Tests for capology_pipeline/transfermarkt.py — TransfermarktClient logic.

Tests the market value resolution logic without network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from capology_pipeline.transfermarkt import TransfermarktClient


@pytest.fixture
def tm_client():
    mock_http = MagicMock()
    return TransfermarktClient(mock_http)


class TestHistoryMap:
    def test_parses_market_value_history(self, tm_client):
        tm_client.http.get_tm_json.return_value = {
            "marketValueHistory": [
                {"clubId": "131", "marketValue": 50_000_000},
                {"clubId": "131", "marketValue": 80_000_000},  # later entry overwrites
                {"clubId": "27", "marketValue": 60_000_000},
            ]
        }
        result = tm_client.history_map("12345")
        assert result == {"131": 80_000_000, "27": 60_000_000}

    def test_returns_empty_on_missing_id(self, tm_client):
        result = tm_client.history_map(None)
        assert result == {}

    def test_returns_empty_on_nan_id(self, tm_client):
        result = tm_client.history_map(float("nan"))
        assert result == {}

    def test_returns_empty_on_empty_response(self, tm_client):
        tm_client.http.get_tm_json.return_value = {}
        result = tm_client.history_map("123")
        assert result == {}

    def test_returns_empty_on_null_history(self, tm_client):
        tm_client.http.get_tm_json.return_value = {"marketValueHistory": None}
        result = tm_client.history_map("123")
        assert result == {}

    def test_skips_entries_without_club_id(self, tm_client):
        tm_client.http.get_tm_json.return_value = {
            "marketValueHistory": [
                {"clubId": None, "marketValue": 50_000_000},
                {"clubId": "131", "marketValue": 80_000_000},
            ]
        }
        result = tm_client.history_map("123")
        assert result == {"131": 80_000_000}


class TestTransfersMap:
    def test_parses_transfers(self, tm_client):
        tm_client.http.get_tm_json.return_value = {
            "transfers": [
                {"date": "2022-07-01", "clubTo": {"id": "131"}, "marketValue": 30_000_000},
                {"date": "2023-07-01", "clubTo": {"id": "27"}, "marketValue": 50_000_000},
            ]
        }
        result = tm_client.transfers_map("12345")
        assert result == {"131": 30_000_000, "27": 50_000_000}

    def test_returns_empty_on_missing_id(self, tm_client):
        result = tm_client.transfers_map(None)
        assert result == {}

    def test_returns_empty_on_empty_response(self, tm_client):
        tm_client.http.get_tm_json.return_value = {}
        result = tm_client.transfers_map("123")
        assert result == {}

    def test_later_transfer_overwrites_earlier(self, tm_client):
        tm_client.http.get_tm_json.return_value = {
            "transfers": [
                {"date": "2020-01-01", "clubTo": {"id": "131"}, "marketValue": 10_000_000},
                {"date": "2023-01-01", "clubTo": {"id": "131"}, "marketValue": 50_000_000},
            ]
        }
        result = tm_client.transfers_map("123")
        assert result["131"] == 50_000_000

    def test_skips_entries_without_market_value(self, tm_client):
        tm_client.http.get_tm_json.return_value = {
            "transfers": [
                {"date": "2022-01-01", "clubTo": {"id": "131"}, "marketValue": None},
                {"date": "2023-01-01", "clubTo": {"id": "27"}, "marketValue": 40_000_000},
            ]
        }
        result = tm_client.transfers_map("123")
        assert "131" not in result
        assert result["27"] == 40_000_000


class TestResolve:
    def test_prefers_history_over_transfers(self):
        history = {"131": 80_000_000}
        transfers = {"131": 30_000_000}
        assert TransfermarktClient.resolve("131", history, transfers) == 80_000_000

    def test_falls_back_to_transfers(self):
        history = {"27": 60_000_000}
        transfers = {"131": 30_000_000}
        assert TransfermarktClient.resolve("131", history, transfers) == 30_000_000

    def test_returns_none_when_not_found(self):
        history = {"27": 60_000_000}
        transfers = {"27": 30_000_000}
        assert TransfermarktClient.resolve("131", history, transfers) is None

    def test_returns_none_for_empty_team_id(self):
        assert TransfermarktClient.resolve(None, {"131": 50}, {"131": 30}) is None

    def test_returns_none_for_nan_team_id(self):
        assert TransfermarktClient.resolve(float("nan"), {"131": 50}, {}) is None

    def test_handles_float_team_id(self):
        """Team IDs from CSV arrive as floats like 131.0."""
        history = {"131": 80_000_000}
        assert TransfermarktClient.resolve("131.0", history, {}) == 80_000_000
