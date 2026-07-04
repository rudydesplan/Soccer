"""Tests for capology_pipeline/http_client.py — error handling and retry.

These tests mock niquests to verify behavior when sources don't respond
or return errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from capology_pipeline.cache import DiskCache
from capology_pipeline.config import Config
from capology_pipeline.http_client import HttpClient, _retry


class TestRetryConfig:
    def test_total_retries(self):
        r = _retry()
        assert r.total == 4

    def test_backoff_factor(self):
        r = _retry()
        assert r.backoff_factor == 1.5

    def test_status_forcelist(self):
        r = _retry()
        assert 429 in r.status_forcelist
        assert 500 in r.status_forcelist
        assert 502 in r.status_forcelist
        assert 503 in r.status_forcelist
        assert 504 in r.status_forcelist

    def test_only_get_allowed(self):
        r = _retry()
        assert "GET" in r.allowed_methods
        assert "POST" not in r.allowed_methods

    def test_respects_retry_after(self):
        r = _retry()
        assert r.respect_retry_after_header is True


class TestCapologyRequests:
    @pytest.fixture
    def http_client(self, tmp_path):
        config = Config()
        cache = DiskCache(tmp_path / "cache")
        with patch("capology_pipeline.http_client.niquests") as mock_niquests:
            mock_session = MagicMock()
            mock_niquests.Session.return_value = mock_session
            client = HttpClient(config, cache)
            client.capology = mock_session
            client.tm = MagicMock()
            yield client, mock_session, cache

    def test_returns_cached_without_network(self, http_client):
        client, mock_session, cache = http_client
        # Pre-populate cache
        cache.set("capology", "page.html", "<html>cached</html>")
        result = client.get_capology_text("https://example.com/page", "page.html")
        assert result == "<html>cached</html>"
        # No network call made
        mock_session.get.assert_not_called()

    def test_fetches_and_caches_on_miss(self, http_client):
        client, mock_session, cache = http_client
        mock_resp = MagicMock()
        mock_resp.text = "<html>fresh</html>"
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        result = client.get_capology_text("https://example.com/new", "new.html")
        assert result == "<html>fresh</html>"
        # Verify it was cached
        assert cache.get("capology", "new.html") == "<html>fresh</html>"

    def test_raises_on_network_failure(self, http_client):
        client, mock_session, _ = http_client
        mock_session.get.side_effect = ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="Capology request failed"):
            client.get_capology_text("https://example.com/fail", "fail.html")

    def test_raises_on_http_error(self, http_client):
        client, mock_session, _ = http_client
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        mock_session.get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Capology request failed"):
            client.get_capology_text("https://example.com/500", "500.html")

    def test_raises_on_timeout(self, http_client):
        client, mock_session, _ = http_client
        mock_session.get.side_effect = TimeoutError("Read timed out")

        with pytest.raises(RuntimeError, match="Capology request failed"):
            client.get_capology_text("https://example.com/slow", "slow.html")


class TestTransfermarktRequests:
    @pytest.fixture
    def http_client(self, tmp_path):
        config = Config()
        config.tm_port = 9999
        cache = DiskCache(tmp_path / "cache")
        with patch("capology_pipeline.http_client.niquests") as mock_niquests:
            mock_session = MagicMock()
            mock_niquests.Session.return_value = mock_session
            client = HttpClient(config, cache)
            client.tm = mock_session
            client.capology = MagicMock()
            yield client, mock_session, cache

    def test_returns_cached_json(self, http_client):
        client, mock_session, cache = http_client
        cache.set("tm", "123__market_value.json", '{"marketValueHistory": []}')
        result = client.get_tm_json("/players/123/market_value", "123__market_value.json")
        assert result == {"marketValueHistory": []}
        mock_session.get.assert_not_called()

    def test_returns_empty_dict_on_failure(self, http_client):
        """Transfermarkt failures are graceful — return {} instead of raising."""
        client, mock_session, _ = http_client
        mock_session.get.side_effect = ConnectionError("refused")
        result = client.get_tm_json("/players/999/market_value", "999__market_value.json")
        assert result == {}

    def test_refetches_on_corrupt_cache(self, http_client):
        """If cached JSON is invalid, refetch from network."""
        client, mock_session, cache = http_client
        cache.set("tm", "bad.json", "not valid json {{{")
        mock_resp = MagicMock()
        mock_resp.text = '{"fresh": true}'
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        result = client.get_tm_json("/players/1/market_value", "bad.json")
        assert result == {"fresh": True}
        mock_session.get.assert_called_once()

    def test_caches_successful_response(self, http_client):
        client, mock_session, cache = http_client
        mock_resp = MagicMock()
        mock_resp.text = '{"data": 42}'
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        client.get_tm_json("/test", "test.json")
        assert cache.get("tm", "test.json") == '{"data": 42}'


class TestHttpClientClose:
    def test_close_does_not_raise(self, tmp_path):
        config = Config()
        cache = DiskCache(tmp_path / "cache")
        with patch("capology_pipeline.http_client.niquests") as mock_niquests:
            mock_session = MagicMock()
            mock_niquests.Session.return_value = mock_session
            client = HttpClient(config, cache)
            # Should not raise even if close() fails
            mock_session.close.side_effect = Exception("close error")
            client.close()
