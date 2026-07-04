"""Tests for capology_pipeline/cache.py — DiskCache idempotency."""

from __future__ import annotations

import pytest

from capology_pipeline.cache import DiskCache


class TestDiskCacheInit:
    def test_creates_subdirs(self, tmp_path):
        cache = DiskCache(tmp_path / "test_cache")
        assert (tmp_path / "test_cache" / "capology").is_dir()
        assert (tmp_path / "test_cache" / "tm").is_dir()

    def test_idempotent_init(self, tmp_path):
        """Creating cache twice on same dir does not raise."""
        DiskCache(tmp_path / "c")
        DiskCache(tmp_path / "c")
        assert (tmp_path / "c" / "capology").is_dir()


class TestSafeName:
    def test_strips_special_chars(self):
        assert DiskCache.safe_name("/player/foo-bar-123") == "player_foo-bar-123"

    def test_preserves_alphanumeric(self):
        assert DiskCache.safe_name("abc123") == "abc123"

    def test_collapses_multiple_specials(self):
        assert DiskCache.safe_name("a///b///c") == "a_b_c"

    def test_strips_leading_trailing(self):
        assert DiskCache.safe_name("///hello///") == "hello"

    def test_dots_and_hyphens_preserved(self):
        assert DiskCache.safe_name("file.json") == "file.json"
        assert DiskCache.safe_name("a-b") == "a-b"


class TestGetSet:
    def test_get_returns_none_when_not_cached(self, tmp_path):
        cache = DiskCache(tmp_path / "c")
        assert cache.get("capology", "nonexistent.html") is None

    def test_set_then_get_returns_value(self, tmp_path):
        cache = DiskCache(tmp_path / "c")
        cache.set("capology", "page.html", "<html>hello</html>")
        assert cache.get("capology", "page.html") == "<html>hello</html>"

    def test_set_overwrites_existing(self, tmp_path):
        cache = DiskCache(tmp_path / "c")
        cache.set("tm", "data.json", '{"v":1}')
        cache.set("tm", "data.json", '{"v":2}')
        assert cache.get("tm", "data.json") == '{"v":2}'

    def test_get_returns_none_on_corrupt_file(self, tmp_path):
        """If the file exists but can't be read, return None."""
        cache = DiskCache(tmp_path / "c")
        # Create a directory where a file is expected (unreadable as text)
        path = tmp_path / "c" / "capology" / "bad.html"
        path.mkdir(parents=True, exist_ok=True)
        # get should not crash
        result = cache.get("capology", "bad.html")
        # It will either return None or raise — we want graceful None
        assert result is None

    def test_atomic_write_no_partial(self, tmp_path):
        """After set(), the file exists fully (no .tmp leftover)."""
        cache = DiskCache(tmp_path / "c")
        cache.set("capology", "test.html", "full content")
        # The .tmp file should not exist
        tmp_file = tmp_path / "c" / "capology" / "test.html.tmp"
        assert not tmp_file.exists()
        # The real file should exist
        real_file = tmp_path / "c" / "capology" / "test.html"
        assert real_file.exists()
        assert real_file.read_text() == "full content"

    def test_set_handles_write_failure_gracefully(self, tmp_path):
        """If write fails (e.g. read-only dir), set() prints warning but doesn't crash."""
        import os
        cache = DiskCache(tmp_path / "c")
        # Make the subdir read-only
        capology_dir = tmp_path / "c" / "capology"
        os.chmod(capology_dir, 0o444)
        try:
            # Should not raise
            cache.set("capology", "fail.html", "data")
        finally:
            os.chmod(capology_dir, 0o755)

    def test_different_subdirs_are_independent(self, tmp_path):
        cache = DiskCache(tmp_path / "c")
        cache.set("capology", "key.txt", "capology_data")
        cache.set("tm", "key.txt", "tm_data")
        assert cache.get("capology", "key.txt") == "capology_data"
        assert cache.get("tm", "key.txt") == "tm_data"
