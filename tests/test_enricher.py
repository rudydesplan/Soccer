"""Tests for capology_pipeline/enricher.py — idempotency and input validation.

Tests the orchestration logic: chunk/resume, input validation, error handling.
All network I/O is mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from capology_pipeline.config import Config
from capology_pipeline.enricher import PlayerEnricher, REQUIRED_COLUMNS


# --- _read_and_prepare ---

class TestReadAndPrepare:
    def test_renames_player_id_and_team_id(self, tmp_path):
        csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            "player_id": [1, 2],
            "player_name": ["A", "B"],
            "nationality": ["X", "Y"],
            "team_id": [10, 20],
            "team_name": ["T1", "T2"],
            "birth_date": ["2000-01-01", "2001-01-01"],
        })
        df.to_csv(csv, index=False)
        result = PlayerEnricher._read_and_prepare(str(csv))
        assert "transfermarkt_player_id" in result.columns
        assert "transfermarkt_team_id" in result.columns
        assert "player_id" not in result.columns
        assert "team_id" not in result.columns

    def test_accepts_already_renamed(self, tmp_path):
        csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            "transfermarkt_player_id": [1],
            "player_name": ["A"],
            "nationality": ["X"],
            "transfermarkt_team_id": [10],
            "team_name": ["T1"],
            "birth_date": ["2000-01-01"],
        })
        df.to_csv(csv, index=False)
        result = PlayerEnricher._read_and_prepare(str(csv))
        assert "transfermarkt_player_id" in result.columns

    def test_raises_on_missing_columns(self, tmp_path):
        csv = tmp_path / "input.csv"
        df = pd.DataFrame({"player_name": ["A"], "nationality": ["X"]})
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            PlayerEnricher._read_and_prepare(str(csv))

    def test_error_message_lists_missing_columns(self, tmp_path):
        csv = tmp_path / "input.csv"
        df = pd.DataFrame({"player_name": ["A"]})
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError) as exc_info:
            PlayerEnricher._read_and_prepare(str(csv))
        # Should mention specific missing columns
        msg = str(exc_info.value)
        assert "birth_date" in msg or "team_name" in msg


# --- Chunk/resume idempotency ---

class TestChunkResume:
    def _make_input(self, tmp_path, n_rows=10):
        csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            "transfermarkt_player_id": range(n_rows),
            "player_name": [f"Player {i}" for i in range(n_rows)],
            "nationality": ["Spain"] * n_rows,
            "transfermarkt_team_id": [100 + i for i in range(n_rows)],
            "team_name": [f"Team {i}" for i in range(n_rows)],
            "birth_date": ["2000-01-01"] * n_rows,
        })
        df.to_csv(csv, index=False)
        return csv

    def test_raises_on_invalid_chunk_size(self, tmp_path):
        csv = self._make_input(tmp_path)
        enricher = PlayerEnricher(Config())
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            enricher.run_chunked(str(csv), str(tmp_path / "out.csv"), chunk_size=0)

    def test_raises_on_negative_chunk_size(self, tmp_path):
        csv = self._make_input(tmp_path)
        enricher = PlayerEnricher(Config())
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            enricher.run_chunked(str(csv), str(tmp_path / "out.csv"), chunk_size=-5)

    def test_skips_existing_chunks(self, tmp_path):
        """If a chunk file already exists, it should NOT be overwritten."""
        csv = self._make_input(tmp_path, n_rows=6)
        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()

        # Pre-create chunk 1 with known content
        chunk1 = chunks_dir / "chunk_00001.csv"
        pre_df = pd.DataFrame({
            "transfermarkt_player_id": [0, 1, 2],
            "player_name": ["EXISTING_0", "EXISTING_1", "EXISTING_2"],
            "nationality": ["Spain"] * 3,
            "transfermarkt_team_id": [100, 101, 102],
            "team_name": ["Team 0", "Team 1", "Team 2"],
            "birth_date": ["2000-01-01"] * 3,
        })
        pre_df.to_csv(chunk1, index=False)

        # Mock _enrich_frame to just return input unchanged
        with patch.object(PlayerEnricher, '_enrich_frame', side_effect=lambda df: df):
            enricher = PlayerEnricher(Config())
            enricher.run_chunked(
                str(csv), str(tmp_path / "out.csv"),
                chunk_size=3, chunks_dir=str(chunks_dir)
            )

        # Chunk 1 should still have the pre-existing content
        result_chunk1 = pd.read_csv(chunk1)
        assert result_chunk1["player_name"].iloc[0] == "EXISTING_0"

    def test_creates_all_chunks(self, tmp_path):
        csv = self._make_input(tmp_path, n_rows=7)
        chunks_dir = tmp_path / "chunks"

        with patch.object(PlayerEnricher, '_enrich_frame', side_effect=lambda df: df):
            enricher = PlayerEnricher(Config())
            enricher.run_chunked(
                str(csv), str(tmp_path / "out.csv"),
                chunk_size=3, chunks_dir=str(chunks_dir)
            )

        # Should create 3 chunks: 3+3+1
        assert (chunks_dir / "chunk_00001.csv").exists()
        assert (chunks_dir / "chunk_00002.csv").exists()
        assert (chunks_dir / "chunk_00003.csv").exists()

    def test_merged_output_has_correct_row_count(self, tmp_path):
        csv = self._make_input(tmp_path, n_rows=7)
        chunks_dir = tmp_path / "chunks"

        with patch.object(PlayerEnricher, '_enrich_frame', side_effect=lambda df: df):
            enricher = PlayerEnricher(Config())
            result = enricher.run_chunked(
                str(csv), str(tmp_path / "out.csv"),
                chunk_size=3, chunks_dir=str(chunks_dir)
            )

        assert len(result) == 7
        output = pd.read_csv(tmp_path / "out.csv")
        assert len(output) == 7


# --- Capology DOB audit ---

class TestCapologyDobAudit:
    def _frame(self):
        return pd.DataFrame({
            "player_name": ["A", "B", "C"],
            "team_name": ["T1", "T2", "T3"],
            "birth_date": ["2000-01-01", "1995-05-05", "1990-09-09"],
            "capology_url": ["https://cap/a", "https://cap/b", None],
            "annual_fixed_eur": [1_000_000.0, 2_000_000.0, None],
            "status": ["active", "active", None],
        })

    def _audit(self, df, scraped_by_url):
        enricher = PlayerEnricher(Config())
        mock_cap = MagicMock()
        mock_cap.matcher._scrape.side_effect = lambda url: scraped_by_url[url]
        with patch.object(PlayerEnricher, "capology", new_callable=PropertyMock,
                          return_value=mock_cap):
            return enricher.audit_capology_dob(df)

    def test_nulls_out_dob_mismatch(self):
        result = self._audit(self._frame(), {
            "https://cap/a": {"birth_date": "2000-01-01"},   # matches
            "https://cap/b": {"birth_date": "1994-04-04"},   # wrong player
        })
        # Matching row untouched
        assert result.loc[0, "annual_fixed_eur"] == 1_000_000.0
        assert result.loc[0, "capology_url"] == "https://cap/a"
        # Mismatching row: all enrichment nulled
        assert pd.isna(result.loc[1, "annual_fixed_eur"])
        assert pd.isna(result.loc[1, "capology_url"])
        assert pd.isna(result.loc[1, "status"])

    def test_keeps_row_when_capology_dob_missing(self):
        result = self._audit(self._frame(), {
            "https://cap/a": {"birth_date": None},
            "https://cap/b": {"birth_date": "1995-05-05"},
        })
        assert result.loc[0, "annual_fixed_eur"] == 1_000_000.0
        assert result.loc[1, "annual_fixed_eur"] == 2_000_000.0

    def test_keeps_row_when_scrape_fails(self):
        enricher = PlayerEnricher(Config())
        mock_cap = MagicMock()
        mock_cap.matcher._scrape.side_effect = RuntimeError("network down")
        with patch.object(PlayerEnricher, "capology", new_callable=PropertyMock,
                          return_value=mock_cap):
            result = enricher.audit_capology_dob(self._frame())
        assert result.loc[0, "annual_fixed_eur"] == 1_000_000.0

    def test_noop_without_capology_column(self):
        df = pd.DataFrame({"player_name": ["A"], "birth_date": ["2000-01-01"]})
        enricher = PlayerEnricher(Config())
        # Must not touch the (expensive) capology property at all
        with patch.object(PlayerEnricher, "capology", new_callable=PropertyMock) as prop:
            result = enricher.audit_capology_dob(df)
            prop.assert_not_called()
        assert len(result) == 1


# --- Enrichment error handling ---

class TestEnrichmentErrorHandling:
    def test_enrich_frame_preserves_row_count(self, tmp_path):
        """Even if enrichment adds columns, row count must not change."""
        config = Config()
        enricher = PlayerEnricher(config)

        input_df = pd.DataFrame({
            "transfermarkt_player_id": [1, 2, 3],
            "player_name": ["A", "B", "C"],
            "nationality": ["X", "Y", "Z"],
            "transfermarkt_team_id": [10, 20, 30],
            "team_name": ["T1", "T2", "T3"],
            "birth_date": ["2000-01-01"] * 3,
        })

        # Mock both rails to return input with extra columns
        with patch.object(enricher, '_add_capology', side_effect=lambda df: df.assign(capology_url=None)):
            with patch.object(enricher, '_add_market_value', side_effect=lambda df: df.assign(market_value_current_eur=None)):
                result = enricher._enrich_frame(input_df)

        assert len(result) == len(input_df)

    def test_left_join_deduplicates_enrichment(self):
        """If enrichment has duplicate keys, only one row per key is joined."""
        players_df = pd.DataFrame({
            "player_name": ["Alice", "Alice"],
            "team_name": ["Team A", "Team A"],
            "birth_date": ["2000-01-01", "2000-01-01"],
            "competition_id": ["GB1", "GB1"],
        })
        enrich_df = pd.DataFrame({
            "player_name": ["Alice", "Alice"],
            "team_name": ["Team A", "Team A"],
            "birth_date": ["2000-01-01", "2000-01-01"],
            "capology_url": ["url1", "url2"],
        })
        result = PlayerEnricher._left_join(players_df, enrich_df)
        # Should still have 2 rows (one per input row)
        assert len(result) == 2
        # Both should get the same enrichment (first deduplicated match)
        assert result["capology_url"].iloc[0] == result["capology_url"].iloc[1]
