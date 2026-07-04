"""Tests for schemas.py — Pydantic validation models.

Verifies that the schemas correctly validate and reject data.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import (
    PipelineInputRow,
    EnrichedRow,
    ModelFeatureRow,
    PlayerPoolRow,
    BenchmarkRequest,
    BenchmarkResponse,
    ComparablePlayerResponse,
    PlayerSearchResult,
    PlayerDetail,
    SalaryStatus,
    Confidence,
)


# =============================================================================
# Pipeline Input
# =============================================================================

class TestPipelineInputRow:
    def test_valid_row(self):
        row = PipelineInputRow(
            transfermarkt_player_id=937958,
            player_name="Lamine Yamal",
            transfermarkt_team_id=131,
            team_name="FC Barcelona",
            competition_id="ES1",
            competition_country="Spain",
            season="2025-2026",
        )
        assert row.player_name == "Lamine Yamal"

    def test_strips_whitespace(self):
        row = PipelineInputRow(
            transfermarkt_player_id=1,
            player_name="  Lamine Yamal  ",
            transfermarkt_team_id=131,
            team_name="  FC Barcelona  ",
            competition_id="ES1",
            competition_country="Spain",
            season="2025-2026",
        )
        assert row.player_name == "Lamine Yamal"
        assert row.team_name == "FC Barcelona"

    def test_rejects_invalid_player_id(self):
        with pytest.raises(ValidationError, match="transfermarkt_player_id"):
            PipelineInputRow(
                transfermarkt_player_id=-1,
                player_name="Test",
                transfermarkt_team_id=1,
                team_name="Team",
                competition_id="GB1",
                competition_country="England",
                season="2025-2026",
            )

    def test_rejects_empty_player_name(self):
        with pytest.raises(ValidationError, match="player_name"):
            PipelineInputRow(
                transfermarkt_player_id=1,
                player_name="",
                transfermarkt_team_id=1,
                team_name="Team",
                competition_id="GB1",
                competition_country="England",
                season="2025-2026",
            )

    def test_rejects_invalid_season_format(self):
        with pytest.raises(ValidationError, match="season"):
            PipelineInputRow(
                transfermarkt_player_id=1,
                player_name="Test",
                transfermarkt_team_id=1,
                team_name="Team",
                competition_id="GB1",
                competition_country="England",
                season="2025/26",
            )

    def test_allows_optional_fields_as_none(self):
        row = PipelineInputRow(
            transfermarkt_player_id=1,
            player_name="Test",
            transfermarkt_team_id=1,
            team_name="Team",
            competition_id="GB1",
            competition_country="England",
            season="2025-2026",
            position=None,
            main_position=None,
            birth_date=None,
            nationality=None,
            market_value=None,
        )
        assert row.position is None


# =============================================================================
# Enrichment Output
# =============================================================================

class TestEnrichedRow:
    def test_valid_enriched_row(self):
        row = EnrichedRow(
            transfermarkt_player_id=1,
            player_name="Test",
            transfermarkt_team_id=1,
            team_name="Team",
            competition_id="GB1",
            competition_country="England",
            season="2025-2026",
            capology_url="https://www.capology.com/player/test-123",
            annual_fixed_eur=5_000_000,
        )
        assert row.annual_fixed_eur == 5_000_000

    def test_rejects_raw_capology_url(self):
        """The old /player/... bug should be caught."""
        with pytest.raises(ValidationError, match="capology_url"):
            EnrichedRow(
                transfermarkt_player_id=1,
                player_name="Test",
                transfermarkt_team_id=1,
                team_name="Team",
                competition_id="GB1",
                competition_country="England",
                season="2025-2026",
                capology_url="/player/test-123",
            )

    def test_rejects_negative_salary(self):
        with pytest.raises(ValidationError, match="annual_fixed_eur"):
            EnrichedRow(
                transfermarkt_player_id=1,
                player_name="Test",
                transfermarkt_team_id=1,
                team_name="Team",
                competition_id="GB1",
                competition_country="England",
                season="2025-2026",
                annual_fixed_eur=-100,
            )

    def test_allows_all_nulls_for_capology(self):
        """Player not in Capology — all enrichment fields null."""
        row = EnrichedRow(
            transfermarkt_player_id=1,
            player_name="Test",
            transfermarkt_team_id=1,
            team_name="Team",
            competition_id="GB1",
            competition_country="England",
            season="2025-2026",
        )
        assert row.capology_url is None
        assert row.annual_fixed_eur is None


# =============================================================================
# Feature Engineering
# =============================================================================

class TestModelFeatureRow:
    def test_valid_feature_row(self):
        row = ModelFeatureRow(
            main_position="Centre-Forward",
            competition_country="England",
            season_start_year=2025,
            age_months=300,
            contract_length_months=48,
            contract_months_remaining=36,
            has_market_value=1,
            log_market_value_current_eur=18.5,
        )
        assert row.age_months == 300

    def test_rejects_invalid_country(self):
        with pytest.raises(ValidationError, match="competition_country"):
            ModelFeatureRow(competition_country="Brazil")

    def test_rejects_age_too_low(self):
        with pytest.raises(ValidationError, match="age_months"):
            ModelFeatureRow(age_months=100)

    def test_rejects_age_too_high(self):
        with pytest.raises(ValidationError, match="age_months"):
            ModelFeatureRow(age_months=700)

    def test_rejects_binary_flag_out_of_range(self):
        with pytest.raises(ValidationError, match="has_market_value"):
            ModelFeatureRow(has_market_value=2)

    def test_allows_all_none(self):
        """All features can be None (model handles missing)."""
        row = ModelFeatureRow()
        assert row.main_position is None
        assert row.age_months is None


# =============================================================================
# API Contracts
# =============================================================================

class TestBenchmarkRequest:
    def test_valid_by_id(self):
        req = BenchmarkRequest(player_id=0)
        assert req.player_id == 0

    def test_valid_by_name(self):
        req = BenchmarkRequest(player_name="Lamine Yamal")
        assert req.player_name == "Lamine Yamal"

    def test_valid_manual(self):
        req = BenchmarkRequest(
            main_position="Centre-Forward",
            market_value_current_eur=50_000_000,
        )
        assert req.main_position == "Centre-Forward"

    def test_rejects_empty_body(self):
        with pytest.raises(ValidationError, match="Provide player_id"):
            BenchmarkRequest()

    def test_rejects_invalid_range_width(self):
        with pytest.raises(ValidationError, match="range_width"):
            BenchmarkRequest(player_id=0, range_width="invalid")

    def test_rejects_short_name(self):
        with pytest.raises(ValidationError, match="player_name"):
            BenchmarkRequest(player_name="X")


class TestBenchmarkResponse:
    def test_valid_response(self):
        resp = BenchmarkResponse(
            player_name="Test",
            expected_salary_low_eur=5_000_000,
            expected_salary_median_eur=8_000_000,
            expected_salary_high_eur=12_000_000,
            salary_status=SalaryStatus.FAIRLY_PAID,
            benchmark_confidence=Confidence.MEDIUM,
            benchmark_n_comparables=25,
            benchmark_avg_similarity=0.72,
            comparable_level_used=1,
            range_width_used="normal",
        )
        assert resp.salary_status == SalaryStatus.FAIRLY_PAID

    def test_rejects_invalid_status(self):
        with pytest.raises(ValidationError):
            BenchmarkResponse(
                player_name="Test",
                expected_salary_low_eur=5_000_000,
                expected_salary_median_eur=8_000_000,
                expected_salary_high_eur=12_000_000,
                salary_status="INVALID",
                benchmark_confidence="HIGH",
                benchmark_n_comparables=25,
                benchmark_avg_similarity=0.72,
                comparable_level_used=1,
                range_width_used="normal",
            )

    def test_rejects_similarity_above_1(self):
        with pytest.raises(ValidationError, match="benchmark_avg_similarity"):
            BenchmarkResponse(
                player_name="Test",
                expected_salary_low_eur=5_000_000,
                expected_salary_median_eur=8_000_000,
                expected_salary_high_eur=12_000_000,
                salary_status="FAIRLY_PAID",
                benchmark_confidence="HIGH",
                benchmark_n_comparables=25,
                benchmark_avg_similarity=1.5,
                comparable_level_used=1,
                range_width_used="normal",
            )


class TestComparablePlayerResponse:
    def test_valid(self):
        p = ComparablePlayerResponse(
            player_name="Test",
            similarity_score=0.85,
        )
        assert p.similarity_score == 0.85

    def test_rejects_similarity_below_0(self):
        with pytest.raises(ValidationError, match="similarity_score"):
            ComparablePlayerResponse(similarity_score=-0.1)


class TestPlayerSearchResult:
    def test_valid(self):
        r = PlayerSearchResult(id=0, player_name="Test")
        assert r.id == 0


class TestPlayerDetail:
    def test_allows_extra_fields(self):
        """PlayerDetail allows extra fields from the pool CSV."""
        d = PlayerDetail(id=0, player_name="Test", extra_field="value")
        assert d.player_name == "Test"
