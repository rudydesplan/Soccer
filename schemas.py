"""Pydantic validation schemas for the Soccer Salary Benchmark project.

Four schema categories:
1. Pipeline input — validates raw CSV rows before enrichment
2. Enrichment output — validates enriched rows after pipeline
3. Feature engineering — validates model-ready feature DataFrames
4. API contracts — typed request/response models for FastAPI
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# 1. Pipeline Input Schemas
# =============================================================================

class PipelineInputRow(BaseModel):
    """Validates a single row from the input CSV before enrichment.

    Corresponds to data (1).csv columns after ID renaming.
    """
    transfermarkt_player_id: int = Field(..., gt=0, description="Transfermarkt player ID")
    player_name: str = Field(..., min_length=1, description="Player display name")
    position: str | None = Field(None, description="Position group (ATT, MID, DEF, GK)")
    main_position: str | None = Field(None, description="Detailed position")
    birth_date: str | None = Field(None, description="Birth date (ISO or various formats)")
    nationality: str | None = Field(None, description="Player nationality")
    transfermarkt_team_id: int = Field(..., gt=0, description="Transfermarkt team ID")
    team_name: str = Field(..., min_length=1, description="Team display name")
    competition_id: str = Field(..., min_length=1, description="League ID (GB1, ES1, etc.)")
    competition_name: str | None = Field(None, description="League display name")
    competition_country: str = Field(..., min_length=1, description="Country of the league")
    season: str = Field(..., pattern=r"^\d{4}-\d{4}$", description="Season (e.g. 2025-2026)")
    market_value: float | None = Field(None, ge=0, description="Original market value (may be stale)")

    @field_validator("player_name", "team_name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# =============================================================================
# 2. Enrichment Output Schemas
# =============================================================================

class EnrichedRow(BaseModel):
    """Validates a single row after Capology + Transfermarkt enrichment.

    Corresponds to data_full.csv columns.
    """
    transfermarkt_player_id: int = Field(..., gt=0)
    player_name: str = Field(..., min_length=1)
    position: str | None = None
    main_position: str | None = None
    birth_date: str | None = None
    nationality: str | None = None
    transfermarkt_team_id: int = Field(..., gt=0)
    team_name: str = Field(..., min_length=1)
    competition_id: str = Field(..., min_length=1)
    competition_name: str | None = None
    competition_country: str = Field(..., min_length=1)
    season: str = Field(..., pattern=r"^\d{4}-\d{4}$")

    # Capology enrichment (all nullable — player may not be in Capology)
    capology_url: str | None = Field(None, description="Full Capology player URL")
    status: str | None = Field(None, description="Contract status (active, loan, free agent)")
    league: str | None = None
    annual_fixed_eur: float | None = Field(None, ge=0, description="Annual gross salary EUR")
    annual_bonus_eur: float | None = Field(None, ge=0)
    annual_total_eur: float | None = Field(None, ge=0)
    salary_currency: str | None = None
    gross_contract: str | None = None
    signed_date: str | None = None
    expiration_date: str | None = None
    release_clause_eur: float | None = Field(None, ge=0)

    # Transfermarkt enrichment
    market_value_current_eur: float | None = Field(None, ge=0, description="Current market value EUR")

    @field_validator("capology_url")
    @classmethod
    def validate_capology_url(cls, v: str | None) -> str | None:
        if v is not None and not v.startswith("https://"):
            raise ValueError(f"capology_url must start with https://, got: {v[:50]}")
        return v


# =============================================================================
# 3. Feature Engineering Schemas
# =============================================================================

VALID_POSITIONS = {
    "Centre-Forward", "Left Winger", "Right Winger", "Second Striker",
    "Attacking Midfield", "Central Midfield", "Defensive Midfield",
    "Left Midfield", "Right Midfield",
    "Centre-Back", "Left-Back", "Right-Back",
    "Goalkeeper",
    "Midfielder", "Striker", "Defender",
}

VALID_COUNTRIES = {"Italy", "Spain", "Germany", "England", "France"}


class ModelFeatureRow(BaseModel):
    """Validates a single row of model-ready features (data_test.csv).

    All fields are nullable because the model handles missing values.
    """
    main_position: str | None = None
    nationality: str | None = None
    competition_id: str | None = None
    competition_country: str | None = None
    status: str | None = None
    season_start_year: int | None = Field(None, ge=2020, le=2030)
    age_months: float | None = Field(None, ge=150, le=600)
    contract_length_months: float | None = Field(None, ge=0, le=120)
    contract_months_remaining: float | None = Field(None, ge=-36, le=120)
    contract_recency_months: float | None = Field(None, ge=-24, le=120)
    has_contract_dates: int | None = Field(None, ge=0, le=1)
    has_market_value: int | None = Field(None, ge=0, le=1)
    log_market_value_current_eur: float | None = Field(None, ge=0)
    has_release_clause: int | None = Field(None, ge=0, le=1)
    log_release_clause_eur: float | None = Field(None, ge=0)
    log_annual_fixed_eur: float | None = Field(None, ge=0)

    @field_validator("competition_country")
    @classmethod
    def validate_country(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_COUNTRIES:
            raise ValueError(f"Unexpected competition_country: {v}")
        return v


class PlayerPoolRow(ModelFeatureRow):
    """Extends ModelFeatureRow with display/context columns for the benchmark pool."""
    transfermarkt_player_id: int | None = None
    player_name: str | None = None
    team_name: str | None = None
    market_value_current_eur: float | None = Field(None, ge=0)
    annual_fixed_eur: float | None = Field(None, ge=0)


# =============================================================================
# 4. API Contract Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Response body for GET /api/health."""
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    """Standard error body (FastAPI HTTPException shape)."""
    detail: str = Field(..., description="Human-readable error message")


class SalaryStatus(str, Enum):
    OVERPAID = "OVERPAID"
    UNDERPAID = "UNDERPAID"
    FAIRLY_PAID = "FAIRLY_PAID"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BenchmarkRequest(BaseModel):
    """Request body for POST /api/benchmark."""
    player_name: str | None = Field(None, min_length=2, description="Player name lookup")
    player_id: int | None = Field(None, ge=0, description="Player row ID (preferred)")
    # Manual override fields
    main_position: str | None = None
    competition_id: str | None = None
    competition_country: str | None = None
    age_months: float | None = Field(None, ge=150, le=600)
    market_value_current_eur: float | None = Field(None, ge=0)
    annual_fixed_eur: float | None = Field(None, ge=0)
    # Options
    range_width: Literal["normal", "wide"] = "normal"
    full_comparables: bool = Field(False, description="Return all comparables instead of top 10")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"player_id": 1234, "range_width": "normal"},
                {"player_name": "Erling Haaland", "range_width": "wide"},
                {
                    "main_position": "Centre-Forward",
                    "competition_id": "GB1",
                    "competition_country": "England",
                    "age_months": 300,
                    "market_value_current_eur": 50_000_000,
                    "annual_fixed_eur": 10_000_000,
                },
            ]
        }
    )

    @model_validator(mode="after")
    def at_least_one_identifier(self):
        if self.player_id is None and self.player_name is None and self.main_position is None:
            raise ValueError("Provide player_id, player_name, or manual fields (main_position + market_value)")
        return self


class ComparablePlayerResponse(BaseModel):
    """A single comparable player in the benchmark response."""
    id: int | None = None
    player_name: str | None = None
    main_position: str | None = None
    competition_id: str | None = None
    competition_country: str | None = None
    age_months: float | None = None
    market_value_current_eur: float | None = None
    annual_fixed_eur: float | None = None
    similarity_score: float = Field(..., ge=0, le=1)


class BenchmarkResponse(BaseModel):
    """Response body for POST /api/benchmark."""
    player_id: int | None = None
    player_name: str
    main_position: str | None = None
    competition_id: str | None = None
    competition_country: str | None = None
    age_months: float | None = None
    market_value_current_eur: float | None = None

    # Salary range
    expected_salary_low_eur: int = Field(..., ge=0)
    expected_salary_median_eur: int = Field(..., ge=0)
    expected_salary_high_eur: int = Field(..., ge=0)

    # Actual salary
    actual_salary_eur: int | None = Field(None, ge=0)
    salary_percentile: int | None = Field(None, ge=0, le=100)
    salary_status: SalaryStatus

    # Confidence
    benchmark_confidence: Confidence
    benchmark_n_comparables: int = Field(..., ge=0)
    benchmark_n_comparables_with_salary: int | None = Field(None, ge=0)
    benchmark_avg_similarity: float = Field(..., ge=0, le=1)
    comparable_level_used: int = Field(..., ge=1, le=3)
    range_width_used: Literal["normal", "wide"]
    model_used: Literal["full", "no_mv", "no_mv_no_pos", "no_mv_no_age"] = "full"
    benchmark_warning: str | None = None

    # Comparables
    comparable_players: list[ComparablePlayerResponse] = []


class CompetitionOption(BaseModel):
    """A competition option for the manual benchmark form."""
    id: str
    name: str
    country: str | None = None


class BenchmarkOptions(BaseModel):
    """Distinct positions and competitions available in the pool."""
    positions: list[str]
    competitions: list[CompetitionOption]


class PlayerSearchResult(BaseModel):
    """A single player in search results."""
    id: int
    player_name: str
    main_position: str | None = None
    team_name: str | None = None
    competition_id: str | None = None
    competition_country: str | None = None
    nationality: str | None = None
    age_months: float | None = None
    market_value_current_eur: float | None = None
    annual_fixed_eur: float | None = None


class PlayerDetail(BaseModel):
    """Full player detail response."""
    id: int
    transfermarkt_player_id: int | None = None
    player_name: str
    position: str | None = None
    main_position: str | None = None
    birth_date: str | None = None
    nationality: str | None = None
    transfermarkt_team_id: int | None = None
    team_name: str | None = None
    competition_id: str | None = None
    competition_name: str | None = None
    competition_country: str | None = None
    season: str | None = None
    market_value_current_eur: float | None = None
    annual_fixed_eur: float | None = None
    age_months: float | None = None

    model_config = ConfigDict(extra="allow")
