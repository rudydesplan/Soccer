"""Tests for capology_pipeline/capology.py — CapologyParser format resilience.

Tests that the parser gracefully handles missing/changed HTML structure.
"""

from __future__ import annotations

import pytest

from capology_pipeline.capology import CapologyParser
from capology_pipeline.config import ENRICH_COLUMNS


@pytest.fixture
def parser():
    return CapologyParser()


# Sample HTML that mimics Capology's page structure
SAMPLE_HTML_FULL = """
<html>
<script>
var data_active = [{
    "annual_gross_eur": accounting.formatMoney("16670000", "€"),
    "bonus_gross_eur": accounting.formatMoney("3000000", "€"),
    "total_gross_eur": accounting.formatMoney("19670000", "€"),
    "status": "Active"
}];
$('.player-details.dob').html( moment("2007-07-13"));
$('#release').html( accounting.formatMoney("1000000000", "€"));
$('#contract').html( "6-yrs/" + accounting.formatMoney(("100020000"/1e6),{})+'M'+' + '+accounting.formatMoney((("10420000"*"6")/1e6),{})+'M'
</script>
<body>
<div>remaining on his contract with Barcelona (La Liga), expiring Jun 30, 2031</div>
<div class="player-status player-detail grid">
    <div class="player-detail-title">SIGNED</div>
    <div class="player-details">Oct 15, 2024</div>
</div>
<div class="player-status player-detail grid">
    <div class="player-detail-title">EXPIRATION</div>
    <div class="player-details">Jun 30, 2031</div>
</div>
</body>
</html>
"""

SAMPLE_HTML_EMPTY = "<html><body><p>Page not found</p></body></html>"

SAMPLE_HTML_NO_DATA_ACTIVE = """
<html>
<script>
// data_active is missing entirely
$('.player-details.dob').html( moment("2007-07-13"));
</script>
<body></body>
</html>
"""

SAMPLE_HTML_INACTIVE_STATUS = """
<html>
<script>
var data_active = [];
</script>
<body>
<span class="status player-details">Free Agent</span>
</body>
</html>
"""


class TestParserFullPage:
    def test_extracts_salary(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["annual_fixed_eur"] == 16_670_000.0

    def test_extracts_bonus(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["annual_bonus_eur"] == 3_000_000.0

    def test_extracts_total(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["annual_total_eur"] == 19_670_000.0

    def test_extracts_currency(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["salary_currency"] == "EUR"

    def test_extracts_status(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["status"] == "active"

    def test_extracts_release_clause(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["release_clause_eur"] == 1_000_000_000.0

    def test_extracts_league(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["league"] == "La Liga"

    def test_extracts_signed_date(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["signed_date"] == "2024-10-15"

    def test_extracts_expiration_date(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["expiration_date"] == "2031-06-30"

    def test_extracts_capology_url(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["capology_url"] == "https://www.capology.com/player/test-123"

    def test_extracts_gross_contract(self, parser):
        result = parser.parse(SAMPLE_HTML_FULL, "https://www.capology.com/player/test-123")
        assert result["gross_contract"] is not None
        assert "6-yrs/" in result["gross_contract"]


class TestParserEmptyPage:
    """When source format changes or page is empty — graceful nulls."""

    def test_returns_all_enrich_columns(self, parser):
        result = parser.parse(SAMPLE_HTML_EMPTY, "https://www.capology.com/player/x-1")
        for col in ENRICH_COLUMNS:
            assert col in result

    def test_salary_is_none(self, parser):
        result = parser.parse(SAMPLE_HTML_EMPTY, "https://www.capology.com/player/x-1")
        assert result["annual_fixed_eur"] is None

    def test_status_is_none(self, parser):
        result = parser.parse(SAMPLE_HTML_EMPTY, "https://www.capology.com/player/x-1")
        assert result["status"] is None

    def test_release_clause_is_none(self, parser):
        result = parser.parse(SAMPLE_HTML_EMPTY, "https://www.capology.com/player/x-1")
        assert result["release_clause_eur"] is None

    def test_league_is_none(self, parser):
        result = parser.parse(SAMPLE_HTML_EMPTY, "https://www.capology.com/player/x-1")
        assert result["league"] is None

    def test_gross_contract_is_none(self, parser):
        result = parser.parse(SAMPLE_HTML_EMPTY, "https://www.capology.com/player/x-1")
        assert result["gross_contract"] is None


class TestParserMissingDataActive:
    """When the JavaScript data_active variable is missing."""

    def test_salary_is_none(self, parser):
        result = parser.parse(SAMPLE_HTML_NO_DATA_ACTIVE, "https://www.capology.com/player/x-1")
        assert result["annual_fixed_eur"] is None

    def test_birth_date_still_extracted(self, parser):
        result = parser.parse(SAMPLE_HTML_NO_DATA_ACTIVE, "https://www.capology.com/player/x-1")
        assert result["birth_date"] == "2007-07-13"


class TestParserInactiveStatus:
    """Status from DOM fallback (inactive/free agent players)."""

    def test_extracts_status_from_dom(self, parser):
        result = parser.parse(SAMPLE_HTML_INACTIVE_STATUS, "https://www.capology.com/player/x-1")
        assert result["status"] == "free agent"
