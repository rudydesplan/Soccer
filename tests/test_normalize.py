"""Tests for capology_pipeline/normalize.py."""

from __future__ import annotations

import pytest

from capology_pipeline.normalize import (
    compute_age,
    extract_capology_id,
    extract_flag_entity,
    normalize_date,
    normalize_pid,
    normalize_text,
    parse_money,
    symbol_to_iso,
)


# --- normalize_text ---

class TestNormalizeText:
    def test_basic_accents(self):
        assert normalize_text("Müller") == "muller"

    def test_transliteration_o_slash(self):
        assert normalize_text("ø") == "o"
        assert normalize_text("Ø") == "o"

    def test_transliteration_i_dotless(self):
        assert normalize_text("ı") == "i"

    def test_transliteration_l_stroke(self):
        assert normalize_text("ł") == "l"
        assert normalize_text("Ł") == "l"

    def test_transliteration_d_stroke(self):
        assert normalize_text("đ") == "d"
        assert normalize_text("Đ") == "d"

    def test_transliteration_ae(self):
        assert normalize_text("æ") == "ae"
        assert normalize_text("Æ") == "ae"

    def test_transliteration_oe(self):
        assert normalize_text("œ") == "oe"

    def test_transliteration_ss(self):
        assert normalize_text("ß") == "ss"

    def test_transliteration_eth(self):
        assert normalize_text("ð") == "d"

    def test_transliteration_thorn(self):
        result = normalize_text("þ")
        assert result == "th"

    def test_none_returns_none(self):
        assert normalize_text(None) is None

    def test_nan_returns_none(self):
        import numpy as np
        assert normalize_text(np.nan) is None

    def test_empty_string_returns_none(self):
        assert normalize_text("") is None

    def test_strips_punctuation(self):
        assert normalize_text("O'Brien-Smith") == "o brien smith"

    def test_collapses_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_full_name(self):
        assert normalize_text("José María García") == "jose maria garcia"


# --- normalize_date ---

class TestNormalizeDate:
    def test_iso_format(self):
        assert normalize_date("2020-01-15") == "2020-01-15"

    def test_us_format(self):
        assert normalize_date("01/15/2020") == "2020-01-15"

    def test_text_format(self):
        assert normalize_date("Jan 15, 2020") == "2020-01-15"

    def test_european_text(self):
        assert normalize_date("15 January 2020") == "2020-01-15"

    def test_invalid_returns_none(self):
        assert normalize_date("not a date") is None

    def test_none_returns_none(self):
        assert normalize_date(None) is None

    def test_nan_returns_none(self):
        import numpy as np
        assert normalize_date(np.nan) is None


# --- normalize_pid ---

class TestNormalizePid:
    def test_float_string(self):
        assert normalize_pid("12345.0") == "12345"

    def test_integer_string(self):
        assert normalize_pid("12345") == "12345"

    def test_none_returns_none(self):
        assert normalize_pid(None) is None

    def test_nan_returns_none(self):
        import numpy as np
        assert normalize_pid(np.nan) is None

    def test_whitespace(self):
        assert normalize_pid("  12345  ") == "12345"

    def test_empty_string_returns_none(self):
        assert normalize_pid("") is None


# --- extract_capology_id ---

class TestExtractCapologyId:
    def test_normal_link(self):
        assert extract_capology_id("/player/lionel-messi-12345") == 12345

    def test_just_number(self):
        assert extract_capology_id("player-99") == 99

    def test_no_match(self):
        assert extract_capology_id("no-number-here") is None

    def test_none_input(self):
        assert extract_capology_id(None) is None


# --- extract_flag_entity ---

class TestExtractFlagEntity:
    def test_normal_url(self):
        result = extract_flag_entity("https://cdn.example.com/flags/finland.svg")
        assert result == "finland"

    def test_hyphenated_country(self):
        result = extract_flag_entity("https://cdn.example.com/flags/costa-rica.svg")
        assert result == "costa rica"

    def test_empty_returns_none(self):
        assert extract_flag_entity("") is None

    def test_none_returns_none(self):
        assert extract_flag_entity(None) is None


# --- parse_money ---

class TestParseMoney:
    def test_euro_amount(self):
        assert parse_money("€ 1,000,000") == 1000000.0

    def test_plain_number(self):
        assert parse_money("5000000") == 5000000.0

    def test_with_decimals(self):
        assert parse_money("€1,234,567.89") == 1234567.89

    def test_none_returns_none(self):
        assert parse_money(None) is None

    def test_nan_returns_none(self):
        import numpy as np
        assert parse_money(np.nan) is None

    def test_empty_string_returns_none(self):
        assert parse_money("") is None

    def test_junk_returns_none(self):
        assert parse_money("abc") is None


# --- symbol_to_iso ---

class TestSymbolToIso:
    def test_euro(self):
        assert symbol_to_iso("€") == "EUR"

    def test_pound(self):
        assert symbol_to_iso("£") == "GBP"

    def test_dollar(self):
        assert symbol_to_iso("$") == "USD"

    def test_euro_in_string(self):
        assert symbol_to_iso("€100") == "EUR"

    def test_empty_string_returns_none(self):
        assert symbol_to_iso("") is None

    def test_none_returns_none(self):
        assert symbol_to_iso(None) is None

    def test_unknown_symbol(self):
        assert symbol_to_iso("¥") is None


# --- compute_age ---

class TestComputeAge:
    def test_normal_age(self):
        import datetime
        result = compute_age("1990-06-15", as_of=datetime.date(2025, 6, 15))
        assert result == 35

    def test_before_birthday(self):
        import datetime
        result = compute_age("1990-06-15", as_of=datetime.date(2025, 6, 14))
        assert result == 34

    def test_after_birthday(self):
        import datetime
        result = compute_age("1990-06-15", as_of=datetime.date(2025, 6, 16))
        assert result == 35

    def test_none_returns_none(self):
        assert compute_age(None) is None

    def test_invalid_date_returns_none(self):
        assert compute_age("not a date") is None


# --- parse_money edge case ---

class TestParseMoneyEdge:
    def test_multiple_dots_value_error(self):
        # "1.2.3" after stripping non-digit/dot chars → ValueError in float()
        assert parse_money("1.2.3") is None
