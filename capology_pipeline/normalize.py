"""Pure normalisation / parsing helpers (no I/O, unit-testable)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath
from urllib.parse import urlparse

import pandas as pd

_SYMBOL_TO_ISO = {"€": "EUR", "£": "GBP", "$": "USD"}

_TRANSLITERATION = str.maketrans({
    "ø": "o", "Ø": "O",
    "ı": "i", "İ": "I",
    "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th",
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "ß": "ss",
})


def normalize_text(value):
    """Lowercase, strip accents/punctuation. None/empty -> None."""
    if value is None or pd.isna(value):
        return None
    value = str(value).translate(_TRANSLITERATION)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def normalize_date(value):
    """Any recognisable date -> ISO 'YYYY-MM-DD', else None."""
    if value is None or pd.isna(value):
        return None
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    return date.strftime("%Y-%m-%d")


def compute_age(birth_date, as_of=None):
    """Age in years from a birth_date, comparable to Transfermarkt's age."""
    iso = normalize_date(birth_date)
    if iso is None:
        return None
    bd = pd.to_datetime(iso).date()
    ref = as_of or pd.Timestamp.today().date()
    return ref.year - bd.year - ((ref.month, ref.day) < (bd.month, bd.day))


def parse_money(value):
    """'€ 1,000,000,000' / '1000000000' -> 1000000000.0 ; junk/None -> None."""
    if value is None or pd.isna(value):
        return None
    cleaned = re.sub(r"[^\d.]", "", str(value))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def symbol_to_iso(symbol):
    if not symbol:
        return None
    for sign, iso in _SYMBOL_TO_ISO.items():
        if sign in symbol:
            return iso
    return None


def normalize_pid(value):
    """Transfermarkt id (player or club) -> bare integer string, else None."""
    if value is None or pd.isna(value):
        return None
    pid = str(value).strip()
    if pid.endswith(".0"):  # ids arrive from the CSV as floats
        pid = pid[:-2]
    return pid or None


def extract_flag_entity(flag_url):
    """Nationality from a Transfermarkt/Capology flag URL ( .../finland.svg )."""
    if not flag_url:
        return None
    path = urlparse(flag_url).path
    return normalize_text(PurePosixPath(path).stem.replace("-", " "))


def extract_capology_id(link):
    match = re.search(r"-(\d+)$", str(link))
    return int(match.group(1)) if match else None
