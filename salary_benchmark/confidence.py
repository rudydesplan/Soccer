"""Confidence label logic.

Confidence reflects how much we trust the benchmark result.
Key principle: if no comparable player has a KNOWN salary from the same league,
confidence must be LOW regardless of other criteria — we're extrapolating.
"""

from __future__ import annotations


def confidence_label(n_comparables: int,
                     avg_similarity: float,
                     relative_range_width: float,
                     n_comparables_with_salary_same_league: int | None = None) -> str:
    """Return 'high', 'medium', or 'low' confidence.

    Args:
        n_comparables: number of comparable players found.
        avg_similarity: average similarity score (0-1) of comparables.
        relative_range_width: (high - low) / median salary.
        n_comparables_with_salary_same_league: how many comparables have a known
            salary AND are from the same league. If 0, confidence is always LOW
            because we're extrapolating from other leagues.
    """
    # Hard rule: if no comparable has salary data from the same league,
    # we cannot claim medium/high confidence — we're guessing.
    if n_comparables_with_salary_same_league is not None and n_comparables_with_salary_same_league == 0:
        return "low"

    if (n_comparables >= 50
            and avg_similarity >= 0.75
            and relative_range_width < 0.60):
        return "high"

    if (n_comparables >= 20
            and avg_similarity >= 0.60
            and relative_range_width < 1.00):
        return "medium"

    return "low"


def salary_status(actual_eur: float | None,
                  low_eur: float,
                  high_eur: float) -> str:
    """Classify a player's salary vs the expected range."""
    if actual_eur is None:
        return "unknown"
    if actual_eur < low_eur:
        return "underpaid"
    if actual_eur > high_eur:
        return "overpaid"
    return "fairly_paid"
