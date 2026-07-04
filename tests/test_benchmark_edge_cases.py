"""Benchmark edge case tests — real scenarios from our dataset.

These use the real model and player pool to verify the benchmark handles
edge cases found in the actual data without crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
PLAYER_POOL = DATA_DIR / "player_pool.csv"

pytestmark = pytest.mark.skipif(
    not PLAYER_POOL.exists(), reason="player_pool.csv not present"
)


@pytest.fixture(scope="module")
def pool():
    return pd.read_csv(PLAYER_POOL)


class TestPlayerWithSalaryButNoMarketValue:
    """Players with no market value use the no-MV fallback model when it is
    trained; without the fallback artifact the benchmark raises a clear error."""

    def test_fallback_or_clear_error(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        from salary_benchmark.model import fallback_available
        sal_no_mv = pool[pool["annual_fixed_eur"].notna() & pool["market_value_current_eur"].isna()]
        if sal_no_mv.empty:
            pytest.skip("No players with salary but no MV")
        player = sal_no_mv.iloc[0].to_dict()
        if fallback_available():
            result = benchmark_player(player)
            assert result["model_used"] == "no_mv"
            assert result["expected_salary_median_eur"] > 0
            assert "fallback" in (result["benchmark_warning"] or "").lower()
            assert result["benchmark_confidence"] in ("MEDIUM", "LOW")
        else:
            with pytest.raises(ValueError, match="fallback model is not available"):
                benchmark_player(player)

    def test_works_when_mv_provided_as_zero(self, pool):
        """If we default missing MV to 0, benchmark should still work."""
        from salary_benchmark.benchmark import benchmark_player
        sal_no_mv = pool[pool["annual_fixed_eur"].notna() & pool["market_value_current_eur"].isna()]
        if sal_no_mv.empty:
            pytest.skip("No players with salary but no MV")
        player = sal_no_mv.iloc[0].to_dict()
        player["market_value_current_eur"] = 0
        player["log_market_value_current_eur"] = 0.0
        player["has_market_value"] = 0
        result = benchmark_player(player)
        assert result["expected_salary_median_eur"] > 0


class TestPlayerWithMissingPositionOrAge:
    """Real pool players missing position or age route to the dedicated
    fallback variants; players missing both are refused."""

    def test_no_position_routes_to_no_pos_fallback(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        from salary_benchmark.model import fallback_available
        if not fallback_available("no_mv_no_pos"):
            pytest.skip("no_mv_no_pos model not trained")
        no_pos = pool[pool["main_position"].isna() & pool["age_months"].notna()]
        if no_pos.empty:
            pytest.skip("No players missing position")
        result = benchmark_player(no_pos.iloc[0].to_dict())
        assert result["model_used"] == "no_mv_no_pos"
        assert result["expected_salary_median_eur"] > 0
        assert result["benchmark_confidence"] == "LOW"
        assert "position" in (result["benchmark_warning"] or "").lower()

    def test_no_age_routes_to_no_age_fallback(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        from salary_benchmark.model import fallback_available
        if not fallback_available("no_mv_no_age"):
            pytest.skip("no_mv_no_age model not trained")
        no_age = pool[pool["age_months"].isna() & pool["main_position"].notna()]
        if no_age.empty:
            pytest.skip("No players missing age")
        result = benchmark_player(no_age.iloc[0].to_dict())
        assert result["model_used"] == "no_mv_no_age"
        assert result["expected_salary_median_eur"] > 0
        assert result["benchmark_confidence"] == "LOW"
        assert "age" in (result["benchmark_warning"] or "").lower()

    def test_missing_both_is_refused(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        no_both = pool[pool["main_position"].isna() & pool["age_months"].isna()]
        if no_both.empty:
            pytest.skip("No players missing both position and age")
        with pytest.raises(ValueError, match="neither main_position nor age_months"):
            benchmark_player(no_both.iloc[0].to_dict())


class TestPlayerWithVeryLowMarketValue:
    """6,146 players have MV < €100K — benchmark should not produce absurd results."""

    def test_benchmark_works(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        low_mv = pool[
            (pool["market_value_current_eur"] < 100_000)
            & pool["market_value_current_eur"].notna()
            & pool["main_position"].notna()
        ]
        if low_mv.empty:
            pytest.skip("No low MV players")
        player = low_mv.iloc[0].to_dict()
        result = benchmark_player(player)
        assert result["expected_salary_median_eur"] > 0
        # Should not predict millions for a €50K player
        assert result["expected_salary_median_eur"] < 5_000_000


class TestPlayerWithMissingPosition:
    """53 players have no position — benchmark should handle gracefully."""

    def test_missing_position_does_not_crash(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        no_pos = pool[pool["main_position"].isna()]
        if no_pos.empty:
            pytest.skip("No players with missing position")
        player = no_pos.iloc[0].to_dict()
        # Should either work or raise a clear ValueError, not crash
        try:
            result = benchmark_player(player)
            # If it works, result should be valid
            assert "expected_salary_median_eur" in result
        except ValueError:
            # Acceptable — clear error for missing required field
            pass


class TestPlayerWithExpiredContract:
    """Players with expiration_date in the past → negative remaining months."""

    def test_expired_contract_does_not_crash(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        expired = pool[pool["contract_months_remaining"] < 0]
        if expired.empty:
            pytest.skip("No expired contracts")
        player = expired.iloc[0].to_dict()
        result = benchmark_player(player)
        assert result["expected_salary_median_eur"] > 0


class TestDuplicateNames:
    """1,136 duplicate names — benchmark_by_name should pick highest MV."""

    def test_duplicate_name_returns_highest_mv(self, pool):
        from salary_benchmark.benchmark import benchmark_by_name
        # Find a name with multiple entries
        name_counts = pool["player_name"].value_counts()
        dup_names = name_counts[name_counts > 1].index
        if len(dup_names) == 0:
            pytest.skip("No duplicate names")
        name = dup_names[0]
        matches = pool[pool["player_name"] == name]
        expected_mv = matches["market_value_current_eur"].max()

        result = benchmark_by_name(name)
        # Should pick the one with highest market value
        if not np.isnan(expected_mv):
            assert result["market_value_current_eur"] == expected_mv


class TestBenchmarkSampleNocrash:
    """Sample of known-salary players — none should crash."""

    def test_sample_of_salary_players(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        sal_players = pool[
            pool["annual_fixed_eur"].notna()
            & pool["main_position"].notna()
            & pool["market_value_current_eur"].notna()
        ]
        # Test a random sample of 50
        sample = sal_players.sample(n=min(50, len(sal_players)), random_state=42)
        failures = []
        for idx, row in sample.iterrows():
            try:
                result = benchmark_player(row.to_dict())
                assert result["expected_salary_median_eur"] > 0
            except Exception as e:
                failures.append((row["player_name"], str(e)))
        assert failures == [], f"Benchmark failed for: {failures}"


class TestComparableSearchAlwaysFinds:
    """For any player with MV + position, level 3 should find ≥1 comparable."""

    def test_always_finds_comparables(self, pool):
        from salary_benchmark.comparables import find_comparables
        valid = pool[
            pool["main_position"].notna()
            & pool["market_value_current_eur"].notna()
            & pool["age_months"].notna()
        ]
        sample = valid.sample(n=min(30, len(valid)), random_state=42)
        for idx, row in sample.iterrows():
            target = row.to_dict()
            # Exclude self
            pool_filtered = pool[pool.index != idx]
            comps, level = find_comparables(target, pool_filtered, n_min=1)
            assert len(comps) >= 1, f"No comparables for {row['player_name']}"


class TestSalaryStatusDistribution:
    """Among known-salary players, not 100% should be 'overpaid'."""

    def test_status_distribution_reasonable(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        sal_players = pool[
            pool["annual_fixed_eur"].notna()
            & pool["main_position"].notna()
        ]
        sample = sal_players.sample(n=min(100, len(sal_players)), random_state=42)
        statuses = []
        for _, row in sample.iterrows():
            try:
                result = benchmark_player(row.to_dict())
                statuses.append(result["salary_status"])
            except Exception:
                pass
        # Should have at least 2 different statuses
        unique_statuses = set(statuses)
        assert len(unique_statuses) >= 2, f"Only got statuses: {unique_statuses}"


class TestConfidenceNotAlwaysLow:
    """At least some players should get MEDIUM or HIGH confidence."""

    def test_some_medium_or_high_confidence(self, pool):
        from salary_benchmark.benchmark import benchmark_player
        # Focus on top leagues where comparable density is high
        top_league = pool[
            (pool["competition_id"] == "GB1")
            & pool["main_position"].notna()
            & pool["market_value_current_eur"].notna()
        ]
        sample = top_league.sample(n=min(50, len(top_league)), random_state=42)
        confidences = []
        for _, row in sample.iterrows():
            try:
                result = benchmark_player(row.to_dict())
                confidences.append(result["benchmark_confidence"])
            except Exception:
                pass
        # At least one should be MEDIUM or HIGH
        non_low = [c for c in confidences if c != "LOW"]
        # This is a soft check — if all are LOW, it's a signal but not necessarily a bug
        # For GB1 with good density, we expect at least some MEDIUM
        assert len(non_low) >= 1 or len(confidences) == 0, \
            f"All {len(confidences)} benchmarks returned LOW confidence"
