"""Tests for salary_benchmark/explain.py (SHAP explanations).

The integration tests use the real pool + trained models (both are tracked
in the repo) — they verify SHAP additivity against the actual prediction,
which a mock cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from salary_benchmark.benchmark import _load_pool
from salary_benchmark.explain import (
    FEATURE_LABELS,
    _display_value,
    explain_by_id,
    explain_by_name,
    explain_player,
)
from salary_benchmark.model import variant_features


class TestDisplayValue:
    def test_log_market_value_shown_in_eur(self):
        assert _display_value("log_market_value_current_eur", np.log1p(50_000_000)) == "€50.0M"

    def test_small_market_value(self):
        assert _display_value("log_market_value_current_eur", np.log1p(500_000)) == "€500,000"

    def test_age_in_years(self):
        assert _display_value("age_months", 300) == "25.0 years"

    def test_contract_months(self):
        assert _display_value("contract_length_months", 48.0) == "48 months"

    def test_flags(self):
        assert _display_value("has_market_value", 1) == "yes"
        assert _display_value("has_market_value", 0) == "no"

    def test_none_and_nan(self):
        assert _display_value("age_months", None) is None
        assert _display_value("age_months", float("nan")) is None

    def test_categorical_passthrough(self):
        assert _display_value("competition_id", "GB1") == "GB1"


class TestExplainIntegration:
    """Real model + real SHAP. Slower (~seconds on first build)."""

    def test_explain_by_id_shape_and_additivity(self):
        result = explain_by_id(0)

        assert result["model_used"] in ("full", "no_mv", "no_mv_no_pos", "no_mv_no_age")
        assert result["base_salary_eur"] >= 0
        assert result["predicted_salary_eur"] >= 0

        features = result["features"]
        assert len(features) == len(variant_features(result["model_used"]))
        # Sorted by |impact|
        impacts = [abs(f["shap_log"]) for f in features]
        assert impacts == sorted(impacts, reverse=True)
        # Additivity: base * exp(sum of contributions) ≈ prediction
        total_log = sum(f["shap_log"] for f in features)
        reconstructed = np.expm1(np.log1p(result["base_salary_eur"]) + total_log)
        assert reconstructed == pytest.approx(result["predicted_salary_eur"], rel=0.02)

    def test_every_feature_has_label_and_pct(self):
        result = explain_by_id(0)
        for f in result["features"]:
            assert f["label"] == FEATURE_LABELS.get(f["feature"], f["feature"])
            # pct_effect must be consistent with shap_log
            assert f["pct_effect"] == pytest.approx(
                (np.exp(f["shap_log"]) - 1) * 100, abs=0.06
            )

    def test_market_value_dominates_for_star_player(self):
        pool = _load_pool()
        star_id = int(pool["market_value_current_eur"].idxmax())
        result = explain_by_id(star_id)
        assert result["model_used"] == "full"
        top = result["features"][0]
        assert top["feature"] == "log_market_value_current_eur"
        assert top["shap_log"] > 0

    def test_explain_by_name(self):
        pool = _load_pool()
        name = pool.iloc[0]["player_name"]
        result = explain_by_name(name)
        assert result["player_name"] == name

    def test_explain_by_name_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            explain_by_name("zzz-no-such-player-zzz")

    def test_explain_by_id_out_of_range(self):
        with pytest.raises(ValueError, match="not found"):
            explain_by_id(99_999_999)

    def test_manual_player_routes_to_fallback(self):
        result = explain_player({
            "player_name": "custom_player",
            "main_position": "Centre-Forward",
            "age_months": 300,
            "competition_id": "GB1",
            "competition_country": "England",
        })
        assert result["model_used"] == "no_mv"
        # The no_mv variant must not have market value features
        feature_names = {f["feature"] for f in result["features"]}
        assert "log_market_value_current_eur" not in feature_names

    def test_player_missing_pos_and_age_refused(self):
        with pytest.raises(ValueError, match="neither"):
            explain_player({"player_name": "x", "market_value_current_eur": 1_000_000})
