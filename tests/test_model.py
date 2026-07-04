"""Tests for salary_benchmark/model.py — AutoGluon fully mocked."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from salary_benchmark import model
from salary_benchmark.model import (
    predict_batch,
    predict_log_salary,
    predict_salary_eur,
    _prepare_frame,
    FEATURES,
)


@pytest.fixture(autouse=True)
def reset_predictor():
    """Reset the module-level predictor cache before each test."""
    model._PREDICTORS.clear()
    yield
    model._PREDICTORS.clear()


@pytest.fixture
def mock_predictor():
    """Create a mock AutoGluon predictor."""
    predictor = MagicMock()
    predictor.predict.return_value = pd.Series([14.5])
    feature_meta = MagicMock()
    feature_meta.get_features.return_value = FEATURES
    predictor.feature_metadata_in = feature_meta
    return predictor


class TestPredictSalaryEur:
    def test_returns_expm1_of_prediction(self, mock_predictor):
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            result = predict_salary_eur({
                "main_position": "Centre-Forward",
                "age_months": 300,
                "market_value_current_eur": 100_000_000,
                "log_market_value_current_eur": np.log1p(100_000_000),
                "contract_length_months": 48,
            })
            expected = float(np.expm1(14.5))
            assert abs(result - expected) < 1.0

    def test_with_dataframe_input(self, mock_predictor):
        df = pd.DataFrame([{
            "main_position": "Centre-Forward",
            "age_months": 300,
            "contract_length_months": 48,
        }])
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            result = predict_salary_eur(df)
            assert isinstance(result, float)


class TestPredictLogSalary:
    def test_returns_raw_log(self, mock_predictor):
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            result = predict_log_salary({
                "main_position": "Centre-Forward",
                "age_months": 300,
                "contract_length_months": 48,
            })
            assert result == 14.5


class TestPredictBatch:
    def test_returns_series(self, mock_predictor):
        mock_predictor.predict.return_value = pd.Series([14.0, 14.5, 15.0])
        df = pd.DataFrame({
            "main_position": ["Centre-Forward", "Left Winger", "Goalkeeper"],
            "age_months": [300, 280, 350],
            "contract_length_months": [48, 36, 60],
        })
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            result = predict_batch(df)
            assert isinstance(result, pd.Series)
            assert len(result) == 3


class TestPrepareFrame:
    def test_frame_matches_predictor_features(self, mock_predictor):
        """The frame passed to predict must contain exactly the predictor's features."""
        player = {
            "main_position": "Centre-Forward",
            "age_months": 300,
            "contract_length_months": 48,
            "contract_months_remaining": 36,
        }
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            predict_log_salary(player)
            assert mock_predictor.predict.called
            call_args = mock_predictor.predict.call_args[0][0]
            assert list(call_args.columns) == FEATURES

    def test_missing_columns_filled_with_nan(self, mock_predictor):
        """Features not provided should be filled with NaN."""
        player = {"main_position": "Centre-Forward"}
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            predict_log_salary(player)
            call_args = mock_predictor.predict.call_args[0][0]
            # age_months wasn't provided, should be NaN
            assert pd.isna(call_args["age_months"].iloc[0])

    def test_predictor_features_exception_fallback(self, mock_predictor):
        """If feature_metadata_in raises, fall back to the canonical FEATURES list."""
        mock_predictor.feature_metadata_in.get_features.side_effect = RuntimeError("no metadata")
        player = {"main_position": "Centre-Forward", "age_months": 300}
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            predict_log_salary(player)
            call_args = mock_predictor.predict.call_args[0][0]
            assert list(call_args.columns) == FEATURES


class TestModelErrorHandling:
    @pytest.fixture(autouse=True)
    def _reset(self):
        model._PREDICTORS.clear()
        yield
        model._PREDICTORS.clear()

    def test_missing_model_directory(self, tmp_path):
        """_load_predictor raises FileNotFoundError when model dir missing."""
        with patch.dict(model.VARIANTS["full"], {"dir": tmp_path / "nonexistent"}):
            with pytest.raises(FileNotFoundError, match="AutoGluon model not found"):
                model._load_predictor()

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValueError, match="Unknown model variant"):
            model._load_predictor("bogus")

    def test_predict_batch_empty_df(self, mock_predictor):
        """predict_batch returns empty Series for empty DataFrame."""
        df = pd.DataFrame()
        with patch.object(model, "_load_predictor", return_value=mock_predictor):
            result = predict_batch(df)
            assert isinstance(result, pd.Series)
            assert len(result) == 0
