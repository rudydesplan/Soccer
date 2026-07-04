"""Load AutoGluon predictors and predict log_salary → EUR.

Model variants (each trained on a reduced feature set, each paired with its
own out-of-fold calibration — see train_fallback_no_mv.py):
- "full"          — production model, all 15 features incl. market value.
- "no_mv"         — no market-value features. For players without a market value.
- "no_mv_no_pos"  — additionally without main_position. For players missing
                    a position (market value, if any, is also unused).
- "no_mv_no_age"  — no market value and no age_months. For players missing age.

Each step down uses a weaker model with wider calibrated intervals.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_AG_DIR = _MODELS_DIR / "autogluon"

# Canonical feature names used by newly-generated feature files.
FEATURES = [
    "main_position", "nationality", "competition_id", "competition_country",
    "status", "season_start_year", "age_months", "contract_length_months",
    "contract_months_remaining", "contract_recency_months", "has_contract_dates", "has_market_value",
    "log_market_value_current_eur", "has_release_clause", "log_release_clause_eur",
]

MV_FEATURES = ["log_market_value_current_eur", "has_market_value"]
FEATURES_NO_MV = [f for f in FEATURES if f not in MV_FEATURES]

# Per-variant configuration: model directory, features excluded at training
# time, and the command that (re)builds the artifact.
VARIANTS: dict[str, dict] = {
    "full": {
        "dir": _AG_DIR,
        "excluded_features": [],
        "train_hint": "train_autogluon_cpu.py",
    },
    "no_mv": {
        "dir": _MODELS_DIR / "autogluon_no_mv",
        "excluded_features": MV_FEATURES,
        "train_hint": "train_fallback_no_mv.py --variant no_mv",
    },
    "no_mv_no_pos": {
        "dir": _MODELS_DIR / "autogluon_no_mv_no_pos",
        "excluded_features": MV_FEATURES + ["main_position"],
        "train_hint": "train_fallback_no_mv.py --variant no_mv_no_pos",
    },
    "no_mv_no_age": {
        "dir": _MODELS_DIR / "autogluon_no_mv_no_age",
        "excluded_features": MV_FEATURES + ["age_months"],
        "train_hint": "train_fallback_no_mv.py --variant no_mv_no_age",
    },
}

_PREDICTORS: dict[str, object] = {}


def variant_features(variant: str) -> list[str]:
    """Feature list a given variant was trained on."""
    excluded = VARIANTS[variant]["excluded_features"]
    return [f for f in FEATURES if f not in excluded]


def fallback_available(variant: str = "no_mv") -> bool:
    """Whether a fallback model artifact exists on disk."""
    return VARIANTS[variant]["dir"].exists()


def _load_from_dir(ag_dir: Path, train_hint: str):
    if not ag_dir.exists():
        raise FileNotFoundError(
            f"AutoGluon model not found at {ag_dir}. "
            f"Train a model first with {train_hint}"
        )
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon not installed. Run: .venv_ml/bin/pip install autogluon.tabular")
    try:
        return TabularPredictor.load(str(ag_dir), require_version_match=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load AutoGluon model from {ag_dir}: {e}. "
            f"The model files may be corrupt — retrain with {train_hint}"
        ) from e


def _load_predictor(variant: str = "full"):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant!r}")
    if variant not in _PREDICTORS:
        cfg = VARIANTS[variant]
        _PREDICTORS[variant] = _load_from_dir(cfg["dir"], cfg["train_hint"])
    return _PREDICTORS[variant]


def _predictor_features(predictor, variant: str = "full") -> list[str]:
    """Return the features expected by the saved predictor."""
    try:
        return list(predictor.feature_metadata_in.get_features())
    except Exception:
        return variant_features(variant)


def _prepare_frame(player_features: dict | pd.DataFrame, predictor,
                   variant: str = "full") -> pd.DataFrame:
    if isinstance(player_features, dict):
        df = pd.DataFrame([player_features])
    else:
        df = player_features.copy()

    features = _predictor_features(predictor, variant)
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    return df[features]


def predict_salary_eur(player_features: dict | pd.DataFrame, variant: str = "full") -> float:
    """Predict annual fixed salary in EUR for one player.

    Args:
        player_features: dict or single-row DataFrame with model features.
        variant: "full" (default) or "no_mv" fallback.

    Returns:
        Predicted annual fixed salary in EUR (expm1 of log prediction).
    """
    predictor = _load_predictor(variant)
    df = _prepare_frame(player_features, predictor, variant)
    log_pred = predictor.predict(df).iloc[0]
    return float(np.expm1(log_pred))


def predict_log_salary(player_features: dict | pd.DataFrame, variant: str = "full") -> float:
    """Return raw log-scale prediction (for calibration arithmetic)."""
    predictor = _load_predictor(variant)
    df = _prepare_frame(player_features, predictor, variant)
    return float(predictor.predict(df).iloc[0])


def predict_batch(df: pd.DataFrame, variant: str = "full") -> pd.Series:
    """Predict log salary for a batch DataFrame. Returns Series of log predictions."""
    if df.empty:
        return pd.Series(dtype="float64")
    predictor = _load_predictor(variant)
    prepared = _prepare_frame(df, predictor, variant)
    return predictor.predict(prepared)
