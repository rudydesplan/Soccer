"""Compute and load residual calibration for salary range estimation.

Calibration uses OUT-OF-FOLD residuals (5-fold GROUPED CV) to avoid in-sample
leakage. The production model (trained on all salary rows) has seen all
training data, so residuals must come from held-out predictions to produce
honest confidence intervals. Folds are grouped by transfermarkt_player_id so
the same player (duplicated across clubs/competitions) never appears in both
the train and validation side of a fold.

Fold models use the same preset as the production model and a 15-minute
time limit per fold.

Usage:
    python -m salary_benchmark.calibration   # recompute and save
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CALIBRATION_PATH = _MODELS_DIR / "calibration.json"
CALIBRATION_NO_MV_PATH = _MODELS_DIR / "calibration_no_mv.json"
POOL_PATH = Path(__file__).resolve().parents[1] / "player_pool.csv"

# One calibration file per model variant (see salary_benchmark.model.VARIANTS).
CALIBRATION_PATHS: dict[str, Path] = {
    "full": CALIBRATION_PATH,
    "no_mv": CALIBRATION_NO_MV_PATH,
    "no_mv_no_pos": _MODELS_DIR / "calibration_no_mv_no_pos.json",
    "no_mv_no_age": _MODELS_DIR / "calibration_no_mv_no_age.json",
}

_CAL_CACHE: dict[str, dict] = {}

TARGET = "log_annual_fixed_eur"
GROUP_COLUMN = "transfermarkt_player_id"

# Match the production training setup (train_autogluon_cpu.py) as closely as
# a per-fold budget allows: same preset, 15 minutes per fold.
FOLD_TIME_LIMIT = 900
FOLD_PRESET = "experimental_quality"

# NN/foundation model families deliberately excluded everywhere (production
# training, calibration, evaluation, comparison — see requirements-ml.txt).
# Their dependencies may be installed in the venv, but the project keeps one
# consistent LightGBM/XGBoost/sklearn family so fold models match production.
# Without this, AutoGluon burns the 900s fold budget attempting foundation
# models (Mitra even downloads weights) and can fail with "No models were
# trained successfully".
EXCLUDED_MODEL_TYPES = [
    "NN_TORCH", "FASTAI", "CAT", "TABM", "TABDPT", "TABICL", "TABPFNMIX",
    "REALTABPFN-V2", "REALTABPFN-V2.5", "MITRA",
]


def build_calibration(pool_path: Path = POOL_PATH,
                      out_path: Path = CALIBRATION_PATH,
                      n_folds: int = 5,
                      exclude_features: list[str] | None = None) -> dict:
    """Compute OUT-OF-FOLD residuals using grouped cross-validation.

    This avoids two leakage problems:
    - In-sample leakage: the production model trained on all data, so
      computing residuals on training data would underestimate prediction
      error. We train fold models on ~80% each and collect residuals on the
      held-out ~20%, then merge all out-of-fold residuals.
    - Same-player leakage: rows are grouped by transfermarkt_player_id
      (GroupKFold), so a player duplicated under multiple clubs can never
      appear in both train and validation of the same fold.
    """
    from autogluon.tabular import TabularPredictor, TabularDataset
    import shutil

    df = pd.read_csv(pool_path)
    df = df[df["annual_fixed_eur"].notna()].copy().reset_index(drop=True)
    print(f"Computing OUT-OF-FOLD calibration on {len(df)} players "
          f"({n_folds}-fold grouped CV, {FOLD_TIME_LIMIT}s/fold, preset={FOLD_PRESET})...")

    # Features used by the model
    feature_cols = [
        "main_position", "nationality", "competition_id", "competition_country",
        "status", "season_start_year", "age_months", "contract_length_months",
        "contract_months_remaining", "contract_recency_months", "has_contract_dates",
        "has_market_value", "log_market_value_current_eur", "has_release_clause",
        "log_release_clause_eur", TARGET,
    ]
    if exclude_features:
        feature_cols = [c for c in feature_cols if c not in exclude_features]
    model_df = df[[c for c in feature_cols if c in df.columns]].copy()

    grouped = GROUP_COLUMN in df.columns
    if grouped:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(model_df, groups=df[GROUP_COLUMN].values)
    else:
        from sklearn.model_selection import KFold
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(model_df)

    oof_residuals = np.full(len(model_df), np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, 1):
        print(f"  Fold {fold_idx}/{n_folds}: train={len(train_idx)}, val={len(val_idx)}")
        ag_path = f"/tmp/calibration_fold_{fold_idx}"

        train_data = TabularDataset(model_df.iloc[train_idx])
        val_data = TabularDataset(model_df.iloc[val_idx])

        predictor = TabularPredictor(
            label=TARGET,
            path=ag_path,
            eval_metric="rmse",
            problem_type="regression",
            verbosity=0,
        ).fit(
            train_data,
            time_limit=FOLD_TIME_LIMIT,
            presets=FOLD_PRESET,
            auto_stack=True,
            excluded_model_types=EXCLUDED_MODEL_TYPES,
        )

        val_pred = predictor.predict(val_data).values
        val_actual = model_df.iloc[val_idx][TARGET].values
        oof_residuals[val_idx] = val_actual - val_pred

        shutil.rmtree(ag_path, ignore_errors=True)

    # Verify all residuals are filled
    assert not np.isnan(oof_residuals).any(), "Some residuals missing after CV"

    calibration = {
        "n_samples": int(len(oof_residuals)),
        "n_folds": n_folds,
        "method": "grouped_out_of_fold_cv" if grouped else "out_of_fold_cv",
        "group_column": GROUP_COLUMN if grouped else None,
        "fold_time_limit": FOLD_TIME_LIMIT,
        "fold_preset": FOLD_PRESET,
        "excluded_model_types": EXCLUDED_MODEL_TYPES,
        "excluded_features": exclude_features or [],
        "residual_p10": float(np.percentile(oof_residuals, 10)),
        "residual_p25": float(np.percentile(oof_residuals, 25)),
        "residual_p50": float(np.percentile(oof_residuals, 50)),
        "residual_p75": float(np.percentile(oof_residuals, 75)),
        "residual_p90": float(np.percentile(oof_residuals, 90)),
        "residual_mean": float(np.mean(oof_residuals)),
        "residual_std": float(np.std(oof_residuals)),
        "rmse": float(np.sqrt(np.mean(oof_residuals ** 2))),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(calibration, indent=2))
    print(f"Calibration saved to {out_path}")
    print(f"  method: {calibration['method']} ({n_folds} folds)")
    print(f"  p10={calibration['residual_p10']:.3f}  p25={calibration['residual_p25']:.3f}"
          f"  p75={calibration['residual_p75']:.3f}  p90={calibration['residual_p90']:.3f}")
    print(f"  RMSE={calibration['rmse']:.4f} (honest out-of-fold)")
    return calibration


def load_calibration(variant: str = "full") -> dict:
    """Load a variant's calibration from disk. Raises if missing."""
    if variant in _CAL_CACHE:
        return _CAL_CACHE[variant]
    path = CALIBRATION_PATHS.get(variant)
    if path is None:
        raise ValueError(f"Unknown calibration variant: {variant!r}")
    if not path.exists():
        hint = ("python -m salary_benchmark.calibration" if variant == "full"
                else f"python train_fallback_no_mv.py --variant {variant}")
        raise FileNotFoundError(
            f"Calibration for variant '{variant}' not found at {path}. "
            f"Build it first: {hint}"
        )
    _CAL_CACHE[variant] = json.loads(path.read_text())
    return _CAL_CACHE[variant]


def load_calibration_no_mv() -> dict:
    """Load the no-market-value fallback calibration (kept for back-compat)."""
    return load_calibration("no_mv")


def salary_range(log_pred: float, width: str = "normal", variant: str = "full") -> dict:
    """Convert a log-scale prediction into a EUR salary range.

    Args:
        log_pred: log-scale prediction from the model.
        width: "normal" (p25/p75) or "wide" (p10/p90).
        variant: model variant name ("full", "no_mv", "no_mv_no_pos",
            "no_mv_no_age") — each has its own out-of-fold residual
            calibration; the fallbacks' are wider.

    Returns:
        dict with low, median, high in EUR.
    """
    cal = load_calibration(variant)
    if width == "wide":
        low_key, high_key = "residual_p10", "residual_p90"
    else:
        low_key, high_key = "residual_p25", "residual_p75"

    return {
        "expected_salary_low_eur": float(np.expm1(log_pred + cal[low_key])),
        "expected_salary_median_eur": float(np.expm1(log_pred)),
        "expected_salary_high_eur": float(np.expm1(log_pred + cal[high_key])),
    }


if __name__ == "__main__":
    build_calibration()
