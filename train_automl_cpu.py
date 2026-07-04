#!/usr/bin/env python
"""Step 2a — AutoML CPU version (FLAML) — run locally.

Best practices applied (from FLAML docs):
  - seed=42                  : reproducibility
  - metric="rmse"            : appropriate for log-scale salary target
  - log_file_name            : enables learning curve + retrain_from_log
  - starting_points="data"   : Zero-Shot warm start (faster convergence)
  - eval_method="cv"         : cross-validation (small dataset ~3400 rows)
  - n_splits=5               : 5-fold CV
  - split_type="group"       : inner folds grouped by player id (no duplicate-player leakage)
  - ensemble=True            : stacked ensemble after search
  - automl.pickle()          : recommended over pickle.dump()
  - learning curve plot      : from log file via get_output_from_log()
  - feature importance plot  : via automl.feature_importances_

Usage:
    .venv_ml/bin/python train_automl_cpu.py [--input data_test.csv] [--time 120]

Outputs:
    models/flaml_model.pkl
    models/flaml_results.json
    models/flaml_best_config_per_estimator.json
    models/flaml_feature_importance.csv
    models/plots/flaml_learning_curve.png
    models/plots/flaml_feature_importance.png
    flaml_run.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "log_annual_fixed_eur"
CATEGORICAL = ["main_position", "nationality", "competition_id", "competition_country", "status"]
NUMERIC = [
    "season_start_year", "age_months", "contract_length_months",
    "contract_months_remaining", "contract_recency_months", "has_contract_dates", "has_market_value",
    "log_market_value_current_eur", "has_release_clause", "log_release_clause_eur",
]


def plot_learning_curve(log_file: str, time_budget: int, out: Path):
    """Plot FLAML search learning curve from log file."""
    from flaml.automl.data import get_output_from_log
    try:
        (
            time_history,
            best_valid_loss_history,
            valid_loss_history,
            config_history,
            metric_history,
        ) = get_output_from_log(filename=log_file, time_budget=time_budget)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(
            time_history,
            1 - np.array(best_valid_loss_history),
            where="post",
            color="#2ecc71",
            linewidth=2,
            label="Best validation R²",
        )
        ax.scatter(
            time_history,
            1 - np.array(valid_loss_history),
            alpha=0.3,
            s=15,
            color="steelblue",
            label="Trial R²",
        )
        ax.set_xlabel("Wall Clock Time (s)")
        ax.set_ylabel("Validation R²")
        ax.set_title("FLAML — Search Learning Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")
    except Exception as e:
        print(f"  Skipped learning curve: {e}")


def plot_feature_importance(importances, feature_names: list[str], best_name: str, out: Path):
    """Plot feature importances from the best FLAML model."""
    try:
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(5, len(fi_df) * 0.4)))
        ax.barh(fi_df["feature"], fi_df["importance"], color="steelblue", edgecolor="white")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"FLAML — Feature Importance ({best_name})")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")
    except Exception as e:
        print(f"  Skipped feature importance plot: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_test.csv")
    parser.add_argument("--time", type=int, default=1800, help="FLAML budget in seconds (default: 30 min)")
    args = parser.parse_args(argv)

    out_dir = Path("models")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    log_file = "flaml_run.log"

    # --- Load data ---
    df = pd.read_csv(args.input)
    df = df[df[TARGET].notna()].copy()
    features = CATEGORICAL + NUMERIC
    X = df[features].copy()
    for col in CATEGORICAL:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df[TARGET]
    print(f"Train rows: {len(df)}")

    # Grouped split by player ID to prevent leakage from duplicate rows
    if "transfermarkt_player_id" in df.columns:
        from sklearn.model_selection import GroupShuffleSplit
        groups = df["transfermarkt_player_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups[train_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        groups_train = None

    # --- FLAML AutoML with best practices ---
    from flaml import AutoML

    automl = AutoML()
    settings = {
        "task": "regression",
        "metric": "rmse",                    # appropriate for log-scale target
        "time_budget": args.time,
        "seed": 42,                          # reproducibility
        "log_file_name": log_file,           # enables learning curve + retrain_from_log
        "starting_points": "data",           # Zero-Shot warm start
        "eval_method": "cv",                 # cross-validation (small dataset)
        "n_splits": 5,                       # 5-fold CV
        "ensemble": True,                    # stacked ensemble after search
        "early_stop": True,                  # stop if no improvement detected
        "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree", "histgb"],
        "verbose": 1,
    }
    # Group the inner CV folds by player id so duplicated players cannot leak
    # across folds during hyperparameter selection.
    if groups_train is not None:
        settings["split_type"] = "group"

    print(f"\n=== FLAML AutoML (budget={args.time}s, cv=5, ensemble=True, early_stop=True) ===")
    if groups_train is not None:
        automl.fit(X_train, y_train, groups=groups_train, **settings)
    else:
        automl.fit(X_train, y_train, **settings)

    # --- Evaluate on test set ---
    y_pred = automl.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== RESULTS ===")
    print(f"  Best estimator : {automl.best_estimator}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  MAE            : {mae:.4f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  Best config    : {automl.best_config}")
    print(f"  Best val loss  : {automl.best_loss:.4f}")
    print(f"  Search time    : {automl.best_config_train_time:.2f}s")

    # --- Save results ---
    results = {
        "best_estimator": automl.best_estimator,
        "rmse": rmse, "mae": mae, "r2": r2,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "time_budget": args.time,
    }
    (out_dir / "flaml_results.json").write_text(json.dumps(results, indent=2))

    # Save best config per estimator (useful for warm start)
    try:
        (out_dir / "flaml_best_config_per_estimator.json").write_text(
            json.dumps(automl.best_config_per_estimator, indent=2)
        )
        print(f"\n  Best config per estimator saved.")
    except Exception as e:
        print(f"  Could not save best_config_per_estimator: {e}")

    # --- Feature importance CSV ---
    feature_names = list(X_train.columns)
    try:
        # When ensemble=True, automl.model is a StackingRegressor.
        # Use the best base estimator (xgboost/lgbm) from the stacker.
        stacker = automl.model
        best_name = automl.best_estimator
        base_est = stacker.named_estimators_.get(best_name)
        if base_est is None:
            # fallback: first estimator with feature_importances_
            for name, est in stacker.named_estimators_.items():
                try:
                    _ = est.feature_importances_
                    base_est = est
                    best_name = name
                    break
                except Exception:
                    continue
        importances = base_est.feature_importances_
        feature_names = list(automl.feature_names_in_)  # FLAML's actual feature names after preprocessing
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(out_dir / "flaml_feature_importance.csv", index=False)
        print(f"\n=== FEATURE IMPORTANCE (top 15, from {best_name}) ===")
        print(fi_df.head(15).to_string(index=False))
    except Exception as e:
        print(f"  Feature importance not available: {e}")
        fi_df = pd.DataFrame()
        importances = None
        best_name = automl.best_estimator
        feature_names = list(automl.feature_names_in_)

    # --- Plots ---
    print("\n=== PLOTS ===")
    plot_learning_curve(log_file, args.time, plots_dir / "flaml_learning_curve.png")

    if not fi_df.empty:
        plot_feature_importance(
            importances, feature_names, best_name,
            plots_dir / "flaml_feature_importance.png"
        )

    # --- Save model (recommended method) ---
    automl.pickle(str(out_dir / "flaml_model.pkl"))
    print(f"\nModel saved to {out_dir}/flaml_model.pkl")
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
