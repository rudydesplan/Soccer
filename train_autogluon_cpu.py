#!/usr/bin/env python
"""AutoGluon TabularPredictor — CPU version (Mac M5 / local machine).

Two-phase training (evaluation and production are kept separate):

  Phase 1 — EVALUATION: grouped 80/20 holdout split (grouped by
    transfermarkt_player_id so the same player never appears in both train
    and test). A model is trained on the train split only and evaluated on
    the held-out 20%. These are the honest metrics reported in the README.
    The evaluation model is saved to models/autogluon_holdout_eval so the
    holdout metrics can be reproduced.

  Phase 2 — PRODUCTION: a final model is trained on ALL salary rows (no
    holdout) and saved to models/autogluon. This is the model served by the
    benchmark engine. Its quality estimate comes from Phase 1.

Best practices applied (from AutoGluon 1.5 docs):
  - eval_metric="rmse"       : appropriate for log-scale salary target
  - auto_stack=True          : bagging + stacking for best accuracy
  - refit_full()             : retrain best model on the full fit data
  - save_space()             : remove unnecessary model artifacts

Note: ag_args_fit={"num_gpus": 1} removed — Apple Silicon GPU (Metal/MPS)
is not supported by AutoGluon. All training runs on CPU.

Note: neural-net and foundation-model families (torch/fastai/catboost/tabpfn/
tabm) are excluded via excluded_model_types — the same exclusion list used by
calibration, evaluation, and comparison — so the production ensemble stays a
LightGBM/XGBoost/sklearn family regardless of what is installed in the venv.

Usage:
    .venv_ml/bin/python train_autogluon_cpu.py [--input data_test.csv] [--time 10800]

Outputs:
    models/autogluon/                    — production model (trained on ALL salary rows)
    models/autogluon_holdout_eval/       — evaluation model (grouped holdout train split)
    models/autogluon_leaderboard.csv     — evaluation leaderboard (holdout test scores)
    models/autogluon_feature_importance.csv
    models/autogluon_results.json        — holdout metrics + full training metadata
    models/plots/autogluon_*.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "log_annual_fixed_eur"
GROUP_COLUMN = "transfermarkt_player_id"
CATEGORICAL = ["main_position", "nationality", "competition_id", "competition_country", "status"]
NUMERIC = [
    "season_start_year", "age_months", "contract_length_months",
    "contract_months_remaining", "contract_recency_months", "has_contract_dates", "has_market_value",
    "log_market_value_current_eur", "has_release_clause", "log_release_clause_eur",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_of_date(input_path: Path) -> str | None:
    """Read as_of_date from the feature build's metadata sidecar if present."""
    meta_path = input_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text()).get("as_of_date")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    actual_eur = np.expm1(y_true)
    pred_eur = np.expm1(y_pred)
    pct_error = np.abs(actual_eur - pred_eur) / actual_eur
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(pct_error.mean() * 100),
        "median_ape": float(np.median(pct_error) * 100),
        "within_20_pct": float((pct_error <= 0.20).mean() * 100),
        "within_50_pct": float((pct_error <= 0.50).mean() * 100),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_test.csv")
    parser.add_argument("--time", type=int, default=10800,
                        help="time limit in seconds PER PHASE (default: 3h)")
    parser.add_argument("--preset", default="experimental_quality",
                        choices=["experimental_quality", "best_quality", "best_v150", "best",
                                 "high_quality", "high_v150", "high", "good_quality", "good", "medium"],
                        help="AutoGluon preset (default: experimental_quality — best CPU quality)")
    args = parser.parse_args(argv)

    out_dir = Path("models")
    plots_dir = out_dir / "plots"
    ag_prod_dir = str(out_dir / "autogluon")
    ag_eval_dir = str(out_dir / "autogluon_holdout_eval")
    out_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    try:
        import autogluon.tabular
        from autogluon.tabular import TabularPredictor, TabularDataset
    except ImportError:
        raise ImportError(
            "AutoGluon not installed. Run:\n"
            "  .venv_ml/bin/pip install autogluon.tabular"
        )

    # --- Load data ---
    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    df = df[df[TARGET].notna()].copy()
    n_salary_total = len(df)

    # Grouped train/test split by player ID to prevent leakage from duplicate rows
    grouped_split = GROUP_COLUMN in df.columns
    if grouped_split:
        from sklearn.model_selection import GroupShuffleSplit
        groups = df[GROUP_COLUMN].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    features = CATEGORICAL + NUMERIC + [TARGET]
    all_df = df[features]
    train_df = train_df[features]
    test_df = test_df[features]
    print(f"Salary rows total: {n_salary_total}  (eval split: train={len(train_df)}, test={len(test_df)})")

    # Keep the production model family consistent with the calibration fold
    # models (salary_benchmark/calibration.py): exclude NN/foundation families
    # even though their dependencies may be installed in the venv.
    from salary_benchmark.calibration import EXCLUDED_MODEL_TYPES

    fit_kwargs = dict(
        time_limit=args.time,
        presets=args.preset,
        auto_stack=True,           # bagging + stacking for best accuracy
        excluded_model_types=EXCLUDED_MODEL_TYPES,
        # no ag_args_fit num_gpus — CPU only
    )

    print(f"\n=== AutoGluon TabularPredictor ===")
    print(f"  preset     : {args.preset}")
    print(f"  time_limit : {args.time}s per phase")
    print(f"  eval_metric: rmse")
    print(f"  auto_stack : True (bagging + stacking)")
    print(f"  GPU        : disabled (CPU only — Apple Silicon not supported)")

    # ------------------------------------------------------------------ #
    # Phase 1 — EVALUATION model (grouped holdout train split only)
    # ------------------------------------------------------------------ #
    print(f"\n=== PHASE 1: EVALUATION MODEL (train={len(train_df)} rows, "
          f"{'grouped' if grouped_split else 'row-level'} holdout) ===")
    eval_predictor = TabularPredictor(
        label=TARGET,
        path=ag_eval_dir,
        eval_metric="rmse",
        problem_type="regression",
    ).fit(TabularDataset(train_df), **fit_kwargs)

    # refit_full so the eval model follows the same procedure as production
    try:
        eval_predictor.refit_full()
    except Exception as e:
        print(f"  refit_full skipped: {e}")

    test_data = TabularDataset(test_df)
    y_pred = eval_predictor.predict(test_data)
    y_test = test_df[TARGET]
    eval_metrics = holdout_metrics(np.array(y_test), np.array(y_pred))

    print(f"\n=== GROUPED HOLDOUT RESULTS (n={len(test_df)}) ===")
    print(f"  Best model : {eval_predictor.model_best}")
    for k, v in eval_metrics.items():
        print(f"  {k:14s}: {v:.4f}")

    # --- Leaderboard (holdout test scores) ---
    print("\n=== LEADERBOARD (holdout test) ===")
    lb = eval_predictor.leaderboard(test_data, silent=True)
    lb.to_csv(out_dir / "autogluon_leaderboard.csv", index=False)
    print(lb[["model", "score_test", "score_val", "fit_time"]].head(15).to_string(index=False))

    # --- Feature importance (permutation with p-values, on holdout test) ---
    print("\n=== FEATURE IMPORTANCE ===")
    try:
        fi = eval_predictor.feature_importance(test_data, silent=True)
        fi.to_csv(out_dir / "autogluon_feature_importance.csv")
        print(fi.head(15).to_string())

        # Plot 1 — Feature importance bar chart
        fi_plot = fi.head(15).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, len(fi_plot) * 0.4)))
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in fi_plot["importance"]]
        ax.barh(fi_plot.index, fi_plot["importance"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Permutation Importance (drop in RMSE if feature shuffled)")
        ax.set_title(f"AutoGluon — Feature Importance ({eval_predictor.model_best})")
        fig.tight_layout()
        fig.savefig(plots_dir / "autogluon_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: autogluon_feature_importance.png")
    except Exception as e:
        print(f"  Feature importance failed: {e}")

    # --- Plot 2 — Leaderboard bar chart (model comparison) ---
    try:
        lb_plot = lb[lb["score_test"].notna()].sort_values("score_test", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, max(5, len(lb_plot) * 0.4)))
        colors = ["#e74c3c" if i == 0 else "steelblue" for i in range(len(lb_plot))]
        ax.barh(lb_plot["model"], -lb_plot["score_test"], color=colors, edgecolor="white")
        ax.set_xlabel("RMSE (lower is better)")
        ax.set_title("AutoGluon — Model Leaderboard (grouped holdout test)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(plots_dir / "autogluon_leaderboard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: autogluon_leaderboard.png")
    except Exception as e:
        print(f"  Leaderboard plot failed: {e}")

    # --- Plot 3 — Prediction error (actual vs predicted, holdout) ---
    try:
        y_pred_arr = np.array(y_pred)
        y_test_arr = np.array(y_test)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Actual vs Predicted
        axes[0].scatter(y_test_arr, y_pred_arr, alpha=0.4, s=15, color="steelblue")
        lims = [min(y_test_arr.min(), y_pred_arr.min()), max(y_test_arr.max(), y_pred_arr.max())]
        axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        axes[0].set_xlabel("Actual log_annual_fixed_eur")
        axes[0].set_ylabel("Predicted log_annual_fixed_eur")
        axes[0].set_title("Prediction Error (grouped holdout)")
        axes[0].legend()

        # Residuals distribution
        residuals = y_test_arr - y_pred_arr
        axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
        axes[1].set_xlabel("Residual (actual - predicted)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Residuals Distribution")

        fig.suptitle(f"AutoGluon — {eval_predictor.model_best}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "autogluon_prediction_error.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: autogluon_prediction_error.png")
    except Exception as e:
        print(f"  Prediction error plot failed: {e}")

    eval_best_model = eval_predictor.model_best
    try:
        eval_predictor.save_space()
    except Exception:
        pass
    del eval_predictor

    # ------------------------------------------------------------------ #
    # Phase 2 — PRODUCTION model (ALL salary rows, no holdout)
    # ------------------------------------------------------------------ #
    print(f"\n=== PHASE 2: PRODUCTION MODEL (all {n_salary_total} salary rows) ===")
    prod_predictor = TabularPredictor(
        label=TARGET,
        path=ag_prod_dir,
        eval_metric="rmse",
        problem_type="regression",
    ).fit(TabularDataset(all_df), **fit_kwargs)

    print("\n=== FIT SUMMARY (production) ===")
    prod_predictor.fit_summary(verbosity=1)

    print("\n=== REFIT FULL (production: retrain best model on all fit data) ===")
    try:
        refit_map = prod_predictor.refit_full()
        print(f"  Refit map: {refit_map}")
    except Exception as e:
        print(f"  refit_full skipped: {e}")

    # --- Plot 4 — Ensemble model structure (production) ---
    try:
        ensemble_fig = prod_predictor.plot_ensemble_model(prune_unused_nodes=True)
        if ensemble_fig is not None:
            ensemble_fig.savefig(plots_dir / "autogluon_ensemble_structure.png",
                                 dpi=150, bbox_inches="tight")
            plt.close(ensemble_fig)
            print(f"  Saved: autogluon_ensemble_structure.png")
    except Exception as e:
        print(f"  Ensemble plot skipped: {e}")

    # --- Save results JSON (holdout metrics + full reproducibility metadata) ---
    results = {
        # Top-level metrics = grouped holdout evaluation (kept for compatibility)
        "best_model": prod_predictor.model_best,
        "preset": args.preset,
        "rmse": eval_metrics["rmse"],
        "mae": eval_metrics["mae"],
        "r2": eval_metrics["r2"],
        "time_limit": args.time,
        # Reproducibility metadata
        "created_at": datetime.now(timezone.utc).isoformat(),
        "autogluon_version": autogluon.tabular.__version__,
        "input_file": str(input_path),
        "input_sha256": _sha256(input_path),
        "as_of_date": _as_of_date(input_path),
        "target": TARGET,
        "features": CATEGORICAL + NUMERIC,
        "excluded_model_types": EXCLUDED_MODEL_TYPES,
        "n_salary_rows_total": n_salary_total,
        "evaluation": {
            "split": "grouped_holdout" if grouped_split else "row_level_holdout",
            "grouped_split": grouped_split,
            "group_column": GROUP_COLUMN if grouped_split else None,
            "test_size": 0.2,
            "random_state": 42,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "best_model": eval_best_model,
            "model_path": ag_eval_dir,
            **eval_metrics,
        },
        "production": {
            "trained_on": "all_salary_rows",
            "n_train": n_salary_total,
            "best_model": prod_predictor.model_best,
            "model_path": ag_prod_dir,
        },
    }
    (out_dir / "autogluon_results.json").write_text(json.dumps(results, indent=2))

    # --- save_space: remove unnecessary artifacts ---
    print("\n=== SAVE SPACE ===")
    try:
        prod_predictor.save_space()
        disk_usage_mb = prod_predictor.disk_usage() / (1024 * 1024)
        print(f"  Disk usage after save_space: {disk_usage_mb:.1f} MB")
    except Exception as e:
        print(f"  save_space skipped: {e}")

    print(f"\nProduction model (all {n_salary_total} rows) saved to {ag_prod_dir}/")
    print(f"Evaluation model (holdout train split) saved to {ag_eval_dir}/")
    print(f"Leaderboard saved to {out_dir}/autogluon_leaderboard.csv")
    print(f"Results + metadata saved to {out_dir}/autogluon_results.json")


if __name__ == "__main__":
    main()
