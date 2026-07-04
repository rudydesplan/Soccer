#!/usr/bin/env python
"""Cross-validated model comparison — compare models on identical folds.

Compares AutoGluon, FLAML, and Ridge on identical K-fold splits — grouped by
transfermarkt_player_id when available (stratified fallback otherwise) — with
paired statistical tests for significance.

NOTE: This is NOT A/B testing (which requires live traffic splitting).
This is offline cross-validated comparison — the standard ML practice for
model selection before deployment.

Produces:
    models/model_comparison_results.json
    models/plots/model_comparison.png
    models/plots/model_comparison_per_fold.png

Usage:
    .venv_ml/bin/python model_comparison.py [--input data_test.csv] [--folds 5]
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
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

TARGET = "log_annual_fixed_eur"
CATEGORICAL = ["main_position", "nationality", "competition_id", "competition_country", "status"]
NUMERIC = [
    "season_start_year", "age_months", "contract_length_months",
    "contract_months_remaining", "contract_recency_months", "has_contract_dates", "has_market_value",
    "log_market_value_current_eur", "has_release_clause", "log_release_clause_eur",
]


def _stratification_bins(df: pd.DataFrame) -> np.ndarray:
    country = df["competition_country"].fillna("Unknown").values
    tier = np.where(df[TARGET] < np.log1p(1_000_000), "low",
           np.where(df[TARGET] < np.log1p(5_000_000), "mid", "high"))
    labels = np.array([f"{c}_{t}" for c, t in zip(country, tier)])
    counts = pd.Series(labels).value_counts()
    rare = set(counts[counts < 5].index)
    return np.array(["_rare_" if l in rare else l for l in labels])


def train_autogluon(train_df, test_df, fold_idx, groups=None):
    from autogluon.tabular import TabularPredictor, TabularDataset
    from salary_benchmark.calibration import EXCLUDED_MODEL_TYPES
    import shutil
    path = f"/tmp/model_comparison_ag_fold_{fold_idx}"
    predictor = TabularPredictor(
        label=TARGET, path=path, eval_metric="rmse",
        problem_type="regression", verbosity=0,
    ).fit(TabularDataset(train_df), time_limit=180, presets="high_quality",
          auto_stack=True,
          # same NN/foundation-family exclusions as production + calibration
          excluded_model_types=EXCLUDED_MODEL_TYPES)
    y_pred = predictor.predict(TabularDataset(test_df)).values
    shutil.rmtree(path, ignore_errors=True)
    return y_pred


def train_flaml(train_df, test_df, fold_idx, groups=None):
    from flaml import AutoML
    features = CATEGORICAL + NUMERIC
    X_train = train_df[features].copy()
    y_train = train_df[TARGET]
    X_test = test_df[features].copy()
    for col in CATEGORICAL:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    automl = AutoML()
    # Same time budget as AutoGluon for a fair comparison. Inner CV is grouped
    # by player id so duplicated players cannot leak across inner folds during
    # hyperparameter selection.
    fit_kwargs = dict(task="regression", metric="rmse",
                      time_budget=180, seed=42, verbose=0,
                      eval_method="cv", n_splits=3)
    if groups is not None:
        fit_kwargs.update(split_type="group", groups=groups)
    automl.fit(X_train, y_train, **fit_kwargs)
    return automl.predict(X_test)


def train_ridge(train_df, test_df, fold_idx, groups=None):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    features = CATEGORICAL + NUMERIC
    # Ridge cannot handle NaN — impute numerics (median) and encode missing
    # categories to -1, unlike the tree ensembles which handle NaN natively.
    ct = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), CATEGORICAL),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
    ])
    pipe = Pipeline([("prep", ct), ("model", Ridge(alpha=1.0))])
    pipe.fit(train_df[features], train_df[TARGET])
    return pipe.predict(test_df[features])


def evaluate_predictions(y_true, y_pred):
    actual_eur = np.expm1(y_true)
    pred_eur = np.expm1(y_pred)
    pct_error = np.abs(actual_eur - pred_eur) / actual_eur
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(pct_error.mean() * 100),
        "within_20_pct": float((pct_error <= 0.20).mean() * 100),
        "within_50_pct": float((pct_error <= 0.50).mean() * 100),
    }


def paired_test(scores_a, scores_b):
    """Wilcoxon signed-rank test for paired fold scores."""
    diff = np.array(scores_a) - np.array(scores_b)
    if np.all(diff == 0):
        return {"statistic": 0, "p_value": 1.0, "significant": False}
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b)
        return {"statistic": float(stat), "p_value": float(p), "significant": p < 0.05}
    except Exception:
        return {"statistic": None, "p_value": None, "significant": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-validated model comparison: AutoGluon vs FLAML vs Ridge.")
    parser.add_argument("--input", default="data_test.csv")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    out_dir = Path("models")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[df[TARGET].notna()].copy()
    features = CATEGORICAL + NUMERIC + [TARGET]
    print(f"Model comparison dataset: {len(df)} rows")

    # Grouped splits by player ID to prevent same-player leakage
    if "transfermarkt_player_id" in df.columns:
        from sklearn.model_selection import GroupKFold
        groups = df["transfermarkt_player_id"].values
        model_df = df[features].copy()
        kf = GroupKFold(n_splits=args.folds)
        split_iter = list(kf.split(model_df, groups=groups))
    else:
        groups = None
        model_df = df[features].copy()
        strat_labels = _stratification_bins(model_df)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        split_iter = list(skf.split(model_df, strat_labels))

    models = {
        "AutoGluon": train_autogluon,
        "FLAML": train_flaml,
        "Ridge": train_ridge,
    }

    all_results = {name: [] for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{args.folds}")
        print(f"{'='*60}")

        train_df = model_df.iloc[train_idx].copy()
        test_df = model_df.iloc[test_idx].copy()
        y_true = test_df[TARGET].values
        groups_train = groups[train_idx] if groups is not None else None

        for name, train_fn in models.items():
            print(f"  Training {name}...", end=" ", flush=True)
            try:
                y_pred = train_fn(train_df, test_df, fold_idx, groups=groups_train)
                metrics = evaluate_predictions(y_true, y_pred)
                metrics["fold"] = fold_idx
                all_results[name].append(metrics)
                print(f"RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                all_results[name].append({"fold": fold_idx, "rmse": None, "error": str(e)})

    # --- Summary ---
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)

    summary = {}
    for name in models:
        valid = [r for r in all_results[name] if r.get("rmse") is not None]
        if not valid:
            errors = {r.get("error") for r in all_results[name] if r.get("error")}
            print(f"\n  ⚠ {name}: FAILED on all folds — excluded from comparison. "
                  f"Errors: {sorted(errors)}")
            continue
        if len(valid) < args.folds:
            print(f"\n  ⚠ {name}: only {len(valid)}/{args.folds} folds succeeded")
        summary[name] = {
            "rmse_mean": float(np.mean([r["rmse"] for r in valid])),
            "rmse_std": float(np.std([r["rmse"] for r in valid])),
            "r2_mean": float(np.mean([r["r2"] for r in valid])),
            "mae_mean": float(np.mean([r["mae"] for r in valid])),
            "mape_mean": float(np.mean([r["mape"] for r in valid])),
            "within_20_mean": float(np.mean([r["within_20_pct"] for r in valid])),
            "within_50_mean": float(np.mean([r["within_50_pct"] for r in valid])),
            "folds_completed": len(valid),
        }
        print(f"\n  {name}:")
        print(f"    RMSE: {summary[name]['rmse_mean']:.4f} ± {summary[name]['rmse_std']:.4f}")
        print(f"    R²:   {summary[name]['r2_mean']:.4f}")
        print(f"    MAPE: {summary[name]['mape_mean']:.1f}%")
        print(f"    Within ±20%: {summary[name]['within_20_mean']:.1f}%")

    # --- Statistical tests ---
    # Pair scores BY FOLD ID: if model A failed on fold 2 and model B on fold 4,
    # only the folds where BOTH succeeded are compared (otherwise the "paired"
    # test would compare mismatched folds).
    print("\n  Paired tests (Wilcoxon signed-rank, fold-aligned):")
    tests = {}
    model_names = [n for n in models if n in summary]
    rmse_by_fold = {
        name: {r["fold"]: r["rmse"] for r in all_results[name] if r.get("rmse") is not None}
        for name in model_names
    }
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i+1:]:
            common_folds = sorted(set(rmse_by_fold[name_a]) & set(rmse_by_fold[name_b]))
            if len(common_folds) >= 3:
                valid_a = [rmse_by_fold[name_a][f] for f in common_folds]
                valid_b = [rmse_by_fold[name_b][f] for f in common_folds]
                test_result = paired_test(valid_a, valid_b)
                test_result["folds_compared"] = common_folds
                key = f"{name_a}_vs_{name_b}"
                tests[key] = test_result
                winner = name_a if np.mean(valid_a) < np.mean(valid_b) else name_b
                sig = "✓ significant" if test_result["significant"] else "✗ not significant"
                print(f"    {key}: p={test_result['p_value']:.4f} → {winner} wins ({sig}) "
                      f"[folds {common_folds}]")

    # --- Determine winner ---
    if summary:
        winner = min(summary.keys(), key=lambda k: summary[k]["rmse_mean"])
        print(f"\n  🏆 WINNER: {winner} (lowest mean RMSE among "
              f"{len(summary)}/{len(models)} models that completed)")
    else:
        winner = None

    # --- Plots ---
    if summary:
        # Comparison bar chart
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        names = list(summary.keys())
        for ax, metric, label in zip(axes, ["rmse_mean", "r2_mean", "within_20_mean"],
                                     ["RMSE (lower=better)", "R² (higher=better)", "Within ±20% (higher=better)"]):
            values = [summary[n][metric] for n in names]
            colors = ["#2ecc71" if n == winner else "steelblue" for n in names]
            ax.bar(names, values, color=colors, edgecolor="white")
            ax.set_ylabel(label)
            ax.set_title(label)
        fig.suptitle("Cross-Validated Model Comparison", fontweight="bold")
        fig.tight_layout()
        fig.savefig(plots_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Per-fold line chart
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in names:
            valid = [r for r in all_results[name] if r.get("rmse") is not None]
            folds = [r["fold"] for r in valid]
            rmses = [r["rmse"] for r in valid]
            ax.plot(folds, rmses, marker="o", label=name, linewidth=2)
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE per Fold — Model Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "model_comparison_per_fold.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Save results ---
    report = {
        "n_folds": args.folds,
        "n_samples": len(df),
        "models": summary,
        "statistical_tests": tests,
        "winner": winner,
        "per_fold": {name: all_results[name] for name in models},
    }
    report_path = out_dir / "model_comparison_results.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n✓ Results saved to {report_path}")
    print(f"✓ Plots saved to {plots_dir}/model_comparison*.png")


if __name__ == "__main__":
    main()
