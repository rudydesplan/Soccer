#!/usr/bin/env python
"""Rigorous model evaluation with cross-validation, segment analysis, and business metrics.

Produces:
    models/evaluation_report.json
    models/plots/evaluation_cv_scores.png
    models/plots/evaluation_segment_league.png
    models/plots/evaluation_segment_position.png
    models/plots/evaluation_segment_salary_tier.png
    models/plots/evaluation_calibration_coverage.png

Usage:
    .venv_ml/bin/python model_evaluation.py [--input data_test.csv] [--folds 5]
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

TARGET = "log_annual_fixed_eur"
CATEGORICAL = ["main_position", "nationality", "competition_id", "competition_country", "status"]
NUMERIC = [
    "season_start_year", "age_months", "contract_length_months",
    "contract_months_remaining", "contract_recency_months", "has_contract_dates", "has_market_value",
    "log_market_value_current_eur", "has_release_clause", "log_release_clause_eur",
]

POSITION_GROUPS = {
    "Centre-Forward": "attacker", "Left Winger": "attacker", "Right Winger": "attacker",
    "Second Striker": "attacker", "Attacking Midfield": "midfielder",
    "Central Midfield": "midfielder", "Defensive Midfield": "midfielder",
    "Left Midfield": "midfielder", "Right Midfield": "midfielder",
    "Centre-Back": "defender", "Left-Back": "defender", "Right-Back": "defender",
    "Goalkeeper": "goalkeeper",
}


def _salary_tier(log_salary: float) -> str:
    eur = np.expm1(log_salary)
    if eur < 1_000_000:
        return "<€1M"
    elif eur < 5_000_000:
        return "€1-5M"
    elif eur < 15_000_000:
        return "€5-15M"
    else:
        return ">€15M"


def _stratification_bins(df: pd.DataFrame) -> np.ndarray:
    """Create stratification labels combining league + salary tier."""
    country = df["competition_country"].fillna("Unknown").values
    tier = np.array([_salary_tier(v) for v in df[TARGET].values])
    return np.array([f"{c}_{t}" for c, t in zip(country, tier)])


def cross_validate(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """Run grouped K-fold cross-validation using AutoGluon.

    Groups by transfermarkt_player_id to prevent same-player leakage.
    """
    from autogluon.tabular import TabularPredictor, TabularDataset
    from salary_benchmark.calibration import EXCLUDED_MODEL_TYPES

    features = CATEGORICAL + NUMERIC + [TARGET]

    # Use grouped splits if player ID available
    has_groups = "transfermarkt_player_id" in df.columns
    if has_groups:
        from sklearn.model_selection import GroupKFold
        groups = df["transfermarkt_player_id"].values
        kf = GroupKFold(n_splits=n_folds)
        split_iter = kf.split(df, groups=groups)
    else:
        strat_labels = _stratification_bins(df)
        label_counts = pd.Series(strat_labels).value_counts()
        rare = set(label_counts[label_counts < n_folds].index)
        strat_labels = np.array(["_rare_" if l in rare else l for l in strat_labels])
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = skf.split(df, strat_labels)

    model_df = df[features].copy()
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}/{n_folds}")
        print(f"{'='*60}")

        train_data = TabularDataset(model_df.iloc[train_idx])
        test_data = TabularDataset(model_df.iloc[test_idx])

        predictor = TabularPredictor(
            label=TARGET,
            path=f"/tmp/ag_eval_fold_{fold_idx}",
            eval_metric="rmse",
            problem_type="regression",
            verbosity=0,
        ).fit(
            train_data,
            time_limit=300,  # 5 min per fold for evaluation
            presets="high_quality",
            auto_stack=True,
            # NN/foundation families excluded everywhere (installed in the
            # venv but excluded via excluded_model_types — see requirements-ml.txt)
            excluded_model_types=EXCLUDED_MODEL_TYPES,
        )

        y_pred = predictor.predict(test_data).values
        y_true = model_df.iloc[test_idx][TARGET].values

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # Business metrics
        actual_eur = np.expm1(y_true)
        pred_eur = np.expm1(y_pred)
        pct_error = np.abs(actual_eur - pred_eur) / actual_eur
        within_20 = float((pct_error <= 0.20).mean() * 100)
        within_50 = float((pct_error <= 0.50).mean() * 100)
        mape = float(pct_error.mean() * 100)

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "within_20_pct": within_20,
            "within_50_pct": within_50,
            "mape": mape,
            "best_model": predictor.model_best,
        })
        print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
        print(f"  Within ±20%: {within_20:.1f}%  Within ±50%: {within_50:.1f}%  MAPE: {mape:.1f}%")

        # Cleanup
        import shutil
        shutil.rmtree(f"/tmp/ag_eval_fold_{fold_idx}", ignore_errors=True)

    return {"folds": fold_results}


def segment_analysis(df: pd.DataFrame) -> dict:
    """Evaluate model performance by segment using the production model."""
    from salary_benchmark.model import predict_batch

    y_true = df[TARGET].values
    y_pred = predict_batch(df).values
    residuals = y_true - y_pred

    segments = {}

    # By league
    league_results = {}
    for country in df["competition_country"].dropna().unique():
        mask = df["competition_country"] == country
        if mask.sum() < 10:
            continue
        idx = mask.values
        league_results[country] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_true[idx], y_pred[idx]))),
            "mae": float(mean_absolute_error(y_true[idx], y_pred[idx])),
            "r2": float(r2_score(y_true[idx], y_pred[idx])),
            "mean_residual": float(residuals[idx].mean()),
        }
    segments["by_league"] = league_results

    # By position group
    pos_results = {}
    pos_groups = df["main_position"].map(lambda p: POSITION_GROUPS.get(p, "other"))
    for group in pos_groups.unique():
        mask = (pos_groups == group).values
        if mask.sum() < 10:
            continue
        pos_results[group] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
        }
    segments["by_position_group"] = pos_results

    # By salary tier
    tier_results = {}
    tiers = np.array([_salary_tier(v) for v in y_true])
    for tier in ["<€1M", "€1-5M", "€5-15M", ">€15M"]:
        mask = tiers == tier
        if mask.sum() < 10:
            continue
        tier_results[tier] = {
            "n": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
            "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "r2": float(r2_score(y_true[mask], y_pred[mask])),
            "mape": float((np.abs(np.expm1(y_true[mask]) - np.expm1(y_pred[mask])) / np.expm1(y_true[mask])).mean() * 100),
        }
    segments["by_salary_tier"] = tier_results

    return segments


def holdout_calibration_coverage(df: pd.DataFrame) -> dict | None:
    """Coverage on the same grouped holdout split used for training evaluation."""
    from autogluon.tabular import TabularPredictor, TabularDataset
    from salary_benchmark.calibration import load_calibration

    eval_path = Path("models/autogluon_holdout_eval")
    if not eval_path.exists() or "transfermarkt_player_id" not in df.columns:
        return None

    from sklearn.model_selection import GroupShuffleSplit
    groups = df["transfermarkt_player_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(df, groups=groups))
    test_df = df.iloc[test_idx]

    cal = load_calibration()
    y_true = test_df[TARGET].values
    y_pred = TabularPredictor.load(str(eval_path)).predict(TabularDataset(test_df)).values

    low_normal = y_pred + cal["residual_p25"]
    high_normal = y_pred + cal["residual_p75"]
    low_wide = y_pred + cal["residual_p10"]
    high_wide = y_pred + cal["residual_p90"]

    return {
        "scope": "grouped_holdout_eval_model",
        "n_test": len(test_df),
        "normal_range_actual_coverage": float(((y_true >= low_normal) & (y_true <= high_normal)).mean()),
        "wide_range_actual_coverage": float(((y_true >= low_wide) & (y_true <= high_wide)).mean()),
    }


def calibration_coverage(df: pd.DataFrame) -> dict:
    """Check if calibrated ranges actually cover the expected % of outcomes."""
    from salary_benchmark.model import predict_batch
    from salary_benchmark.calibration import load_calibration

    cal = load_calibration()
    y_true = df[TARGET].values
    y_pred = predict_batch(df).values

    # Normal range (p25/p75 → should cover ~50%)
    low_normal = y_pred + cal["residual_p25"]
    high_normal = y_pred + cal["residual_p75"]
    covered_normal = ((y_true >= low_normal) & (y_true <= high_normal)).mean()

    # Wide range (p10/p90 → should cover ~80%)
    low_wide = y_pred + cal["residual_p10"]
    high_wide = y_pred + cal["residual_p90"]
    covered_wide = ((y_true >= low_wide) & (y_true <= high_wide)).mean()

    return {
        "scope": "in_sample_production_model",
        "normal_range_expected_coverage": 0.50,
        "normal_range_actual_coverage": float(covered_normal),
        "wide_range_expected_coverage": 0.80,
        "wide_range_actual_coverage": float(covered_wide),
    }


def calibration_quality_label(normal_coverage: float) -> str:
    """Judge calibration quality against the 50% target for the normal range."""
    return "good" if abs(normal_coverage - 0.50) < 0.05 else "needs_adjustment"


def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(42)
    boot = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_low": float(np.percentile(boot, alpha * 100)),
        "ci_high": float(np.percentile(boot, (1 - alpha) * 100)),
    }


def plot_cv_scores(fold_results: list[dict], out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    folds = [f["fold"] for f in fold_results]

    for ax, metric, label in zip(axes, ["rmse", "mae", "r2"], ["RMSE", "MAE", "R²"]):
        values = [f[metric] for f in fold_results]
        ax.bar(folds, values, color="steelblue", edgecolor="white")
        ax.axhline(np.mean(values), color="red", linestyle="--", label=f"Mean: {np.mean(values):.4f}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(label)
        ax.set_title(f"{label} per fold")
        ax.legend()

    fig.suptitle("Cross-Validation Scores (5-fold, grouped by player)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation_cv_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segments(segments: dict, out_dir: Path):
    # League segment
    league = segments["by_league"]
    if league:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = sorted(league.keys(), key=lambda k: league[k]["rmse"])
        rmses = [league[k]["rmse"] for k in names]
        ns = [league[k]["n"] for k in names]
        bars = ax.barh(names, rmses, color="steelblue", edgecolor="white")
        for bar, n in zip(bars, ns):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"n={n}", va="center", fontsize=9)
        ax.set_xlabel("RMSE")
        ax.set_title("Model Performance by League")
        fig.tight_layout()
        fig.savefig(out_dir / "evaluation_segment_league.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Position segment
    pos = segments["by_position_group"]
    if pos:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = sorted(pos.keys(), key=lambda k: pos[k]["rmse"])
        rmses = [pos[k]["rmse"] for k in names]
        ax.barh(names, rmses, color="#2ecc71", edgecolor="white")
        ax.set_xlabel("RMSE")
        ax.set_title("Model Performance by Position Group")
        fig.tight_layout()
        fig.savefig(out_dir / "evaluation_segment_position.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Salary tier segment
    tier = segments["by_salary_tier"]
    if tier:
        fig, ax = plt.subplots(figsize=(8, 4))
        order = ["<€1M", "€1-5M", "€5-15M", ">€15M"]
        names = [t for t in order if t in tier]
        mapes = [tier[t]["mape"] for t in names]
        ax.bar(names, mapes, color="#e74c3c", edgecolor="white")
        ax.set_ylabel("MAPE (%)")
        ax.set_title("Prediction Error by Salary Tier")
        fig.tight_layout()
        fig.savefig(out_dir / "evaluation_segment_salary_tier.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rigorous model evaluation.")
    parser.add_argument("--input", default="data_test.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation (fast mode)")
    args = parser.parse_args(argv)

    out_dir = Path("models")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df[df[TARGET].notna()].copy()
    print(f"Evaluation dataset: {len(df)} rows with known salary")

    report = {"n_samples": len(df)}

    # --- Grouped holdout metrics (from train_autogluon_cpu.py evaluation phase) ---
    ag_results_path = out_dir / "autogluon_results.json"
    if ag_results_path.exists():
        try:
            ag_results = json.loads(ag_results_path.read_text())
            if "evaluation" in ag_results:
                report["grouped_holdout"] = ag_results["evaluation"]
                print("\n  Grouped holdout metrics (from training evaluation phase):")
                for metric in ["rmse", "mae", "r2", "mape", "within_20_pct", "within_50_pct"]:
                    if metric in ag_results["evaluation"]:
                        print(f"    {metric:15s}: {ag_results['evaluation'][metric]:.4f}")
        except json.JSONDecodeError:
            print("  ⚠ Could not parse models/autogluon_results.json")

    # --- Cross-validation ---
    if not args.skip_cv:
        print("\n" + "="*60)
        print("  CROSS-VALIDATION")
        print("="*60)
        cv_results = cross_validate(df, n_folds=args.folds)
        report["cross_validation"] = cv_results

        fold_results = cv_results["folds"]
        report["cv_summary"] = {
            "rmse": bootstrap_ci([f["rmse"] for f in fold_results]),
            "mae": bootstrap_ci([f["mae"] for f in fold_results]),
            "r2": bootstrap_ci([f["r2"] for f in fold_results]),
            "within_20_pct": bootstrap_ci([f["within_20_pct"] for f in fold_results]),
            "within_50_pct": bootstrap_ci([f["within_50_pct"] for f in fold_results]),
            "mape": bootstrap_ci([f["mape"] for f in fold_results]),
        }
        plot_cv_scores(fold_results, plots_dir)
        print(f"\n  CV Summary:")
        for metric in ["rmse", "mae", "r2", "within_20_pct", "within_50_pct", "mape"]:
            s = report["cv_summary"][metric]
            print(f"    {metric:15s}: {s['mean']:.4f} ± {s['std']:.4f}  [{s['ci_low']:.4f}, {s['ci_high']:.4f}]")

    # --- Segment analysis (using production model, IN-SAMPLE) ---
    print("\n" + "="*60)
    print("  SEGMENT ANALYSIS (in-sample, production model)")
    print("="*60)
    segments = segment_analysis(df)
    segments["scope"] = "in_sample_production_model"
    report["segments"] = segments
    plot_segments(segments, plots_dir)

    print("\n  By league:")
    for k, v in sorted(segments["by_league"].items(), key=lambda x: x[1]["rmse"]):
        print(f"    {k:12s}: RMSE={v['rmse']:.4f}  R²={v['r2']:.4f}  n={v['n']}")

    print("\n  By position group:")
    for k, v in sorted(segments["by_position_group"].items(), key=lambda x: x[1]["rmse"]):
        print(f"    {k:12s}: RMSE={v['rmse']:.4f}  R²={v['r2']:.4f}  n={v['n']}")

    print("\n  By salary tier:")
    for k, v in segments["by_salary_tier"].items():
        print(f"    {k:8s}: RMSE={v['rmse']:.4f}  MAPE={v['mape']:.1f}%  n={v['n']}")

    # --- Calibration coverage ---
    print("\n" + "="*60)
    print("  CALIBRATION COVERAGE")
    print("="*60)
    cal_cov = calibration_coverage(df)
    report["calibration_coverage"] = cal_cov
    holdout_cov = holdout_calibration_coverage(df)
    if holdout_cov:
        report["calibration_coverage_holdout"] = holdout_cov
        print(f"  Holdout normal coverage: {holdout_cov['normal_range_actual_coverage']*100:.1f}%")
        print(f"  Holdout wide coverage:   {holdout_cov['wide_range_actual_coverage']*100:.1f}%")
    print(f"  Normal range (p25-p75): expected 50%, actual {cal_cov['normal_range_actual_coverage']*100:.1f}%")
    print(f"  Wide range (p10-p90):   expected 80%, actual {cal_cov['wide_range_actual_coverage']*100:.1f}%")

    # Judge quality on the honest holdout coverage when available; in-sample
    # coverage is optimistic by construction and must not drive this label.
    quality_basis = holdout_cov if holdout_cov else cal_cov
    report["calibration_quality"] = {
        "label": calibration_quality_label(quality_basis["normal_range_actual_coverage"]),
        "basis": quality_basis["scope"],
        "normal_range_actual_coverage": quality_basis["normal_range_actual_coverage"],
    }
    print(f"  Quality: {report['calibration_quality']['label']} "
          f"(judged on {report['calibration_quality']['basis']})")

    # --- Business metrics (production model, IN-SAMPLE) ---
    # The production model trained on all salary rows, so these are optimistic
    # in-sample numbers. Honest out-of-sample numbers live in grouped_holdout
    # and cv_summary.
    print("\n" + "="*60)
    print("  BUSINESS METRICS (in-sample, production model)")
    print("="*60)
    from salary_benchmark.model import predict_batch
    y_true = df[TARGET].values
    y_pred = predict_batch(df).values
    actual_eur = np.expm1(y_true)
    pred_eur = np.expm1(y_pred)
    pct_error = np.abs(actual_eur - pred_eur) / actual_eur

    business = {
        "scope": "in_sample_production_model",
        "within_20_pct": float((pct_error <= 0.20).mean() * 100),
        "within_50_pct": float((pct_error <= 0.50).mean() * 100),
        "within_100_pct": float((pct_error <= 1.00).mean() * 100),
        "mape": float(pct_error.mean() * 100),
        "median_ape": float(np.median(pct_error) * 100),
    }
    report["business_metrics"] = business
    print(f"  Predictions within ±20% of actual: {business['within_20_pct']:.1f}%")
    print(f"  Predictions within ±50% of actual: {business['within_50_pct']:.1f}%")
    print(f"  MAPE: {business['mape']:.1f}%")
    print(f"  Median APE: {business['median_ape']:.1f}%")

    # --- Save report ---
    report_path = out_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n✓ Report saved to {report_path}")
    print(f"✓ Plots saved to {plots_dir}/evaluation_*.png")


if __name__ == "__main__":
    main()
