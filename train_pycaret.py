#!/usr/bin/env python
"""Step 1 — PyCaret 4.0a8 compare_models → tune_model → coefficients → plots.

Regression plots available (4.0a8):
  residuals, residuals_distribution, prediction_error, learning,
  feature, permutation, shap_summary, shap_beeswarm

Usage:
    .venv_ml/bin/python train_pycaret.py [--input data_test.csv]

Outputs:
    models/pycaret_compare_results.csv
    models/pycaret_linear_results.csv
    models/pycaret_coefficients.csv
    models/pycaret_formula.txt
    models/pycaret_best_model.pkl
    models/pycaret_best_linear.pkl
    models/plots/                        (all regression diagnostic plots)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "log_annual_fixed_eur"
CATEGORICAL = ["main_position", "nationality", "competition_id", "competition_country", "status"]
NUMERIC = [
    "season_start_year", "age_months", "contract_length_months",
    "contract_months_remaining", "contract_recency_months", "has_contract_dates", "has_market_value",
    "log_market_value_current_eur", "has_release_clause", "log_release_clause_eur",
]

# All regression plots available in PyCaret 4.0a8
REGRESSION_PLOTS = [
    "residuals",
    "residuals_distribution",
    "prediction_error",
    "learning",
    "feature",
    "permutation",
    "shap_summary",
    "shap_beeswarm",
]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_test.csv")
    args = parser.parse_args(argv)

    out_dir = Path("models")
    plots_dir = out_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(args.input)
    df = df[df[TARGET].notna()].copy()

    # PyCaret 4.0a8 only supports row-level train/test split and KFold CV
    # (its fold_strategy argument is ignored by the native setup), so grouped
    # splits cannot be injected. To prevent same-player leakage — duplicated
    # players appearing under multiple clubs on both sides of a split — keep
    # exactly one row per player before fitting.
    if "transfermarkt_player_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["transfermarkt_player_id"], keep="first")
        print(f"Deduplicated players: {before} -> {len(df)} rows "
              f"(one row per transfermarkt_player_id, prevents split leakage)")

    features = CATEGORICAL + NUMERIC + [TARGET]
    df = df[features]
    print(f"Train rows: {len(df)}")

    # --- PyCaret 4.0a8 ---
    from pycaret.regression import RegressionExperiment

    exp = RegressionExperiment(
        target=TARGET,
        session_id=42,
        train_size=0.8,
        fold=5,
        normalize=True,
        transformation=True,
        feature_selection=True,
        verbose=True,
    )
    exp.fit(df)

    # --- compare_models: all models ---
    print("\n=== COMPARE ALL MODELS ===")
    compare_result = exp.compare_models(sort="RMSE")
    compare_df = exp.pull()
    compare_df.to_csv(out_dir / "pycaret_compare_results.csv", index=False)
    print(compare_df.to_string())
    best_overall = compare_result.best

    # --- Best linear model ---
    print("\n=== BEST LINEAR MODEL ===")
    linear_result = exp.compare_models(
        include=["ridge", "lasso", "en", "lr"],
        sort="RMSE",
    )
    linear_df = exp.pull()
    linear_df.to_csv(out_dir / "pycaret_linear_results.csv", index=False)
    print(linear_df.to_string())
    best_linear = linear_result.best

    # --- Tune best linear model ---
    print("\n=== TUNE BEST LINEAR MODEL ===")
    tuned_linear = exp.tune_model(best_linear, optimize="RMSE", n_iter=50)
    tuned_df = exp.pull()
    print(tuned_df.to_string())

    # choose_better manually
    from sklearn.metrics import mean_squared_error
    X_test = exp.get_config("X_test")
    y_test = exp.get_config("y_test")
    pipeline_tuned    = tuned_linear.pipeline
    pipeline_original = best_linear
    rmse_tuned    = np.sqrt(mean_squared_error(y_test, pipeline_tuned.predict(X_test)))
    rmse_original = np.sqrt(mean_squared_error(y_test, pipeline_original.predict(X_test)))
    print(f"\n  choose_better: tuned={rmse_tuned:.4f}  original={rmse_original:.4f}")
    final_linear = pipeline_tuned if rmse_tuned < rmse_original else pipeline_original
    print(f"  → keeping {'tuned' if rmse_tuned < rmse_original else 'original'}")

    # --- Extract coefficients ---
    print("\n=== COEFFICIENTS ===")
    try:
        model = final_linear[-1]
        feature_names = exp.get_config("X_train_transformed").columns.tolist()
        coef = model.coef_
        intercept = float(model.intercept_)

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
        }).sort_values("abs_coefficient", ascending=False)

        coef_df.to_csv(out_dir / "pycaret_coefficients.csv", index=False)
        print(f"Intercept: {intercept:.4f}")
        print(coef_df.head(20).to_string(index=False))

        lines = ["log_annual_fixed_eur =", f"  {intercept:+.4f}  (intercept)"]
        for _, row in coef_df.head(15).iterrows():
            lines.append(f"  {row['coefficient']:+.4f} × {row['feature']}")
        lines.append("  + ...")
        formula = "\n".join(lines)
        (out_dir / "pycaret_formula.txt").write_text(formula)
        print(f"\nFormula:\n{formula}")

    except Exception as e:
        print(f"Could not extract coefficients: {e}")

    # --- Plots via native PyCaret 4.0a8 plot_model ---
    print("\n=== PLOTS (Ridge — native PyCaret 4.0a8) ===")
    for plot_name in REGRESSION_PLOTS:
        out_path = str(plots_dir / f"ridge_{plot_name}.png")
        try:
            exp.plot_model(final_linear, plot=plot_name, save=out_path)
            print(f"  Saved: ridge_{plot_name}.png")
        except Exception as e:
            print(f"  Skipped {plot_name}: {e}")

    # --- evaluate_model: full diagnostic bundle for Ridge ---
    print("\n=== EVALUATE MODEL (Ridge) ===")
    try:
        figs = exp.evaluate_model(final_linear)
        for name, fig in figs.items():
            out_path = plots_dir / f"ridge_eval_{name}.png"
            try:
                fig.write_image(str(out_path))
                print(f"  Saved: ridge_eval_{name}.png")
            except Exception as e:
                print(f"  Could not save {name}: {e}")
    except Exception as e:
        print(f"  evaluate_model failed: {e}")

    # --- Plots via native PyCaret 4.0a8 plot_model for LightGBM ---
    print("\n=== PLOTS (LightGBM — native PyCaret 4.0a8) ===")
    for plot_name in REGRESSION_PLOTS:
        out_path = str(plots_dir / f"lgbm_{plot_name}.png")
        try:
            exp.plot_model(best_overall, plot=plot_name, save=out_path)
            print(f"  Saved: lgbm_{plot_name}.png")
        except Exception as e:
            print(f"  Skipped {plot_name}: {e}")

    # --- evaluate_model: full diagnostic bundle for LightGBM ---
    print("\n=== EVALUATE MODEL (LightGBM) ===")
    try:
        figs = exp.evaluate_model(best_overall)
        for name, fig in figs.items():
            out_path = plots_dir / f"lgbm_eval_{name}.png"
            try:
                fig.write_image(str(out_path))
                print(f"  Saved: lgbm_eval_{name}.png")
            except Exception as e:
                print(f"  Could not save {name}: {e}")
    except Exception as e:
        print(f"  evaluate_model failed: {e}")

    # --- Save models ---
    exp.save_model(best_overall, str(out_dir / "pycaret_best_model"))
    exp.save_model(tuned_linear, str(out_dir / "pycaret_best_linear"))
    print(f"\nAll plots saved to {plots_dir}/")
    print(f"Models saved to {out_dir}/")


if __name__ == "__main__":
    main()
