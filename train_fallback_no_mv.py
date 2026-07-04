#!/usr/bin/env python
"""Train a fallback model variant + its calibration.

Market value is the full model's strongest feature (permutation importance
0.596), but 924 of 19,476 pool players have no market value — and a further
handful also miss their position or age. Each fallback variant is an
AutoGluon model trained WITHOUT the missing features, paired with its own,
honestly wider, out-of-fold calibration:

    no_mv          — no market-value features (default)
    no_mv_no_pos   — additionally without main_position
    no_mv_no_age   — additionally without age_months

Three phases per variant:
  1. EVALUATION — the exact same grouped 80/20 holdout split as the full
     model (same group column, same random_state), so the fallback's metrics
     are directly comparable on identical test players. The comparison against
     the full model's stored holdout metrics is printed and saved.
  2. PRODUCTION — a fallback model trained on ALL salary rows, saved to
     models/autogluon_<variant>.
  3. CALIBRATION — grouped 5-fold out-of-fold residuals for the fallback
     feature set, saved to models/calibration_<variant>.json.

Usage:
    .venv_ml/bin/python train_fallback_no_mv.py [--variant no_mv]
        [--input data_test.csv] [--time 1800]

Outputs (per variant):
    models/autogluon_<variant>/            — production fallback model
    models/calibration_<variant>.json      — fallback residual calibration
    models/fallback_<variant>_results.json — holdout metrics + comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from salary_benchmark.calibration import CALIBRATION_PATHS, EXCLUDED_MODEL_TYPES, build_calibration
from salary_benchmark.model import VARIANTS, variant_features

TARGET = "log_annual_fixed_eur"
GROUP_COLUMN = "transfermarkt_player_id"

FALLBACK_VARIANTS = [v for v in VARIANTS if v != "full"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    parser.add_argument("--variant", default="no_mv", choices=FALLBACK_VARIANTS,
                        help="which fallback variant to train (default: no_mv)")
    parser.add_argument("--input", default="data_test.csv")
    parser.add_argument("--time", type=int, default=1800,
                        help="time limit in seconds per training phase (default: 30 min)")
    parser.add_argument("--preset", default="experimental_quality")
    args = parser.parse_args(argv)

    from autogluon.tabular import TabularPredictor, TabularDataset
    import autogluon.tabular

    variant = args.variant
    excluded_features = VARIANTS[variant]["excluded_features"]
    model_features = variant_features(variant)

    out_dir = Path("models")
    ag_prod_dir = str(VARIANTS[variant]["dir"])
    calibration_path = CALIBRATION_PATHS[variant]
    out_dir.mkdir(exist_ok=True)

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    df = df[df[TARGET].notna()].copy()
    n_salary_total = len(df)

    # Same grouped holdout split as train_autogluon_cpu.py (identical seed →
    # identical test players, so metrics are directly comparable).
    from sklearn.model_selection import GroupShuffleSplit
    groups = df[GROUP_COLUMN].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    features = model_features + [TARGET]
    all_df = df[features]
    train_df = df.iloc[train_idx][features]
    test_df = df.iloc[test_idx][features]
    print(f"Variant: {variant}")
    print(f"Salary rows: {n_salary_total} (train={len(train_df)}, test={len(test_df)})")
    print(f"Features: {len(model_features)} (removed: {excluded_features})")

    fit_kwargs = dict(
        time_limit=args.time,
        presets=args.preset,
        auto_stack=True,
        excluded_model_types=EXCLUDED_MODEL_TYPES,
    )

    # ------------------------------------------------------------------ #
    # Phase 1 — EVALUATION on the shared grouped holdout
    # ------------------------------------------------------------------ #
    print(f"\n=== PHASE 1: EVALUATION ({variant} model, grouped holdout) ===")
    eval_dir = f"/tmp/ag_{variant}_eval"
    eval_predictor = TabularPredictor(
        label=TARGET, path=eval_dir, eval_metric="rmse",
        problem_type="regression",
    ).fit(TabularDataset(train_df), **fit_kwargs)
    try:
        eval_predictor.refit_full()
    except Exception as e:
        print(f"  refit_full skipped: {e}")

    y_pred = eval_predictor.predict(TabularDataset(test_df)).values
    y_test = test_df[TARGET].values
    metrics = holdout_metrics(y_test, y_pred)
    eval_best = eval_predictor.model_best

    print(f"\n=== {variant.upper()} HOLDOUT RESULTS (n={len(test_df)}) ===")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v:.4f}")

    # Side-by-side vs the full model on the identical holdout
    comparison = {}
    full_results_path = out_dir / "autogluon_results.json"
    if full_results_path.exists():
        full_eval = json.loads(full_results_path.read_text()).get("evaluation", {})
        print(f"\n=== FULL vs {variant.upper()} (same grouped holdout, n={len(test_df)}) ===")
        print(f"  {'metric':14s}  {'full (with MV)':>15s}  {variant:>17s}")
        for k in ["rmse", "mae", "r2", "mape", "median_ape", "within_20_pct", "within_50_pct"]:
            if k in full_eval:
                print(f"  {k:14s}  {full_eval[k]:>15.4f}  {metrics[k]:>17.4f}")
                comparison[k] = {"full": full_eval[k], variant: metrics[k]}

    import shutil
    del eval_predictor
    shutil.rmtree(eval_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Phase 2 — PRODUCTION fallback (all salary rows)
    # ------------------------------------------------------------------ #
    print(f"\n=== PHASE 2: PRODUCTION FALLBACK (all {n_salary_total} salary rows) ===")
    prod_predictor = TabularPredictor(
        label=TARGET, path=ag_prod_dir, eval_metric="rmse",
        problem_type="regression",
    ).fit(TabularDataset(all_df), **fit_kwargs)
    try:
        prod_predictor.refit_full()
    except Exception as e:
        print(f"  refit_full skipped: {e}")
    try:
        prod_predictor.save_space()
    except Exception:
        pass

    # ------------------------------------------------------------------ #
    # Phase 3 — CALIBRATION (grouped OOF residuals, variant feature set)
    # ------------------------------------------------------------------ #
    print(f"\n=== PHASE 3: FALLBACK CALIBRATION (grouped out-of-fold) ===")
    calibration = build_calibration(
        out_path=calibration_path,
        exclude_features=excluded_features,
    )

    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "autogluon_version": autogluon.tabular.__version__,
        "input_file": str(input_path),
        "input_sha256": _sha256(input_path),
        "variant": variant,
        "preset": args.preset,
        "time_limit": args.time,
        "target": TARGET,
        "features": model_features,
        "excluded_features": excluded_features,
        "excluded_model_types": EXCLUDED_MODEL_TYPES,
        "n_salary_rows_total": n_salary_total,
        "evaluation": {
            "split": "grouped_holdout",
            "group_column": GROUP_COLUMN,
            "test_size": 0.2,
            "random_state": 42,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "best_model": eval_best,
            **metrics,
        },
        "comparison_vs_full_model": comparison,
        "production": {
            "trained_on": "all_salary_rows",
            "n_train": n_salary_total,
            "best_model": prod_predictor.model_best,
            "model_path": ag_prod_dir,
        },
        "calibration": {
            "path": str(calibration_path),
            "residual_p25": calibration["residual_p25"],
            "residual_p75": calibration["residual_p75"],
            "residual_p10": calibration["residual_p10"],
            "residual_p90": calibration["residual_p90"],
        },
    }
    results_path = out_dir / f"fallback_{variant}_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nFallback model saved to {ag_prod_dir}/")
    print(f"Fallback calibration saved to {calibration_path}")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
