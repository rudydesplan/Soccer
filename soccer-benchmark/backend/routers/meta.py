"""Model card / transparency endpoint.

Serves the numbers behind the "About the model" page from the training
artifacts in models/, so the page stays truthful when models are retrained
instead of hardcoding metrics that drift.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schemas import ErrorResponse, ModelCardResponse
from salary_benchmark.explain import FEATURE_LABELS
from salary_benchmark.model import VARIANTS, fallback_available

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODELS_DIR = _PROJECT_ROOT / "models"

_TOP_FEATURES = 8

router = APIRouter(tags=["meta"])

# Artifacts are written at training time and never change while the server
# runs, so the assembled card is cached after the first request.
_CARD_CACHE: dict | None = None


def _load_json(name: str) -> dict:
    path = _MODELS_DIR / name
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact {name} not found — retrain the model to generate it.",
        )
    return json.loads(path.read_text())


def _feature_importance() -> list[dict]:
    path = _MODELS_DIR / "autogluon_feature_importance.csv"
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    features = []
    for row in rows:
        name = row.get("") or row.get("Unnamed: 0") or row.get("feature")
        if not name:
            continue
        features.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "importance": round(float(row["importance"]), 4),
        })
    features.sort(key=lambda r: -r["importance"])
    return features[:_TOP_FEATURES]


def _coverage() -> dict:
    from salary_benchmark.benchmark import _load_pool

    try:
        pool = _load_pool()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Player pool not available on the server")

    return {
        "n_rows": int(len(pool)),
        "n_players": int(pool["transfermarkt_player_id"].nunique()),
        "n_with_salary": int(pool["annual_fixed_eur"].notna().sum()),
        "n_with_market_value": int(pool["market_value_current_eur"].notna().sum()),
        "countries": sorted(pool["competition_country"].dropna().unique().tolist()),
        "n_leagues": int(pool["competition_id"].nunique()),
        "seasons": sorted(int(s) for s in pool["season_start_year"].dropna().unique()),
        "n_positions": int(pool["main_position"].nunique()),
    }


def _build_card() -> dict:
    results = _load_json("autogluon_results.json")
    evaluation = results.get("evaluation") or _load_json("evaluation_report.json").get("grouped_holdout", {})
    calibration = _load_json("calibration.json")

    return {
        "model_name": results.get("best_model", "unknown"),
        "framework": "AutoGluon",
        "framework_version": results.get("autogluon_version"),
        "trained_at": results.get("created_at"),
        "target": results.get("target", "log_annual_fixed_eur"),
        "variants": {name: (name == "full" or fallback_available(name))
                     for name in ("full", *VARIANTS.keys())},
        "metrics": {
            "split": evaluation.get("split", "grouped_holdout"),
            "n_train": evaluation.get("n_train", 0),
            "n_test": evaluation.get("n_test", 0),
            "r2": round(evaluation["r2"], 4),
            "rmse_log": round(evaluation["rmse"], 4),
            "mae_log": round(evaluation["mae"], 4),
            "median_ape_pct": round(evaluation["median_ape"], 1),
            "within_20_pct": round(evaluation["within_20_pct"], 1),
            "within_50_pct": round(evaluation["within_50_pct"], 1),
        },
        "calibration": {
            "method": calibration.get("method", "unknown"),
            "n_folds": calibration.get("n_folds", 1),
            "n_samples": calibration.get("n_samples", 0),
            "residual_p10": round(calibration["residual_p10"], 4),
            "residual_p25": round(calibration["residual_p25"], 4),
            "residual_p50": round(calibration["residual_p50"], 4),
            "residual_p75": round(calibration["residual_p75"], 4),
            "residual_p90": round(calibration["residual_p90"], 4),
        },
        "top_features": _feature_importance(),
        "coverage": _coverage(),
    }


@router.get(
    "/meta/model-card",
    response_model=ModelCardResponse,
    responses={503: {"model": ErrorResponse, "description": "Model artifacts or pool missing"}},
)
def model_card():
    """Transparency data for the model card page.

    Everything is read from training artifacts (models/*.json, feature
    importance CSV) and the player pool — no hardcoded metrics. Holdout
    metrics use a grouped split (no player appears in both train and test).
    """
    global _CARD_CACHE
    if _CARD_CACHE is None:
        _CARD_CACHE = _build_card()
    return _CARD_CACHE
