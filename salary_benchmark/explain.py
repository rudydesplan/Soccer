"""SHAP explanations for individual salary predictions.

Answers "why is this player's expected salary high/low?" with per-feature
contributions — e.g. "market value €170M pushes the estimate up 15×, playing
in the Premier League adds 35%".

Implementation notes:
- AutoGluon predictors are ensembles (LightGBM + WeightedEnsemble), so
  TreeExplainer can't be used directly. We use the model-agnostic
  PermutationExplainer over `predictor.predict`.
- SHAP's tabular maskers need numeric arrays, so categorical features are
  integer-encoded against the pool's vocabulary before masking and decoded
  back to strings inside the prediction wrapper.
- The model predicts log1p(salary). SHAP values are therefore additive in
  log-space, which makes each contribution MULTIPLICATIVE on the salary
  scale: exp(shap) is the factor a feature applies to the estimate. We report
  both the raw log contribution and that multiplicative percentage.
- One explainer per variant is built lazily and cached, with a fixed-seed
  background sample of the pool (the "typical player" baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import _clean_record, _load_pool, select_model_variant
from .model import _load_predictor, variant_features

# Small background keeps latency ~1-2s per explanation. Larger backgrounds
# sharpen the baseline slightly but scale masked evaluations linearly.
_BACKGROUND_SIZE = 20
_MAX_EVALS = 64
_SEED = 42

_EXPLAINERS: dict[str, dict] = {}

# Feature names a sporting director understands.
FEATURE_LABELS = {
    "main_position": "Position",
    "nationality": "Nationality",
    "competition_id": "League",
    "competition_country": "League country",
    "status": "Contract status",
    "season_start_year": "Season",
    "age_months": "Age",
    "contract_length_months": "Contract length",
    "contract_months_remaining": "Contract months remaining",
    "contract_recency_months": "Contract recency",
    "has_contract_dates": "Contract dates known",
    "has_market_value": "Market value known",
    "log_market_value_current_eur": "Market value",
    "has_release_clause": "Release clause known",
    "log_release_clause_eur": "Release clause",
}


def _display_value(feature: str, value) -> str | None:
    """Human-readable value for a feature (undo log transforms etc.)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if feature in ("log_market_value_current_eur", "log_release_clause_eur"):
        eur = float(np.expm1(value))
        return f"€{eur / 1_000_000:.1f}M" if eur >= 1_000_000 else f"€{eur:,.0f}"
    if feature == "age_months":
        return f"{float(value) / 12:.1f} years"
    if feature in ("contract_length_months", "contract_months_remaining", "contract_recency_months"):
        return f"{float(value):.0f} months"
    if feature in ("has_contract_dates", "has_market_value", "has_release_clause"):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _build_explainer(variant: str) -> dict:
    """Build (and cache) the SHAP explainer bundle for one model variant."""
    import shap

    pool = _load_pool()
    features = variant_features(variant)
    predictor = _load_predictor(variant)

    cat_cols = [f for f in features if pool[f].dtype == object]
    vocab = {f: pd.Index(pool[f].dropna().unique()) for f in cat_cols}

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for f in cat_cols:
            codes = vocab[f].get_indexer(df[f]).astype(float)
            codes[df[f].isna().to_numpy()] = np.nan
            # Unseen categories (indexer -1, not NaN) map to NaN too:
            # AutoGluon treats them as missing rather than crashing.
            codes[codes < 0] = np.nan
            out[f] = codes
        return out.astype(float)

    def decode(X: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=features).astype(float)
        for f in cat_cols:
            codes = df[f]
            values = pd.Series(index=df.index, dtype=object)
            valid = codes.notna()
            if valid.any():
                idx = codes[valid].astype(int).clip(0, len(vocab[f]) - 1)
                values[valid] = vocab[f][idx]
            df[f] = values
        return df

    def predict_fn(X: np.ndarray) -> np.ndarray:
        return predictor.predict(decode(X)).to_numpy()

    background = encode(pool[features].sample(_BACKGROUND_SIZE, random_state=_SEED))
    masker = shap.maskers.Independent(background.to_numpy(), max_samples=_BACKGROUND_SIZE)
    explainer = shap.PermutationExplainer(predict_fn, masker)

    return {"explainer": explainer, "encode": encode, "features": features}


def _get_explainer(variant: str) -> dict:
    if variant not in _EXPLAINERS:
        _EXPLAINERS[variant] = _build_explainer(variant)
    return _EXPLAINERS[variant]


def explain_player(player: dict) -> dict:
    """Explain the salary prediction for one player.

    Routes to the same model variant the benchmark uses, so the explanation
    always describes the model that actually produced the player's range.

    Returns a dict with the log-space baseline/prediction (converted to EUR)
    and per-feature contributions sorted by impact.
    """
    player = _clean_record(dict(player))  # NaN → None so routing sees "missing"
    variant = select_model_variant(player)
    bundle = _get_explainer(variant)
    features = bundle["features"]

    row_raw = pd.DataFrame([{f: player.get(f) for f in features}])
    row = bundle["encode"](row_raw)

    sv = bundle["explainer"](row.to_numpy(), max_evals=_MAX_EVALS, silent=True)
    base_log = float(sv.base_values[0])
    contributions = sv.values[0]
    pred_log = base_log + float(contributions.sum())

    feature_rows = []
    for feature, shap_log in zip(features, contributions):
        shap_log = float(shap_log)
        feature_rows.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": _display_value(feature, row_raw.iloc[0][feature]),
            "shap_log": round(shap_log, 4),
            # exp(shap) is the multiplicative factor this feature applies to
            # the salary estimate; expressed as +35% / -20%.
            "pct_effect": round((float(np.exp(shap_log)) - 1.0) * 100.0, 1),
        })
    feature_rows.sort(key=lambda r: -abs(r["shap_log"]))

    return {
        "player_name": player.get("player_name", "unknown"),
        "model_used": variant,
        "base_salary_eur": int(round(float(np.expm1(base_log)))),
        "predicted_salary_eur": int(round(float(np.expm1(pred_log)))),
        "features": feature_rows,
    }


def explain_by_id(player_id: int) -> dict:
    """Explain the prediction for a pool player by row id."""
    pool = _load_pool()
    if player_id < 0 or player_id >= len(pool):
        raise ValueError(f"Player id {player_id} not found in player_pool.csv")
    return explain_player(pool.iloc[player_id].to_dict())


def explain_by_name(player_name: str) -> dict:
    """Explain the prediction for a pool player by name."""
    from .benchmark import resolve_player_by_name
    return explain_player(resolve_player_by_name(player_name))
