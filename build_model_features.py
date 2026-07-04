#!/usr/bin/env python
"""Build model features or a benchmark player pool from an enriched CSV.

Default usage:
    ./.venv/bin/python build_model_features.py \
        --input data_full.csv \
        --output data_test.csv

Player-pool usage:
    ./.venv/bin/python build_model_features.py \
        --input data_full.csv \
        --output player_pool.csv \
        --mode pool

Categorical columns are intentionally kept as text; encode them later inside
ML pipelines such as PyCaret, FLAML, or AutoGluon.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_DROP_COLUMNS = [
    "season_start_date",
    "player_name",
    "transfermarkt_team_id",
    "market_value_current_eur",
    "expiration_date",
    "signed_date",
    "gross_contract",
    "release_clause_eur",
    "salary_currency",
    "annual_fixed_eur",
    "annual_bonus_eur",
    "annual_total_eur",
    "league",
    "capology_url",
    "season",
    "competition_name",
    "team_name",
    "birth_date",
    "position",
]

_CONTRACT_YEARS_RE = re.compile(r"^\s*(\d+)\s*-\s*yrs?\b", re.IGNORECASE)
_SEASON_YEAR_RE = re.compile(r"(\d{4})")


def parse_season_start_year(season: object) -> float:
    """Extract 2025 from values like '2025-2026' or '2025/26'."""
    if pd.isna(season):
        return np.nan
    match = _SEASON_YEAR_RE.search(str(season))
    if not match:
        return np.nan
    return int(match.group(1))


def parse_season_start(season: object) -> pd.Timestamp | pd.NaT:
    """Convert '2025-2026' or '2025/26' to season start date 2025-07-01."""
    year = parse_season_start_year(season)
    if pd.isna(year):
        return pd.NaT
    return pd.Timestamp(year=int(year), month=7, day=1)


def parse_contract_length_months(gross_contract: object) -> float:
    """Parse values like '4-yrs/€54.0M + 13.5M' -> 48 months.

    This is the contract length encoded in Capology's gross-contract string,
    not date-derived remaining months.
    """
    if pd.isna(gross_contract):
        return np.nan
    match = _CONTRACT_YEARS_RE.search(str(gross_contract))
    if not match:
        return np.nan
    return float(int(match.group(1)) * 12)


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Fractional calendar months between two date series."""
    start = pd.to_datetime(start, errors="coerce")
    end = pd.to_datetime(end, errors="coerce")
    valid = start.notna() & end.notna()
    result = pd.Series(np.nan, index=start.index, dtype="float64")

    whole_months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)

    # pd.to_timedelta(..., unit="M") is intentionally unsupported, so use DateOffset row-wise.
    month_start = pd.Series(
        [s + pd.DateOffset(months=int(m)) if ok else pd.NaT
         for s, m, ok in zip(start, whole_months, valid)],
        index=start.index,
    )
    next_month = pd.Series(
        [d + pd.DateOffset(months=1) if pd.notna(d) else pd.NaT for d in month_start],
        index=start.index,
    )
    fraction = (end - month_start).dt.total_seconds() / (next_month - month_start).dt.total_seconds()
    result[valid] = whole_months[valid] + fraction[valid]
    return result.round(2)


def _rename_source_ids(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "player_id" in df.columns:
        rename["player_id"] = "transfermarkt_player_id"
    if "team_id" in df.columns:
        rename["team_id"] = "transfermarkt_team_id"
    return df.rename(columns=rename) if rename else df


def add_numeric_features(df: pd.DataFrame, as_of_date: object | None = None) -> pd.DataFrame:
    _REQUIRED_COLUMNS = {
        "season", "birth_date", "signed_date", "expiration_date",
        "gross_contract", "market_value_current_eur", "release_clause_eur",
        "annual_fixed_eur",
    }
    df = _rename_source_ids(df.copy())
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame missing required column(s): {sorted(missing)}. "
            f"Expected: {sorted(_REQUIRED_COLUMNS)}"
        )

    season_start = df["season"].map(parse_season_start)
    birth_date = pd.to_datetime(df["birth_date"], errors="coerce", utc=True).dt.tz_localize(None)
    signed_date = pd.to_datetime(df["signed_date"], errors="coerce", utc=True).dt.tz_localize(None)
    expiration_date = pd.to_datetime(df["expiration_date"], errors="coerce", utc=True).dt.tz_localize(None)
    as_of = pd.Timestamp.today().normalize() if as_of_date is None else pd.Timestamp(as_of_date).normalize()
    as_of_series = pd.Series(as_of, index=df.index)

    df["season_start_year"] = df["season"].map(parse_season_start_year).astype("Int64")
    df["season_start_date"] = season_start
    df["age_months"] = months_between(birth_date, season_start)
    df["contract_length_months"] = df["gross_contract"].map(parse_contract_length_months)
    df["contract_months_remaining"] = months_between(as_of_series, expiration_date)
    df["contract_recency_months"] = months_between(signed_date, season_start)
    df["has_contract_dates"] = (signed_date.notna() & expiration_date.notna()).astype(int)

    market_value = pd.to_numeric(df["market_value_current_eur"], errors="coerce")
    df["market_value_current_eur"] = market_value
    df["has_market_value"] = market_value.notna().astype(int)
    df["log_market_value_current_eur"] = np.log1p(market_value.clip(lower=0))

    release_clause = pd.to_numeric(df["release_clause_eur"], errors="coerce")
    df["release_clause_eur"] = release_clause
    df["has_release_clause"] = release_clause.notna().astype(int)
    df["log_release_clause_eur"] = np.log1p(release_clause.clip(lower=0))

    annual_fixed = pd.to_numeric(df["annual_fixed_eur"], errors="coerce")
    df["annual_fixed_eur"] = annual_fixed
    df["log_annual_fixed_eur"] = np.log1p(annual_fixed.clip(lower=0))

    return df


def parse_drop_columns(value: str | None) -> list[str]:
    if value is None:
        return MODEL_DROP_COLUMNS
    value = value.strip()
    if not value:
        return []
    return [col.strip() for col in value.split(",") if col.strip()]


def build_model_features(
    input_csv: str | Path,
    output_csv: str | Path,
    drop_columns: list[str] | None = None,
    as_of_date: object | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = add_numeric_features(df, as_of_date=as_of_date)

    to_drop = [col for col in (drop_columns or []) if col in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Validate output schema (all rows; warn, don't reject)
    try:
        from schemas import ModelFeatureRow
        errors = 0
        for _, row in df.iterrows():
            try:
                ModelFeatureRow(**{k: (None if pd.isna(v) else v)
                                  for k, v in row.items()
                                  if k in ModelFeatureRow.model_fields})
            except Exception:
                errors += 1
        if errors:
            print(f"  ⚠ Feature validation: {errors}/{len(df)} rows have warnings")
    except ImportError:
        pass

    df.to_csv(output_csv, index=False)
    return df


def build_player_pool(
    input_csv: str | Path,
    output_csv: str | Path,
    as_of_date: object | None = None,
) -> pd.DataFrame:
    """Build the richer player pool used by the benchmark/comparable engine."""
    df = pd.read_csv(input_csv)
    df = add_numeric_features(df, as_of_date=as_of_date)
    if "season_start_date" in df.columns:
        df = df.drop(columns=["season_start_date"])
    df.to_csv(output_csv, index=False)
    return df


def build_features(
    input_csv: str | Path,
    output_csv: str | Path,
    drop_columns: list[str] | None = None,
    as_of_date: object | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for model-feature generation."""
    return build_model_features(input_csv, output_csv, drop_columns, as_of_date=as_of_date)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build model features or player pool from enriched soccer CSV.")
    parser.add_argument("--input", default="data_full.csv", help="input enriched CSV")
    parser.add_argument("--output", default="data_test.csv", help="output CSV")
    parser.add_argument(
        "--mode",
        default="model",
        choices=["model", "pool"],
        help="model=data_test-style features; pool=player_pool with source columns + features",
    )
    parser.add_argument(
        "--drop-columns",
        default=None,
        help=(
            "model mode only: comma-separated columns to drop after feature creation; "
            "default drops modelling passthrough columns; pass '' to keep all"
        ),
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="date used to compute contract_months_remaining; default is today's date",
    )
    args = parser.parse_args(argv)

    # Validate --as-of-date early
    if args.as_of_date is not None:
        try:
            pd.Timestamp(args.as_of_date)
        except (ValueError, TypeError):
            print(f"Error: Invalid date format '{args.as_of_date}'. Use YYYY-MM-DD.", file=sys.stderr)
            sys.exit(1)

    try:
        if args.mode == "pool":
            out = build_player_pool(args.input, args.output, as_of_date=args.as_of_date)
            n_with_salary = out["annual_fixed_eur"].notna().sum()
            print(f"Wrote {args.output}: {len(out)} rows, {len(out.columns)} columns")
            print(f"  Players with known salary : {n_with_salary}")
            print(f"  Players without salary    : {out['annual_fixed_eur'].isna().sum()}")
        else:
            out = build_model_features(
                args.input,
                args.output,
                parse_drop_columns(args.drop_columns),
                as_of_date=args.as_of_date,
            )
            print(f"Wrote {args.output}: {len(out)} rows, {len(out.columns)} columns")

        # Write metadata sidecar
        import json
        from datetime import datetime
        as_of_used = args.as_of_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        meta = {
            "input": args.input,
            "output": args.output,
            "mode": args.mode,
            "as_of_date": str(as_of_used),
            "generated_at": datetime.now().isoformat(),
            "rows": len(out),
            "columns": len(out.columns),
        }
        meta_path = Path(args.output).with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  Metadata: {meta_path}")
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Cannot write output file: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
