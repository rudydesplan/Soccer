#!/usr/bin/env python
"""CLI for salary benchmarking.

Usage:
    # By player name (must exist in player_pool.csv):
    .venv_ml/bin/python run_benchmark.py --player "Lamine Yamal"

    # By player name with wide range:
    .venv_ml/bin/python run_benchmark.py --player "Erling Haaland" --range wide

    # Custom player (not in pool):
    .venv_ml/bin/python run_benchmark.py \\
        --main-position "Centre-Forward" \\
        --competition-id GB1 \\
        --competition-country England \\
        --age-months 299 \\
        --market-value 200000000
"""

from __future__ import annotations

import argparse
import json
import sys

from salary_benchmark.benchmark import benchmark_by_name, benchmark_player


def fmt_eur(v):
    if v is None:
        return "N/A"
    if v >= 1_000_000:
        return f"€{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"€{v/1_000:.0f}K"
    return f"€{v:.0f}"


def print_benchmark(result: dict):
    print("\n" + "=" * 60)
    print(f"  SALARY BENCHMARK — {result['player_name']}")
    print("=" * 60)
    print(f"  Position    : {result['main_position']}")
    print(f"  Competition : {result['competition_id']} ({result['competition_country']})")
    print(f"  Age         : {result['age_months']:.0f} months ({result['age_months']/12:.1f} yrs)" if result['age_months'] else "  Age         : N/A")
    print(f"  Market value: {fmt_eur(result['market_value_current_eur'])}")
    print()
    print(f"  Expected salary range ({result['range_width_used']}):")
    print(f"    Low    : {fmt_eur(result['expected_salary_low_eur'])}")
    print(f"    Median : {fmt_eur(result['expected_salary_median_eur'])}")
    print(f"    High   : {fmt_eur(result['expected_salary_high_eur'])}")
    print()
    if result["actual_salary_eur"] is not None:
        print(f"  Actual salary : {fmt_eur(result['actual_salary_eur'])}")
        print(f"  Status        : {result['salary_status'].upper()}")
        if result["salary_percentile"] is not None:
            print(f"  Percentile    : {result['salary_percentile']}th (among comparables)")
    print()
    fallback_labels = {
        "no_mv": "FALLBACK (no market value — wider, less precise range)",
        "no_mv_no_pos": "FALLBACK (no market value or position — much wider range)",
        "no_mv_no_age": "FALLBACK (no market value or age — much wider range)",
    }
    if result.get("model_used") in fallback_labels:
        print(f"  Model         : {fallback_labels[result['model_used']]}")
    print(f"  Confidence    : {result['benchmark_confidence'].upper()}")
    print(f"  Comparables   : {result['benchmark_n_comparables']} players "
          f"(level {result['comparable_level_used']}, "
          f"avg similarity {result['benchmark_avg_similarity']:.2f})")
    print()
    if result["comparable_players"]:
        print(f"  Top comparable players:")
        for p in result["comparable_players"][:5]:
            name = p.get("player_name", "?")
            pos  = p.get("main_position", "?")
            comp = p.get("competition_id", "?")
            mv   = fmt_eur(p.get("market_value_current_eur"))
            sal  = fmt_eur(p.get("annual_fixed_eur"))
            sim  = p.get("similarity_score", 0)
            print(f"    {name:<28} {pos:<20} {comp:<5} MV={mv:<10} Salary={sal:<10} sim={sim:.2f}")
    print("=" * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Salary benchmark for a football player.")
    parser.add_argument("--player", default=None, help="Player name (must exist in player_pool.csv)")
    parser.add_argument("--range", default="normal", choices=["normal", "wide"],
                        dest="range_width", help="Salary range width (normal=p25/p75, wide=p10/p90)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    # Manual player override
    parser.add_argument("--main-position",       default=None)
    parser.add_argument("--competition-id",      default=None)
    parser.add_argument("--competition-country", default=None)
    parser.add_argument("--age-months",          type=float, default=None)
    parser.add_argument("--market-value",        type=float, default=None)
    parser.add_argument("--actual-salary",       type=float, default=None)
    args = parser.parse_args(argv)

    try:
        if args.player:
            result = benchmark_by_name(args.player, range_width=args.range_width)
        else:
            if not all([args.main_position, args.competition_id,
                        args.competition_country]) or args.age_months is None:
                parser.error("Provide --player OR all of: --main-position, --competition-id, "
                             "--competition-country, --age-months (and ideally --market-value; "
                             "without it the weaker no-MV fallback model is used)")
            import numpy as np
            mv = args.market_value
            player = {
                "player_name":              "custom_player",
                "main_position":            args.main_position,
                "competition_id":           args.competition_id,
                "competition_country":      args.competition_country,
                "age_months":               args.age_months,
                "market_value_current_eur": mv,
                "log_market_value_current_eur": float(np.log1p(mv)) if mv is not None else None,
                "has_market_value":         1 if mv is not None else 0,
                "annual_fixed_eur":         args.actual_salary,
            }
            result = benchmark_player(player, range_width=args.range_width)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print_benchmark(result)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Hint: Build the player pool first with:", file=sys.stderr)
        print("  python build_model_features.py --mode pool --input data_full.csv --output player_pool.csv", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Hint: Use the ML venv: .venv_ml/bin/python run_benchmark.py ...", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
