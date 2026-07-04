"""Ad-hoc benchmark: time Capology vs Transfermarkt rails on a sample CSV.

Usage:
    python bench.py [players_bench.csv] [--workers N] [--tm-pool M] [--no-clear]

WARNING: by default this deletes the ENTIRE disk cache (potentially 10GB+ of
downloaded pages) so timings reflect real network work. Pass --no-clear to
keep the cache and measure warm-cache performance instead.
"""

import sys
import time
import shutil
from pathlib import Path

import pandas as pd

from capology_pipeline.config import Config
from capology_pipeline.enricher import PlayerEnricher


def clear_cache(cfg):
    """Delete the ENTIRE cache directory for a cold-cache measurement."""
    if cfg.cache_dir.exists():
        shutil.rmtree(cfg.cache_dir)
    print(f"[bench] cleared cache dir {cfg.cache_dir}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    input_csv = args[0] if args else "players_bench.csv"

    workers = None
    tm_pool = None
    clear = True
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
        elif a.startswith("--tm-pool="):
            tm_pool = int(a.split("=")[1])
        elif a == "--no-clear":
            clear = False

    cfg = Config()
    if workers:
        cfg.capology_workers = workers
    if tm_pool:
        cfg.tm_pool = tm_pool

    print(f"[bench] input={input_csv} capology_workers={cfg.capology_workers} tm_pool={cfg.tm_pool}")

    if clear:
        clear_cache(cfg)

    with PlayerEnricher(cfg) as enr:
        players_df = enr._read_and_prepare(input_csv)
        n = len(players_df)

        # index load (one-off) timed separately
        t0 = time.time()
        _ = enr.capology  # triggers index load
        t_index = time.time() - t0

        t0 = time.time()
        cap_df = enr._add_capology(players_df)
        t_cap = time.time() - t0

        t0 = time.time()
        full_df = enr._add_market_value(cap_df)
        t_tm = time.time() - t0

    matched = full_df["capology_url"].notna().sum()
    mv = full_df["market_value_current_eur"].notna().sum()

    print("\n==================== BENCH RESULTS ====================")
    print(f"rows                     : {n}")
    print(f"capology index load      : {t_index:6.1f}s (one-off)")
    print(f"CAPOLOGY rail            : {t_cap:6.1f}s  ({t_cap/n*1000:.0f} ms/row)  matched={matched}/{n}")
    print(f"TRANSFERMARKT rail       : {t_tm:6.1f}s  ({t_tm/n*1000:.0f} ms/row)  mv_found={mv}/{n}")
    print(f"TOTAL (excl. index)      : {t_cap + t_tm:6.1f}s")
    print("=======================================================")


if __name__ == "__main__":
    main()
