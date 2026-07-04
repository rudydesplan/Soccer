#!/usr/bin/env python
"""Patch capology_url=null rows in existing data_full_chunks/ files.

For every chunk CSV, finds rows where capology_url is null, re-runs the
Capology matcher with the improved matching logic, and fills in all Capology
enrichment columns in-place. The chunk file is overwritten atomically.

Usage:
    ./.venv/bin/python patch_capology_nulls.py [--chunks-dir data_full_chunks]
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from capology_pipeline.cache import DiskCache
from capology_pipeline.capology import CapologyClient
from capology_pipeline.config import Config, ENRICH_COLUMNS
from capology_pipeline.http_client import HttpClient
from capology_pipeline.normalize import normalize_date, normalize_text


def patch_chunks(chunks_dir: Path, workers: int, config: Config) -> None:
    chunk_files = sorted(chunks_dir.glob("chunk_*.csv"))
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return

    cache = DiskCache(config.cache_dir)
    http = HttpClient(config, cache)
    cap = CapologyClient(config, http)
    print(f"Loaded Capology index: {len(cap.index_df)} players")

    total_null = 0
    total_fixed = 0

    for chunk_file in chunk_files:
        df = pd.read_csv(chunk_file)
        null_mask = df["capology_url"].isna()
        n_null = int(null_mask.sum())
        if n_null == 0:
            continue

        total_null += n_null
        null_rows = df[null_mask].copy()
        print(f"\n{chunk_file.name}: {n_null} null rows to patch")

        counter = {"n": 0}
        lock = threading.Lock()

        def _task(row_dict: dict) -> dict:
            rec = cap.enrich(row_dict)
            with lock:
                counter["n"] += 1
                if counter["n"] % 50 == 0:
                    print(f"  {chunk_file.name}: patched {counter['n']}/{n_null}")
            return rec

        row_dicts = [r.to_dict() for _, r in null_rows.iterrows()]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            records = list(ex.map(_task, row_dicts))

        enrich_df = pd.DataFrame.from_records(records)

        # Build join keys on the patch frame
        def _keys(frame: pd.DataFrame):
            return (
                frame["player_name"].map(normalize_text),
                frame["team_name"].map(normalize_text),
                frame["birth_date"].map(normalize_date),
            )

        enrich_df["_nk"], enrich_df["_tk"], enrich_df["_bk"] = _keys(enrich_df)
        enrich_df = enrich_df.drop(columns=["player_name", "team_name", "birth_date"])
        enrich_df = enrich_df.drop_duplicates(subset=["_nk", "_tk", "_bk"])

        df["_nk"], df["_tk"], df["_bk"] = _keys(df)

        # Only update rows that were null
        null_idx = df.index[null_mask]
        patch_sub = df.loc[null_idx].merge(
            enrich_df, on=["_nk", "_tk", "_bk"], how="left", suffixes=("_old", "")
        )

        for col in ENRICH_COLUMNS:
            if col in patch_sub.columns:
                df[col] = df[col].astype(object)
                # Only overwrite with non-null patch values (don't erase partial data)
                new_vals = patch_sub[col].values
                for i, idx in enumerate(null_idx):
                    if pd.notna(new_vals[i]):
                        df.at[idx, col] = new_vals[i]

        df = df.drop(columns=["_nk", "_tk", "_bk"])

        fixed = int(df.loc[null_idx, "capology_url"].notna().sum())
        total_fixed += fixed
        print(f"  {chunk_file.name}: fixed {fixed}/{n_null}")

        # Atomic write
        tmp = chunk_file.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(chunk_file)

    http.close()
    print(f"\nDone. Total null={total_null}, fixed={total_fixed}, "
          f"still null={total_null - total_fixed}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Patch capology_url=null rows in chunk files.")
    parser.add_argument("--chunks-dir", default="data_full_chunks", help="directory of chunk CSVs")
    parser.add_argument("--workers", type=int, default=None, help="Capology thread workers")
    args = parser.parse_args(argv)

    config = Config()
    if args.workers is not None:
        config.capology_workers = args.workers

    patch_chunks(Path(args.chunks_dir), config.capology_workers, config)


if __name__ == "__main__":
    main()
