"""PlayerEnricher — orchestrates both rails and writes data_full.csv.

Rail A (Capology salary): scrape once per unique player, concurrently, then
  left-join onto the CSV on (player_name, team_name, birth_date).
Rail B (Transfermarkt market value): one /market_value per unique player id,
  /transfers only when a club is missing from the value history; resolved per
  row by club id.

Used as a context manager so the local Transfermarkt server is started before
and stopped after enrichment:

    with PlayerEnricher(Config()) as enricher:
        enricher.run("players.csv", "data_full.csv")
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .cache import DiskCache
from .capology import CapologyClient
from .config import Config, ENRICH_COLUMNS
from .http_client import HttpClient
from .normalize import normalize_date, normalize_pid, normalize_text
from .transfermarkt import TransfermarktClient, TransfermarktServer

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "player_name", "nationality", "team_name", "birth_date",
    "transfermarkt_player_id", "transfermarkt_team_id",
}


class PlayerEnricher:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.cache = DiskCache(self.config.cache_dir)
        self.http = HttpClient(self.config, self.cache)
        self.server = TransfermarktServer(self.config)
        self._capology: CapologyClient | None = None
        self.tm = TransfermarktClient(self.http)

    # -- context manager: manage the TM server lifecycle --
    def __enter__(self) -> "PlayerEnricher":
        self.server.start()
        return self

    def __exit__(self, *exc):
        self.server.stop()
        self.http.close()

    # -- lazy Capology client (loads the 46k index once) --
    @property
    def capology(self) -> CapologyClient:
        if self._capology is None:
            self._capology = CapologyClient(self.config, self.http)
            print(f"Loaded Capology index: {len(self._capology.index_df)} players")
        return self._capology

    # -- main entry --
    def run(self, input_csv: str | None = None, output_csv: str | None = None) -> pd.DataFrame:
        input_csv = input_csv or self.config.input_csv
        output_csv = output_csv or self.config.output_csv

        players_df = self._read_and_prepare(input_csv)
        full_df = self._enrich_frame(players_df)
        full_df.to_csv(output_csv, index=False)
        print(f"Done. Output written to {output_csv} ({len(full_df)} rows)")
        return full_df

    def run_chunked(
        self,
        input_csv: str | None = None,
        output_csv: str | None = None,
        chunk_size: int = 1000,
        chunks_dir: str | Path | None = None,
    ) -> pd.DataFrame:
        input_csv = input_csv or self.config.input_csv
        output_csv = output_csv or self.config.output_csv
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        players_df = self._read_and_prepare(input_csv)
        total = len(players_df)
        out_path = Path(output_csv)
        chunks_path = Path(chunks_dir) if chunks_dir else out_path.with_suffix("").with_name(
            f"{out_path.stem}_chunks"
        )
        chunks_path.mkdir(parents=True, exist_ok=True)

        chunk_files = []
        n_chunks = (total + chunk_size - 1) // chunk_size
        print(
            f"Chunk/resume mode: {total} rows, chunk_size={chunk_size}, "
            f"chunks={n_chunks}, dir={chunks_path}"
        )

        for i, start in enumerate(range(0, total, chunk_size), start=1):
            end = min(start + chunk_size, total)
            chunk_file = chunks_path / f"chunk_{i:05d}.csv"
            chunk_files.append(chunk_file)
            if chunk_file.exists():
                print(f"Chunk {i}/{n_chunks}: skip existing {chunk_file} ({start}:{end})")
                continue

            print(f"Chunk {i}/{n_chunks}: enrich rows {start}:{end}")
            chunk_df = players_df.iloc[start:end].copy()
            enriched = self._enrich_frame(chunk_df)
            enriched.to_csv(chunk_file, index=False)
            print(f"Chunk {i}/{n_chunks}: written {chunk_file} ({len(enriched)} rows)")

        missing = [p for p in chunk_files if not p.exists()]
        if missing:
            raise RuntimeError(f"Missing chunk outputs: {missing}")

        full_df = pd.concat((pd.read_csv(p) for p in chunk_files), ignore_index=True)
        if len(full_df) != total:
            raise RuntimeError(
                f"chunk merge row count mismatch: expected {total}, got {len(full_df)}"
            )
        # Resumed chunk files may predate matcher fixes; audit the merged
        # frame so stale enrichment can never re-enter data_full.csv.
        full_df = self.audit_capology_dob(full_df)
        full_df.to_csv(output_csv, index=False)
        print(f"Done. Merged {len(chunk_files)} chunks into {output_csv} ({len(full_df)} rows)")
        return full_df

    def audit_capology_dob(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-enrichment audit: null out Capology enrichment whose scraped
        birth date contradicts the row's source birth date.

        The matcher validates DOB at match time, but rows resumed from older
        chunk files (or produced by older matcher versions) can still carry a
        wrong player's salary. This re-checks every enriched row against the
        (disk-cached) Capology page and nulls out mismatches.
        """
        if "capology_url" not in df.columns or "birth_date" not in df.columns:
            return df
        mask = df["capology_url"].notna()
        if not mask.any():
            return df

        df = df.copy()
        enrich_cols = [c for c in ENRICH_COLUMNS if c in df.columns]
        dob_cache: dict[str, str | None] = {}

        def capology_dob(url: str) -> str | None:
            if url not in dob_cache:
                try:
                    dob_cache[url] = self.capology.matcher._scrape(url).get("birth_date")
                except Exception:
                    dob_cache[url] = None
            return dob_cache[url]

        bad_idx = []
        for idx in df.index[mask]:
            source_dob = normalize_date(df.at[idx, "birth_date"])
            if not source_dob:
                continue
            cap_dob = capology_dob(df.at[idx, "capology_url"])
            if cap_dob and cap_dob != source_dob:
                bad_idx.append(idx)

        if bad_idx:
            print(f"  Capology DOB audit: nulling {len(bad_idx)} rows with mismatched birth dates")
            df.loc[bad_idx, enrich_cols] = None
        else:
            print(f"  Capology DOB audit: {int(mask.sum())} enriched rows checked, no mismatches")
        return df

    def _enrich_frame(self, players_df: pd.DataFrame) -> pd.DataFrame:
        full_df = self._add_capology(players_df)
        full_df = self.audit_capology_dob(full_df)
        full_df = self._add_market_value(full_df)

        assert len(full_df) == len(players_df), "enrichment must not change the CSV row count"

        # drop unused passthrough columns (e.g. stale source market_value); input untouched
        to_drop = [c for c in self.config.drop_columns if c in full_df.columns]
        if to_drop:
            full_df = full_df.drop(columns=to_drop)
        return full_df

    # -- steps --
    @staticmethod
    def _read_and_prepare(input_csv: str) -> pd.DataFrame:
        df = pd.read_csv(input_csv)
        rename = {}
        if "player_id" in df.columns:
            rename["player_id"] = "transfermarkt_player_id"
        if "team_id" in df.columns:
            rename["team_id"] = "transfermarkt_team_id"
        if rename:
            df = df.rename(columns=rename)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

        # Validate ALL rows with Pydantic schema (log warnings, don't reject)
        try:
            from schemas import PipelineInputRow
            errors = []
            for idx, row in df.iterrows():
                try:
                    PipelineInputRow(**{k: (None if pd.isna(v) else v) for k, v in row.items()
                                       if k in PipelineInputRow.model_fields})
                except Exception as e:
                    errors.append((idx, str(e)[:100]))
            if errors:
                print(f"  ⚠ Input validation: {len(errors)}/{len(df)} rows with warnings "
                      f"(first: row {errors[0][0]}: {errors[0][1]})")
        except ImportError:
            pass  # schemas not available — skip validation

        return df

    def _add_capology(self, players_df: pd.DataFrame) -> pd.DataFrame:
        unique_players = players_df.drop_duplicates(
            subset=["player_name", "team_name", "birth_date"]
        )
        total = len(unique_players)
        workers = self.config.capology_workers
        print(f"Capology: scraping {total} unique players "
              f"(from {len(players_df)} rows) with {workers} workers")

        counter = {"n": 0, "errors": 0}
        lock = threading.Lock()

        def _task(row_dict):
            rec = self.capology.enrich(row_dict)
            with lock:
                counter["n"] += 1
                if counter["n"] % 100 == 0:
                    print(f"  Capology {counter['n']}/{total}")
            return rec

        # touch the lazy client once (single-threaded) so the index loads safely
        _ = self.capology
        row_dicts = [r.to_dict() for _, r in unique_players.iterrows()]
        records = []
        empty_record = {c: None for c in ENRICH_COLUMNS}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_task, rd) for rd in row_dicts]
            for i, future in enumerate(futures):
                try:
                    records.append(future.result())
                except Exception as e:
                    counter["errors"] += 1
                    player_name = row_dicts[i].get("player_name", "unknown")
                    logger.warning(f"Capology enrichment failed for '{player_name}': {e}")
                    # Return empty enrichment so the row is not lost
                    join_keys = {
                        "player_name": row_dicts[i].get("player_name"),
                        "team_name": row_dicts[i].get("team_name"),
                        "birth_date": normalize_date(row_dicts[i].get("birth_date")),
                    }
                    records.append({**join_keys, **empty_record})

        if counter["errors"] > 0:
            print(f"  Capology: {counter['errors']}/{total} players failed (continued with empty data)")

        enrich_df = pd.DataFrame.from_records(records)
        return self._left_join(players_df, enrich_df)

    @staticmethod
    def _left_join(players_df: pd.DataFrame, enrich_df: pd.DataFrame) -> pd.DataFrame:
        def keys(frame):
            return (
                frame["player_name"].map(normalize_text),
                frame["team_name"].map(normalize_text),
                frame["birth_date"].map(normalize_date),
            )

        players_df = players_df.copy()
        players_df["_name_key"], players_df["_team_key"], players_df["_bd_key"] = keys(players_df)
        enrich_df["_name_key"], enrich_df["_team_key"], enrich_df["_bd_key"] = keys(enrich_df)
        enrich_df = enrich_df.drop(columns=["player_name", "team_name", "birth_date"])
        enrich_df = enrich_df.drop_duplicates(subset=["_name_key", "_team_key", "_bd_key"])
        merged = players_df.merge(enrich_df, on=["_name_key", "_team_key", "_bd_key"], how="left")
        return merged.drop(columns=["_name_key", "_team_key", "_bd_key"])

    def _add_market_value(self, players_df: pd.DataFrame) -> pd.DataFrame:
        # club ids each player id must resolve (from the CSV rows)
        needed_clubs: dict[str, set] = {}
        for _, row in players_df.iterrows():
            pid = normalize_pid(row.get("transfermarkt_player_id"))
            cid = normalize_pid(row.get("transfermarkt_team_id"))
            if pid:
                needed_clubs.setdefault(pid, set())
                if cid:
                    needed_clubs[pid].add(cid)

        ids = list(needed_clubs.keys())
        pool = self.config.tm_pool
        print(f"Transfermarkt: {len(ids)} unique ids — value history "
              f"(pool={pool}), transfers only when needed")

        with ThreadPoolExecutor(max_workers=pool) as ex:
            history_maps = dict(zip(ids, ex.map(self.tm.history_map, ids)))

        need_transfers = [
            pid for pid in ids
            if needed_clubs[pid] - set(history_maps.get(pid, {}).keys())
        ]
        print(f"  transfers needed for {len(need_transfers)}/{len(ids)} ids")
        with ThreadPoolExecutor(max_workers=pool) as ex:
            transfers_maps = dict(zip(need_transfers, ex.map(self.tm.transfers_map, need_transfers)))

        def row_mv(row):
            pid = normalize_pid(row.get("transfermarkt_player_id"))
            return self.tm.resolve(
                row.get("transfermarkt_team_id"),
                history_maps.get(pid, {}),
                transfers_maps.get(pid, {}),
            )

        players_df = players_df.copy()
        players_df["market_value_current_eur"] = players_df.apply(row_mv, axis=1)
        return players_df
