"""Capology rail: page parsing, Splink matching, and per-player enrichment."""

from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from splink import DuckDBAPI, Linker, SettingsCreator, block_on
import splink.comparison_library as cl

from .config import Config, ENRICH_COLUMNS
from .http_client import HttpClient
from .normalize import (
    extract_capology_id,
    extract_flag_entity,
    normalize_date,
    normalize_text,
    parse_money,
    symbol_to_iso,
)

logging.getLogger("splink").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
# Parser — pure HTML/JS -> fields (no network, unit-testable)
# --------------------------------------------------------------------------- #
class CapologyParser:
    _MONEY_JS_RE = re.compile(
        r'"([a-z_0-9]+)"\s*:\s*accounting\.formatMoney\(\s*"(\d+)"\s*,\s*"([^"]*)"'
    )
    # League is not in data_active; it is in the summary sentence, e.g.
    #   "... remaining on his contract with Barcelona (La Liga), expiring ..."
    _LEAGUE_SENTENCE_RE = re.compile(r"contract with\s+[^(]+?\(([^)]+)\)")
    _STATUS_JS_RE = re.compile(
        r'"status"\s*:\s*"(?:<span[^>]*>)?\s*([A-Za-z ]+?)\s*(?:</span>)?"'
    )
    _DOB_JS_RE = re.compile(
        r"\.player-details\.dob'\)\.html\(\s*moment\(\s*\"(\d{4}-\d{2}-\d{2})\""
    )
    _RELEASE_JS_RE = re.compile(
        r"#release'\)\.html\(\s*accounting\.formatMoney\(\s*\"(\d+)\""
    )
    # #contract: "6-yrs/" + formatMoney(("100020000"/1e6),..)+'M'+' + '+formatMoney((("10420000"*"6")/1e6),..)+'M'
    _CONTRACT_JS_RE = re.compile(
        r"#contract'\)\.html\(\s*\"([^\"]*yrs/)\"\s*\+\s*accounting\.formatMoney\(\(\"(\d+)\""
        r"[^)]*\)[^']*'M'\s*\+\s*' \+ '\s*\+\s*accounting\.formatMoney\(\(\(\"(\d+)\"\*\"(\d+)\""
    )

    def parse(self, html: str, capology_url: str) -> dict:
        obj = self._first_data_active_object(html)
        dob_m = self._DOB_JS_RE.search(html)

        fields = {c: None for c in ENRICH_COLUMNS}
        fields["capology_url"] = capology_url
        # helpers for Stage-2 disambiguation (not exported):
        fields["club"] = None
        fields["birth_date"] = normalize_date(dob_m.group(1)) if dob_m else None

        fields["status"] = self._status(html, obj)
        fields["signed_date"], fields["expiration_date"] = self._signed_expiration(html)

        release = self._RELEASE_JS_RE.search(html)
        fields["release_clause_eur"] = parse_money(release.group(1)) if release else None
        fields["gross_contract"] = self._gross_contract(html)

        league = self._LEAGUE_SENTENCE_RE.search(html)
        fields["league"] = league.group(1).strip() if league else None

        if obj:
            money, symbols = {}, {}
            for key, val, sym in self._MONEY_JS_RE.findall(obj):
                if key not in money:
                    money[key] = int(val)
                    symbols[key] = sym
            fields["annual_fixed_eur"] = float(money["annual_gross_eur"]) if "annual_gross_eur" in money else None
            fields["annual_bonus_eur"] = float(money["bonus_gross_eur"]) if "bonus_gross_eur" in money else None
            fields["annual_total_eur"] = float(money["total_gross_eur"]) if "total_gross_eur" in money else None
            fields["salary_currency"] = symbol_to_iso(symbols.get("annual_gross_eur"))
            club = re.search(r"href='/club/([^/']+)/salaries", obj)
            fields["club"] = club.group(1).replace("-", " ") if club else None

        return fields

    @staticmethod
    def _first_data_active_object(html: str) -> str | None:
        idx = html.find("var data_active")
        if idx == -1:
            return None
        segment = html[idx : idx + 6000]
        brace = segment.find("{")
        if brace == -1:
            return None
        end = re.search(r"\}\s*,\s*\{|\}\s*\]", segment[brace:])
        return segment[brace : brace + (end.start() if end else len(segment))]

    def _status(self, html: str, obj: str | None) -> str | None:
        """active/loan -> data_active JS; inactive -> server-rendered DOM node."""
        if obj:
            m = self._STATUS_JS_RE.search(obj)
            if m:
                return m.group(1).strip().lower()
        node = BeautifulSoup(html, "lxml").select_one("span.status.player-details")
        if node:
            text = node.get_text(strip=True)
            return text.lower() if text else None
        return None

    @staticmethod
    def _signed_expiration(html: str):
        """SIGNED / EXPIRATION are server-rendered in the DOM detail grid."""
        soup = BeautifulSoup(html, "lxml")
        signed = expiration = None
        for block in soup.select("div.player-status.player-detail.grid"):
            title = block.select_one(".player-detail-title")
            value = block.select_one(".player-details")
            if not title or not value:
                continue
            label = title.get_text(" ", strip=True).lower()
            text = value.get_text(" ", strip=True) or None
            if label == "signed":
                signed = text
            elif label == "expiration":
                expiration = text
        return normalize_date(signed), normalize_date(expiration)

    def _gross_contract(self, html: str) -> str | None:
        m = self._CONTRACT_JS_RE.search(html)
        if not m:
            return None
        prefix, total_fixed, bonus_per_yr, years = m.groups()
        try:
            fixed_m = float(total_fixed) / 1_000_000
            bonus_total_m = (float(bonus_per_yr) * float(years)) / 1_000_000
        except (TypeError, ValueError):
            return None
        return f"{prefix}\u20ac{fixed_m:.1f}M + {bonus_total_m:.1f}M"


# --------------------------------------------------------------------------- #
# Matcher — Splink two-stage player -> capology url
# --------------------------------------------------------------------------- #
class CapologyMatcher:
    def __init__(self, index_df: pd.DataFrame, client: HttpClient, parser: CapologyParser):
        self.index_df = index_df
        self.client = client
        self.parser = parser

    def match_url(self, player: dict) -> str | None:
        candidates = self._prefilter(player)
        if candidates.empty:
            return None

        nat = normalize_text(player.get("nationality"))
        if nat is not None:
            narrowed = candidates[candidates["nationality_norm"] == nat]
            if not narrowed.empty:
                candidates = narrowed

        if len(candidates) == 1:
            only_url = candidates.iloc[0]["capology_url"]
            if self._candidate_matches_player(player, only_url):
                return only_url
            return None

        # Build a unique_id -> capology_url lookup so Splink results (which
        # return unique_id = raw link path) can be resolved to full URLs.
        uid_to_url = dict(zip(candidates["unique_id"], candidates["capology_url"]))

        # Stage 1: Splink on name + nationality
        source = pd.DataFrame([{
            "unique_id": "source_player",
            "player_name_norm": normalize_text(player.get("player_name")),
            "nationality_norm": normalize_text(player.get("nationality")),
        }])
        stage1 = candidates.reindex(columns=["unique_id", "player_name_norm", "nationality_norm"])
        best_uid = self._run_splink(
            source, stage1,
            blocking_cols=["player_name_norm"],
            comparisons=[cl.NameComparison("player_name_norm"), cl.ExactMatch("nationality_norm")],
        )
        best_url = uid_to_url.get(best_uid) if best_uid else None

        # Stage 2: team_name + birth_date (requires hydrating candidate pages)
        refined = self._disambiguate(player, candidates)
        for url in (refined, best_url):
            if url and self._candidate_matches_player(player, url):
                return url
        return None

    def _prefilter(self, player: dict) -> pd.DataFrame:
        name_norm = normalize_text(player.get("player_name"))
        if not name_norm:
            return self.index_df.iloc[0:0]

        exact = self.index_df[self.index_df["player_name_norm"] == name_norm].copy()
        if not exact.empty:
            return exact

        # Capology often uses fuller or alternate spellings:
        #   Junior Kroupi -> Eli Junior Kroupi
        #   Yann Bisseck -> Yann Aurel Bisseck
        #   Pio Esposito -> Francesco Pio Esposito
        #   Yéremy Pino -> Yeremi Pino
        # Use a conservative fallback candidate set on the last token, then let
        # nationality + Splink/disambiguation pick the winner.
        tokens = name_norm.split()
        if not tokens:
            return self.index_df.iloc[0:0]
        first = tokens[0]
        last = tokens[-1]
        name_series = self.index_df["player_name_norm"].fillna("")
        mask = name_series.str.contains(last, regex=False) | name_series.map(
            lambda x: SequenceMatcher(None, name_norm, x).ratio() >= 0.86
        )

        # Common nickname contraction in Brazilian/Portuguese names:
        #   Savinho -> Sávio
        # Keep it as a candidate only; birth date/team validation still decides.
        nickname = None
        if len(tokens) == 1 and name_norm.endswith("inho"):
            nickname = f"{name_norm[:-4]}io"
            mask = mask | (name_series == nickname)

        # Some Capology entries are mononyms while Transfermarkt has a fuller
        # display name, e.g. Johnny Cardoso -> Johnny. Only add this fallback
        # when the first-token candidate set is small enough to hydrate safely.
        first_token_mask = name_series.map(lambda x: first in (x or "").split())
        if int(first_token_mask.sum()) <= 25:
            mask = mask | first_token_mask

        candidates = self.index_df[mask].copy()
        if candidates.empty:
            return candidates

        token_set = set(tokens)
        candidates["_name_ratio"] = candidates["player_name_norm"].map(
            lambda x: SequenceMatcher(None, name_norm, x or "").ratio()
        )
        candidates["_token_overlap"] = candidates["player_name_norm"].map(
            lambda x: len(token_set & set((x or "").split())) / len(token_set)
        )
        keep_mask = (
            (candidates["_token_overlap"] >= 0.5)
            | (candidates["_name_ratio"] >= 0.86)
            | (
                (candidates["_token_overlap"] >= (1 / len(token_set)))
                & candidates["player_name_norm"].map(lambda x: len((x or "").split()) == 1)
            )
        )
        if nickname:
            keep_mask = keep_mask | (candidates["player_name_norm"] == nickname)
        candidates = candidates[keep_mask]
        return candidates.drop(columns=["_name_ratio", "_token_overlap"])

    def _candidate_matches_player(self, player: dict, url: str) -> bool:
        try:
            fields = self._scrape(url)
        except Exception:
            return False
        source_birth_date = normalize_date(player.get("birth_date"))
        if source_birth_date and fields.get("birth_date") != source_birth_date:
            return False
        source_team = normalize_text(player.get("team_name"))
        candidate_team = normalize_text(fields.get("club"))
        if source_team and candidate_team:
            source_tokens = set(source_team.split())
            candidate_tokens = set(candidate_team.split())
            if not (source_tokens & candidate_tokens):
                return False
        return True

    def _disambiguate(self, player: dict, candidates: pd.DataFrame) -> str | None:
        hydrated = []
        for _, row in candidates.iterrows():
            try:
                fields = self._scrape(row["capology_url"])
            except Exception:
                continue
            hydrated.append({
                "unique_id": row["capology_url"],
                "team_name_norm": normalize_text(fields.get("club")),
                "birth_date": fields.get("birth_date"),
            })
            time.sleep(0.5)
        if not hydrated:
            return None
        candidate_df = pd.DataFrame(hydrated)
        source_df = pd.DataFrame([{
            "unique_id": "source_player",
            "team_name_norm": normalize_text(player.get("team_name")),
            "birth_date": normalize_date(player.get("birth_date")),
        }])
        return self._run_splink(
            source_df, candidate_df,
            blocking_cols=["team_name_norm", "birth_date"],
            comparisons=[
                cl.ExactMatch("team_name_norm"),
                cl.DateOfBirthComparison("birth_date", input_is_string=True, invalid_dates_as_null=True),
            ],
        )

    def _scrape(self, url: str) -> dict:
        slug = self.client.cache.safe_name(url.rstrip("/").rsplit("/", 1)[-1]) + ".html"
        html = self.client.get_capology_text(url, slug)
        return self.parser.parse(html, url)

    @staticmethod
    def _run_splink(source_df, candidate_df, blocking_cols, comparisons) -> str | None:
        settings = SettingsCreator(
            link_type="link_only",
            blocking_rules_to_generate_predictions=[block_on(c) for c in blocking_cols],
            comparisons=comparisons,
            retain_intermediate_calculation_columns=True,
        )
        linker = Linker(
            [source_df, candidate_df], settings,
            db_api=DuckDBAPI(), set_up_basic_logging=False,
            input_table_aliases=["source", "capology"],
        )
        preds = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
        if preds.empty:
            return None
        preds = preds.sort_values(["match_probability", "match_weight"], ascending=False)
        best = preds.iloc[0]
        for col in ("unique_id_l", "unique_id_r"):
            val = best.get(col)
            if col in preds.columns and pd.notna(val) and val != "source_player":
                return val
        return None


# --------------------------------------------------------------------------- #
# Client — index loading + per-player enrichment
# --------------------------------------------------------------------------- #
class CapologyClient:
    def __init__(self, config: Config, client: HttpClient):
        self.config = config
        self.client = client
        self.parser = CapologyParser()
        self.index_df = self._load_index()
        self.matcher = CapologyMatcher(self.index_df, client, self.parser)

    def _load_index(self) -> pd.DataFrame:
        payload_text = self.client.get_capology_text(
            self.config.capology_index_url, "search_players.json"
        )
        import json
        payload = json.loads(payload_text)
        rows = []
        for p in payload:
            name, link, flag = p.get("name"), p.get("link"), p.get("flag")
            if not name or not link:
                continue
            rows.append({
                "unique_id": link,
                "capology_name": name,
                "player_name_norm": normalize_text(name),
                "nationality_norm": extract_flag_entity(flag),
                "capology_player_id": extract_capology_id(link),
                "capology_url": urljoin(self.config.capology_base_url, link),
            })
        return pd.DataFrame(rows).drop_duplicates(subset=["unique_id"]).reset_index(drop=True)

    def enrich(self, player: dict) -> dict:
        """player dict -> enrichment dict carrying the natural join keys."""
        join_keys = {
            "player_name": player.get("player_name"),
            "team_name": player.get("team_name"),
            "birth_date": normalize_date(player.get("birth_date")),
        }
        empty = {c: None for c in ENRICH_COLUMNS}

        url = self.matcher.match_url(player)
        if not url:
            return {**join_keys, **empty}
        try:
            scraped = self.matcher._scrape(url)
        except Exception as exc:
            print(f"  ! scrape failed for {url}: {exc}")
            return {**join_keys, **empty, "capology_url": url}

        # Final safety net: never keep enrichment whose Capology birth date
        # contradicts the source birth date (wrong-player match).
        capology_dob = scraped.get("birth_date")
        source_dob = join_keys["birth_date"]
        if capology_dob and source_dob and capology_dob != source_dob:
            print(f"  ! DOB mismatch for {url}: capology={capology_dob} "
                  f"source={source_dob} — dropping enrichment")
            return {**join_keys, **empty}

        return {**join_keys, **{c: scraped.get(c) for c in ENRICH_COLUMNS}}
