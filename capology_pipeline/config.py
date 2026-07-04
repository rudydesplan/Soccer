"""Configuration for the Capology / Transfermarkt enrichment pipeline.

All tunables live here; every field can be overridden via an environment
variable so a run can be reconfigured without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

# Enrichment columns produced for each player (order preserved in the output).
ENRICH_COLUMNS = [
    "capology_url",
    "status",
    "league",
    "annual_fixed_eur",
    "annual_bonus_eur",
    "annual_total_eur",
    "salary_currency",
    "gross_contract",
    "signed_date",
    "expiration_date",
    "release_clause_eur",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- Capology ---
    capology_base_url: str = "https://www.capology.com"
    capology_index_url: str = "https://www.capology.com/static/files/search_players.json"
    capology_workers: int = field(default_factory=lambda: _env_int("CAPOLOGY_WORKERS", 12))

    # --- Transfermarkt (local API, managed by TransfermarktServer) ---
    tm_host: str = field(default_factory=lambda: os.environ.get("TM_HOST", "127.0.0.1"))
    tm_port: int = field(default_factory=lambda: _env_int("TM_PORT", 8010))
    tm_pool: int = field(default_factory=lambda: _env_int("TM_POOL", 16))
    # Local TM server (uvicorn). It is I/O-bound (scrapes Transfermarkt live), so
    # multiple worker processes multiply concurrent throughput. 8 workers was the
    # measured sweet spot on a 10-core machine.
    tm_server_workers: int = field(default_factory=lambda: _env_int("TM_SERVER_WORKERS", 8))
    # If set, use this base URL and DO NOT manage a server (e.g. public instance).
    tm_base_url_override: str | None = field(
        default_factory=lambda: os.environ.get("TRANSFERMARKT_API")
    )

    # --- Vendored transfermarkt-api repo (auto-cloned + venv-bootstrapped) ---
    tm_repo_url: str = "https://github.com/felipeall/transfermarkt-api.git"
    tm_repo_commit: str = "bee4c49628b60f64d99137a675179ff2e6e843b4"  # pinned for reproducibility
    vendor_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "vendor")
    tm_python: str = field(default_factory=lambda: os.environ.get("TM_PYTHON", "python3.13"))

    # --- Cache / IO ---
    cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CACHE_DIR", str(_PROJECT_ROOT / ".cache")))
    )
    input_csv: str = "data (1).csv"  # raw Transfermarkt export shipped with the repo
    output_csv: str = "data_full.csv"

    # Columns to drop from the output (never from the input). The enrichment is a
    # real left-join that keeps every CSV column; list here only what to prune.
    # market_value is the CSV's own (stale) value — we drop it in favour of the
    # freshly fetched market_value_current_eur.
    drop_columns: tuple[str, ...] = ("market_value",)

    @property
    def tm_repo_dir(self) -> Path:
        return self.vendor_dir / "transfermarkt-api"

    @property
    def tm_base_url(self) -> str:
        """Base URL clients should hit (override wins over the managed server)."""
        return self.tm_base_url_override or f"http://{self.tm_host}:{self.tm_port}"
