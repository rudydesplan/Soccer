"""OOP Capology + Transfermarkt player enrichment pipeline."""

from .config import Config
from .enricher import PlayerEnricher

__all__ = ["Config", "PlayerEnricher"]
