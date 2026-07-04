"""Disk cache — idempotent + resumable.

Every network payload is cached to disk keyed by url/id. A crash mid-run resumes
without re-downloading; parsers can be improved and re-run offline.
"""

from __future__ import annotations

import re
from pathlib import Path

_SUBDIRS = ("capology", "tm")


class DiskCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        for sub in _SUBDIRS:
            (self.cache_dir / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_name(key) -> str:
        """Filesystem-safe cache filename fragment."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(key)).strip("_")

    def get(self, subdir: str, key: str) -> str | None:
        path = self.cache_dir / subdir / key
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def set(self, subdir: str, key: str, text: str) -> None:
        """Atomic write (temp + rename) so an interrupted run can't corrupt cache."""
        path = self.cache_dir / subdir / key
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            print(f"  ! cache write failed for {path}: {exc}")
