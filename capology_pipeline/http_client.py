"""HTTP client wrapping niquests sessions + disk cache.

Two sessions:
  * capology — public site behind Cloudflare: HTTP/2 keep-alive + thread pool,
    NOT aggressive multiplexing (polite). Thread-safe.
  * transfermarkt — local API we control: HTTP/2 multiplexing for throughput.
"""

from __future__ import annotations

import json

import niquests
from niquests.packages.urllib3.util import Retry

from .cache import DiskCache
from .config import Config, REQUEST_HEADERS


def _retry() -> Retry:
    return Retry(
        total=4,
        backoff_factor=1.5,  # 0s, 1.5s, 3s, 6s ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


class HttpClient:
    def __init__(self, config: Config, cache: DiskCache):
        self.config = config
        self.cache = cache

        self.capology = niquests.Session(
            retries=_retry(), pool_maxsize=config.capology_workers
        )
        self.capology.headers.update(REQUEST_HEADERS)

        self.tm = niquests.Session(
            retries=_retry(), multiplexed=True, pool_maxsize=config.tm_pool
        )
        self.tm.headers.update(REQUEST_HEADERS)

    def close(self) -> None:
        for s in (self.capology, self.tm):
            try:
                s.close()
            except Exception:
                pass

    # -- Capology (text/HTML, cached) --
    def get_capology_text(self, url: str, cache_key: str, timeout: int = 60) -> str:
        cached = self.cache.get("capology", cache_key)
        if cached is not None:
            return cached
        try:
            resp = self.capology.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Capology request failed for {url}: {e}"
            ) from e
        self.cache.set("capology", cache_key, resp.text)
        return resp.text

    # -- Transfermarkt (JSON, cached, no throttle: local has no rate limit) --
    def get_tm_json(self, path: str, cache_key: str, timeout: int = 60) -> dict:
        cached = self.cache.get("tm", cache_key)
        if cached is not None:
            try:
                return json.loads(cached)
            except Exception:
                pass  # corrupt cache -> refetch
        try:
            resp = self.tm.get(f"{self.config.tm_base_url}{path}", timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            self.cache.set("tm", cache_key, text)
            return json.loads(text)
        except Exception as exc:
            print(f"  ! transfermarkt GET {path} failed: {exc}")
            return {}
