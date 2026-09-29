"""Shared HTTP layer for every ingestor.

Design rules (TRD §4):
  * exponential backoff, 3 tries, then raise FetchError -> caller falls back to
    the bundled sample. A failed feed degrades a pill; it never breaks the app.
  * every raw response is cached to data/cache/ with a TTL, so a re-run of
    `make seed` costs nothing and works on a plane.

TLS note: we pin `verify=certifi.where()` explicitly. Some Python installs ship a
broken/stale default OpenSSL bundle (the macOS python.org build in particular),
and a CERTIFICATE_VERIFY_FAILED at seed time looks exactly like "the API is
down". Being explicit removes a whole class of confusing support reports.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import certifi
import httpx
from loguru import logger

from vayu_core.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_TIMEOUT = 60.0
USER_AGENT = "VAYU/0.1 (air-quality research prototype; contact via GitHub repo)"


class FetchError(RuntimeError):
    """A source could not be reached or returned something unusable."""


class RateLimiter:
    """Minimum spacing between real network calls to one host.

    Deliberately applied *after* the cache check, not before: a re-seed that is
    fully cached must not pay 20 minutes of sleeps for requests it never makes.
    """

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        gap = self.min_interval_s - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


@dataclass
class CacheEntry:
    path: Path
    fresh: bool


def _cache_path(key: str, suffix: str) -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}{suffix}"


def _cached(key: str, suffix: str, ttl: timedelta | None) -> CacheEntry | None:
    p = _cache_path(key, suffix)
    if not p.exists():
        return None
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
    return CacheEntry(path=p, fresh=(ttl is None or age < ttl))


def fetch_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    ttl: timedelta | None = timedelta(hours=6),
    tries: int = 3,
    timeout: float = DEFAULT_TIMEOUT,
    cache_key: str | None = None,
    limiter: "RateLimiter | None" = None,
) -> str:
    """GET with retry + on-disk caching. Raises FetchError after `tries`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key or f"{url}?{json.dumps(params or {}, sort_keys=True)}"

    hit = _cached(key, ".txt", ttl)
    if hit and hit.fresh:
        logger.debug(f"cache hit {url}")
        return hit.path.read_text()

    last: Exception | None = None
    for attempt in range(1, tries + 1):
        if limiter:
            limiter.wait()
        try:
            with httpx.Client(
                verify=certifi.where(),
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT, **(headers or {})},
            ) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                _cache_path(key, ".txt").write_text(r.text)
                return r.text
        except Exception as exc:  # noqa: BLE001 - any failure means "try again, then fall back"
            last = exc
            # A 429 is a *quota window*, not congestion: retrying 1s later just
            # burns another rejection. Back off past the window instead.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            wait = 65.0 if status == 429 else 2**attempt * 0.5
            logger.warning(
                f"fetch failed ({attempt}/{tries}) {url.split('?')[0]}: "
                f"{type(exc).__name__}{f' {status}' if status else ''} — retrying in {wait:.0f}s"
            )
            if attempt < tries:
                time.sleep(wait)

    # Stale cache beats no data: a day-old response with an honest "cached" pill
    # is far more useful than an empty map.
    if hit:
        logger.info(f"using stale cache for {url} after {tries} failures")
        return hit.path.read_text()

    raise FetchError(f"{url}: {last}") from last


def fetch_json(url: str, **kw: Any) -> Any:
    txt = fetch_text(url, **kw)
    try:
        return json.loads(txt)
    except json.JSONDecodeError as exc:
        raise FetchError(f"{url}: response was not JSON ({txt[:120]!r})") from exc
