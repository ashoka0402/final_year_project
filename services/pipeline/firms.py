"""NASA FIRMS active-fire detections (VIIRS). Free key required.

Feeds the Attributor (Phase 3): fires falling inside a ward's back-trajectory
cone are the evidence behind an `open_burning` attribution share, and their FRP
(fire radiative power) drives the emission estimate the plume model uses.

Fallback without a key: data/samples/fires_{city}.csv, the bundled 7-day extract.
The fires layer then shows a 'sample' pill with the extract's own timestamps —
App Flow §7 requires the disclosure rather than silently ageing data.

Free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from vayu_core.config import REPO_ROOT, CityConfig, get_settings

from .http import FetchError, RateLimiter, fetch_text

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Two feeds, and picking the wrong one silently returns nothing:
#   NRT ("near real time") only covers roughly the last two months.
#   SP  ("standard processing") is the archive, and is what a pinned demo clock
#       in the past needs. Asking NRT for 2025-11-03 yields an empty CSV, not an
#       error — the same silent-empty trap as the retired OpenAQ sensors.
SENSOR_NRT = "VIIRS_SNPP_NRT"
SENSOR_ARCHIVE = "VIIRS_SNPP_SP"

# Day-range caps differ per feed and the API only tells you by rejecting the
# request: the archive replies 400 "Invalid day range. Expects [1..5]", so a
# 10-day chunk silently loses that whole span to a warning.
MAX_DAY_RANGE_NRT = 10
MAX_DAY_RANGE_ARCHIVE = 5

# Fires are attributed to a city if they fall in a box around it: stubble
# burning that poisons Delhi happens in Punjab/Haryana, well outside the
# municipal bbox. 2 degrees ~ 220 km covers the transport range the
# trajectory module reasons over (24h at typical winter wind speeds).
SEARCH_PAD_DEG = 2.0

# 5000 transactions / 10 min. A window backfill is ~15 requests, so this is
# courtesy rather than necessity.
_limiter = RateLimiter(0.5)


def available() -> bool:
    return bool(get_settings().firms_api_key)


def sample_path(city: CityConfig):
    return REPO_ROOT / "data" / "samples" / f"fires_{city.id}.csv"


def _normalise(df: pd.DataFrame, city: CityConfig, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["frp"] = pd.to_numeric(df.get("frp"), errors="coerce")
    out["confidence"] = df.get("confidence", "n").astype(str)
    ts = pd.to_datetime(
        df["acq_date"].astype(str) + " " + df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        utc=True,
        errors="coerce",
    )
    out["acq_ts"] = ts
    out["sensor"] = df.get("instrument", "VIIRS")
    out["city"] = city.id
    out["source"] = source
    out = out.dropna(subset=["lat", "lon", "acq_ts"])
    # Deterministic id so re-seeding is idempotent (same pixel = same row).
    out["fire_id"] = (
        city.id
        + ":"
        + out["lat"].round(5).astype(str)
        + ","
        + out["lon"].round(5).astype(str)
        + "@"
        + out["acq_ts"].dt.strftime("%Y%m%d%H%M")
    )
    return out[["city", "fire_id", "lat", "lon", "frp", "confidence", "acq_ts", "sensor", "source"]].drop_duplicates("fire_id")


def _area(city: CityConfig) -> str:
    w, s, e, n = city.bbox
    return f"{w - SEARCH_PAD_DEG},{s - SEARCH_PAD_DEG},{e + SEARCH_PAD_DEG},{n + SEARCH_PAD_DEG}"


def _chunk(key: str, sensor: str, area: str, days: int, start: date | None) -> pd.DataFrame:
    url = f"{BASE}/{key}/{sensor}/{area}/{days}"
    if start is not None:
        url += f"/{start.isoformat()}"
    txt = fetch_text(url, ttl=timedelta(days=30) if start else timedelta(hours=3), limiter=_limiter)
    if "latitude" not in txt[:200]:
        # FIRMS answers 200 with an error sentence rather than a CSV header.
        raise FetchError(f"FIRMS: {txt[:110]!r}")
    df = pd.read_csv(io.StringIO(txt))
    return df if "latitude" in df.columns else pd.DataFrame()


def fetch_window(city: CityConfig, start: date, end: date) -> tuple[pd.DataFrame, str]:
    """Archive detections over [start, end], paged in 5-day chunks (the cap).

    Used when the demo clock is pinned to the past: the whole training window
    needs fire coverage so `upwind_fire_frp_24h` is a real feature rather than a
    column of zeros.
    """
    settings = get_settings()
    if not available():
        return pd.DataFrame(), "unavailable"

    area = _area(city)
    frames: list[pd.DataFrame] = []
    cur = start
    chunks = failures = 0
    while cur <= end:
        span = min(MAX_DAY_RANGE_ARCHIVE, (end - cur).days + 1)
        chunks += 1
        try:
            df = _chunk(settings.firms_api_key, SENSOR_ARCHIVE, area, span, cur)
            if not df.empty:
                frames.append(df)
        except (FetchError, ValueError) as exc:
            failures += 1
            logger.warning(f"[{city.id}] FIRMS archive {cur} (+{span}d): {exc}")
        cur += timedelta(days=span)

    # A handful of empty days is normal; losing a large share of the window is a
    # partial backfill wearing a success badge. Fires drive the burning
    # attribution, so say it loudly rather than under-count Punjab.
    if failures and failures / chunks > 0.2:
        logger.error(
            f"[{city.id}] FIRMS: {failures}/{chunks} chunks failed — fire coverage is "
            "incomplete and attribution will under-count open burning"
        )

    if not frames:
        return pd.DataFrame(), "unavailable"
    raw = pd.concat(frames, ignore_index=True)
    out = _normalise(raw, city, "firms-archive")
    sample_path(city).parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(sample_path(city), index=False)
    logger.info(
        f"[{city.id}] FIRMS archive: {len(out)} detections {start} → {end} "
        f"(total FRP {out['frp'].sum():.0f} MW)"
    )
    return out, "live"


def fetch_fires(city: CityConfig, days: int = 7) -> tuple[pd.DataFrame, str]:
    """Return (fires, status) where status is 'live' | 'sample' | 'unavailable'."""
    settings = get_settings()
    area = _area(city)

    if available():
        try:
            df = _chunk(settings.firms_api_key, SENSOR_NRT, area, min(days, MAX_DAY_RANGE_NRT), None)
            if not df.empty:
                out = _normalise(df, city, "firms-live")
                sample_path(city).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(sample_path(city), index=False)  # refresh the bundle
                logger.info(f"[{city.id}] FIRMS: {len(out)} live detections")
                return out, "live"
            logger.info(f"[{city.id}] FIRMS: no active fires in the last {days}d")
            return pd.DataFrame(), "live"
        except (FetchError, ValueError, KeyError) as exc:
            logger.warning(f"[{city.id}] FIRMS live fetch failed: {exc}")
    else:
        logger.info(f"[{city.id}] FIRMS_API_KEY not set — using bundled sample fires")

    p = sample_path(city)
    if p.exists():
        try:
            return _normalise(pd.read_csv(p), city, "sample"), "sample"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{city.id}] bundled fires unreadable: {exc}")

    # No key and no bundle: the layer hides rather than inventing fire pixels.
    # Fabricating a fire would fabricate an enforcement target.
    return pd.DataFrame(), "unavailable"
