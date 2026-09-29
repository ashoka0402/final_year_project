"""OpenAQ v3 — measured station observations. Requires a free key.

When OPENAQ_API_KEY is present this becomes the *preferred* source for station
identity and history, because it is measured rather than modelled: it upgrades
the historical series from CAMS reanalysis to real CPCB/DPCC observations.
Without a key, every call here is skipped and the seeder falls back to
cpcb.py (identity + current) + airquality.py (CAMS history).

Free key: https://openaq.org — no cost, instant.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from vayu_core.config import CityConfig, get_settings
from vayu_core.db import SOURCE_OPENAQ
from vayu_core.qc import screen

from .http import FetchError, RateLimiter, fetch_json

BASE = "https://api.openaq.org/v3"

# OpenAQ parameter name -> ours.
# Narrowed to what the science actually consumes: pm25 is the forecast target,
# pm10 and no2 are features (and no2 carries the traffic/industry signal in
# attribution). Delhi stations expose ~18 sensors each including wind and
# temperature; fetching all of them would triple the backfill for data we get
# from Open-Meteo anyway.
PARAM_MAP = {"pm25": "pm25", "pm10": "pm10", "no2": "no2"}

# OpenAQ's free tier allows ~60 req/min. A backfill is ~1000 requests, and a 429
# storm would surface as *partial history* (the worst failure mode for a
# forecaster) rather than a clean error, so we stay a margin under it.
_limiter = RateLimiter(1.1)


def available() -> bool:
    return bool(get_settings().openaq_api_key)


def _headers() -> dict[str, str]:
    return {"X-API-Key": get_settings().openaq_api_key}


def _live_sensors(location_id: int, start: date, end: date) -> dict[str, int]:
    """Pick, per parameter, the sensor that actually covers [start, end].

    Why this needs its own request: many CPCB stations expose several sensors
    for one parameter — R K Puram has two pm25, ids 35 and 12234787. The
    `/locations` payload carries no coverage dates, so any id-based tie-break is
    a coin flip. It is worse than a coin flip: sensor 35 ran 2016→2018 and
    returns zero rows, so "lowest id" deterministically selects the dead
    instrument and yields a silently *partial* backfill — the worst failure mode
    for a forecaster, because it looks like success.

    `/locations/{id}/sensors` returns datetimeFirst/datetimeLast, so we select on
    real coverage and skip retired sensors entirely. That costs one request per
    location and saves far more by never querying dead sensors.
    """
    try:
        payload = fetch_json(
            f"{BASE}/locations/{location_id}/sensors",
            headers=_headers(),
            ttl=timedelta(days=7),
            limiter=_limiter,
        )
    except FetchError as exc:
        logger.warning(f"OpenAQ sensors for location {location_id}: {exc}")
        return {}

    best: dict[str, tuple[int, str]] = {}
    for s in payload.get("results", []) or []:
        p = ((s.get("parameter") or {}).get("name") or "").lower()
        if p not in PARAM_MAP:
            continue
        first = ((s.get("datetimeFirst") or {}) or {}).get("utc")
        last = ((s.get("datetimeLast") or {}) or {}).get("utc")
        if not first or not last:
            continue
        # Keep only sensors whose coverage overlaps the window we want.
        if first[:10] > end.isoformat() or last[:10] < start.isoformat():
            continue
        param = PARAM_MAP[p]
        if param not in best or last > best[param][1]:
            best[param] = (int(s["id"]), last)
    return {k: v[0] for k, v in best.items()}


def fetch_locations(city: CityConfig, start: date, end: date) -> pd.DataFrame:
    """Stations inside the city bbox that have sensors covering [start, end]."""
    if not available():
        return pd.DataFrame()

    # bbox is min X, min Y, max X, max Y in WGS84 (lon/lat order) — our "wsen".
    payload = fetch_json(
        f"{BASE}/locations",
        params={"bbox": city.bbox_str("wsen"), "limit": 1000},
        headers=_headers(),
        ttl=timedelta(days=1),
        limiter=_limiter,
    )
    results = payload.get("results", []) or []
    logger.info(f"[{city.id}] OpenAQ: {len(results)} locations in bbox — checking sensor coverage")

    rows = []
    skipped = 0
    for loc in results:
        coords = loc.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue
        sensors = _live_sensors(int(loc["id"]), start, end)
        if not sensors:
            skipped += 1
            continue
        rows.append(
            {
                "city": city.id,
                "station_id": f"{city.id}:oaq{loc['id']}",
                "name": (loc.get("name") or f"OpenAQ {loc['id']}").strip(),
                "lat": float(lat),
                "lon": float(lon),
                "provider": f"OpenAQ · {(loc.get('provider') or {}).get('name', 'unknown')}",
                "first_seen": None,
                "last_seen": None,
                "sensors": sensors,
            }
        )
    logger.info(
        f"[{city.id}] OpenAQ: {len(rows)} stations with live coverage "
        f"({skipped} skipped — retired or outside the window)"
    )
    return pd.DataFrame(rows)


def fetch_measurements(city: CityConfig, stations: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Hourly measurements per sensor over [start, end].

    OpenAQ paginates hard; we cap pages per sensor so one slow city cannot stall
    a seed. Partial history is fine — the Forecaster tolerates gaps.
    """
    if not available() or stations.empty:
        return pd.DataFrame()

    total_sensors = sum(len(getattr(st, "sensors", {}) or {}) for st in stations.itertuples())
    logger.info(
        f"[{city.id}] OpenAQ backfill: {total_sensors} sensors x {(end - start).days}d "
        f"— throttled to ~{60 / _limiter.min_interval_s:.0f} req/min, this takes a few minutes"
    )

    frames: list[pd.DataFrame] = []
    done = 0
    for st in stations.itertuples():
        for param, sensor_id in (getattr(st, "sensors", {}) or {}).items():
            page = 1
            while page <= 12:  # 12 x 1000h ~ covers a long backfill
                try:
                    payload = fetch_json(
                        f"{BASE}/sensors/{sensor_id}/hours",
                        params={
                            "datetime_from": f"{start.isoformat()}T00:00:00Z",
                            "datetime_to": f"{end.isoformat()}T23:59:59Z",
                            "limit": 1000,
                            "page": page,
                        },
                        headers=_headers(),
                        # A past window never changes; cache hard so a re-seed is free.
                        ttl=timedelta(days=30),
                        limiter=_limiter,
                    )
                except FetchError as exc:
                    logger.warning(f"[{city.id}] OpenAQ sensor {sensor_id} p{page}: {exc}")
                    break

                results = payload.get("results", []) or []
                if not results:
                    break
                rows = []
                for r in results:
                    ts = ((r.get("period") or {}).get("datetimeFrom") or {}).get("utc")
                    val = r.get("value")
                    if ts is None or val is None:
                        continue
                    rows.append({"ts": ts, "value": float(val)})
                if rows:
                    df = pd.DataFrame(rows)
                    df["ts"] = pd.to_datetime(df["ts"], utc=True)
                    df["city"] = city.id
                    df["station_id"] = st.station_id
                    df["param"] = param
                    df["unit"] = "mg/m3" if param == "co" else "ug/m3"
                    df["source"] = SOURCE_OPENAQ
                    frames.append(df)
                if len(results) < 1000:
                    break
                page += 1
            done += 1
            if done % 25 == 0:
                logger.info(f"[{city.id}] OpenAQ backfill: {done}/{total_sensors} sensors")

    if not frames:
        logger.warning(
            f"[{city.id}] OpenAQ returned no measurements — check the key and the date window; "
            "falling back to CAMS reanalysis"
        )
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)[
        ["city", "station_id", "param", "ts", "value", "unit", "source"]
    ]
    out = out.drop_duplicates(subset=["city", "station_id", "param", "ts"])
    out = screen(out, f"[{city.id}] OpenAQ")

    # A partial backfill is the dangerous outcome: it looks like success but
    # trains the forecaster on a fraction of the network. Say so loudly.
    got = out["station_id"].nunique()
    if got < len(stations):
        logger.warning(
            f"[{city.id}] OpenAQ returned data for only {got}/{len(stations)} stations — "
            "the rest reported nothing for this window"
        )
    logger.info(f"[{city.id}] OpenAQ: {len(out):,} measured rows across {got} stations")
    return out
