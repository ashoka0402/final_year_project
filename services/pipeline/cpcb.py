"""CPCB CAAQMS real-time feed via data.gov.in.

This is the authoritative Indian source: the same CAAQMS network that feeds
SAMEER. The resource gives, per station and pollutant, the current min/max/avg
with real station names and coordinates — including the ones a Delhi
commissioner would name out loud (Anand Vihar, Punjabi Bagh, ITO).

VAYU uses it for two things:
  1. Station identity (name + lat/lon) — the station layer on the map.
  2. Current measured air quality -> the 'live' AQI path.

CRITICAL — what `avg_value` actually contains:
This resource publishes CPCB *sub-indices* (per-pollutant AQI), not
concentrations, despite field names that suggest otherwise. Verified against the
feed itself: CO reads ~52 nationally, which is impossible as mg/m3 (severe
poisoning) and impossible as ug/m3 (below ambient CO), but is exactly 1.05 mg/m3
when read as a sub-index — and CAMS independently reports 1.36 mg/m3 at the same
station-hour. Reading these as concentrations puts every Delhi station at
"AQI 500 Severe" in the middle of monsoon.

So: station AQI is taken directly as max(sub-index) — exact, no round-trip — and
concentrations are recovered by inverting the published CPCB breakpoint table.
Those concentrations are 24h-averages (8h for CO/O3), which is what the index is
defined on; they are not hourly values and are labelled accordingly.

The portal's published demo key works without signup, which is what keeps the
"no keys required" promise honest rather than aspirational. It is rate-limited
and caps `limit` at 10 rows/request, so we page and tolerate a truncated page
set rather than losing the whole city.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from vayu_core.aqi import AqiResult, aqi_from_sub_indices, concentration_from_sub_index
from vayu_core.config import CityConfig, get_settings
from vayu_core.db import SOURCE_CPCB_LIVE
from vayu_core.qc import screen

from .http import FetchError, fetch_json

RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
PAGE = 10  # the public demo key ignores larger values

# data.gov.in pollutant_id -> our param names (vayu_core.aqi.BREAKPOINTS)
PARAM_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "OZONE": "o3",
}


def _station_id(city_id: str, station_name: str) -> str:
    """Stable, readable id: the feed has no station id of its own, and joining on
    a free-text name across runs is fragile."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in station_name.split(",")[0].strip().lower())
    slug = "-".join(filter(None, slug.split("-")))[:40]
    return f"{city_id}:{slug}"


def _parse_ts(raw: str, tz: str) -> datetime | None:
    """The feed stamps 'DD-MM-YYYY HH:MM:SS' in India Standard Time, unlabelled."""
    try:
        naive = datetime.strptime(raw.strip(), "%d-%m-%Y %H:%M:%S")
    except (ValueError, AttributeError):
        return None
    return naive.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def fetch_stations(city: CityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stations_df, measurements_df) of current CPCB readings.

    Raises FetchError if the feed is unreachable, so the seeder can fall back.
    """
    settings = get_settings()
    # Two selection modes:
    #   * cpcb_city_filter set  -> server-side filter by one city name (Delhi,
    #     Lucknow). Fast: the feed returns only that city.
    #   * not set (NCR)         -> fetch the whole national feed and select by the
    #     config bbox below. NCR spans ~19 municipalities under different city
    #     names, so a single name filter can't capture it; the bbox can.
    city_filter = city.sources.get("cpcb_city_filter")
    bbox_mode = not city_filter
    # The national feed is ~3,500 records; a single city is a few dozen. Page
    # far enough for the whole feed only when in bbox mode.
    max_offset = 5000 if bbox_mode else 2000

    records: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        params = {
            "api-key": settings.data_gov_key,
            "format": "json",
            "limit": PAGE,
            "offset": offset,
        }
        if city_filter:
            params["filters[city]"] = city_filter
        try:
            payload = fetch_json(
                BASE,
                params=params,
                # Short TTL: this is the live layer. Long enough that paging
                # through ~30 requests in one seed doesn't hammer the shared key.
                ttl=timedelta(minutes=20),
            )
        except FetchError as exc:
            # The shared demo key rate-limits partway through a city on a bad
            # day. Half a city of real stations beats none — keep what we have
            # and let the caller label the layer accordingly.
            if records:
                logger.warning(
                    f"[{city.id}] CPCB paging stopped at offset {offset} ({exc}); "
                    f"continuing with {len(records)} records already fetched"
                )
                break
            raise
        if total is None:
            total = int(payload.get("total", 0) or 0)
            logger.info(
                f"[{city.id}] CPCB feed reports {total} records "
                f"({'bbox mode' if bbox_mode else repr(city_filter)})"
            )
        batch = payload.get("records", []) or []
        if not batch:
            break
        records.extend(batch)
        offset += PAGE
        if offset > max_offset:  # guard against a runaway pager
            break

    if not records:
        raise FetchError(f"CPCB feed returned no records ({city_filter or 'bbox mode'})")

    stations: dict[str, dict] = {}
    meas: list[dict] = []
    w, s, e, n = city.bbox

    for r in records:
        name = (r.get("station") or "").strip()
        if not name:
            continue
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        # A few CAAQMS rows carry coordinates outside the municipal bbox
        # (satellite towns filed under the parent city). Keep the map honest.
        if not (w <= lon <= e and s <= lat <= n):
            continue

        sid = _station_id(city.id, name)
        ts = _parse_ts(r.get("last_update", ""), city.timezone)
        stations.setdefault(
            sid,
            {
                "city": city.id,
                "station_id": sid,
                "name": name.split(",")[0].strip(),
                "lat": lat,
                "lon": lon,
                "provider": "CPCB CAAQMS (data.gov.in)",
                "first_seen": ts,
                "last_seen": ts,
            },
        )

        param = PARAM_MAP.get((r.get("pollutant_id") or "").strip().upper())
        raw_val = r.get("avg_value")
        if param is None or ts is None or raw_val in (None, "", "NA"):
            continue
        try:
            index_value = float(raw_val)
        except (TypeError, ValueError):
            continue

        # avg_value is a CPCB sub-index (see module docstring). Invert it to the
        # 24h-average concentration the index is defined on.
        conc = concentration_from_sub_index(param, index_value)
        if conc is None:
            continue

        meas.append(
            {
                "city": city.id,
                "station_id": sid,
                "param": param,
                "ts": ts,
                "value": conc,
                "unit": "mg/m3" if param == "co" else "ug/m3",
                "source": SOURCE_CPCB_LIVE,
                "sub_index": index_value,
            }
        )

    st_df = pd.DataFrame(list(stations.values()))
    me_df = pd.DataFrame(meas).drop_duplicates(subset=["city", "station_id", "param", "ts"])
    me_df = screen(me_df, f"[{city.id}] CPCB")
    logger.info(f"[{city.id}] CPCB: {len(st_df)} stations, {len(me_df)} current measurements")
    return st_df, me_df


def station_aqi(measurements: pd.DataFrame) -> dict[str, AqiResult]:
    """Current CPCB AQI per station, taken straight from the published indices."""
    if measurements.empty or "sub_index" not in measurements.columns:
        return {}
    out: dict[str, AqiResult] = {}
    for sid, grp in measurements.groupby("station_id"):
        latest = grp.sort_values("ts").groupby("param")["sub_index"].last()
        res = aqi_from_sub_indices(latest.to_dict())
        if res is not None:
            out[str(sid)] = res
    return out
