"""Weather from Open-Meteo — historical archive + forecast. No API key, ever.

Fetched on the city's weather grid (config: weather_grid.nx x ny) rather than at
a single point, because the back-trajectory (TRD 5.2) integrates a *field*: it
bilinearly interpolates wind between grid points as it steps backwards. A single
city-centre wind vector would make every trajectory a straight line and the
attribution worthless.

100 m winds are fetched alongside 10 m because plume transport above the canopy
tracks the 100 m level much more closely; TRD 5.2 prefers it for trajectories.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from vayu_core.config import CityConfig, get_settings

from .http import FetchError, RateLimiter, fetch_json

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo mirrors every endpoint on a `customer-` host for paid keys. VAYU
# needs neither: the free hosts served the entire archive. `_endpoint()` swaps
# hosts only if OPEN_METEO_API_KEY happens to be set.
def _endpoint(url: str) -> tuple[str, dict]:
    key = get_settings().open_meteo_api_key
    if not key:
        return url, {}
    return url.replace("https://", "https://customer-", 1), {"apikey": key}

# Archives what the forecast actually SAID at the time, rather than what later
# turned out to be true. This is what keeps the backtest honest: the Forecaster's
# strongest features are the weather at the target hour (fx_*), and feeding it
# reanalysis would hand it perfect foresight no operator has, inflating the
# reported skill. With this endpoint the model inherits the weather model's own
# error — the skill you would actually get on the day.
HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "boundary_layer_height",
    "precipitation",
    "surface_pressure",
]

# The airshed grid exists solely so back-trajectories have wind to integrate
# through — nothing reads its temperature or rain. Open-Meteo weights a request
# by (locations x hours x variables), and the airshed has 9x the points of the
# city grid, so fetching all nine variables there is what tips the seed into 429.
HOURLY_WIND_ONLY = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
]

COLS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "rh",
    "wind_speed_10m": "wind_speed_10m",
    "wind_direction_10m": "wind_dir_10m",
    "wind_speed_100m": "wind_speed_100m",
    "wind_direction_100m": "wind_dir_100m",
    "boundary_layer_height": "pblh",
    "precipitation": "precip",
    "surface_pressure": "pressure",
}

BATCH = 25  # the 5x5 city grid fits in a single request

# Open-Meteo is free and key-less but weights a request by (locations x hours x
# variables), so one 276-day x 25-point x 9-variable call is enormous and earns a
# 429. Chunking by time keeps each call small, and the limiter spaces them out.
# Getting this wrong is silent-partial again: Lucknow came back with 0 weather
# rows while Delhi succeeded, purely because Delhi ran first.
DAYS_PER_REQUEST = 45
_limiter = RateLimiter(1.5)


def _to_frame(city: CityConfig, block: dict, i: int, j: int, kind: str, grid: str = "city") -> pd.DataFrame:
    hourly = block.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({"ts": pd.to_datetime(pd.Series(times), utc=True)})
    for var, col in COLS.items():
        vals = hourly.get(var)
        # Absent variables stay NaN rather than breaking the frame: the airshed
        # grid deliberately requests wind only.
        df[col] = pd.Series(vals, dtype="float64") if vals else pd.Series([pd.NA] * len(df), dtype="Float64")
    df["city"] = city.id
    df["grid"] = grid
    df["grid_i"] = i
    df["grid_j"] = j
    df["kind"] = kind
    return df


def _fetch(
    city: CityConfig,
    url: str,
    extra: dict,
    kind: str,
    ttl: timedelta,
    grid: str = "city",
    hourly: list[str] | None = None,
) -> pd.DataFrame:
    pts = city.grid_points() if grid == "city" else city.airshed_points()
    frames: list[pd.DataFrame] = []

    # Split a long date range into chunks so each request stays under
    # Open-Meteo's per-call weighting.
    spans: list[dict] = []
    if "start_date" in extra and "end_date" in extra:
        s0 = date.fromisoformat(extra["start_date"])
        e0 = date.fromisoformat(extra["end_date"])
        cur = s0
        while cur <= e0:
            nxt = min(cur + timedelta(days=DAYS_PER_REQUEST - 1), e0)
            spans.append({**extra, "start_date": cur.isoformat(), "end_date": nxt.isoformat()})
            cur = nxt + timedelta(days=1)
    else:
        spans = [extra]

    failures = 0
    for span in spans:
        for b in range(0, len(pts), BATCH):
            chunk = pts[b : b + BATCH]
            try:
                endpoint, auth = _endpoint(url)
                payload = fetch_json(
                    endpoint,
                    params={
                        "latitude": ",".join(f"{lat:.4f}" for _, _, lat, _ in chunk),
                        "longitude": ",".join(f"{lon:.4f}" for _, _, _, lon in chunk),
                        "hourly": ",".join(hourly or HOURLY),
                        "timezone": "UTC",
                        **auth,
                        **span,
                    },
                    ttl=ttl,
                    timeout=180.0,
                    limiter=_limiter,
                )
            except FetchError as exc:
                failures += 1
                logger.warning(f"[{city.id}] weather {grid}/{kind} {span.get('start_date')}: {exc}")
                continue
            blocks = payload if isinstance(payload, list) else [payload]
            for (i, j, _, _), block in zip(chunk, blocks):
                f = _to_frame(city, block, i, j, kind, grid)
                if not f.empty:
                    frames.append(f)

    if failures:
        logger.error(
            f"[{city.id}] weather {grid}/{kind}: {failures} request(s) failed — "
            "coverage has holes and forecast features will be NaN there"
        )
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    ordered = ["city", "grid", "grid_i", "grid_j", "ts", *COLS.values(), "kind"]
    out = out[ordered].drop_duplicates(subset=["city", "grid", "grid_i", "grid_j", "ts", "kind"])
    logger.info(f"[{city.id}] weather {grid}/{kind}: {len(out):,} rows over {len(pts)} grid points")
    return out


def fetch_history(city: CityConfig, start: date, end: date) -> pd.DataFrame:
    return _fetch(
        city,
        ARCHIVE,
        {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "hist",
        ttl=timedelta(days=30),
    )


def fetch_forecast(city: CityConfig, days: int = 4) -> pd.DataFrame:
    """Live 4-day forecast (TRD §4). Only meaningful when DEMO_MODE is off —
    in DEMO_MODE 'now' is pinned to a past date, and the forecast window is
    served from the archive instead so the timeline stays coherent."""
    return _fetch(city, FORECAST, {"forecast_days": days}, "forecast", ttl=timedelta(hours=1))


def fetch_airshed(city: CityConfig, start: date, end: date) -> pd.DataFrame:
    """Wide, coarse wind field for back-trajectories.

    Fetched over a short window (a few days around the pinned clock) rather than
    the whole training span: only trajectories read it, and 225 points x months
    of hours would bloat the committed bundle for no benefit.
    """
    return _fetch(
        city,
        ARCHIVE,
        {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "hist",
        ttl=timedelta(days=30),
        grid="airshed",
        hourly=HOURLY_WIND_ONLY,
    )


def fetch_historical_forecast(city: CityConfig, start: date, end: date) -> pd.DataFrame:
    """What the weather forecast SAID over [start, end] — the honest fx_* source.

    Stored as kind='forecast'. Every hour of the training and holdout window
    needs this: the Forecaster joins weather at the *target* hour, so a window
    where forecast weather is missing trains and evaluates on NaN features.
    Getting this wrong is not subtle — a 30-day holdout with only 14 days of
    weather made VAYU lose to persistence by 54%.
    """
    return _fetch(
        city,
        HISTORICAL_FORECAST,
        {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "forecast",
        ttl=timedelta(days=30),
    )
