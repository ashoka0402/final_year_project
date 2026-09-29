"""On-demand forecast scoring — the engine behind the live clock / date picker.

The seeded demo carries forecasts stamped `run_ts = DEMO_NOW`. Once the operator
can move the clock (L1b: live wall clock, or the date picker's `?as_of=`), the
app needs forecasts for *that* instant, which may not be in the table yet.

Windowed scoring (L1a) made a single-city score ~1.5 s, fast enough to do this
inside a request. So instead of "read the latest run", the API asks for **the run
at the app's current hour**, and this module scores it on the spot if it's
missing. Going back in time selects (or scores) the run for that past hour;
coming back to now selects the live one. Every downstream surface — nowcast,
horizons, alerts, GRAP stage, ROI — keys off the same `current_run_ts`, so they
all move together to the chosen instant.

Idempotent by construction: a run is keyed by (city, run_ts) floored to the hour,
so a second request for the same hour finds it and returns instantly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from vayu_core.config import CityConfig, get_settings, load_city
from vayu_core.db import read_conn, upsert_df, write_conn
from vayu_core.forecast.features import SCORING_WINDOW_DAYS
from vayu_core.forecast.model import Forecaster
from vayu_core.forecast.run import run_forecast

# Weather must reach the far horizon: model_frame attaches the forecast weather
# valid at t+72 h, so the scoring window has to include rows up to a bit past it.
_FORECAST_HORIZON_H = 72
_WX_FUTURE_PAD = timedelta(hours=_FORECAST_HORIZON_H + 8)
# A day of slack over the feature window so every lag/rolling term is covered.
_HIST_PAD = timedelta(days=SCORING_WINDOW_DAYS + 1)

# One Forecaster per process — loading the 9 boosters costs ~150 ms and they are
# immutable between retrains.
_FC: Forecaster | None = None


def _forecaster() -> Forecaster:
    global _FC
    if _FC is None:
        _FC = Forecaster()
    return _FC


def reset_forecaster() -> None:
    """Drop the cached models — call after a retrain swaps the artifacts."""
    global _FC
    _FC = None


def align_hour(at: datetime) -> datetime:
    """Floor `at` to the hour (UTC). Runs are hour-granular so repeated requests
    within the same hour reuse one scoring pass. DEMO_NOW is on the hour, so the
    seeded run (run_ts == DEMO_NOW) is selected unchanged in demo mode."""
    at = at if at.tzinfo else at.replace(tzinfo=get_settings().now().tzinfo)
    return at.astimezone(get_settings().now().tzinfo).replace(minute=0, second=0, microsecond=0)


def run_exists(city_id: str, run_ts: datetime) -> bool:
    with read_conn() as con:
        row = con.execute(
            "SELECT 1 FROM forecasts WHERE city = ? AND run_ts = ? LIMIT 1",
            [city_id, run_ts],
        ).fetchone()
    return row is not None


def _load_window(city: CityConfig, at: datetime) -> tuple[pd.DataFrame, ...]:
    """Load only the slice of data the score at `at` needs (fast path)."""
    lo, hi = at - _HIST_PAD, at + _WX_FUTURE_PAD
    with read_conn() as con:
        meas = con.execute(
            """SELECT city, station_id, param, ts, value FROM measurements
               WHERE city = ? AND ts >= ? AND ts <= ?""",
            [city.id, at - _HIST_PAD, at],
        ).df()
        wx = con.execute(
            "SELECT * FROM weather_hourly WHERE city = ? AND ts >= ? AND ts <= ?",
            [city.id, lo, hi],
        ).df()
        st = con.execute(
            "SELECT city, station_id, name, lat, lon FROM stations WHERE city = ?", [city.id]
        ).df()
        wards = con.execute(
            "SELECT city, ward_id, name, centroid_lat, centroid_lon, population FROM wards WHERE city = ?",
            [city.id],
        ).df()
        fires = con.execute(
            "SELECT * FROM fires WHERE city = ? AND acq_ts >= ? AND acq_ts <= ?",
            [city.id, at - _HIST_PAD, at],
        ).df()
    for df in (meas, wx):
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return meas, wx, st, wards, fires


def ensure_forecasts(city_id: str, at: datetime) -> datetime | None:
    """Guarantee a forecast run exists for the hour of `at`; return its run_ts.

    Returns the aligned run_ts if a run exists or was scored, or None if scoring
    was impossible (no trained model, or no station reported within the window of
    `at` — e.g. a date the archive doesn't cover). Callers treat None as "no
    forecast for this instant" and fall back to their empty/404 path.
    """
    run_ts = align_hour(at)
    if run_exists(city_id, run_ts):
        return run_ts

    fc = _forecaster()
    if not fc.available:
        logger.warning(f"[{city_id}] no trained model — cannot score on demand")
        return None

    city = load_city(city_id)
    meas, wx, st, wards, fires = _load_window(city, run_ts)
    if meas.empty or st.empty or wards.empty:
        logger.warning(f"[{city_id}] no data in window for {run_ts:%Y-%m-%d %H:%M} — cannot score")
        return None

    df = run_forecast(city, meas, wx, st, wards, fires, run_ts, forecaster=fc)
    if df.empty:
        logger.warning(f"[{city_id}] on-demand score at {run_ts:%Y-%m-%d %H:%M} produced no rows")
        return None

    with write_conn() as con:
        n = upsert_df(con, "forecasts", df, ["city", "ward_id", "run_ts", "target_ts"])
    logger.info(f"[{city_id}] on-demand scored {n:,} rows for {run_ts:%Y-%m-%d %H:%M UTC}")
    return run_ts


def current_run_ts(city_id: str) -> datetime | None:
    """The run_ts every read surface should use: the run for the app's current
    hour (per the clock override / demo pin / wall clock), scored if missing."""
    return ensure_forecasts(city_id, get_settings().now())
