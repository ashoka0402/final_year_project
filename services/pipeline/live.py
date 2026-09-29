"""Periodic live CPCB refresh, run only in LIVE mode (DEMO_MODE=false).

DEMO_MODE keeps this off entirely (services/api/main.py's lifespan already
documents why: "a background refresh can never move the ground under a live
demo"). With DEMO_MODE=false, `settings.now()` is real wall-clock time, so
without this the app would show a real "now" against data that never
advances — a station's last reading getting older every minute with nothing
refilling it. This is deliberately the light path: current-hour station
readings only, not a full historical-weather reseed, which is too heavy to
run on a schedule.
"""

from __future__ import annotations

from loguru import logger

from vayu_core.config import load_city
from vayu_core.db import set_data_status, upsert_df, write_conn

from . import cpcb

MEAS_COLS = ["city", "station_id", "param", "ts", "value", "unit", "source"]


def refresh_live_measurements(city_ids: tuple[str, ...]) -> None:
    """Pull the current CPCB snapshot for each city. One city's failure (feed
    down, rate-limited) is logged and skipped rather than taking the others
    down with it — this runs unattended on a timer, with no one watching to
    retry by hand."""
    for city_id in city_ids:
        try:
            city = load_city(city_id)
            stations, current = cpcb.fetch_stations(city)
            with write_conn() as con:
                n_s = upsert_df(con, "stations", stations.drop(columns=["sensors"], errors="ignore"), ["station_id"])
                n_m = upsert_df(con, "measurements", current.reindex(columns=MEAS_COLS), ["city", "station_id", "param", "ts"])
                set_data_status(con, city.id, "measurements", "live", f"CPCB live · {n_m} rows", n_m)
                set_data_status(con, city.id, "stations", "live", f"CPCB live · {n_s} stations", n_s)
            logger.info(f"[live] {city_id}: refreshed {n_m} measurement rows / {n_s} stations")
        except Exception:  # noqa: BLE001 — a scheduled job must survive one bad tick
            logger.exception(f"[live] {city_id}: refresh failed, keeping last-known readings")
