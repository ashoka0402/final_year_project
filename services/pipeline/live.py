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

import pandas as pd
from loguru import logger

from vayu_core.config import load_city
from vayu_core.db import read_conn, set_data_status, upsert_df, write_conn

from . import airquality, cpcb, openaq

MEAS_COLS = ["city", "station_id", "param", "ts", "value", "unit", "source"]


def refresh_live_measurements(city_ids: tuple[str, ...]) -> None:
    """Refresh current measured air quality with explicit source priority.

    Official CPCB/MPCB-compatible CAAQMS is attempted first. If it is
    unavailable, OpenAQ v3 is used as the measured secondary source. If both
    fail, the last-known database values remain untouched.
    """
    for city_id in city_ids:
        city = load_city(city_id)

        try:
            stations, current = cpcb.fetch_stations(city)
            if stations.empty or current.empty:
                raise RuntimeError("official source returned no usable current measurements")
            source_label = "Official CPCB CAAQMS observation"
            source_status = "live"
        except Exception as official_exc:  # noqa: BLE001 - secondary measured source
            logger.warning(f"[live] {city_id}: official feed unavailable: {official_exc}")
            try:
                stations, current = openaq.fetch_current(city)
                if stations.empty or current.empty:
                    raise RuntimeError("OpenAQ returned no usable current measurements")
                source_label = "OpenAQ v3 measured observation (secondary source)"
                source_status = "live"
            except Exception as oa_exc:  # noqa: BLE001 - final modelled fallback
                logger.warning(f"[live] {city_id}: OpenAQ fallback unavailable: {oa_exc}")
                try:
                    with read_conn() as con:
                        stations = con.execute(
                            "SELECT city, station_id, name, lat, lon, provider FROM stations WHERE city = ?",
                            [city.id],
                        ).df()
                    current = airquality.fetch_history(city, stations, pd.Timestamp.now(tz="UTC").date(), pd.Timestamp.now(tz="UTC").date())
                    now = pd.Timestamp.utcnow()
                    current["ts"] = pd.to_datetime(current["ts"], utc=True)
                    current = current[(current["ts"] <= now) & (current["ts"] >= now - pd.Timedelta(hours=6))]
                    if stations.empty or current.empty:
                        raise RuntimeError("CAMS returned no usable current values")
                    source_label = "CAMS/Open-Meteo modelled fallback"
                    source_status = "cams"
                except Exception as cams_exc:  # noqa: BLE001 - keep last-known data
                    logger.warning(f"[live] {city_id}: CAMS fallback unavailable: {cams_exc}")
                    logger.warning(f"[live] {city_id}: keeping last-known readings")
                    continue

        if stations.empty or current.empty:
            logger.warning(
                f"[live] {city_id}: source returned no usable current measurements; "
                "keeping last-known readings"
            )
            continue

        try:
            with write_conn() as con:
                n_s = upsert_df(
                    con,
                    "stations",
                    stations.drop(columns=["sensors"], errors="ignore"),
                    ["station_id"],
                )
                n_m = upsert_df(
                    con,
                    "measurements",
                    current.reindex(columns=MEAS_COLS),
                    ["city", "station_id", "param", "ts"],
                )
                set_data_status(
                    con,
                    city.id,
                    "measurements",
                    source_status,
                    f"{source_label} · {n_m} rows",
                    n_m,
                )
                set_data_status(
                    con,
                    city.id,
                    "stations",
                    "live",
                    f"{source_label} · {n_s} stations",
                    n_s,
                )
            logger.info(
                f"[live] {city_id}: refreshed {n_m} measurement rows / "
                f"{n_s} stations via {source_label}"
            )
        except Exception:  # noqa: BLE001 - preserve scheduler loop
            logger.exception(
                f"[live] {city_id}: database update failed, keeping last-known readings"
            )
