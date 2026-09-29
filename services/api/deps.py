"""Shared read helpers for the API routers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from fastapi import HTTPException

from vayu_core.config import CityConfig, load_city
from vayu_core.db import read_conn


def get_city(city_id: str) -> CityConfig:
    try:
        return load_city(city_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown city '{city_id}'") from None


def read_wards(city_id: str, with_geom: bool = False) -> pd.DataFrame:
    cols = "city, ward_id, name, centroid_lat, centroid_lon, area_km2, population, pop_source"
    if with_geom:
        cols += ", geom_geojson"
    with read_conn() as con:
        return con.execute(
            f"SELECT {cols} FROM wards WHERE city = ? ORDER BY ward_id", [city_id]
        ).df()


def read_stations(city_id: str) -> pd.DataFrame:
    with read_conn() as con:
        return con.execute(
            "SELECT city, station_id, name, lat, lon, provider FROM stations WHERE city = ? ORDER BY station_id",
            [city_id],
        ).df()


def read_measurements(city_id: str, at: datetime, lookback: timedelta = timedelta(hours=6)) -> pd.DataFrame:
    """Only the window the snapshot needs — the table holds ~600k rows per city
    and the endpoint has a 300ms warm budget (TRD §10)."""
    with read_conn() as con:
        df = con.execute(
            """SELECT city, station_id, param, ts, value, unit, source
               FROM measurements
               WHERE city = ? AND ts <= ? AND ts >= ?""",
            [city_id, at, at - lookback],
        ).df()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def read_data_status(city_id: str) -> pd.DataFrame:
    with read_conn() as con:
        return con.execute(
            "SELECT source, status, detail, rows_loaded, fetched_ts FROM data_status WHERE city = ? ORDER BY source",
            [city_id],
        ).df()
