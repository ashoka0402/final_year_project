"""Train (if needed) and score forecasts — the step `make seed` runs after ingest.

Kept out of seed.py so it can also be triggered by the orchestrator's run-cycle
(App Flow §5) without re-ingesting anything.
"""

from __future__ import annotations

import sys

import pandas as pd
from loguru import logger

from vayu_core.config import CityConfig, get_settings, list_cities
from vayu_core.db import read_conn, upsert_df, write_conn
from vayu_core.forecast.features import build_features
from vayu_core.forecast.model import ARTIFACT_DIR, Forecaster, train
from vayu_core.forecast.run import run_forecast


def _load(city: CityConfig) -> tuple[pd.DataFrame, ...]:
    with read_conn() as con:
        meas = con.execute(
            "SELECT city, station_id, param, ts, value FROM measurements WHERE city = ?", [city.id]
        ).df()
        wx = con.execute("SELECT * FROM weather_hourly WHERE city = ?", [city.id]).df()
        st = con.execute(
            "SELECT city, station_id, name, lat, lon FROM stations WHERE city = ?", [city.id]
        ).df()
        wards = con.execute(
            "SELECT city, ward_id, name, centroid_lat, centroid_lon, population FROM wards WHERE city = ?",
            [city.id],
        ).df()
        fires = con.execute("SELECT * FROM fires WHERE city = ?", [city.id]).df()
    for df in (meas, wx):
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return meas, wx, st, wards, fires


def train_and_score(force_train: bool = False) -> int:
    settings = get_settings()
    at = settings.now()

    loaded: dict[str, tuple] = {}
    frames: dict[str, pd.DataFrame] = {}
    weathers: dict[str, pd.DataFrame] = {}
    for city in list_cities():
        meas, wx, st, wards, fires = _load(city)
        if meas.empty or st.empty or wards.empty:
            logger.warning(f"[{city.id}] insufficient data — skipping")
            continue
        loaded[city.id] = (meas, wx, st, wards, fires)
        frames[city.id] = build_features(city, meas, wx, st, fires)
        weathers[city.id] = wx

    if not frames:
        logger.error("nothing to train on — run the ingest step first")
        return 1

    if force_train or not Forecaster().available:
        logger.info("training forecast models…")
        train(frames, weathers)
    else:
        logger.info(f"reusing trained models in {ARTIFACT_DIR.name}/ (use --retrain to rebuild)")

    fc = Forecaster()
    if not fc.available:
        logger.error("no models available after training")
        return 1

    total = 0
    for city in list_cities():
        if city.id not in loaded:
            continue
        meas, wx, st, wards, fires = loaded[city.id]
        df = run_forecast(city, meas, wx, st, wards, fires, at, forecaster=fc)
        if df.empty:
            continue
        with write_conn() as con:
            total += upsert_df(con, "forecasts", df, ["city", "ward_id", "run_ts", "target_ts"])
    logger.success(f"scored {total:,} ward-forecast rows at {at:%Y-%m-%d %H:%M UTC}")
    return 0


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> {message}", level="INFO")
    return train_and_score(force_train="--retrain" in sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
