"""`make seed` — one command from a clean clone to a working offline app.

Order matters: wards and stations first (they define the geometry everything
else hangs off), then the series, then the optional layers. Every stage is
independently degradable — a stage that fails records its status and the app
still runs, with an honest pill instead of a crash (PRD F2, App Flow §7).

Re-running is cheap and safe: HTTP responses are cached, sample parquet is
reused, and every write is INSERT OR REPLACE. `--force` re-downloads.
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from vayu_core.config import REPO_ROOT, CityConfig, get_settings, list_cities
from vayu_core.db import (
    init_db,
    set_data_status,
    upsert_df,
    write_conn,
)

from . import airquality, cpcb, firms, meteo, openaq, osm, permits, roads, s5p
from .http import FetchError
from .wards import load_wards

SAMPLES = REPO_ROOT / "data" / "samples"

# measurements columns that live in DuckDB (sub_index is a CPCB-only extra we
# keep in the parquet bundle for traceability but not in the shared schema).
MEAS_COLS = ["city", "station_id", "param", "ts", "value", "unit", "source"]


def _sample(name: str) -> Path:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    return SAMPLES / name


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"bundled {path.name} unreadable: {exc}")
        return pd.DataFrame()


def seed_city(city: CityConfig, force: bool = False) -> dict[str, str]:
    settings = get_settings()
    now = settings.now()
    statuses: dict[str, str] = {}

    logger.info(f"─── seeding {city.name} (now={now:%Y-%m-%d %H:%M %Z}, demo_mode={settings.demo_mode}) ───")

    # ---- 1. Wards -----------------------------------------------------------
    wards_df, wards_status = load_wards(city, force=force)
    if wards_df.empty:
        logger.error(f"[{city.id}] no wards — skipping city")
        return {"wards": "unavailable"}
    statuses["wards"] = wards_status

    # ---- 2. Stations + current readings -------------------------------------
    st_path, me_path = _sample(f"stations_{city.id}.parquet"), _sample(f"measurements_{city.id}.parquet")
    stations = pd.DataFrame()
    current = pd.DataFrame()
    aq_status = "sample"

    start, end = airquality.demo_window(now)

    if openaq.available():
        try:
            stations = openaq.fetch_locations(city, start, end)
            aq_status = "live"
        except FetchError as exc:
            logger.warning(f"[{city.id}] OpenAQ locations failed: {exc}")

    if stations.empty:
        try:
            stations, current = cpcb.fetch_stations(city)
            aq_status = "live"
        except FetchError as exc:
            logger.warning(f"[{city.id}] CPCB unavailable ({exc}) — falling back to bundle")
            stations = _read_parquet(st_path)
            aq_status = "sample"

    if stations.empty:
        logger.error(f"[{city.id}] no stations from any source — skipping city")
        return {**statuses, "stations": "unavailable"}

    stations.drop(columns=["sensors"], errors="ignore").to_parquet(st_path, index=False)
    statuses["stations"] = aq_status

    # ---- 3. Historical hourly series ----------------------------------------
    history = pd.DataFrame()
    hist_status = "sample"

    if not force and me_path.exists():
        history = _read_parquet(me_path)
        if not history.empty:
            logger.info(f"[{city.id}] reusing bundled series ({len(history):,} rows) — use --force to refresh")
            hist_status = "sample"

    if history.empty:
        if openaq.available():
            try:
                history = openaq.fetch_measurements(city, stations, start, end)
                hist_status = "live"
            except FetchError as exc:
                logger.warning(f"[{city.id}] OpenAQ history failed: {exc}")
        if history.empty:
            try:
                history = airquality.fetch_history(city, stations, start, end)
                hist_status = "cams"
            except FetchError as exc:
                logger.warning(f"[{city.id}] CAMS history failed: {exc}")

    # ---- 3b. Past stubble seasons (training only) ---------------------------
    # Without these the model has never seen a November; see
    # airquality.historical_winters() for the measured consequence.
    extra: list[pd.DataFrame] = []
    if openaq.available() and settings.demo_mode:
        for w_start, w_end in airquality.historical_winters():
            try:
                w_stations = openaq.fetch_locations(city, w_start, w_end)
                if w_stations.empty:
                    continue
                w_hist = openaq.fetch_measurements(city, w_stations, w_start, w_end)
                if not w_hist.empty:
                    extra.append(w_hist)
                    logger.info(
                        f"[{city.id}] historical winter {w_start:%Y-%m} → {w_end:%Y-%m}: "
                        f"{len(w_hist):,} rows across {w_hist['station_id'].nunique()} stations"
                    )
            except FetchError as exc:
                logger.warning(f"[{city.id}] historical winter {w_start:%Y} failed: {exc}")

    frames = [f for f in (history, current, *extra) if not f.empty]
    measurements = (
        pd.concat([f.reindex(columns=MEAS_COLS + ["sub_index"], fill_value=None) for f in frames], ignore_index=True)
        if frames
        else pd.DataFrame(columns=MEAS_COLS)
    )
    if not measurements.empty:
        measurements = measurements.drop_duplicates(subset=["city", "station_id", "param", "ts"], keep="last")
        measurements.to_parquet(me_path, index=False)
    statuses["measurements"] = hist_status

    # ---- 4. Weather ---------------------------------------------------------
    wx_path = _sample(f"weather_{city.id}.parquet")
    weather = pd.DataFrame()
    wx_status = "sample"
    if not force and wx_path.exists():
        weather = _read_parquet(wx_path)
    if weather.empty:
        try:
            # Both fields must span the WHOLE window, not stop at "now": the
            # bundle reaches 21 days past the demo clock so the backtest and
            # outcome verification have ground truth, and every one of those
            # hours needs weather or the features are NaN.
            hist = meteo.fetch_history(city, start, end)
            # With a pinned demo clock, kind='forecast' comes from the historical
            # forecast archive — what the forecast actually said at the time, not
            # reanalysis, which would hand the model perfect foresight.
            fc = (
                meteo.fetch_historical_forecast(city, start, end)
                if settings.demo_mode
                else meteo.fetch_forecast(city)
            )
            # Wide field for back-trajectories: a few days either side of "now"
            # is enough, since only trajectories read it.
            air = meteo.fetch_airshed(
                city, (now - timedelta(days=4)).date(), (now + timedelta(days=4)).date()
            )
            # Weather for the past stubble seasons too, or those training rows
            # carry NaN wind and the model learns "November is bad" without the
            # mechanism — stagnation under an inversion — that makes it bad.
            past: list[pd.DataFrame] = []
            if settings.demo_mode:
                for w_start, w_end in airquality.historical_winters():
                    past.append(meteo.fetch_history(city, w_start, w_end))
                    # The historical-forecast archive does not reach back to
                    # 2016, so reanalysis stands in for fx_* on those rows. That
                    # only affects TRAINING; the holdout still uses the real
                    # forecast, so the reported skill stays honest.
                    past.append(
                        meteo.fetch_history(city, w_start, w_end).assign(kind="forecast")
                    )
            weather = pd.concat(
                [f for f in (hist, fc, air, *past) if not f.empty], ignore_index=True
            )
            wx_status = "live"
            if not weather.empty:
                weather.to_parquet(wx_path, index=False)
        except FetchError as exc:
            logger.warning(f"[{city.id}] weather fetch failed: {exc}")
    statuses["weather"] = wx_status

    # ---- 5. Fires (optional key) --------------------------------------------
    # With a pinned demo clock, "the last 7 days" is the wrong question — the NRT
    # feed doesn't reach back to Nov 2025 and answers with an empty CSV. Pull the
    # archive across the whole window instead, so the fire feature and the
    # attribution evidence both have real detections to work with.
    if settings.demo_mode and firms.available():
        fires, fires_status = firms.fetch_window(city, start, end)
        if fires.empty:
            fires, fires_status = firms.fetch_fires(city)
    else:
        fires, fires_status = firms.fetch_fires(city)
    statuses["fires"] = fires_status

    # ---- 6. OSM context (no key) --------------------------------------------
    try:
        _, osm_status = osm.fetch_osm(city, force=force)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{city.id}] OSM failed: {exc}")
        osm_status = "unavailable"
    statuses["osm"] = osm_status

    # ---- 6b. Roads (traffic proxy) + permits (curated) ----------------------
    try:
        roads_gj, roads_status = roads.fetch_roads(city, force=force)
        road_df = roads.density_per_ward(city, roads_gj, wards_df)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{city.id}] roads failed: {exc}")
        road_df, roads_status = pd.DataFrame(), "unavailable"
    statuses["roads"] = roads_status

    try:
        permit_df, permits_status = permits.build_permits(city, wards_df, force=force)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{city.id}] permits failed: {exc}")
        permit_df, permits_status = pd.DataFrame(), "unavailable"
    statuses["permits"] = permits_status

    # ---- 7. Sentinel-5P (optional) ------------------------------------------
    statuses["s5p"] = s5p.fetch_no2(city).status

    # ---- 8. Load into DuckDB ------------------------------------------------
    with write_conn() as con:
        n_w = upsert_df(con, "wards", wards_df, ["city", "ward_id"])
        n_s = upsert_df(
            con,
            "stations",
            stations.drop(columns=["sensors"], errors="ignore"),
            ["city", "station_id"],
        )
        n_m = upsert_df(
            con,
            "measurements",
            measurements[MEAS_COLS] if not measurements.empty else measurements,
            ["city", "station_id", "param", "ts"],
        )
        n_x = upsert_df(con, "weather_hourly", weather, ["city", "grid", "grid_i", "grid_j", "ts", "kind"])
        n_f = upsert_df(con, "fires", fires, ["city", "fire_id"])
        n_r = upsert_df(con, "ward_roads", road_df, ["city", "ward_id"])
        n_p = upsert_df(con, "permits", permit_df, ["city", "permit_id"])

        detail = {
            "wards": f"{n_w} wards · {city.wards.attribution or 'bundled'}",
            "stations": f"{n_s} stations · {stations['provider'].iloc[0] if n_s else 'n/a'}",
            "measurements": f"{n_m:,} rows · {', '.join(sorted(measurements['source'].dropna().unique())) if n_m else 'none'}",
            "weather": f"{n_x:,} rows · Open-Meteo",
            "fires": f"{n_f} detections · NASA FIRMS VIIRS",
            "osm": "OpenStreetMap via Overpass",
            "roads": f"{n_r} wards scored · OSM major roads",
            "permits": f"{n_p} curated sites on real OSM construction landuse — SAMPLE DATA",
            "s5p": "Sentinel-5P NO2 (GEE)",
        }
        rows = {"wards": n_w, "stations": n_s, "measurements": n_m, "weather": n_x,
                "fires": n_f, "roads": n_r, "permits": n_p}
        for source, status in statuses.items():
            set_data_status(con, city.id, source, status, detail.get(source, ""), rows.get(source, 0), now)

    logger.success(
        f"[{city.id}] {n_w} wards · {n_s} stations · {n_m:,} measurements · {n_x:,} weather · {n_f} fires"
    )
    return statuses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed VAYU with bundled + live data")
    parser.add_argument("--force", action="store_true", help="re-download everything, ignoring bundles")
    parser.add_argument("--city", help="seed only this city id")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <8}</level> {message}", level="INFO")

    init_db()
    cities = [c for c in list_cities() if not args.city or c.id == args.city]
    if not cities:
        logger.error(f"no city matched '{args.city}'")
        return 1

    failed = []
    for city in cities:
        try:
            seed_city(city, force=args.force)
        except Exception as exc:  # noqa: BLE001 - one bad city must not sink the seed
            logger.exception(f"[{city.id}] seeding failed: {exc}")
            failed.append(city.id)

    # One executed order per city so /verify has something to show on a fresh
    # clone (PRD E1). Verdicts are computed from real readings, badged "seeded".
    try:
        from .seed_demo import run as seed_demo_run

        seed_demo_run(cities)
    except Exception as exc:  # noqa: BLE001 - the demo record is a nicety, not the seed
        logger.warning(f"demo record seeding skipped: {exc}")

    if failed:
        logger.error(f"seed finished with failures: {', '.join(failed)}")
        return 1
    logger.success("seed complete — run `make dev`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
