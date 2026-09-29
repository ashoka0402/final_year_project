"""Surface AQI dataset assembly.

Synthetic satellite_grid + measurements + weather_hourly throughout, so every
expected sample count and value is known by construction — these tests do not
depend on any real ingest having run, and stay fast enough for every commit.

The training loop itself (PyTorch) is exercised once at the bottom on a tiny
synthetic dataset, checking the pipeline runs and produces the right shapes and
report fields — not that it achieves any particular accuracy, which is what the
real, documented evaluation against real data is for.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vayu_core.config import get_settings, load_city, load_region
from vayu_core.db import init_db, write_conn
from vayu_core.national import surface_aqi as SA


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """`get_settings()` is lru_cache'd, so a bare `monkeypatch.setenv` after the
    first test to call it has no effect — every later test silently keeps
    reading the FIRST test's tmp db. Clearing the cache on entry and exit is
    what actually isolates each test's database, found by a test in this exact
    file failing only when run alongside its siblings, not in isolation."""
    path = tmp_path / "t.duckdb"
    monkeypatch.setenv("VAYU_DB_PATH", str(path))
    get_settings.cache_clear()
    init_db(path)
    yield path
    get_settings.cache_clear()


@pytest.fixture()
def region():
    return load_region("india")


def _seed_satellite(con, region_id: str, products: list[str], cells: list[tuple[float, float]], days: list[dt.date]):
    rows = []
    for prod in products:
        for glat, glon in cells:
            for d in days:
                rows.append(
                    {"region": region_id, "product": prod, "grid_lat": glat, "grid_lon": glon,
                     "date": d, "value": 1e-4, "unit": "mol/m^2", "n_obs": 6, "source": "test"}
                )
    from vayu_core.db import upsert_df

    upsert_df(con, "satellite_grid", pd.DataFrame(rows), ["region", "product", "grid_lat", "grid_lon", "date"])


def test_available_channels_detects_what_is_actually_ingested(isolated_db):
    with write_conn() as con:
        _seed_satellite(con, "india", ["hcho", "no2"], [(28.625, 77.125)], [dt.date(2025, 11, 1)])
        got = SA.available_channels(con, "india", wanted=("hcho", "no2", "so2", "co"))
    assert got == ("hcho", "no2"), "must report only products that actually have rows, in requested order"


def test_satellite_patch_falls_back_to_day_mean_for_missing_cells(isolated_db):
    """Regression for a real bug: an early version returned None (dropping the
    whole sample) whenever ANY requested channel had zero rows for the
    region, rather than only when NO fallback existed at all."""
    region = load_region("india")
    day = dt.date(2025, 11, 1)
    glat, glon = region.snap(28.6, 77.1)
    with write_conn() as con:
        # Only the CENTRE cell is populated; the other 8 cells of the 3x3
        # patch are missing and must be filled from the day's mean instead of
        # dropping the sample.
        _seed_satellite(con, "india", ["hcho"], [(glat, glon)], [day])
        patch = SA._satellite_patch(con, region, 28.6, 77.1, day, ("hcho",))
    assert patch is not None
    assert patch.shape == (1, 3, 3)
    assert np.isfinite(patch).all(), "missing cells must be filled, not left as NaN"


def test_satellite_patch_returns_none_when_region_has_no_data_that_day(isolated_db):
    from vayu_core.db import read_conn

    region = load_region("india")
    with read_conn() as con:
        patch = SA._satellite_patch(con, region, 28.6, 77.1, dt.date(2025, 11, 1), ("hcho",))
    assert patch is None


def test_build_dataset_produces_time_ordered_consecutive_windows(isolated_db):
    """The LOOKBACK_DAYS window must be CALENDAR-consecutive days, not just
    row-consecutive — a station with a gap in its readings must not silently
    stitch together two different weeks into one 'sequence'."""
    region = load_region("india")
    city = load_city("lucknow")

    days = [dt.date(2025, 10, 1) + dt.timedelta(days=i) for i in range(10)]
    glat, glon = region.snap(26.85, 80.95)
    cells = [(round(glat + a * region.grid_deg, 4), round(glon + b * region.grid_deg, 4))
             for a in (-1, 0, 1) for b in (-1, 0, 1)]

    with write_conn() as con:
        _seed_satellite(con, "india", list(SA.ALL_SAT_CHANNELS[:2]), cells, days)
        con.execute(
            "INSERT INTO stations (city, station_id, name, lat, lon, provider) VALUES (?,?,?,?,?,?)",
            ["lucknow", "test:station1", "Test Station", 26.85, 80.95, "test"],
        )
        for i, d in enumerate(days):
            con.execute(
                "INSERT INTO measurements (city, station_id, param, ts, value, unit, source) VALUES (?,?,?,?,?,?,?)",
                ["lucknow", "test:station1", "pm25", dt.datetime.combine(d, dt.time(6)), 50.0 + i, "ug/m3", "test"],
            )
        for i, d in enumerate(days):
            for j, (gi, gj, glat_c, glon_c) in enumerate(city.grid_points()):
                con.execute(
                    """INSERT INTO weather_hourly
                       (city, grid, grid_i, grid_j, ts, temp_c, rh, wind_speed_10m, wind_dir_10m,
                        wind_speed_100m, wind_dir_100m, pblh, precip, pressure, kind)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    ["lucknow", "city", gi, gj, dt.datetime.combine(d, dt.time(6)),
                     25.0, 50.0, 2.0, 180.0, 3.0, 180.0, 800.0, 0.0, 1000.0, "hist"],
                )

    ds = SA.build_dataset(region, cities=("lucknow",), channels=SA.ALL_SAT_CHANNELS[:2])
    assert len(ds.y) == 10 - SA.LOOKBACK_DAYS, "one sample per day once a full lookback window exists"
    assert list(ds.dates) == sorted(ds.dates), "samples must come out in date order"
    for i in range(1, len(ds.dates)):
        assert (ds.dates[i] - ds.dates[i - 1]).days == 1, "consecutive samples must be consecutive calendar days"


def test_train_runs_end_to_end_on_a_tiny_synthetic_dataset():
    """Not a claim about accuracy — a claim that the pipeline (model, time
    split, standardisation, holdout evaluation, persistence baseline) runs
    without shape errors and returns every documented report field."""
    rng = np.random.default_rng(0)
    n, lookback, channels, side = 40, SA.LOOKBACK_DAYS, ("hcho", "no2"), 3

    X_sat = rng.normal(1e-4, 1e-5, size=(n, lookback, len(channels), side, side)).astype("float32")
    X_met = rng.normal(0, 1, size=(n, lookback, len(SA.MET_COLS) + 1)).astype("float32")
    y = (50 + rng.normal(0, 5, size=n)).astype("float32")
    dates = [dt.date(2025, 10, 1) + dt.timedelta(days=i) for i in range(n)]

    ds = SA.Dataset(X_sat=X_sat, X_met=X_met, y=y, dates=dates, station_ids=["s"] * n, channels=channels)
    _, _, report = SA.train(ds, holdout_days=10, epochs=3)

    assert report.n_train + report.n_holdout <= n
    assert report.n_holdout > 0
    for field in ("rmse", "mae", "r", "baseline_rmse"):
        assert np.isfinite(getattr(report, field)) or field == "r"  # r can be nan on tiny n
