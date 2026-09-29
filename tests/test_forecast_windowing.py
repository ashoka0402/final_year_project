"""Windowed vs full-history feature build (Phase L — going live).

Live forecasting and the date picker need scoring to take seconds, not ~80s. The
fix builds features only over the ~10 days before `at`. These tests pin the two
properties that make that safe:

  1. The windowed forecast is IDENTICAL to the full-history one (it must be a
     speedup, not a different answer).
  2. A station whose last reading is stale (years old) never drives a "now"
     forecast — the bug the windowing surfaced: 2016/2018 readings were being
     fed into today's nowcast.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vayu_core.config import load_city
from vayu_core.forecast.model import Forecaster
from vayu_core.forecast.run import _latest_station_rows, run_forecast
from vayu_core.forecast.features import SCORING_WINDOW_DAYS, build_features

AT = pd.Timestamp("2025-11-03T06:00Z")


@pytest.fixture(scope="module")
def loaded():
    from services.pipeline.score import _load

    city = load_city("delhi")
    return city, _load(city)


@pytest.fixture(scope="module")
def fc():
    f = Forecaster()
    if not f.available:
        pytest.skip("no trained model artifacts")
    return f


def test_windowed_forecast_matches_full_history(loaded, fc):
    city, (meas, wx, st, wards, fires) = loaded
    full = run_forecast(city, meas, wx, st, wards, fires, AT, forecaster=fc, windowed=False)
    win = run_forecast(city, meas, wx, st, wards, fires, AT, forecaster=fc, windowed=True)
    assert not full.empty and not win.empty
    m = full.merge(win, on=["ward_id", "horizon_h"], suffixes=("_f", "_w"))
    assert len(m) == len(full)
    for col in ("p10", "p50", "p90", "aqi_p50"):
        assert (m[f"{col}_f"] - m[f"{col}_w"]).abs().max() == 0, f"{col} diverged"


def test_stale_stations_never_drive_a_nowcast(loaded):
    """A reading older than the scoring window must be excluded as a source."""
    city, (meas, wx, st, wards, fires) = loaded
    feats = build_features(city, meas, wx, st, fires, since=None)  # full history
    latest = _latest_station_rows(feats, AT)
    assert not latest.empty
    age_days = (AT - pd.to_datetime(latest["ts"], utc=True)).dt.total_seconds() / 86400
    assert age_days.max() <= SCORING_WINDOW_DAYS, (
        "a stale (years-old) station leaked into the current forecast sources"
    )
