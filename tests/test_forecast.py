"""Forecaster tests — leakage, lag correctness, and band coherence.

These target the failure modes that are invisible at runtime: a leaking feature
or a mis-aligned lag produces a *better-looking* model, not a crash, and would
sail through a demo while making the headline RMSE a lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vayu_core.config import load_city
from vayu_core.forecast.features import (
    FEATURE_COLUMNS,
    PM_LAGS,
    add_target,
    build_features,
)


def _synthetic(hours: int = 400, stations: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A known series: pm25 = station_offset + hour_index, so every lag is exact."""
    ts = pd.date_range("2025-10-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for s in range(stations):
        for i, t in enumerate(ts):
            rows.append(
                {"city": "delhi", "station_id": f"s{s}", "param": "pm25", "ts": t, "value": float(s * 1000 + i)}
            )
    meas = pd.DataFrame(rows)
    st = pd.DataFrame(
        [
            {"city": "delhi", "station_id": f"s{s}", "name": f"S{s}", "lat": 28.5 + 0.1 * s, "lon": 77.1}
            for s in range(stations)
        ]
    )
    return meas, st


def test_lags_are_hours_not_rows():
    """A gap in the feed must not silently turn lag24 into "24 rows back"."""
    meas, st = _synthetic(hours=200, stations=1)
    # Punch a 5-hour hole in the middle of the series.
    hole = meas["ts"].iloc[100:105]
    meas = meas[~meas["ts"].isin(hole)]

    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st)
    f = f.dropna(subset=["pm25", "pm25_lag1"])
    # pm25 increments by 1 per hour, so lag1 must be exactly pm25 - 1 everywhere
    # the previous hour exists — never pm25 - 6 because rows were skipped.
    diff = (f["pm25"] - f["pm25_lag1"]).dropna()
    assert set(np.round(diff.unique(), 6)) <= {1.0}, "lag misaligned across a gap"


@pytest.mark.parametrize("lag", PM_LAGS)
def test_each_lag_looks_back_exactly_that_many_hours(lag):
    meas, st = _synthetic(hours=300, stations=1)
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st)
    f = f.dropna(subset=["pm25", f"pm25_lag{lag}"])
    diff = (f["pm25"] - f[f"pm25_lag{lag}"]).round(6).unique()
    assert set(diff) <= {float(lag)}


def test_rolling_means_exclude_the_current_hour():
    """A rolling window that includes t leaks the value being predicted from."""
    meas, st = _synthetic(hours=200, stations=1)
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st).dropna(subset=["pm25_roll6"])
    row = f.iloc[50]
    # Series is i -> i, so a 6h mean over the PREVIOUS 6 hours ending at t-1 is
    # mean(t-6 .. t-1) = t - 3.5. If it wrongly included t it would be t - 2.5.
    assert row["pm25_roll6"] == pytest.approx(row["pm25"] - 3.5, abs=1e-6)


def test_target_is_strictly_in_the_future():
    meas, st = _synthetic(hours=200, stations=1)
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st)
    d = add_target(f, 24).dropna(subset=["y"])
    # y must be the value 24h ahead: y - pm25 == 24 for our synthetic ramp.
    assert set((d["y"] - d["pm25"]).round(6).unique()) <= {24.0}


def test_no_feature_is_the_target_in_disguise():
    """No feature column may correlate perfectly with y — that is leakage."""
    meas, st = _synthetic(hours=400, stations=2)
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st)
    d = add_target(f, 24).dropna(subset=["y", "pm25_lag1"])
    for col in ("pm25_lag1", "pm25_roll6", "pm25_roll24"):
        # On a pure ramp these are highly correlated with y by construction, but
        # none may be *identical* to it.
        assert not np.allclose(d[col].to_numpy(), d["y"].to_numpy()), f"{col} equals the target"


def test_features_are_stable_and_documented():
    from vayu_core.forecast.features import FEATURE_LABELS

    # Every model input needs a plain-English label for the "Why?" panel (A4).
    missing = [c for c in FEATURE_COLUMNS if c not in FEATURE_LABELS]
    assert not missing, f"features without a human label: {missing}"


def test_build_features_survives_a_station_with_no_data():
    meas, st = _synthetic(hours=100, stations=1)
    st = pd.concat(
        [st, pd.DataFrame([{"city": "delhi", "station_id": "ghost", "name": "Ghost", "lat": 28.7, "lon": 77.3}])],
        ignore_index=True,
    )
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st)
    assert not f.empty  # a dead station must not sink the frame


def test_quantile_band_is_ordered_after_prediction():
    """p10 <= p50 <= p90 must hold per row even though the models are independent."""
    from vayu_core.forecast.model import Forecaster

    fc = Forecaster()
    if not fc.available:
        pytest.skip("no trained models yet — run `make seed`")
    meas, st = _synthetic(hours=200, stations=2)
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st).dropna(subset=["pm25_lag1"])
    from vayu_core.forecast.features import model_frame
    from vayu_core.forecast.model import add_city_code

    # The model expects the horizon's fx_* columns; predicting off the base
    # frame alone would raise rather than silently mispredict.
    rows = model_frame(f.tail(20), pd.DataFrame(), 24, with_target=False)
    out = fc.predict(add_city_code(rows, "delhi"), 24)
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()
    assert (out >= 0).all().all(), "negative concentration is unphysical"


def test_predict_is_anchored_to_the_current_level():
    """The models emit a residual; predict() must add the anchor back.

    This is what lets the forecaster represent a 500 µg/m³ morning that no
    training row ever reached — trees cannot extrapolate a level target, and a
    level-target model lost to persistence by 53% on real Delhi data.
    """
    from vayu_core.forecast.features import model_frame
    from vayu_core.forecast.model import Forecaster, add_city_code, anchor_of

    fc = Forecaster()
    if not fc.available:
        pytest.skip("no trained models yet — run `make seed`")

    meas, st = _synthetic(hours=200, stations=1)
    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st).dropna(subset=["pm25_lag1"])
    rows = add_city_code(model_frame(f.tail(5), pd.DataFrame(), 24, with_target=False), "delhi")

    out = fc.predict(rows, 24)
    anchor = anchor_of(rows).to_numpy()
    # Predictions must track the anchor, not collapse to a learned constant.
    assert np.all(np.abs(out["p50"].to_numpy() - anchor) < 400), "prediction detached from its anchor"

    # Shift the anchor far above anything in training; the prediction must move
    # with it rather than saturate.
    hot = rows.copy()
    for c in ("pm25", "pm25_lag1", "pm25_lag3", "pm25_roll6", "pm25_roll24"):
        hot[c] = hot[c] + 900.0
    hot_out = fc.predict(hot, 24)
    assert (hot_out["p50"].to_numpy() > out["p50"].to_numpy() + 500).all(), (
        "a 900 µg/m³ higher anchor must raise the forecast — this is the "
        "extrapolation failure the residual target exists to prevent"
    )


def test_anchor_falls_back_to_lag_when_the_hour_is_missing():
    from vayu_core.forecast.model import anchor_of

    df = pd.DataFrame({"pm25": [100.0, np.nan], "pm25_lag1": [99.0, 77.0]})
    a = anchor_of(df)
    assert a.iloc[0] == 100.0
    assert a.iloc[1] == 77.0, "a dropped hour must not drop the row"


def test_lags_do_not_span_a_multi_year_gap():
    """Delhi's record runs 2016-2018, then resumes 2025 on a new instrument.

    Reindexing across that hole would both explode memory (~60k empty hours per
    station) and let `pm25_lag24` reach back years for "yesterday".
    """
    early = pd.date_range("2016-11-01", periods=72, freq="h", tz="UTC")
    late = pd.date_range("2025-11-01", periods=72, freq="h", tz="UTC")
    rows = [
        {"city": "delhi", "station_id": "s0", "param": "pm25", "ts": t, "value": 400.0 + i}
        for i, t in enumerate(early)
    ] + [
        {"city": "delhi", "station_id": "s0", "param": "pm25", "ts": t, "value": 100.0 + i}
        for i, t in enumerate(late)
    ]
    meas = pd.DataFrame(rows)
    st = pd.DataFrame([{"city": "delhi", "station_id": "s0", "name": "S0", "lat": 28.6, "lon": 77.1}])

    f = build_features(load_city("delhi"), meas, pd.DataFrame(), st)
    # Only the two real blocks may exist — not nine years of empty hours.
    assert len(f) < 200, f"reindex spanned the gap: {len(f):,} rows"

    # No 2025 row may carry a lag sourced from 2016.
    late_rows = f[f["ts"] >= "2025-01-01"].dropna(subset=["pm25_lag24"])
    assert (late_rows["pm25_lag24"] < 300).all(), "a lag reached across the 7-year gap"
