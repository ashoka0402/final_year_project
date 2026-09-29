"""The vectorised upwind/fire features must equal the naive definition.

`_add_fires` and `_add_upwind` were rewritten from row-by-row Python loops into
numpy broadcasts because the naive form is ~864M iterations on real data and
hangs the seed. A fast rewrite that quietly changes the numbers is worse than a
slow one, so these tests re-derive both features the obvious way and demand an
exact match.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from vayu_core.config import load_city
from vayu_core.forecast.features import (
    UPWIND_HALF_ANGLE,
    UPWIND_MIN_KM,
    build_features,
)
from vayu_core.geo import angular_diff, bearing_deg, haversine_km

CITY = load_city("delhi")


def _stations() -> pd.DataFrame:
    # A north-south line so "upwind" is unambiguous under a northerly wind.
    return pd.DataFrame(
        [
            {"city": "delhi", "station_id": "s_south", "name": "South", "lat": 28.50, "lon": 77.10},
            {"city": "delhi", "station_id": "s_mid", "name": "Mid", "lat": 28.65, "lon": 77.10},
            {"city": "delhi", "station_id": "s_north", "name": "North", "lat": 28.80, "lon": 77.10},
        ]
    )


def _measurements(hours: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2025-11-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for k, sid in enumerate(("s_south", "s_mid", "s_north")):
        for i, t in enumerate(ts):
            rows.append(
                {"city": "delhi", "station_id": sid, "param": "pm25", "ts": t, "value": 50.0 + 10 * k + i}
            )
    return pd.DataFrame(rows)


def _weather(direction_deg: float, speed: float = 12.0, hours: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2025-11-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for i, j, _lat, _lon in CITY.grid_points():
        for t in ts:
            rows.append(
                {
                    "city": "delhi", "grid": "city", "grid_i": i, "grid_j": j, "ts": t,
                    "wind_speed_100m": speed, "wind_dir_100m": direction_deg,
                    "wind_speed_10m": speed, "wind_dir_10m": direction_deg,
                    "pblh": 500.0, "rh": 50.0, "temp_c": 20.0, "precip": 0.0, "pressure": 1010.0,
                    "kind": "hist",
                }
            )
    return pd.DataFrame(rows)


def _naive_fires(df: pd.DataFrame, fires: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    """The definition, written the slow obvious way."""
    st = {s.station_id: (s.lat, s.lon) for s in stations.itertuples()}
    f = fires.copy()
    f["acq_ts"] = pd.to_datetime(f["acq_ts"], utc=True)
    upwind = (np.degrees(np.arctan2(-df["u"], -df["v"])) + 360) % 360

    out = np.zeros(len(df))
    for idx, (sid, ts, bear) in enumerate(zip(df["station_id"], df["ts"], upwind)):
        if np.isnan(bear) or sid not in st:
            continue
        lat_s, lon_s = st[sid]
        total = 0.0
        for fr in f.itertuples():
            age_h = (pd.Timestamp(ts) - pd.Timestamp(fr.acq_ts)).total_seconds() / 3600.0
            if not (0 <= age_h <= 24):
                continue
            if haversine_km(lat_s, lon_s, fr.lat, fr.lon) > 50:
                continue
            if angular_diff(bearing_deg(lat_s, lon_s, fr.lat, fr.lon), bear) <= UPWIND_HALF_ANGLE:
                total += float(fr.frp)
        out[idx] = total
    return out


def test_vectorised_fires_match_the_naive_definition():
    stations = _stations()
    meas = _measurements()
    # Wind from the north (0 deg) -> upwind bearing is 0, i.e. look north.
    wx = _weather(0.0)

    ts0 = pd.Timestamp("2025-11-01T12:00Z")
    fires = pd.DataFrame(
        [
            # North of s_mid, within 50 km and 24h -> should count.
            {"lat": 28.90, "lon": 77.10, "frp": 12.0, "acq_ts": ts0},
            # South of s_mid (downwind) -> must NOT count.
            {"lat": 28.40, "lon": 77.10, "frp": 99.0, "acq_ts": ts0},
            # North but far beyond 50 km -> must NOT count.
            {"lat": 30.20, "lon": 77.10, "frp": 50.0, "acq_ts": ts0},
            # North, near, but 3 days stale -> must NOT count.
            {"lat": 28.88, "lon": 77.12, "frp": 40.0, "acq_ts": ts0 - pd.Timedelta(days=3)},
        ]
    )

    f = build_features(CITY, meas, wx, stations, fires)
    got = f["upwind_fire_frp_24h"].to_numpy()
    want = _naive_fires(f, fires, stations)

    assert np.allclose(got, want), "vectorised fire feature diverges from the definition"
    assert got.max() > 0, "the test fixture should produce at least one upwind fire hit"
    # The 99 MW downwind fire must never appear.
    assert got.max() == pytest.approx(12.0)


def test_fires_outside_the_cone_and_window_contribute_zero():
    stations = _stations()
    meas = _measurements()
    wx = _weather(0.0)  # wind from north
    ts0 = pd.Timestamp("2025-11-01T12:00Z")
    # Only a downwind fire exists.
    fires = pd.DataFrame([{"lat": 28.40, "lon": 77.10, "frp": 500.0, "acq_ts": ts0}])
    f = build_features(CITY, meas, wx, stations, fires)
    assert f["upwind_fire_frp_24h"].max() == 0.0


def test_no_fire_data_gives_zero_not_nan():
    """Zero means "no fires upwind" and is a real value the model can learn from."""
    f = build_features(CITY, _measurements(), _weather(0.0), _stations(), pd.DataFrame())
    assert (f["upwind_fire_frp_24h"] == 0.0).all()
    assert not f["upwind_fire_frp_24h"].isna().any()


def test_upwind_station_is_the_one_actually_upwind():
    stations = _stations()
    meas = _measurements()

    # Wind FROM the north: s_mid's upwind neighbour must be s_north.
    f = build_features(CITY, meas, _weather(0.0), stations, pd.DataFrame())
    mid = f[(f["station_id"] == "s_mid") & f["upwind_pm25"].notna()]
    north = f[f["station_id"] == "s_north"].set_index("ts")["pm25"]
    assert not mid.empty
    for r in mid.head(10).itertuples():
        assert r.upwind_pm25 == pytest.approx(north.loc[r.ts])

    # Flip the wind to come FROM the south: it must switch to s_south.
    f2 = build_features(CITY, meas, _weather(180.0), stations, pd.DataFrame())
    mid2 = f2[(f2["station_id"] == "s_mid") & f2["upwind_pm25"].notna()]
    south = f2[f2["station_id"] == "s_south"].set_index("ts")["pm25"]
    assert not mid2.empty
    for r in mid2.head(10).itertuples():
        assert r.upwind_pm25 == pytest.approx(south.loc[r.ts])


def test_upwind_respects_the_distance_band():
    """Stations closer than 5 km or beyond 50 km are not upwind candidates."""
    stations = pd.DataFrame(
        [
            {"city": "delhi", "station_id": "a", "name": "A", "lat": 28.65, "lon": 77.10},
            # ~1 km north — too close, inside the exclusion radius.
            {"city": "delhi", "station_id": "b", "name": "B", "lat": 28.659, "lon": 77.10},
        ]
    )
    ts = pd.date_range("2025-11-01", periods=40, freq="h", tz="UTC")
    meas = pd.DataFrame(
        [
            {"city": "delhi", "station_id": s, "param": "pm25", "ts": t, "value": 100.0}
            for s in ("a", "b")
            for t in ts
        ]
    )
    d = haversine_km(28.65, 77.10, 28.659, 77.10)
    assert d < UPWIND_MIN_KM  # fixture sanity

    f = build_features(CITY, meas, _weather(0.0, hours=40), stations, pd.DataFrame())
    a = f[f["station_id"] == "a"]
    assert a["upwind_pm25"].isna().all(), "a station 1 km away must not count as upwind"
