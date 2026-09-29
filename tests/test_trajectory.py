"""Back-trajectory tests (TRD §10: "known uniform wind field → straight-line
trajectory of correct length/bearing").

The physics here is unobservable in the UI — a trajectory 3.6x too long (km/h
read as m/s) still draws a plausible line on a map and would point enforcement
at the wrong district. So the numbers are pinned against analytic truth.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from vayu_core.attribution.trajectory import (
    WindField,
    back_trajectory,
    km_per_deg_lon,
)
from vayu_core.config import load_city


def _uniform_weather(city, speed_kmh: float, direction_deg: float, hours: int = 48) -> pd.DataFrame:
    """A wind field that is the same everywhere and at every hour.

    Built on the AIRSHED grid, because that is what back-trajectories run on in
    production — a 24h path leaves the city bbox within two hours.
    """
    ts = pd.date_range("2025-11-01", periods=hours, freq="h", tz="UTC")
    rows = []
    for i, j, _lat, _lon in city.airshed_points():
        for t in ts:
            rows.append(
                {
                    "city": city.id, "grid": "airshed", "grid_i": i, "grid_j": j, "ts": t,
                    "wind_speed_100m": speed_kmh, "wind_dir_100m": direction_deg,
                    "wind_speed_10m": speed_kmh, "wind_dir_10m": direction_deg,
                    "kind": "hist",
                }
            )
    return pd.DataFrame(rows)


def test_uniform_wind_gives_a_straight_line_of_correct_length():
    city = load_city("delhi")
    # 20 km/h from the north-west (315 deg) — air travels to the south-east.
    wx = _uniform_weather(city, 20.0, 315.0)
    field = WindField(city, wx)
    assert field.available

    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 12, field)

    # Length = speed x duration, within 2% (TRD §10).
    assert t.length_km == pytest.approx(20.0 * 12, rel=0.02)
    assert t.mean_speed_kmh == pytest.approx(20.0, rel=0.02)

    # Straightness must be checked on the GROUND, not in lon/lat degrees: a
    # ground-straight path curves in degree space because km-per-degree-longitude
    # varies with latitude. Project to local km first.
    lat0 = float(t.polyline[0][1])
    pts = [
        (
            (float(p[0]) - float(t.polyline[0][0])) * km_per_deg_lon(lat0),
            (float(p[1]) - lat0) * 110.574,
        )
        for p in t.polyline
    ]
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    span = math.hypot(x1 - x0, y1 - y0)
    for x, y in pts:
        # Perpendicular distance from the chord, in km.
        deviation = abs((x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)) / span
        assert deviation < 0.5, f"uniform wind must give a straight path (off by {deviation:.2f} km)"


def test_back_trajectory_goes_upwind_not_downwind():
    city = load_city("delhi")
    # Wind FROM the north-west: the source of this air lies to the NORTH-WEST,
    # so the back-trajectory must travel north and west (lat up, lon down).
    field = WindField(city, _uniform_weather(city, 15.0, 315.0))
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 12, field)

    start_lon, start_lat = float(t.polyline[0][0]), float(t.polyline[0][1])
    end_lon, end_lat = float(t.polyline[-1][0]), float(t.polyline[-1][1])
    assert end_lat > start_lat, "should trace back toward the north"
    assert end_lon < start_lon, "should trace back toward the west"


@pytest.mark.parametrize(
    "direction, dlat, dlon",
    [
        (0.0, 1, 0),    # wind from the north -> source is north
        (180.0, -1, 0), # from the south      -> source is south
        (90.0, 0, 1),   # from the east       -> source is east
        (270.0, 0, -1), # from the west       -> source is west
    ],
)
def test_cardinal_directions_trace_to_the_right_quadrant(direction, dlat, dlon):
    city = load_city("delhi")
    field = WindField(city, _uniform_weather(city, 12.0, direction))
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 6, field)
    d_lat = float(t.polyline[-1][1]) - float(t.polyline[0][1])
    d_lon = float(t.polyline[-1][0]) - float(t.polyline[0][0])
    if dlat:
        assert math.copysign(1, d_lat) == dlat and abs(d_lat) > 1e-3
    if dlon:
        assert math.copysign(1, d_lon) == dlon and abs(d_lon) > 1e-3


def test_timestamps_run_backwards_from_the_start():
    city = load_city("delhi")
    field = WindField(city, _uniform_weather(city, 10.0, 270.0))
    start = pd.Timestamp("2025-11-02T00:00Z")
    t = back_trajectory(city, "W1", 28.65, 77.10, start, 6, field)
    times = [pd.Timestamp(p[2]) for p in t.polyline]
    assert times[0] == start
    assert all(times[i] > times[i + 1] for i in range(len(times) - 1)), "time must run backwards"
    assert (times[0] - times[-1]) == pd.Timedelta(hours=6)


def test_cone_widens_with_distance_and_is_a_closed_ring():
    city = load_city("delhi")
    field = WindField(city, _uniform_weather(city, 20.0, 315.0))
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 24, field)

    assert t.cone and t.cone[0] == t.cone[-1], "cone ring must be closed"

    # Width perpendicular to the path must grow as we move away from the ward.
    n = len(t.polyline)
    def width_at(i: int) -> float:
        left, right = t.cone[i], t.cone[len(t.cone) - 2 - i]
        lat = float(t.polyline[i][1])
        dx = (left[0] - right[0]) * km_per_deg_lon(lat)
        dy = (left[1] - right[1]) * 110.574
        return math.hypot(dx, dy)

    near, far = width_at(2), width_at(n - 3)
    assert far > near, "cone must widen with distance from the ward"
    assert near < 5.0, "cone should start narrow at the ward"


def test_stagnant_air_is_flagged_not_silently_drawn():
    """Calm wind makes attribution meaningless — the UI must be told."""
    city = load_city("delhi")
    field = WindField(city, _uniform_weather(city, 0.5, 90.0))
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 24, field)
    assert t.stagnant is True
    assert t.length_km < 15


def test_no_wind_data_returns_empty_not_a_crash():
    city = load_city("delhi")
    field = WindField(city, pd.DataFrame())
    assert not field.available
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 12, field)
    assert t.polyline == [] and t.cone == []


def test_trajectory_may_leave_the_city_bbox():
    """Stubble smoke comes from Punjab — clamping the field must not trap the
    parcel inside the municipal boundary."""
    city = load_city("delhi")
    field = WindField(city, _uniform_weather(city, 25.0, 315.0))
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 24, field)
    w, s, e, n = city.bbox
    end_lon, end_lat = float(t.polyline[-1][0]), float(t.polyline[-1][1])
    assert end_lat > n or end_lon < w, "24h upwind at 25 km/h must exit the city bbox"


def test_geojson_shape_is_valid():
    city = load_city("delhi")
    field = WindField(city, _uniform_weather(city, 15.0, 315.0))
    t = back_trajectory(city, "W1", 28.65, 77.10, pd.Timestamp("2025-11-02T00:00Z"), 12, field)
    gj = t.to_geojson()
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 2
    line, cone = gj["features"]
    assert line["geometry"]["type"] == "LineString"
    assert cone["geometry"]["type"] == "Polygon"
    assert len(line["geometry"]["coordinates"]) > 10
