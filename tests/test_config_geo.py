"""Config-driven-city and geometry tests.

PRD G1 says a new city is one config file. The cheapest way to keep that true is
to assert it over *every* file in config/cities, so a city added later is held to
the same contract without anyone remembering to add a test.
"""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from vayu_core.config import list_cities, load_city
from vayu_core.geo import (
    angular_diff,
    bearing_deg,
    haversine_km,
    idw,
    polygon_area_km2,
    representative_point,
)


def test_both_demo_cities_are_configured():
    ids = {c.id for c in list_cities()}
    assert {"delhi", "lucknow"} <= ids


@pytest.mark.parametrize("city", list_cities(), ids=lambda c: c.id)
def test_city_config_is_self_describing(city):
    w, s, e, n = city.bbox
    assert w < e and s < n, "bbox must be [west, south, east, north]"
    lon, lat = city.map_center
    assert w <= lon <= e and s <= lat <= n, "map centre must sit inside the bbox"
    assert city.wards.id_property and city.wards.name_property
    assert city.population.total > 0 and city.population.source
    assert "en" in city.languages and len(city.languages) >= 3  # PRD D2


def test_unknown_city_raises_keyerror():
    with pytest.raises(KeyError):
        load_city("atlantis")


def test_weather_grid_points_lie_inside_bbox_and_are_unique():
    city = load_city("delhi")
    pts = city.grid_points()
    assert len(pts) == city.weather_grid.nx * city.weather_grid.ny
    w, s, e, n = city.bbox
    for _, _, lat, lon in pts:
        assert w < lon < e and s < lat < n
    assert len({(i, j) for i, j, _, _ in pts}) == len(pts)


def test_haversine_against_known_distance():
    # Delhi (Connaught Place) -> Lucknow, ~ 420 km great-circle.
    d = haversine_km(28.6315, 77.2167, 26.8467, 80.9462)
    assert 400 < d < 440


def test_bearing_cardinal_directions():
    assert bearing_deg(28.0, 77.0, 29.0, 77.0) == pytest.approx(0, abs=0.5)  # north
    assert bearing_deg(28.0, 77.0, 28.0, 78.0) == pytest.approx(90, abs=0.5)  # east


def test_angular_diff_wraps_around_north():
    assert angular_diff(350, 10) == pytest.approx(20)
    assert angular_diff(10, 350) == pytest.approx(20)
    assert angular_diff(0, 180) == pytest.approx(180)


def test_polygon_area_km2_matches_analytic_square():
    # ~0.1 deg square near Delhi: width shrinks by cos(lat).
    lat0 = 28.6
    poly = Polygon([(77.0, lat0), (77.1, lat0), (77.1, lat0 + 0.1), (77.0, lat0 + 0.1)])
    expected = (111.32 * 0.1 * math.cos(math.radians(lat0 + 0.05))) * (110.57 * 0.1)
    assert polygon_area_km2(poly) == pytest.approx(expected, rel=0.02)


def test_representative_point_is_inside_a_concave_polygon():
    # C-shape: the true centroid falls outside the ward.
    c = Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)])
    assert not c.contains(c.centroid)
    lat, lon = representative_point(c)
    from shapely.geometry import Point

    assert c.contains(Point(lon, lat))


def test_idw_returns_exact_value_at_a_station():
    vals, near = idw([(28.6, 77.2)], [(28.6, 77.2), (28.7, 77.3)], [100.0, 200.0])
    assert vals[0] == pytest.approx(100.0)
    assert near[0] == pytest.approx(0.0, abs=1e-3)


def test_idw_is_bounded_by_its_inputs_and_favours_the_near_station():
    vals, _ = idw([(28.60, 77.20)], [(28.61, 77.20), (28.90, 77.20)], [100.0, 300.0])
    assert 100.0 < vals[0] < 300.0
    assert vals[0] < 150.0, "nearer station must dominate at p=2"


def test_idw_reports_distance_for_far_wards_and_nan_beyond_cutoff():
    vals, near = idw([(28.0, 77.0)], [(28.6, 77.2)], [100.0], max_km=25)
    assert np.isnan(vals[0]), "beyond max_km there is no honest estimate"
    assert near[0] > 25

def test_idw_with_no_stations_is_nan_not_a_crash():
    vals, near = idw([(28.6, 77.2)], [], [])
    assert np.isnan(vals[0]) and math.isinf(near[0])
