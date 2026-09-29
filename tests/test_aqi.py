"""CPCB AQI conversion tests.

These pin the breakpoints in TRD 5.1. If someone "tidies" the table, the demo's
headline number (AQI 312) changes meaning — so the band edges are asserted
explicitly rather than round-tripped.
"""

import pytest

from vayu_core.aqi import (
    aqi_from_concentrations,
    aqi_from_pm25,
    category_for,
    sub_index,
)


@pytest.mark.parametrize(
    "pm25, expected",
    [
        (0, 0),
        (30, 50),  # top of Good
        (60, 100),  # top of Satisfactory
        (90, 200),  # top of Moderate
        (120, 300),  # top of Poor
        (250, 400),  # top of Very Poor
        (500, 500),  # top of Severe
    ],
)
def test_pm25_band_edges_are_exact(pm25, expected):
    assert aqi_from_pm25(pm25) == expected


def test_pm25_interpolates_linearly_inside_a_band():
    # Midpoint of the 60-90 band (-> 101-200) is 75 ug/m3 -> ~150.
    assert aqi_from_pm25(75) == pytest.approx(150, abs=1)


def test_demo_episode_pm25_maps_into_severe():
    # The real CAMS peak driving the golden flow (~333 ug/m3 on 2025-12-14)
    # must land in Severe, i.e. above the 300 threshold the alert keys on.
    aqi = aqi_from_pm25(333.4)
    assert aqi is not None and aqi > 400
    assert category_for(aqi)[0] == "Severe"


def test_above_top_breakpoint_clamps_at_500_not_extrapolated():
    assert aqi_from_pm25(900) == 500


def test_missing_and_negative_are_none_not_zero():
    # None means "no data"; 0 would be a claim of clean air.
    assert aqi_from_pm25(None) is None
    assert sub_index("pm25", -5) is None
    assert sub_index("unobtainium", 10) is None


def test_worst_pollutant_wins():
    r = aqi_from_concentrations({"pm25": 40, "pm10": 300})  # ~68 vs ~250
    assert r is not None
    assert r.dominant_param == "pm10"
    assert r.aqi == pytest.approx(250, abs=2)


def test_requires_a_particulate_to_report_station_aqi():
    # CPCB will not issue an AQI off gases alone; neither will we.
    assert aqi_from_concentrations({"co": 5, "no2": 50}) is None
    assert aqi_from_concentrations({}) is None
    assert aqi_from_concentrations({"pm25": 55}) is not None


@pytest.mark.parametrize(
    "aqi, label",
    [
        (25, "Good"),
        (75, "Satisfactory"),
        (150, "Moderate"),
        (250, "Poor"),
        (312, "Very Poor"),
        (450, "Severe"),
    ],
)
def test_categories_match_cpcb_bands(aqi, label):
    assert category_for(aqi)[0] == label


def test_sub_index_is_continuous_across_band_joins():
    # No gap at a breakpoint: 30.0 and 30.001 must not jump by a whole category.
    assert abs(aqi_from_pm25(30) - aqi_from_pm25(30.001)) < 2
