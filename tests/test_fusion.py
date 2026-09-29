"""Evidence-fusion and confidence tests (TRD 5.3).

Attribution is the claim VAYU stakes its credibility on: a share here becomes an
enforcement order against a real location. So the properties asserted are the
ones that make the formula defensible rather than decorative —

  * a fire outside the cone must contribute nothing (no blaming upwind-of-nothing)
  * shares must sum to 100 and follow the evidence
  * stagnant air must REFUSE to attribute rather than guess
  * every share must carry clickable evidence (PRD B2)
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from vayu_core.attribution.confidence import (
    category_confidence,
    station_agreement_score,
    wind_stability,
)
from vayu_core.attribution.fusion import attribute
from vayu_core.attribution.trajectory import WindField, back_trajectory
from vayu_core.config import load_city

CITY = load_city("delhi")
AT = pd.Timestamp("2025-11-03T06:00Z")
WARD_LAT, WARD_LON = 28.65, 77.10


def _weather(city, speed_kmh: float, direction_deg: float, hours: int = 48) -> pd.DataFrame:
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


def _traj(speed=18.0, direction=315.0, hours=12):
    """Wind FROM the north-west -> the cone opens to the north-west."""
    field = WindField(CITY, _weather(CITY, speed, direction))
    return back_trajectory(CITY, "W1", WARD_LAT, WARD_LON, AT, hours, field)


def _fire(lat, lon, frp=40.0, age_h=3):
    return {"lat": lat, "lon": lon, "frp": frp, "confidence": "h", "acq_ts": AT - timedelta(hours=age_h)}


def test_fire_inside_the_cone_drives_open_burning():
    t = _traj()
    # Place a fire on the trajectory path itself — guaranteed inside the cone.
    mid = t.polyline[len(t.polyline) // 2]
    fires = pd.DataFrame([_fire(float(mid[1]), float(mid[0]))])

    a = attribute(CITY, "W1", "Test Ward", WARD_LAT, WARD_LON, t, AT, fires=fires)
    burn = next((c for c in a.categories if c.category == "open_burning"), None)
    assert burn is not None and burn.share_pct > 0
    assert burn.evidence, "a share with no evidence is unusable (PRD B2)"
    assert burn.evidence[0].type == "fire"
    assert burn.evidence[0].lat is not None and burn.evidence[0].lon is not None


def test_fire_downwind_of_the_ward_contributes_nothing():
    """The cone opens upwind. A fire the other way cannot have caused this air."""
    t = _traj(direction=315.0)  # wind from NW -> cone to the NW
    # Put a fire to the SOUTH-EAST, i.e. downwind.
    fires = pd.DataFrame([_fire(WARD_LAT - 0.4, WARD_LON + 0.4, frp=500.0)])

    a = attribute(CITY, "W1", "Test Ward", WARD_LAT, WARD_LON, t, AT, fires=fires)
    burn = next((c for c in a.categories if c.category == "open_burning"), None)
    assert burn is None or burn.share_pct == 0, "a downwind fire was blamed for upwind air"


def test_shares_sum_to_100():
    t = _traj()
    mid = t.polyline[len(t.polyline) // 2]
    fires = pd.DataFrame([_fire(float(mid[1]), float(mid[0]))])
    a = attribute(
        CITY, "W1", "Test Ward", WARD_LAT, WARD_LON, t, AT,
        fires=fires, road_density=12.0, no2=70.0, regional_pm_proxy=1.0,
    )
    assert a.categories
    assert sum(c.share_pct for c in a.categories) == pytest.approx(100.0, abs=1.5)


def test_more_fire_power_raises_the_burning_share():
    t = _traj()
    mid = t.polyline[len(t.polyline) // 2]
    lat, lon = float(mid[1]), float(mid[0])

    def burn_share(frp):
        fires = pd.DataFrame([_fire(lat, lon, frp=frp)])
        a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, fires=fires, road_density=10.0)
        c = next((x for x in a.categories if x.category == "open_burning"), None)
        return c.share_pct if c else 0.0

    assert burn_share(200.0) > burn_share(10.0), "share must follow the evidence"


def test_older_fires_count_less_than_fresh_ones():
    t = _traj()
    mid = t.polyline[len(t.polyline) // 2]
    lat, lon = float(mid[1]), float(mid[0])

    def burn_share(age_h):
        fires = pd.DataFrame([_fire(lat, lon, frp=100.0, age_h=age_h)])
        a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, fires=fires, road_density=10.0)
        c = next((x for x in a.categories if x.category == "open_burning"), None)
        return c.share_pct if c else 0.0

    assert burn_share(1) > burn_share(20), "recency decay is not applied"


def test_fires_outside_the_lookback_window_are_ignored():
    t = _traj()
    mid = t.polyline[len(t.polyline) // 2]
    # 3 days old — outside the 24h window entirely.
    fires = pd.DataFrame([_fire(float(mid[1]), float(mid[0]), frp=900.0, age_h=72)])
    a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, fires=fires, road_density=10.0)
    burn = next((c for c in a.categories if c.category == "open_burning"), None)
    assert burn is None or burn.share_pct == 0


def test_stagnant_air_refuses_to_attribute():
    """App Flow §3.2: say "local sources dominant", never draw a confident donut."""
    t = _traj(speed=0.4)
    assert t.stagnant
    a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, road_density=20.0)
    assert a.stagnant is True
    assert a.categories == []
    assert a.note and "stagnant" in a.note.lower()


def test_no_evidence_gives_an_honest_note_not_a_fake_split():
    t = _traj()
    a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT)  # no fires/roads/permits
    # Regional transport may still score off trajectory geometry alone; if
    # nothing at all scores, we must say so rather than invent a split.
    if not a.categories:
        assert a.note


def test_non_compliant_permit_is_weighted_double():
    t = _traj()
    # Construction dust is LOCAL — it settles within km, so the fusion decays it
    # over 10 km. Put the site a few km up the path, not 100 km away.
    near = t.polyline[3]
    lat, lon = float(near[1]), float(near[0])

    def share(compliant):
        p = pd.DataFrame([{
            "permit_id": "X-1", "name": "Site", "site_type": "Tower", "lat": lat, "lon": lon,
            "status": "active", "dust_control_compliant": compliant, "last_inspected": "2025-10-01",
        }])
        a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, permits=p, road_density=10.0)
        c = next((x for x in a.categories if x.category == "construction"), None)
        return c.share_pct if c else 0.0

    assert share(False) > share(True), "a non-compliant site must outweigh a compliant one"


def test_permit_evidence_carries_its_sample_badge():
    """The badge must travel with the data, so the UI cannot forget it."""
    t = _traj()
    near = t.polyline[3]
    p = pd.DataFrame([{
        "permit_id": "X-1", "name": "Site", "site_type": "Tower",
        "lat": float(near[1]), "lon": float(near[0]), "status": "active",
        "dust_control_compliant": False, "last_inspected": "2025-10-01",
    }])
    a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, permits=p)
    c = next(x for x in a.categories if x.category == "construction")
    assert "sample" in c.evidence[0].source.lower()


def test_rush_hour_raises_the_traffic_share():
    t = _traj()
    # 08:30 IST vs 03:30 IST — same ward, same roads.
    rush = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t,
                     pd.Timestamp("2025-11-03T03:00Z"), road_density=20.0)  # 08:30 IST
    night = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t,
                      pd.Timestamp("2025-11-02T22:00Z"), road_density=20.0)  # 03:30 IST
    r = next(c.share_pct for c in rush.categories if c.category == "traffic")
    n = next(c.share_pct for c in night.categories if c.category == "traffic")
    assert r > n, "rush-hour factor is not being applied"


def test_regional_share_appears_when_the_path_leaves_the_city():
    # 24h at 18 km/h definitely exits Delhi's bbox.
    t = _traj(hours=24)
    a = attribute(CITY, "W1", "W", WARD_LAT, WARD_LON, t, AT, regional_pm_proxy=1.0, road_density=5.0)
    reg = next((c for c in a.categories if c.category == "regional_transport"), None)
    assert reg is not None and reg.share_pct > 0
    assert reg.evidence and "outside the city" in reg.evidence[0].label


# ---- confidence -------------------------------------------------------------

def test_confidence_rises_with_evidence_and_wind_quality():
    good = _traj(speed=18.0)
    low = category_confidence("open_burning", 1, good)
    high = category_confidence("open_burning", 10, good)
    assert 0 <= low <= 1 and 0 <= high <= 1
    assert high > low


def test_stagnant_wind_collapses_confidence():
    calm = _traj(speed=0.4)
    steady = _traj(speed=18.0)
    assert wind_stability(calm) == 0.0
    assert wind_stability(steady) > 0.5
    assert category_confidence("open_burning", 8, calm) < category_confidence("open_burning", 8, steady)


def test_proxy_categories_are_less_confident_than_measured_ones():
    """A fire pixel is observed; a traffic share is inferred. Say so."""
    t = _traj()
    assert category_confidence("open_burning", 5, t) > category_confidence("traffic", 5, t)


def test_station_agreement_scoring():
    assert station_agreement_score([100, 101, 99]) > 0.9      # tight
    assert station_agreement_score([50, 300, 120]) < 0.5      # scattered
    assert station_agreement_score([100]) == 0.5              # can't corroborate itself
    assert station_agreement_score([]) == 0.5
