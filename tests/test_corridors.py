"""Economic corridors + the federated exchange format.

The properties worth pinning: corridors actually span state boundaries (that is
the whole point), geometry is not distorted by treating degrees as metres, and
the wire payload is self-describing enough for another agency to consume safely.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from vayu_core.config import load_region
from vayu_core.national import corridors as CO


def test_corridors_load_and_span_multiple_states():
    """A corridor confined to one state would defeat the purpose — the brief is
    explicitly about coordination ACROSS boundaries."""
    cs = CO.load_corridors("india")
    assert len(cs) >= 5
    assert all(len(c.states) >= 2 for c in cs)
    igp = CO.get_corridor("agra_kanpur_igp")
    assert igp is not None and len(igp.states) >= 6


def test_corridor_contains_cities_on_its_route():
    igp = CO.get_corridor("agra_kanpur_igp")
    for name, lat, lon in [("Delhi", 28.61, 77.10), ("Kanpur", 26.45, 80.35),
                           ("Kolkata", 22.57, 88.36), ("Ludhiana", 30.90, 75.85)]:
        assert igp.contains(lat, lon), f"{name} should lie on the IGP spine"
    # Somewhere far off-route must NOT be included.
    assert not igp.contains(13.08, 80.27), "Chennai is not on the IGP spine"


def test_longitude_is_scaled_by_latitude():
    """Degrees are not metres: at 30N a degree of longitude is ~87km against
    ~111km for latitude. Without the cos(lat) correction an east-west corridor
    would sample a visibly narrower strip than a north-south one."""
    d_lon = CO._dist_to_segment(1.0, 30.0, 0.0, 30.0, 0.0, 31.0)
    d_lat = CO._dist_to_segment(0.0, 31.0, 0.0, 30.0, 0.0, 31.0)
    assert d_lon < 1.0, "a degree of longitude at 30N must count as less than a full degree"
    assert d_lat == 0.0


def test_cells_are_inside_the_region_grid():
    region = load_region("india")
    cells = CO.get_corridor("cbic").cells(region)
    lats, lons = region.grid_axes()
    assert cells, "corridor must cover at least some grid cells"
    assert all(la in set(lats) and lo in set(lons) for la, lo in cells)


def _frame(cells, day, value=3e-4):
    return pd.DataFrame(
        [{"grid_lat": a, "grid_lon": b, "date": day, "value": value, "n_obs": 6} for a, b in cells]
    )


def test_bulletin_payload_is_self_describing():
    """A bare float in a shared feed is how cross-agency pipelines silently
    disagree — units and provenance must travel with the numbers."""
    region = load_region("india")
    c = CO.get_corridor("cbic")
    day = dt.date(2025, 11, 21)
    cells = c.cells(region)[:10]

    fires = pd.DataFrame(
        [{"grid_lat": cells[0][0], "grid_lon": cells[0][1], "date": day,
          "fire_count": 7, "frp_sum": 120.0}]
    )
    p = CO.build_bulletin(c, region, day, _frame(cells, day), fires).to_payload()

    assert p["schema"] == CO.SCHEMA_VERSION
    assert p["hcho"]["unit"] == "mol m-2"
    assert "Sentinel-5P" in p["hcho"]["source"]
    assert "FIRMS" in p["fire"]["source"]
    assert p["fire"]["count"] == 7
    assert p["corridor"]["states"] and p["date"] == day.isoformat()


def test_coverage_distinguishes_clean_from_unobserved():
    """A corridor the satellite could not see must not read as a clean one."""
    region = load_region("india")
    c = CO.get_corridor("cbic")
    day = dt.date(2025, 11, 21)
    all_cells = c.cells(region)

    seen = CO.build_bulletin(c, region, day, _frame(all_cells, day), pd.DataFrame()).to_payload()
    partial = CO.build_bulletin(
        c, region, day, _frame(all_cells[:3], day), pd.DataFrame()
    ).to_payload()

    assert seen["coverage"]["coverage_pct"] == 100.0
    assert partial["coverage"]["coverage_pct"] < 20.0
    assert partial["coverage"]["cells_total"] == seen["coverage"]["cells_total"]


def test_only_that_day_and_that_corridor_are_counted():
    region = load_region("india")
    c = CO.get_corridor("cbic")
    day = dt.date(2025, 11, 21)
    cells = c.cells(region)[:5]

    # One row inside on the target day, one inside on a different day, and one
    # far outside the corridor on the target day.
    rows = _frame(cells, day)
    rows = pd.concat([rows, _frame(cells, dt.date(2025, 11, 22))], ignore_index=True)
    rows = pd.concat(
        [rows, pd.DataFrame([{"grid_lat": 30.125, "grid_lon": 75.375, "date": day,
                              "value": 9e-4, "n_obs": 6}])],
        ignore_index=True,
    )
    p = CO.build_bulletin(c, region, day, rows, pd.DataFrame()).to_payload()
    assert p["coverage"]["cells_observed"] == len(cells), "leaked another day or another corridor"


def test_empty_inputs_do_not_crash():
    region = load_region("india")
    p = CO.build_bulletin(
        CO.get_corridor("cbic"), region, dt.date(2025, 11, 21), pd.DataFrame(), pd.DataFrame()
    ).to_payload()
    assert p["coverage"]["cells_observed"] == 0
    assert p["hcho"]["mean"] is None
    assert p["fire"]["count"] == 0
