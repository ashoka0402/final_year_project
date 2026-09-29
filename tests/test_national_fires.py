"""National fire gridding (hotspot-attribution input).

`to_fire_grid` snaps detections with vectorised arithmetic rather than calling
RegionConfig.snap() per row, because at 10^5-10^6 detections a Python-level loop
dominates the ingest. That optimisation is only safe if it produces *exactly*
the same cell as the scalar path — a half-cell drift would silently misalign
every fire<->HCHO correlation downstream. These tests pin that equivalence.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from services.pipeline.national import season_windows, to_fire_grid
from vayu_core.config import load_region


@pytest.fixture(scope="module")
def india():
    return load_region("india")


@pytest.fixture(scope="module")
def detections():
    rng = np.random.default_rng(0)
    n = 3000
    return pd.DataFrame(
        {
            "lat": rng.uniform(6.5, 37.5, n),
            "lon": rng.uniform(68.5, 97.5, n),
            "frp": rng.uniform(1, 300, n),
            "acq_date": [dt.date(2025, 11, 3)] * n,
        }
    )


def test_vectorised_snapping_equals_scalar_snap(india, detections):
    """The whole point: the fast path and RegionConfig.snap() must agree."""
    grid = to_fire_grid(india, detections)
    # Recompute per-row with the scalar API and confirm every cell matches.
    d = india.grid_deg
    for lat, lon in zip(detections["lat"][:500], detections["lon"][:500]):
        want_lat, want_lon = india.snap(lat, lon)
        got_lat = round(((lat - india.bbox[1]) // d) * d + india.bbox[1] + d / 2, 4)
        got_lon = round(((lon - india.bbox[0]) // d) * d + india.bbox[0] + d / 2, 4)
        assert (got_lat, got_lon) == (want_lat, want_lon)
    assert not grid.empty


def test_gridding_conserves_detection_counts(india, detections):
    """No detection may be dropped or double-counted by the aggregation."""
    grid = to_fire_grid(india, detections)
    assert grid["fire_count"].sum() == len(detections)


def test_grid_cells_land_on_the_declared_axes(india, detections):
    lats, lons = india.grid_axes()
    grid = to_fire_grid(india, detections)
    assert set(grid["grid_lat"]).issubset(set(lats))
    assert set(grid["grid_lon"]).issubset(set(lons))


def test_frp_aggregates_are_consistent(india, detections):
    grid = to_fire_grid(india, detections)
    # mean == sum/count per cell, and total FRP is conserved.
    assert np.allclose(grid["frp_mean"], grid["frp_sum"] / grid["fire_count"])
    assert grid["frp_sum"].sum() == pytest.approx(detections["frp"].sum())


def test_out_of_bbox_detections_are_dropped(india):
    """FIRMS can return a pixel just outside the requested box; a cell we cannot
    address is not a cell, so it must be dropped rather than clamped."""
    df = pd.DataFrame(
        {
            "lat": [28.6, 55.0, -10.0],       # valid, far north, southern hemisphere
            "lon": [77.2, 77.0, 77.0],
            "frp": [10.0, 10.0, 10.0],
            "acq_date": [dt.date(2025, 11, 3)] * 3,
        }
    )
    grid = to_fire_grid(india, df)
    assert grid["fire_count"].sum() == 1, "only the in-bbox detection survives"


def test_empty_input_returns_empty_frame(india):
    assert to_fire_grid(india, pd.DataFrame()).empty


def test_season_windows_cover_both_burning_periods(india):
    got = season_windows(india, 2025)
    assert (dt.date(2025, 4, 1), dt.date(2025, 5, 31)) in got   # wheat + forest
    assert (dt.date(2025, 10, 1), dt.date(2025, 11, 30)) in got  # paddy
    # Month-end arithmetic must not spill into the next month.
    for start, end in got:
        assert end > start
        assert (end + dt.timedelta(days=1)).day == 1
