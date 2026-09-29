"""National layer: region config + analysis-grid arithmetic.

The grid is load-bearing — satellite retrievals, fire counts, CPCB stations and
model predictions are all joined by snapping to a cell centre. If `snap` is off
by half a cell, every one of those joins silently misaligns, so these tests pin
the arithmetic rather than trusting it.
"""

from __future__ import annotations

import pytest

from vayu_core.config import load_region


@pytest.fixture(scope="module")
def india():
    return load_region("india")


def test_grid_axes_are_cell_centres_inside_the_bbox(india):
    lats, lons = india.grid_axes()
    w, s, e, n = india.bbox
    d = india.grid_deg

    assert lats and lons
    # Centres sit half a cell inside the edges — never on the boundary itself.
    assert lats[0] == pytest.approx(s + d / 2)
    assert lons[0] == pytest.approx(w + d / 2)
    assert lats[-1] < n and lons[-1] < e
    # Uniform spacing, no drift from repeated addition.
    assert lats[1] - lats[0] == pytest.approx(d)
    assert lons[-1] - lons[-2] == pytest.approx(d)


def test_grid_size_matches_axes(india):
    ny, nx = india.grid_size()
    lats, lons = india.grid_axes()
    assert (ny, nx) == (len(lats), len(lons))


def test_snap_lands_on_a_real_axis_value_and_is_idempotent(india):
    """A snapped point must BE a grid cell, and snapping it again must not move
    it — otherwise repeated joins would walk a value across cells."""
    lats, lons = set(india.grid_axes()[0]), set(india.grid_axes()[1])
    for lat, lon in [(28.61, 77.21), (30.90, 75.85), (26.14, 91.74), (13.08, 80.27), (23.0, 80.0)]:
        cell = india.snap(lat, lon)
        assert cell[0] in lats, f"lat {cell[0]} not on the grid"
        assert cell[1] in lons, f"lon {cell[1]} not on the grid"
        assert india.snap(*cell) == cell, "snap must be idempotent"


def test_snap_keeps_points_within_half_a_cell(india):
    """The assigned centre must be the *nearest* cell, not a neighbour."""
    d = india.grid_deg
    for lat, lon in [(28.61, 77.21), (19.076, 72.877), (12.97, 77.59)]:
        clat, clon = india.snap(lat, lon)
        assert abs(clat - lat) <= d / 2 + 1e-9
        assert abs(clon - lon) <= d / 2 + 1e-9


def test_source_regions_resolve_for_known_cities(india):
    """Source-region identification; these are the anchors."""
    assert india.source_region_for(*india.snap(30.90, 75.85)) == "igp_northwest"   # Ludhiana
    assert india.source_region_for(*india.snap(26.14, 91.74)) == "northeast_forest"  # Guwahati
    assert india.source_region_for(*india.snap(13.08, 80.27)) == "peninsular"      # Chennai
    # A point in the Arabian Sea belongs to no source region.
    assert india.source_region_for(18.0, 68.5) is None


def test_burning_seasons_match_the_indian_agricultural_calendar(india):
    assert india.is_burning_season(10) == "kharif_burning"   # paddy residue
    assert india.is_burning_season(11) == "kharif_burning"
    assert india.is_burning_season(4) == "rabi_burning"      # wheat residue + forest fires
    assert india.is_burning_season(7) is None                # monsoon: nothing burns


def test_every_product_declares_units_and_a_band(india):
    """A retrieval with no unit is a number with no meaning."""
    assert "hcho" in india.products, "HCHO is the hotspot detector's whole subject"
    for name, p in india.products.items():
        assert p.band, f"{name} has no band"
        assert p.unit, f"{name} has no unit"
        assert p.collection, f"{name} has no collection"
