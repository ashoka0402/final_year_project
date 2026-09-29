"""S5P L3 -> analysis-grid binning (surface-AQI + hotspot-detection input).

Network-free: `to_grid` is pure array maths, so it is tested with synthetic
rasters that make each property checkable by construction.

The subtle correctness risk here is orientation. A north-up GeoTIFF stores row 0
at the NORTH edge, so naively pairing rows with ascending latitudes flips the
country upside down — Kerala's values would be filed under Kashmir's cells and
every hotspot would appear in the wrong place while all the *statistics* still
looked perfectly reasonable. `test_north_up_orientation_is_preserved` is the
guard against exactly that silent failure.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from services.pipeline.satellite import GEE_ONLY_PRODUCTS, to_grid
from vayu_core.config import load_region

DAY = dt.date(2025, 11, 3)


@pytest.fixture(scope="module")
def india():
    return load_region("india")


def _raster(india, fill=1.0, shape=(320, 300)):
    """A fully-valid raster covering exactly the region bbox."""
    return np.ma.MaskedArray(np.full(shape, fill, dtype="float64"), mask=False)


def test_constant_field_grids_to_that_constant(india):
    """A uniform field must average to itself in every cell — no scaling drift."""
    g = to_grid(india, _raster(india, fill=2.5e-4), "hcho", DAY, "mol/m^2")
    assert not g.empty
    assert np.allclose(g["value"], 2.5e-4)


def test_cells_land_on_declared_axes_and_cover_the_grid(india):
    g = to_grid(india, _raster(india), "hcho", DAY, "mol/m^2")
    lats, lons = india.grid_axes()
    assert set(g["grid_lat"]).issubset(set(lats))
    assert set(g["grid_lon"]).issubset(set(lons))
    # A full raster should populate essentially the whole grid.
    assert len(g) > 0.98 * len(lats) * len(lons)


def test_masked_pixels_are_excluded_not_zero_filled(india):
    """Cloud / failed retrievals must not be averaged in as zero — that would
    manufacture artificially clean cells."""
    arr = _raster(india, fill=4.0)
    arr.mask = np.zeros(arr.shape, dtype=bool)
    arr.mask[:160, :] = True  # blank the northern half

    g = to_grid(india, arr, "hcho", DAY, "mol/m^2")
    # Every surviving cell keeps the true value, not a zero-diluted mean.
    assert np.allclose(g["value"], 4.0)
    # And the blanked half genuinely produced no rows.
    assert len(g) < 0.6 * len(india.grid_axes()[0]) * len(india.grid_axes()[1])


def test_n_obs_records_backing_pixel_count(india):
    """Coverage must stay visible rather than implied."""
    g = to_grid(india, _raster(india), "hcho", DAY, "mol/m^2")
    # 0.1deg source into a coarser grid => several pixels per cell.
    assert g["n_obs"].min() >= 1
    assert g["n_obs"].mean() > 1
    assert g["n_obs"].sum() == pytest.approx(320 * 300, rel=0.02)


def test_north_up_orientation_is_preserved(india):
    """Row 0 of a north-up raster is the NORTH edge.

    Put a marker in the top rows; it must come back at HIGH latitude. If the
    orientation were flipped this test fails while means/counts stay perfect —
    which is precisely how such a bug hides.
    """
    arr = _raster(india, fill=1.0)
    arr[:20, :] = 99.0  # northern strip

    g = to_grid(india, arr, "hcho", DAY, "mol/m^2")
    hot = g[g["value"] > 50]
    cool = g[g["value"] < 50]
    assert not hot.empty, "marker vanished"
    assert hot["grid_lat"].min() > cool["grid_lat"].max(), "raster is vertically flipped"


def test_empty_and_fully_masked_inputs_return_empty(india):
    assert to_grid(india, np.ma.MaskedArray(np.zeros((0, 0))), "hcho", DAY, "u").empty
    blank = _raster(india)
    blank.mask = np.ones(blank.shape, dtype=bool)
    assert to_grid(india, blank, "hcho", DAY, "u").empty


def test_metadata_columns_are_stamped(india):
    g = to_grid(india, _raster(india), "hcho", DAY, "mol/m^2")
    assert (g["region"] == india.id).all()
    assert (g["product"] == "hcho").all()
    assert (g["date"] == DAY).all()
    assert (g["unit"] == "mol/m^2").all()
    assert (g["source"] == "s5p-tropomi-dlr").all()


def test_no2_and_co_are_flagged_as_unavailable_from_dlr():
    """The surface-AQI model wants NO2 and CO; DLR does not publish them. The gap is
    declared in code so it cannot be silently forgotten."""
    assert "no2" in GEE_ONLY_PRODUCTS
    assert "co" in GEE_ONLY_PRODUCTS
