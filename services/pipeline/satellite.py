"""S5P/TROPOMI L3 columnar retrievals over India.

Source: DLR's public S5P L3 archive. It needs **no credentials** — unlike
Google Earth Engine — which is why it is the primary path here:

    https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/YYYY/MM/DD/
        S5P_DLR_NRTI_01_L3_<PROD>_<YYYYMMDD>/
            S5P_DLR_NRTI_01_L3_<PROD>_<YYYYMMDD>.json   <- STAC metadata
            S5P_DLR_NRTI_01_L3_<PROD>_<YYYYMMDD>_<prod>.tif  <- Cloud-Optimized GeoTIFF

Two properties of that archive shape this module:

1. The GeoTIFFs are **Cloud-Optimized** (512x512 zstd tiles, EPSG:4326, global
   1800x3600 at 0.1deg). So we HTTP range-read only the tiles covering India
   rather than pulling the ~28 MB global file. Measured from here: a full
   download runs at ~39 KB/s (minutes per file), while the India window costs
   ~50 s. The network — not GDAL — is the floor.

2. Filenames embed a processing stream and version (`NRTI_01`) that can change.
   Rather than hard-code the pattern we read the tiny STAC JSON (~6 KB) and
   take the asset href from it, so a stream change does not silently 404.

Products available from DLR: AI ALH AOD ASSA CF COT CTH H2O HCHO O3 SO2 SO2LH
UVI. **NO2 and CO are not published here** — the surface-AQI model's full
multi-pollutant set needs Earth Engine for those two, via `ingest_day_gee`
below. Stated, not
silently skipped: `ingest_day` dispatches to GEE automatically when a
`GEE_SERVICE_ACCOUNT_JSON` is configured, and logs (not raises) when it is not.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from datetime import date, timedelta

import certifi
import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.config import RegionConfig, get_settings
from vayu_core.db import upsert_df, write_conn

BASE = "https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3"

# Products DLR publishes, mapped to the STAC asset key that holds the raster.
DLR_PRODUCTS: dict[str, str] = {
    "hcho": "hcho",
    "o3": "o3",
    "so2": "so2",
    "aod": "aod",
}
# Named so the gap is legible in code, not just prose.
GEE_ONLY_PRODUCTS = ("no2", "co")

_HTTP_TIMEOUT = 120

# Python on macOS often cannot find the system root certificates, so a bare
# urlopen fails DLR's HTTPS with CERTIFICATE_VERIFY_FAILED. certifi ships the
# roots — the same fix the CPCB and search ingestors use.
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# GDAL knobs that make /vsicurl range-reads sane: don't list the directory,
# keep a read cache, and open with a big enough first byte-range to land the
# header + tile index in one request.
_GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
    "GDAL_INGESTED_BYTES_AT_OPEN": "131072",
    "GDAL_CACHEMAX": "512",
}


def _apply_gdal_env() -> None:
    os.environ.update(_GDAL_ENV)
    # GDAL bundles its own curl and does not inherit Python's SSL context, so
    # it needs to be pointed at the same CA bundle independently.
    os.environ.setdefault("GDAL_HTTP_CAINFO", certifi.where())
    os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())


# DLR publishes a day under one of two processing streams, and which one varies
# day to day. OFFL ("offline") is the fully reprocessed product with the better
# calibration; NRTI ("near-real-time") is the fast preliminary one. We try OFFL
# first for quality and fall back to NRTI — hardcoding NRTI silently lost 3 of
# 56 days in the first season ingest, all of which exist as OFFL.
STREAMS = ("OFFL_01", "NRTI_01")


def stac_url(product: str, day: date, stream: str = "NRTI_01") -> str:
    prod = product.upper()
    stamp = day.strftime("%Y%m%d")
    d = f"{BASE}/{day:%Y/%m/%d}/S5P_DLR_{stream}_L3_{prod}_{stamp}"
    return f"{d}/S5P_DLR_{stream}_L3_{prod}_{stamp}.json"


def asset_href(product: str, day: date) -> tuple[str, float, float] | None:
    """(raster url, scale, nodata) from the day's STAC metadata, or None.

    Reading the STAC first costs ~6 KB and removes the need to guess filenames
    or the scale factor — the raster stores integers that only mean something
    once multiplied by the band's `scale`. Streams are tried in STREAMS order.
    """
    key = DLR_PRODUCTS.get(product, product)
    for stream in STREAMS:
        try:
            with urllib.request.urlopen(
                stac_url(product, day, stream), timeout=_HTTP_TIMEOUT, context=_SSL_CTX
            ) as r:  # noqa: S310
                meta = json.loads(r.read().decode())
        except Exception:  # noqa: BLE001 — a missing stream is normal, try the next
            continue

        asset = (meta.get("assets") or {}).get(key)
        if not asset or not asset.get("href"):
            logger.warning(f"S5P {product} {day} [{stream}]: STAC has no '{key}' asset")
            continue

        bands = asset.get("raster:bands") or [{}]
        scale = float(bands[0].get("scale", 1.0) or 1.0)
        nodata = float(bands[0].get("nodata", np.nan))
        return asset["href"], scale, nodata

    logger.warning(f"S5P {product} {day}: not published in any of {STREAMS}")
    return None


def read_window(href: str, region: RegionConfig, scale: float, nodata: float) -> np.ma.MaskedArray | None:
    """Range-read the region's window from the global COG, scaled to real units."""
    _apply_gdal_env()
    import rasterio
    from rasterio.windows import from_bounds

    w, s, e, n = region.bbox
    try:
        with rasterio.open(f"/vsicurl/{href}") as ds:
            arr = ds.read(1, window=from_bounds(w, s, e, n, ds.transform), masked=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"S5P read failed for {href.rsplit('/', 1)[-1]}: {exc}")
        return None

    if np.isfinite(nodata):
        arr = np.ma.masked_where(np.isclose(arr.filled(np.nan), nodata, rtol=1e-6), arr)
    return arr.astype("float64") * scale


def to_grid(region: RegionConfig, arr: np.ma.MaskedArray, product: str, day: date, unit: str) -> pd.DataFrame:
    """Bin the 0.1deg source pixels onto the region's analysis grid (cell means).

    The source is 0.1deg and the analysis grid is coarser, so each cell averages
    several native pixels. Masked (retrieval-failed / cloud) pixels are excluded
    from the mean rather than treated as zero — averaging in a zero would drag
    a cell's column density toward nothing and manufacture a false clean patch.
    `n_obs` records how many real pixels backed each cell so coverage stays
    visible downstream instead of being implied.
    """
    if arr is None or arr.size == 0:
        return pd.DataFrame()

    w, s, e, n = region.bbox
    ny, nx = arr.shape
    # The window is the bbox, so pixel centres are evenly spaced across it.
    # Row 0 is the NORTH edge in a north-up raster — hence the descending lats.
    lat_edges = np.linspace(n, s, ny + 1)
    lon_edges = np.linspace(w, e, nx + 1)
    plat = (lat_edges[:-1] + lat_edges[1:]) / 2
    plon = (lon_edges[:-1] + lon_edges[1:]) / 2

    lon_mesh, lat_mesh = np.meshgrid(plon, plat)
    valid = ~np.ma.getmaskarray(arr)
    if not valid.any():
        return pd.DataFrame()

    d = region.grid_deg
    glat = (((lat_mesh[valid] - s) // d) * d + s + d / 2).round(4)
    glon = (((lon_mesh[valid] - w) // d) * d + w + d / 2).round(4)

    df = pd.DataFrame({"grid_lat": glat, "grid_lon": glon, "value": np.asarray(arr[valid], dtype="float64")})
    agg = df.groupby(["grid_lat", "grid_lon"], as_index=False).agg(
        value=("value", "mean"), n_obs=("value", "size")
    )
    lats, lons = region.grid_axes()
    agg = agg[agg["grid_lat"].between(min(lats), max(lats)) & agg["grid_lon"].between(min(lons), max(lons))]
    if agg.empty:
        return pd.DataFrame()

    agg["region"] = region.id
    agg["product"] = product
    agg["date"] = day
    agg["unit"] = unit
    agg["source"] = "s5p-tropomi-dlr"
    return agg[["region", "product", "grid_lat", "grid_lon", "date", "value", "unit", "n_obs", "source"]]


_EE_READY: bool | None = None  # tri-state cache: None=untried, True/False=result


def _ee_available() -> bool:
    """Lazily initialise Earth Engine with the configured service account.

    Cached after the first attempt so a missing/broken key logs once per
    process, not once per product-day. `ee.Initialize` itself is the expensive
    and the fallible step (network + the project's EE registration), so it
    happens exactly once.
    """
    global _EE_READY
    if _EE_READY is not None:
        return _EE_READY

    key_path = get_settings().gee_service_account_json
    if not key_path:
        logger.info("GEE_SERVICE_ACCOUNT_JSON not set — no2/co stay unavailable")
        _EE_READY = False
        return False

    try:
        import ee

        info = json.loads(open(key_path).read())  # noqa: PTH123 — reading a local config path
        creds = ee.ServiceAccountCredentials(info["client_email"], key_path)
        ee.Initialize(creds)
        _EE_READY = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Earth Engine init failed ({type(exc).__name__}: {exc}) — no2/co stay unavailable")
        _EE_READY = False
    return _EE_READY


# GEE images can carry masked (no-retrieval) pixels; unmasking to this sentinel
# before download makes "no data" unambiguous in the downloaded GeoTIFF, the
# same role `nodata` plays for the DLR path's `read_window`.
_GEE_NODATA = -9999.0

# ~0.1deg at the equator. Matches DLR's native grid so a day sourced from GEE
# bins onto `to_grid` with the same effective resolution as one sourced from
# DLR — the two are meant to be compared cell-for-cell, not just coexist.
_GEE_SCALE_M = 11132


def ingest_day_gee(region: RegionConfig, product: str, day: date) -> int:
    """NO2/CO for one region-day, via Earth Engine. Mirrors the DLR path's
    shape (fetch -> to_grid -> upsert) so both sources land in the same table
    with the same schema and can be queried without a source-specific branch.
    """
    if not _ee_available():
        return 0
    spec = region.products.get(product)
    if spec is None:
        logger.warning(f"no product spec for {product!r} in region {region.id!r} — cannot ingest via GEE")
        return 0

    import ee

    w, s, e, n = region.bbox
    geom = ee.Geometry.Rectangle([w, s, e, n])
    day_str, next_str = day.isoformat(), (day + timedelta(days=1)).isoformat()

    coll = ee.ImageCollection(spec.collection).select(spec.band).filterDate(day_str, next_str).filterBounds(geom)

    # A day with no TROPOMI orbit crossing the region (this happens for NO2
    # noticeably more than CO — NO2's UV-VIS retrieval is filtered out by cloud
    # and aerosol interference far more aggressively than CO's SWIR channel, a
    # real sensor characteristic and not a bug) makes `.mean()` return a
    # zero-band image. `.unmask()` on that then raises rather than returning an
    # empty result, so the empty-collection case is handled explicitly here
    # instead of being funnelled through a generic try/except that would log it
    # identically to a real network failure.
    if coll.size().getInfo() == 0:
        logger.warning(f"GEE {product} {day}: no orbit passes over {region.id} this day")
        return 0

    # .mean() over the day's orbits: TROPOMI's OFFL collection can carry more
    # than one overlapping pass per day, and averaging overlaps is the same
    # choice DLR's own L3 product makes rather than keeping only the latest.
    img = coll.mean().unmask(_GEE_NODATA)

    try:
        url = img.getDownloadURL(
            {"region": geom, "scale": _GEE_SCALE_M, "format": "GEO_TIFF", "crs": "EPSG:4326"}
        )
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT, context=_SSL_CTX) as r:  # noqa: S310
            tif_bytes = r.read()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"GEE {product} {day} download failed: {exc}")
        return 0

    import rasterio

    try:
        with rasterio.io.MemoryFile(tif_bytes) as mem, mem.open() as ds:
            arr = ds.read(1, masked=False).astype("float64")
            arr = np.ma.masked_where(np.isclose(arr, _GEE_NODATA, rtol=1e-6), arr)
            arr *= spec.scale
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"GEE {product} {day}: could not read downloaded GeoTIFF: {exc}")
        return 0

    grid = to_grid(region, arr, product, day, spec.unit)
    if grid.empty:
        logger.warning(f"GEE {product} {day}: no valid pixels over {region.id}")
        return 0
    grid["source"] = "s5p-tropomi-gee"

    with write_conn() as con:
        n = upsert_df(con, "satellite_grid", grid, ["region", "product", "grid_lat", "grid_lon", "date"])
    logger.info(f"[{region.id}] {product} {day}: {n:,} cells (mean {grid['value'].mean():.3e} {spec.unit}) via GEE")
    return n


def ingest_day(region: RegionConfig, product: str, day: date) -> int:
    """Fetch one product-day for the region and persist it. Returns rows written."""
    if product in GEE_ONLY_PRODUCTS:
        return ingest_day_gee(region, product, day)

    found = asset_href(product, day)
    if found is None:
        return 0
    href, scale, nodata = found

    arr = read_window(href, region, scale, nodata)
    if arr is None:
        return 0

    spec = region.products.get(product)
    unit = spec.unit if spec else "mol/m^2"
    grid = to_grid(region, arr, product, day, unit)
    if grid.empty:
        logger.warning(f"S5P {product} {day}: no valid pixels over {region.id}")
        return 0

    with write_conn() as con:
        n = upsert_df(con, "satellite_grid", grid, ["region", "product", "grid_lat", "grid_lon", "date"])
    logger.info(f"[{region.id}] {product} {day}: {n:,} cells (mean {grid['value'].mean():.3e} {unit})")
    return n


def ingest_range(region: RegionConfig, product: str, start: date, end: date, step_days: int = 1) -> int:
    """Ingest a product over [start, end]. Each day is independent — a missing
    or corrupt day is logged and skipped rather than aborting the window."""
    total = 0
    day = start
    days = ok = 0
    while day <= end:
        days += 1
        try:
            n = ingest_day(region, product, day)
            total += n
            ok += 1 if n else 0
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"S5P {product} {day} failed: {exc}")
        day += timedelta(days=step_days)
    logger.success(f"[{region.id}] {product}: {total:,} cell-days from {ok}/{days} days ({start} → {end})")
    return total
