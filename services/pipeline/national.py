"""National-layer ingestion: India-wide gridded inputs.

The city pipeline fetches a ~50 km airshed around one municipality. This module
fetches the whole country and lays it on the RegionConfig analysis grid, which
is what hotspot detection ("map spatio-temporal HCHO", "correlate with fire
counts") actually needs.

Currently implemented:
  * FIRMS fire detections over India -> `fire_grid` (daily count / FRP per cell)

Satellite columns (S5P HCHO/NO2/SO2/CO/O3, AOD) live in `satellite.py` and need
Google Earth Engine credentials; this module deliberately does not depend on
them, so fire analysis works with only a FIRMS key.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from vayu_core.config import RegionConfig, get_settings
from vayu_core.db import upsert_df, write_conn

from . import firms
from .http import FetchError

# FIRMS caps an archive request at 5 days; the whole of India is a large area,
# so we page tightly and let a failed chunk degrade that window rather than the
# whole ingest.
CHUNK_DAYS = 5


def _region_area(region: RegionConfig) -> str:
    """FIRMS area string (w,s,e,n) for the region — no padding.

    The city ingest pads by 2 deg so a back-trajectory leaving the city still
    finds upwind fires. A national grid has no "upwind outside the domain" in
    the same sense: the bbox IS the study area, and padding would pull in
    detections we have no grid cells to hold.
    """
    w, s, e, n = region.bbox
    return f"{w},{s},{e},{n}"


def fetch_fires(region: RegionConfig, start: date, end: date) -> pd.DataFrame:
    """Raw VIIRS detections over the region for [start, end].

    Returns a frame with lat/lon/frp/acq_ts, or empty if FIRMS is unavailable.
    Partial windows are logged loudly — an under-counted fire season would
    silently weaken every correlation computed downstream.
    """
    settings = get_settings()
    if not firms.available():
        logger.warning("FIRMS key absent — national fire ingest skipped")
        return pd.DataFrame()

    area = _region_area(region)
    frames: list[pd.DataFrame] = []
    cur = start
    chunks = failures = 0
    while cur <= end:
        span = min(CHUNK_DAYS, (end - cur).days + 1)
        chunks += 1
        try:
            df = firms._chunk(settings.firms_api_key, firms.SENSOR_ARCHIVE, area, span, cur)
            if not df.empty:
                frames.append(df)
        except (FetchError, ValueError) as exc:
            failures += 1
            logger.warning(f"[{region.id}] FIRMS {cur} (+{span}d): {exc}")
        cur += timedelta(days=span)

    if failures and chunks and failures / chunks > 0.2:
        logger.error(
            f"[{region.id}] FIRMS: {failures}/{chunks} chunks failed — national fire "
            "coverage is incomplete; hotspot correlations will under-count"
        )
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(raw["latitude"], errors="coerce"),
            "lon": pd.to_numeric(raw["longitude"], errors="coerce"),
            "frp": pd.to_numeric(raw.get("frp"), errors="coerce"),
            "acq_date": pd.to_datetime(raw["acq_date"], errors="coerce").dt.date,
        }
    ).dropna(subset=["lat", "lon", "acq_date"])
    logger.info(f"[{region.id}] FIRMS: {len(out):,} detections {start} → {end}")
    return out


def to_fire_grid(region: RegionConfig, fires: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw detections onto the analysis grid (count + FRP per cell/day).

    Snapping is vectorised arithmetic rather than a per-row `snap()` call: at
    ~10^5-10^6 detections a Python-level loop dominates the whole ingest.
    """
    if fires.empty:
        return pd.DataFrame()

    w, s, _, _ = region.bbox
    d = region.grid_deg
    lat = fires["lat"].to_numpy()
    lon = fires["lon"].to_numpy()
    # Same formula as RegionConfig.snap(), applied to whole columns at once.
    glat = (((lat - s) // d) * d + s + d / 2).round(4)
    glon = (((lon - w) // d) * d + w + d / 2).round(4)

    g = pd.DataFrame({"grid_lat": glat, "grid_lon": glon, "date": fires["acq_date"], "frp": fires["frp"]})
    # Keep only cells inside the declared grid — FIRMS can return a pixel a hair
    # outside the requested bbox, and a cell we cannot address is not a cell.
    lats, lons = region.grid_axes()
    g = g[g["grid_lat"].between(min(lats), max(lats)) & g["grid_lon"].between(min(lons), max(lons))]
    if g.empty:
        return pd.DataFrame()

    agg = (
        g.groupby(["grid_lat", "grid_lon", "date"], as_index=False)
        .agg(fire_count=("frp", "size"), frp_sum=("frp", "sum"), frp_mean=("frp", "mean"))
    )
    agg["region"] = region.id
    agg["source_region"] = [
        region.source_region_for(la, lo) for la, lo in zip(agg["grid_lat"], agg["grid_lon"])
    ]
    return agg[
        ["region", "grid_lat", "grid_lon", "date", "fire_count", "frp_sum", "frp_mean", "source_region"]
    ]


def ingest_fires(region: RegionConfig, start: date, end: date) -> int:
    """Fetch + grid + persist national fires for a window. Returns rows written."""
    raw = fetch_fires(region, start, end)
    grid = to_fire_grid(region, raw)
    if grid.empty:
        return 0
    with write_conn() as con:
        n = upsert_df(con, "fire_grid", grid, ["region", "grid_lat", "grid_lon", "date"])
    logger.success(
        f"[{region.id}] fire_grid: {n:,} cell-days from {len(raw):,} detections "
        f"({start} → {end})"
    )
    return n


def season_windows(region: RegionConfig, year: int) -> list[tuple[date, date]]:
    """Contiguous [start, end] windows for the region's burning seasons in `year`.

    Hotspot detection is scoped to biomass-burning periods, so this is both
    the scientifically right window and what keeps the archive small.
    """
    out: list[tuple[date, date]] = []
    for season in region.seasons.values():
        months = sorted(season.months)
        if not months:
            continue
        start = date(year, months[0], 1)
        last = months[-1]
        end = date(year + 1, 1, 1) - timedelta(days=1) if last == 12 else date(year, last + 1, 1) - timedelta(days=1)
        out.append((start, end))
    return sorted(out)
