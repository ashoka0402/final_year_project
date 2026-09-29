"""Road density per ward, from OpenStreetMap — the traffic proxy (TRD 5.3).

VAYU cannot see traffic directly: there is no free real-time vehicle-count feed
for Indian cities, and the master prompt rules out CCTV ingestion. What it can
do honestly is measure how much major road a ward contains, which is the
standard proxy for vehicular emission potential, and then modulate it by the
hour of day and by measured NO2 (the tailpipe tracer VAYU *does* observe).

Roads are weighted by class because a motorway carries far more traffic per km
than a secondary road. The weights are a documented judgement, not a
calibration — they appear on the Methodology page as such.

This is a static layer: it is fetched once per city and cached, because road
networks change on the order of years.
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from shapely.geometry import shape
from shapely.strtree import STRtree

from vayu_core.config import REPO_ROOT, CityConfig

from .http import FetchError, fetch_text

ENDPOINT = "https://overpass-api.de/api/interpreter"

# Emission weight per road class, relative to `secondary` = 1.0. Motorways and
# trunks carry the heavy-vehicle share that dominates PM from traffic.
CLASS_WEIGHT = {
    "motorway": 4.0,
    "trunk": 3.0,
    "primary": 2.0,
    "secondary": 1.0,
}

KM_PER_DEG_LAT = 110.574


def sample_path(city: CityConfig) -> Path:
    return REPO_ROOT / "data" / "samples" / f"roads_{city.id}.geojson"


def _query(city: CityConfig) -> str:
    w, s, e, n = city.bbox
    classes = "|".join(CLASS_WEIGHT)
    return f'[out:json][timeout:120];way["highway"~"^({classes})$"]({s},{w},{n},{e});out geom;'


def fetch_roads(city: CityConfig, force: bool = False) -> tuple[dict, str]:
    """Return (geojson, status). Cached to data/samples/roads_{city}.geojson."""
    p = sample_path(city)
    if not force and p.exists():
        try:
            return json.loads(p.read_text()), "sample"
        except json.JSONDecodeError:
            logger.warning(f"[{city.id}] cached roads corrupt — refetching")

    try:
        txt = fetch_text(ENDPOINT, params={"data": _query(city)}, ttl=timedelta(days=30), timeout=180.0)
        payload = json.loads(txt)
    except (FetchError, json.JSONDecodeError) as exc:
        logger.warning(f"[{city.id}] roads unavailable: {exc}")
        if p.exists():
            return json.loads(p.read_text()), "sample"
        return {"type": "FeatureCollection", "features": []}, "unavailable"

    feats = []
    for el in payload.get("elements", []) or []:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        cls = (el.get("tags") or {}).get("highway")
        if cls not in CLASS_WEIGHT:
            continue
        feats.append(
            {
                "type": "Feature",
                "properties": {"highway": cls, "name": (el.get("tags") or {}).get("name")},
                "geometry": {"type": "LineString", "coordinates": [[g["lon"], g["lat"]] for g in geom]},
            }
        )

    gj = {"type": "FeatureCollection", "features": feats}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gj))
    logger.info(f"[{city.id}] roads: {len(feats):,} segments")
    return gj, "live"


def density_per_ward(city: CityConfig, roads: dict, wards: pd.DataFrame) -> pd.DataFrame:
    """Weighted road km per km² of ward.

    Returns columns: city, ward_id, road_km, road_km_weighted, road_density.
    """
    feats = roads.get("features") or []
    if not feats or wards.empty:
        return pd.DataFrame()

    lines = [shape(f["geometry"]) for f in feats]
    weights = [CLASS_WEIGHT.get(f["properties"]["highway"], 1.0) for f in feats]
    tree = STRtree(lines)

    rows = []
    for w in wards.itertuples():
        poly = shape(json.loads(w.geom_geojson))
        km = 0.0
        wkm = 0.0
        # STRtree narrows thousands of segments to the handful that could touch
        # this ward; a full cross-product would be 290 x 7,850 intersections.
        for idx in tree.query(poly):
            line = lines[idx]
            if not line.intersects(poly):
                continue
            clipped = line.intersection(poly)
            length_km = _length_km(clipped)
            km += length_km
            wkm += length_km * weights[idx]
        area = max(float(w.area_km2), 0.01)
        rows.append(
            {
                "city": city.id,
                "ward_id": w.ward_id,
                "road_km": round(km, 3),
                "road_km_weighted": round(wkm, 3),
                "road_density": round(wkm / area, 3),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            f"[{city.id}] road density: {df['road_km'].sum():,.0f} km across {len(df)} wards "
            f"(max {df['road_density'].max():.1f} weighted km/km²)"
        )
    return df


def _length_km(geom) -> float:
    """Length of a (possibly multi-part) geometry in km, on a local projection."""
    if geom.is_empty:
        return 0.0
    if geom.geom_type == "LineString":
        parts = [geom]
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        parts = [g for g in geom.geoms if g.geom_type == "LineString"]
    else:
        return 0.0

    total = 0.0
    for part in parts:
        coords = list(part.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            lat0 = math.radians((y0 + y1) / 2)
            dx = (x1 - x0) * 111.320 * math.cos(lat0)
            dy = (y1 - y0) * KM_PER_DEG_LAT
            total += math.hypot(dx, dy)
    return total
