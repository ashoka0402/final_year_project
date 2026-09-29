"""OpenStreetMap via Overpass — no key.

Pulls the static context layers the Attributor and Herald need:
  * schools / hospitals  -> vulnerability overlay, "notify N schools" (Epic D3)
  * industrial landuse   -> the `industry` attribution category
  * brick kilns          -> tagged as works=brickyard where mappers have done so

Run once per city and cached to data/samples/osm_{city}.geojson; Overpass is a
donated public service and this data changes on the order of months.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from vayu_core.config import REPO_ROOT, CityConfig

from .http import FetchError, fetch_text

ENDPOINT = "https://overpass-api.de/api/interpreter"

FEATURES: dict[str, str] = {
    "school": 'nwr["amenity"="school"]',
    "hospital": 'nwr["amenity"="hospital"]',
    "industrial": 'way["landuse"="industrial"]',
    "brick_kiln": 'nwr["works"="brickyard"]',
}


def sample_path(city: CityConfig) -> Path:
    return REPO_ROOT / "data" / "samples" / f"osm_{city.id}.geojson"


def _query(city: CityConfig) -> str:
    s, w, n, e = city.bbox[1], city.bbox[0], city.bbox[3], city.bbox[2]
    bbox = f"({s},{w},{n},{e})"
    parts = "\n  ".join(f"{sel}{bbox};" for sel in FEATURES.values())
    # `out geom` rather than `out center`: TRD 5.3 attributes industry by
    # AREA of landuse inside the trajectory cone, so a 200-hectare industrial
    # estate must not count the same as a single mapped workshop. Schools and
    # hospitals still only need a point, and `out geom` gives us both.
    return f"[out:json][timeout:120];\n(\n  {parts}\n);\nout geom tags;"


def _classify(tags: dict[str, str]) -> str | None:
    if tags.get("amenity") == "school":
        return "school"
    if tags.get("amenity") == "hospital":
        return "hospital"
    if tags.get("landuse") == "industrial":
        return "industrial"
    if tags.get("works") == "brickyard":
        return "brick_kiln"
    return None


def fetch_osm(city: CityConfig, force: bool = False) -> tuple[dict[str, Any], str]:
    """Return (geojson, status) with status 'live' | 'sample' | 'unavailable'."""
    p = sample_path(city)

    if not force and p.exists():
        try:
            return json.loads(p.read_text()), "sample"
        except json.JSONDecodeError:
            logger.warning(f"[{city.id}] cached OSM extract corrupt — refetching")

    try:
        txt = fetch_text(
            ENDPOINT,
            params={"data": _query(city)},
            ttl=timedelta(days=30),
            timeout=120.0,
        )
        payload = json.loads(txt)
    except (FetchError, json.JSONDecodeError) as exc:
        logger.warning(f"[{city.id}] Overpass unavailable: {exc}")
        if p.exists():
            return json.loads(p.read_text()), "sample"
        return {"type": "FeatureCollection", "features": []}, "unavailable"

    feats: list[dict[str, Any]] = []
    for el in payload.get("elements", []) or []:
        tags = el.get("tags") or {}
        kind = _classify(tags)
        if kind is None:
            continue

        geom = el.get("geometry") or []
        ring = [[float(g["lon"]), float(g["lat"])] for g in geom if "lon" in g and "lat" in g]

        # Industrial landuse keeps its polygon; everything else is a point.
        if kind == "industrial" and len(ring) >= 3:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            lon = sum(p[0] for p in ring) / len(ring)
            lat = sum(p[1] for p in ring) / len(ring)
            geometry: dict[str, Any] = {"type": "Polygon", "coordinates": [ring]}
        else:
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None and ring:
                lon = sum(p[0] for p in ring) / len(ring)
                lat = sum(p[1] for p in ring) / len(ring)
            if lat is None or lon is None:
                continue
            geometry = {"type": "Point", "coordinates": [float(lon), float(lat)]}

        props: dict[str, Any] = {
            "osm_id": f"{el.get('type')}/{el.get('id')}",
            "kind": kind,
            "name": tags.get("name") or kind.replace("_", " ").title(),
            "lon": float(lon),
            "lat": float(lat),
        }
        if geometry["type"] == "Polygon":
            from vayu_core.geo import polygon_area_km2
            from shapely.geometry import shape as _shape

            try:
                props["area_km2"] = round(polygon_area_km2(_shape(geometry)), 4)
            except Exception:  # noqa: BLE001
                props["area_km2"] = 0.0

        feats.append({"type": "Feature", "properties": props, "geometry": geometry})

    gj = {"type": "FeatureCollection", "features": feats}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gj))
    counts: dict[str, int] = {}
    for f in feats:
        counts[f["properties"]["kind"]] = counts.get(f["properties"]["kind"], 0) + 1
    logger.info(f"[{city.id}] OSM: {len(feats)} features {counts}")
    return gj, "live"
