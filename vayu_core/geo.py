"""Geometry helpers shared by the pipeline and the science modules.

Kept dependency-light (shapely for polygons, numpy for the rest) and free of any
city-specific assumption.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance. Used for station->ward weighting and evidence
    distances, where a flat-earth approximation would drift over a city bbox."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2, degrees clockwise from N."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def geom_of(feature: dict[str, Any]) -> BaseGeometry:
    return shape(feature["geometry"])


def polygon_area_km2(geom: BaseGeometry) -> float:
    """Area via an equal-area azimuthal projection centred on the polygon.

    Degrees-squared would badly distort ward areas, and area drives the
    population apportionment, so this is worth doing properly.
    """
    lon0, lat0 = geom.centroid.x, geom.centroid.y
    lat0_r = math.radians(lat0)

    def project(x: float, y: float) -> tuple[float, float]:
        return (
            math.radians(x - lon0) * EARTH_RADIUS_KM * math.cos(lat0_r),
            math.radians(y - lat0) * EARTH_RADIUS_KM,
        )

    from shapely.ops import transform

    return abs(transform(lambda xs, ys: tuple(zip(*[project(x, y) for x, y in zip(xs, ys)])), geom).area)


def representative_point(geom: BaseGeometry) -> tuple[float, float]:
    """(lat, lon) guaranteed to lie inside the polygon.

    A plain centroid can fall outside concave wards (and Delhi has several),
    which would silently mis-assign a ward's forecast to a neighbour.
    """
    p = geom.representative_point()
    return p.y, p.x


def idw(
    targets: Sequence[tuple[float, float]],
    sources: Sequence[tuple[float, float]],
    values: Sequence[float],
    power: float = 2.0,
    k: int = 5,
    max_km: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance-weighted interpolation (TRD 5.1: p=2, k=5 nearest).

    Returns (interpolated_values, distance_to_nearest_source_km). The second
    return exists because a ward >25 km from any station must be watermarked
    low-confidence (App Flow §7) rather than presented like a well-observed one.
    """
    if len(sources) == 0 or len(values) == 0:
        return np.full(len(targets), np.nan), np.full(len(targets), np.inf)

    vals = np.asarray(values, dtype=float)
    out = np.empty(len(targets), dtype=float)
    nearest = np.empty(len(targets), dtype=float)

    for idx, (tlat, tlon) in enumerate(targets):
        d = np.array([haversine_km(tlat, tlon, slat, slon) for slat, slon in sources])
        ok = ~np.isnan(vals)
        if max_km is not None:
            ok &= d <= max_km
        if not ok.any():
            out[idx] = np.nan
            nearest[idx] = float(d.min()) if len(d) else np.inf
            continue

        d_ok, v_ok = d[ok], vals[ok]
        nearest[idx] = float(d_ok.min())

        order = np.argsort(d_ok)[: max(1, k)]
        d_k, v_k = d_ok[order], v_ok[order]

        # Sitting on a station: return it exactly rather than dividing by ~0.
        if d_k[0] < 1e-6:
            out[idx] = float(v_k[0])
            continue

        w = 1.0 / np.power(d_k, power)
        out[idx] = float(np.sum(w * v_k) / np.sum(w))

    return out, nearest


def bbox_of(features: Iterable[dict[str, Any]]) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for f in features:
        minx, miny, maxx, maxy = geom_of(f).bounds
        xs += [minx, maxx]
        ys += [miny, maxy]
    return [min(xs), min(ys), max(xs), max(ys)]
