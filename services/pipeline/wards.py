"""Ward boundary ingestion.

Real municipal ward polygons for both demo cities come from DataMeet's
Municipal_Spatial_Data (Delhi: 290 wards, Lucknow: 112 wards). Because real
boundaries exist for both, VAYU does not need the H3 "analysis zones" fallback
for Delhi or Lucknow — but the fallback is implemented anyway, since the claim
"a new city is one config file" is only true if a city *without* published
boundaries still works.

Population: per-ward Census counts are not published in a machine-readable form
for either city, so we split the Census 2011 city total across wards using the
delimitation principle that drew the boundaries in the first place — municipal
wards are electoral units sized to hold equal population — and record that in
`pop_source`. See `_estimate_population` for why this replaced apportionment by
polygon area, which was actively inverted. Every population number VAYU shows
carries how it was derived: it is an estimate and says so (PRD F2).
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from vayu_core.config import REPO_ROOT, CityConfig
from vayu_core.geo import geom_of, polygon_area_km2, representative_point

from .http import FetchError, fetch_text


def _titlecase(raw: str) -> str:
    """Title-case a ward name without mangling ordinals.

    str.title() turns "Rajiv Gandhi 2nd" into "Rajiv Gandhi 2Nd" because it
    treats a digit as a word boundary. Upstream names arrive in mixed case
    ("DELHI CANTT CHARGE 1", "Sarojni Nagar Part 1"), so we normalise, but only
    capitalise a letter that follows a non-alphanumeric.
    """
    return re.sub(r"(?<![A-Za-z0-9])([a-z])", lambda m: m.group(1).upper(), raw.strip().lower())


def _download_wards(city: CityConfig) -> dict[str, Any] | None:
    if not city.wards.source_url:
        return None
    try:
        # Boundaries change on the order of years; cache hard.
        txt = fetch_text(city.wards.source_url, ttl=timedelta(days=30))
        return json.loads(txt)
    except (FetchError, json.JSONDecodeError) as exc:
        logger.warning(f"[{city.id}] ward download failed: {exc}")
        return None


def _h3_fallback(city: CityConfig) -> dict[str, Any]:
    """Hex tessellation of the bbox, labelled "analysis zones" — never "wards".

    Used only when a city publishes no boundaries. We avoid a hard h3 dependency
    by falling back to a plain lat/lon grid if the library is absent; both are
    labelled the same way in the UI, because both are our construct, not a
    municipal fact.
    """
    w, s, e, n = city.bbox
    feats: list[dict[str, Any]] = []
    try:
        import h3  # type: ignore

        cells = h3.polygon_to_cells(
            h3.LatLngPoly([(s, w), (s, e), (n, e), (n, w)]), city.wards.h3_resolution
        )
        for idx, cell in enumerate(sorted(cells), start=1):
            ring = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"zone_id": f"Z{idx:03d}", "zone_name": f"Analysis zone {idx}"},
                    "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
                }
            )
    except Exception:  # noqa: BLE001
        logger.warning(f"[{city.id}] h3 unavailable — using a rectangular analysis grid")
        rows = cols = 10
        for r in range(rows):
            for c in range(cols):
                x0, x1 = w + (e - w) * c / cols, w + (e - w) * (c + 1) / cols
                y0, y1 = s + (n - s) * r / rows, s + (n - s) * (r + 1) / rows
                idx = r * cols + c + 1
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {"zone_id": f"Z{idx:03d}", "zone_name": f"Analysis zone {idx}"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
                        },
                    }
                )
    return {"type": "FeatureCollection", "features": feats}


def load_wards(city: CityConfig, force: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (wards dataframe, status) and cache the GeoJSON into data/samples/.

    status: 'live' (fetched now) | 'sample' (bundled copy) | 'h3-fallback'.
    """
    local: Path = city.wards_path
    status = "sample"
    gj: dict[str, Any] | None = None

    if force or not local.exists():
        gj = _download_wards(city)
        if gj is not None:
            status = "live"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(gj))
            logger.info(f"[{city.id}] wards downloaded -> {local.relative_to(REPO_ROOT)}")

    if gj is None and local.exists():
        gj = json.loads(local.read_text())
        status = "sample"

    if gj is None:
        if not city.wards.use_h3:
            logger.warning(f"[{city.id}] no ward boundaries available — falling back to analysis zones")
        gj = _h3_fallback(city)
        status = "h3-fallback"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(gj))

    id_prop = city.wards.id_property if status != "h3-fallback" else "zone_id"
    name_prop = city.wards.name_property if status != "h3-fallback" else "zone_name"

    rows = []
    for i, feat in enumerate(gj["features"], start=1):
        props = feat.get("properties", {}) or {}
        geom = geom_of(feat)
        if geom.is_empty:
            continue
        raw_id = props.get(id_prop)
        # Some sources leave the id blank on a handful of features; a positional
        # id keeps them on the map instead of silently dropping a ward.
        ward_id = f"W{str(raw_id).strip().replace(' ', '_')}" if raw_id not in (None, "") else f"W_IDX{i:03d}"
        lat, lon = representative_point(geom)
        rows.append(
            {
                "city": city.id,
                "ward_id": ward_id,
                "name": _titlecase(str(props.get(name_prop) or ward_id)),
                "geom_geojson": json.dumps(feat["geometry"]),
                "centroid_lat": lat,
                "centroid_lon": lon,
                "area_km2": polygon_area_km2(geom),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, status

    # Duplicate ids do occur upstream (Delhi has repeated cantonment charges);
    # make them unique so ward_id stays a real primary key.
    dup = df["ward_id"].duplicated(keep=False)
    if dup.any():
        df.loc[dup, "ward_id"] = df.loc[dup, "ward_id"] + "_" + (df.loc[dup].groupby("ward_id").cumcount() + 1).astype(str)
        logger.info(f"[{city.id}] disambiguated {int(dup.sum())} duplicate ward ids")

    df["population"] = _estimate_population(df, city)
    df["pop_source"] = f"{city.population.source} — {city.population.method}"

    return df, status


def _estimate_population(df: pd.DataFrame, city: CityConfig) -> pd.Series:
    """Split the Census city total across wards by the delimitation principle.

    Municipal wards are not arbitrary polygons — they are electoral units drawn
    to hold EQUAL population. Delhi Municipal Corporation Act 1957 s.5 requires
    the area be divided "in such manner that the population of each of the wards
    shall, so far as practicable, be the same"; UP's Municipal Corporation Act
    1959 governs Lucknow's wards on the same principle. So an equal split is not
    a shrug — it is the statute that drew these boundaries, and it is the best
    estimator available without per-ward Census tables.

    This replaces apportionment by polygon area, which was not merely imprecise
    but *inverted*: it handed Delhi's largest wards the most people, when the
    large wards are precisely the sparse agricultural fringe (Chhawla, 78 km²)
    and the dense inner-city wards are the small ones. That put a 550x spread
    (1,598 to 879,253) across wards the law says should be roughly equal, and it
    propagated straight into the ROI ranking — which multiplies by population and
    would have sent enforcement teams to farmland over dense neighbourhoods.

    Known limitation, surfaced on the Methodology page: "so far as practicable"
    is not "exactly", real wards vary by roughly +/-15%, and Delhi's 290 polygons
    include NDMC and Cantonment areas that are not MCD wards at all. Per-ward
    Census figures do exist in the State Election Commission's delimitation
    orders, as PDFs; ingesting those is the upgrade path.
    """
    n = len(df)
    per_ward = round(city.population.total / n)
    logger.info(
        f"[{city.id}] population: {city.population.total:,} split across {n} wards "
        f"= {per_ward:,}/ward (delimitation principle, not polygon area)"
    )
    return pd.Series([per_ward] * n, index=df.index, dtype=int)
