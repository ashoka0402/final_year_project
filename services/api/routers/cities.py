"""City surfaces: catalogue, current conditions, ward geometry, data status."""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Response

from vayu_core.config import get_settings, list_cities
from vayu_core.db import read_conn
from vayu_core.observations import snapshot

from ..deps import get_city, read_data_status, read_measurements, read_stations, read_wards
from ..schemas import CitySummary, CurrentOut, DataStatus, GeoJSON, StationOut, WardOut

router = APIRouter(prefix="/cities", tags=["cities"])


@router.get("", response_model=list[CitySummary])
def get_cities() -> list[CitySummary]:
    """Every city discovered in config/cities — adding a file adds a city (PRD G1)."""
    with read_conn() as con:
        counts = con.execute(
            """SELECT c.city,
                      (SELECT count(*) FROM wards w WHERE w.city = c.city)    AS wards,
                      (SELECT count(*) FROM stations s WHERE s.city = c.city) AS stations
               FROM (SELECT DISTINCT city FROM wards
                     UNION SELECT DISTINCT city FROM stations) c"""
        ).df()

    lookup = {r.city: r for r in counts.itertuples()} if not counts.empty else {}
    out = []
    for c in list_cities():
        row = lookup.get(c.id)
        out.append(
            CitySummary(
                id=c.id,
                name=c.name,
                timezone=c.timezone,
                bbox=c.bbox,
                map_center=c.map_center,
                map_zoom=c.map_zoom,
                languages=c.languages,
                grap_applicable=c.grap_applicable,
                ward_count=int(getattr(row, "wards", 0) or 0),
                station_count=int(getattr(row, "stations", 0) or 0),
                population=c.population.total,
                population_source=c.population.source,
            )
        )
    return out


@router.get("/{city_id}/current", response_model=CurrentOut)
def get_current(city_id: str) -> CurrentOut:
    """Latest AQI per ward + per station, with per-source freshness (PRD A1/F2)."""
    city = get_city(city_id)
    settings = get_settings()
    at = settings.now()

    wards = read_wards(city.id)
    stations = read_stations(city.id)
    measurements = read_measurements(city.id, at)

    snap = snapshot(city, stations, measurements, wards, at)
    status_df = read_data_status(city.id)

    return CurrentOut(
        city=city.id,
        as_of=snap.as_of,
        demo_mode=settings.demo_mode,
        aqi=snap.city_aqi,
        category=snap.city_category,
        color=snap.city_color,
        sources=snap.sources,
        wards=[WardOut(**w.__dict__) for w in snap.wards],
        stations=[StationOut(**s.__dict__) for s in snap.stations],
        data_status=[
            DataStatus(
                source=r.source,
                status=r.status,
                detail=r.detail or "",
                rows_loaded=int(r.rows_loaded or 0),
                fetched_ts=None if pd.isna(r.fetched_ts) else r.fetched_ts,
            )
            for r in status_df.itertuples()
        ],
    )


@router.get("/{city_id}/wards.geojson", response_model=GeoJSON)
def get_ward_geometry(city_id: str) -> Response:
    """Ward polygons, served separately from values.

    The geometry is ~700 KB for Delhi and never changes between runs, while
    /current changes every refresh. Splitting them lets the browser cache the
    expensive half and keeps the choropleth update small.
    """
    city = get_city(city_id)
    wards = read_wards(city.id, with_geom=True)

    features = [
        {
            "type": "Feature",
            "id": w.ward_id,
            "properties": {
                "ward_id": w.ward_id,
                "name": w.name,
                "population": int(w.population),
                "area_km2": round(float(w.area_km2), 3),
                "centroid": [round(float(w.centroid_lon), 5), round(float(w.centroid_lat), 5)],
            },
            "geometry": json.loads(w.geom_geojson),
        }
        for w in wards.itertuples()
    ]

    payload = {
        "type": "FeatureCollection",
        "attribution": city.wards.attribution,
        "features": features,
    }
    return Response(
        content=json.dumps(payload),
        media_type="application/geo+json",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/{city_id}/data_status", response_model=list[DataStatus])
def get_data_status(city_id: str) -> list[DataStatus]:
    city = get_city(city_id)
    df = read_data_status(city.id)
    return [
        DataStatus(
            source=r.source,
            status=r.status,
            detail=r.detail or "",
            rows_loaded=int(r.rows_loaded or 0),
            fetched_ts=None if pd.isna(r.fetched_ts) else r.fetched_ts,
        )
        for r in df.itertuples()
    ]
