"""Attribution surfaces: source shares with evidence, and the back-trajectory."""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response

from vayu_core.attribution.confidence import station_agreement_score
from vayu_core.attribution.fusion import attribute
from vayu_core.attribution.trajectory import WindField, back_trajectory
from vayu_core.config import REPO_ROOT, CityConfig, get_settings
from vayu_core.db import read_conn

from ..deps import get_city, read_wards

router = APIRouter(prefix="/cities", tags=["attribution"])

TRAJECTORY_HOURS = (6, 12, 24)


def _osm(city_id: str) -> dict | None:
    p = REPO_ROOT / "data" / "samples" / f"osm_{city_id}.geojson"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _wind_field(city: CityConfig) -> WindField:
    """Airshed wind field. Cached per city — building it scans ~50k rows and the
    field is identical for every ward in a run."""
    with read_conn() as con:
        wx = con.execute(
            "SELECT * FROM weather_hourly WHERE city = ? AND grid = 'airshed'", [city.id]
        ).df()
    if not wx.empty:
        wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    return WindField(city, wx)


def _ward_row(city_id: str, ward_id: str):
    wards = read_wards(city_id)
    row = wards[wards["ward_id"] == ward_id]
    if row.empty:
        raise HTTPException(404, f"unknown ward '{ward_id}' in {city_id}")
    return row.iloc[0]


@router.get("/{city_id}/trajectory/{ward_id}")
def get_trajectory(city_id: str, ward_id: str, hours: int = Query(12)) -> Response:
    """GeoJSON polyline + dispersion cone for the map animation (PRD B3)."""
    city = get_city(city_id)
    if hours not in TRAJECTORY_HOURS:
        raise HTTPException(400, f"hours must be one of {list(TRAJECTORY_HOURS)}")

    ward = _ward_row(city.id, ward_id)
    field = _wind_field(city)
    if not field.available:
        raise HTTPException(
            404,
            "No airshed wind field for this city — run `make seed`. Back-trajectories "
            "need the wide grid, not the city grid.",
        )

    at = get_settings().now()
    traj = back_trajectory(
        city, ward_id, float(ward.centroid_lat), float(ward.centroid_lon), at, hours, field
    )
    payload = traj.to_geojson()
    payload["properties"] = {
        "ward_id": ward_id,
        "ward_name": str(ward["name"]),
        "hours": hours,
        "length_km": round(traj.length_km, 1),
        "mean_speed_kmh": round(traj.mean_speed_kmh, 1),
        "stagnant": traj.stagnant,
        "run_ts": at.isoformat(),
    }
    return Response(content=json.dumps(payload), media_type="application/geo+json")


@router.get("/{city_id}/attribution/{ward_id}")
def get_attribution(city_id: str, ward_id: str, hours: int = Query(12)) -> dict:
    """Source shares with clickable evidence (PRD B1/B2)."""
    city = get_city(city_id)
    if hours not in TRAJECTORY_HOURS:
        raise HTTPException(400, f"hours must be one of {list(TRAJECTORY_HOURS)}")

    ward = _ward_row(city.id, ward_id)
    at = get_settings().now()
    field = _wind_field(city)
    if not field.available:
        raise HTTPException(404, "No airshed wind field — run `make seed`.")

    traj = back_trajectory(
        city, ward_id, float(ward.centroid_lat), float(ward.centroid_lon), at, hours, field
    )

    with read_conn() as con:
        fires = con.execute("SELECT * FROM fires WHERE city = ?", [city.id]).df()
        permits = con.execute("SELECT * FROM permits WHERE city = ?", [city.id]).df()
        rd = con.execute(
            "SELECT road_density FROM ward_roads WHERE city = ? AND ward_id = ?", [city.id, ward_id]
        ).fetchone()
        no2 = con.execute(
            """SELECT avg(value) FROM measurements
               WHERE city = ? AND param = 'no2' AND ts BETWEEN ? AND ?""",
            [city.id, at - timedelta(hours=3), at],
        ).fetchone()
        # Station agreement: do nearby monitors corroborate each other right now?
        pm = con.execute(
            """SELECT value FROM measurements
               WHERE city = ? AND param = 'pm25' AND ts BETWEEN ? AND ?""",
            [city.id, at - timedelta(hours=3), at],
        ).df()

    agreement = station_agreement_score(pm["value"].tolist() if not pm.empty else [])

    result = attribute(
        city,
        ward_id,
        str(ward["name"]),
        float(ward.centroid_lat),
        float(ward.centroid_lon),
        traj,
        at,
        fires=fires,
        osm=_osm(city.id),
        permits=permits,
        road_density=float(rd[0]) if rd and rd[0] is not None else 0.0,
        no2=float(no2[0]) if no2 and no2[0] is not None else None,
        regional_pm_proxy=1.0,
        station_agreement=agreement,
    )

    out = result.to_dict()
    out["ward_name"] = str(ward["name"])
    out["station_agreement"] = agreement
    out["trajectory"] = {
        "hours": hours,
        "length_km": round(traj.length_km, 1),
        "mean_speed_kmh": round(traj.mean_speed_kmh, 1),
        "stagnant": traj.stagnant,
    }
    return out


@router.get("/{city_id}/attribution-crosscheck")
def crosscheck(city_id: str) -> dict:
    """Our shares vs published Delhi splits (TRD 5.3) — the Methodology artefact."""
    p = REPO_ROOT / "docs" / "attribution_crosscheck.json"
    if not p.exists():
        raise HTTPException(
            404, "Cross-check has not been run — `make calibrate` generates it."
        )
    return json.loads(p.read_text())
