"""Federated corridor API — the interoperability surface.

Any state agency can consume a corridor bulletin over plain HTTP without
adopting VAYU's database, models or code. The payload is versioned and carries
units and provenance with every number, because cross-agency pipelines fail on
formats far more often than on science.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException, Query

from vayu_core.config import load_region
from vayu_core.db import read_conn
from vayu_core.national import corridors as CO

router = APIRouter(prefix="/corridors", tags=["corridors"])


@router.get("")
def list_corridors(region_id: str = "india") -> dict:
    region = _region(region_id)
    out = []
    for c in CO.load_corridors(region_id):
        out.append(
            {
                "id": c.id, "name": c.name, "states": c.states,
                "cells": len(c.cells(region)), "buffer_deg": c.buffer_deg,
                "waypoints": [list(p) for p in c.waypoints],
            }
        )
    return {"region": region_id, "schema": CO.SCHEMA_VERSION, "count": len(out), "corridors": out}


@router.get("/{corridor_id}/bulletin")
def corridor_bulletin(
    corridor_id: str,
    date: str = Query(..., description="YYYY-MM-DD"),
    region_id: str = "india",
) -> dict:
    """One corridor's federated daily bulletin."""
    region = _region(region_id)
    corridor = CO.get_corridor(corridor_id, region_id)
    if corridor is None:
        raise HTTPException(404, f"Unknown corridor {corridor_id!r}")
    try:
        day = dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(422, f"Could not parse date {date!r} (expected YYYY-MM-DD)") from None

    from vayu_core.national import hotspots as H

    with read_conn() as con:
        hcho = con.execute(
            """SELECT grid_lat, grid_lon, date, value, n_obs FROM satellite_grid
               WHERE region = ? AND product = 'hcho'""",
            [region_id],
        ).df()
        fires = con.execute(
            "SELECT grid_lat, grid_lon, date, fire_count, frp_sum FROM fire_grid WHERE region = ?",
            [region_id],
        ).df()
        citizen = con.execute(
            """SELECT grid_lat, grid_lon, date, may_influence FROM citizen_reports
               WHERE region = ?""",
            [region_id],
        ).df()

    if hcho.empty:
        raise HTTPException(
            404,
            "No satellite data ingested yet — run the S5P ingest for this region.",
        )

    hs = H.detect(hcho[hcho["n_obs"] >= H.DEFAULT_MIN_OBS], fires).hotspots
    bulletin = CO.build_bulletin(corridor, region, day, hcho, fires, hs, citizen)
    return bulletin.to_payload()


def _region(region_id: str):
    try:
        return load_region(region_id)
    except KeyError:
        raise HTTPException(404, f"Unknown region {region_id!r}") from None
