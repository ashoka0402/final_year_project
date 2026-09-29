"""Shared attribution/intervention computation for the API routers.

Both `/attribution/{ward}` and `/interventions` need the same chain: wind field →
back-trajectory → evidence fusion → ROI. Duplicating it would let the two
surfaces drift apart, which is the worst possible failure here — a leaderboard
recommending an action whose evidence the ward sheet doesn't show.

Everything city-wide is cached per run instant. The per-ward work (trajectory,
fusion, plume) is ~25ms, so a single ward is comfortably inside the 300ms warm
budget (TRD §10) and a top-N sweep stays sub-second.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
from fastapi import HTTPException
from loguru import logger

from vayu_core.attribution.confidence import station_agreement_score
from vayu_core.attribution.fusion import WardAttribution, attribute
from vayu_core.attribution.trajectory import Trajectory, WindField, back_trajectory
from vayu_core.config import REPO_ROOT, CityConfig, get_settings
from vayu_core.db import read_conn
from vayu_core.interventions.roi import Leaderboard, build_candidates, prepare_wind

from .deps import read_wards

# How many wards a city-wide leaderboard evaluates, worst-forecast first.
#
# On the demo morning 289 of Delhi's 290 wards are forecast past AQI 300, so
# "flagged" filters nothing and a full sweep costs ~7s — far outside the request
# budget. An operator triages the worst wards anyway, so we rank by forecast
# severity and evaluate the top slice. The response reports `wards_evaluated`
# and `wards_total` so this is visible rather than silently truncated.
CITY_SWEEP_WARDS = 12


def osm_landuse(city_id: str) -> dict | None:
    p = REPO_ROOT / "data" / "samples" / f"osm_{city_id}.geojson"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@lru_cache(maxsize=4)
def wind_field(city_id: str) -> WindField:
    """Airshed wind field — cached per city; building it scans ~50k rows."""
    city = _city(city_id)
    with read_conn() as con:
        wx = con.execute(
            "SELECT * FROM weather_hourly WHERE city = ? AND grid = 'airshed'", [city.id]
        ).df()
    if not wx.empty:
        wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    return WindField(city, wx)


@lru_cache(maxsize=4)
def city_wind(city_id: str) -> pd.DataFrame:
    """Hourly city wind in m/s for the plume. Cached: identical for every ward."""
    with read_conn() as con:
        wx = con.execute(
            "SELECT * FROM weather_hourly WHERE city = ? AND grid = 'city'", [city_id]
        ).df()
    return prepare_wind(wx)


@lru_cache(maxsize=4)
def _fires(city_id: str) -> pd.DataFrame:
    with read_conn() as con:
        return con.execute("SELECT * FROM fires WHERE city = ?", [city_id]).df()


@lru_cache(maxsize=4)
def _permits(city_id: str) -> pd.DataFrame:
    with read_conn() as con:
        return con.execute("SELECT * FROM permits WHERE city = ?", [city_id]).df()


@lru_cache(maxsize=4)
def _road_density(city_id: str) -> dict[str, float]:
    with read_conn() as con:
        df = con.execute(
            "SELECT ward_id, road_density FROM ward_roads WHERE city = ?", [city_id]
        ).df()
    return dict(zip(df["ward_id"], df["road_density"])) if not df.empty else {}


def _city(city_id: str) -> CityConfig:
    from vayu_core.config import load_city

    return load_city(city_id)


def station_agreement(city_id: str) -> float:
    """Do nearby monitors corroborate each other right now?"""
    at = get_settings().now()
    with read_conn() as con:
        pm = con.execute(
            """SELECT value FROM measurements
               WHERE city = ? AND param = 'pm25' AND ts BETWEEN ? AND ?""",
            [city_id, at - pd.Timedelta(hours=3), at],
        ).df()
    return station_agreement_score(pm["value"].tolist() if not pm.empty else [])


def city_aqi(city_id: str) -> int | None:
    """Current city AQI — decides which GRAP stage a citation may invoke."""
    from .scoring import current_run_ts

    run_ts = current_run_ts(city_id)
    if run_ts is None:
        return None
    with read_conn() as con:
        row = con.execute(
            """SELECT max(aqi_p50) FROM forecasts
               WHERE city = ? AND horizon_h = 24 AND run_ts = ?""",
            [city_id, run_ts],
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def ward_attribution(
    city: CityConfig, ward_id: str, hours: int = 12
) -> tuple[WardAttribution, Trajectory, float]:
    """The full attribution chain for one ward."""
    wards = read_wards(city.id)
    row = wards[wards["ward_id"] == ward_id]
    if row.empty:
        raise HTTPException(404, f"unknown ward '{ward_id}' in {city.id}")
    ward = row.iloc[0]

    field = wind_field(city.id)
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
    agreement = station_agreement(city.id)
    result = attribute(
        city,
        ward_id,
        str(ward["name"]),
        float(ward.centroid_lat),
        float(ward.centroid_lon),
        traj,
        at,
        fires=_fires(city.id),
        osm=osm_landuse(city.id),
        permits=_permits(city.id),
        road_density=float(_road_density(city.id).get(ward_id, 0.0)),
        station_agreement=agreement,
    )
    return result, traj, agreement


def ward_leaderboard(city: CityConfig, ward_id: str, hours: int = 12) -> Leaderboard:
    attribution, _, _ = ward_attribution(city, ward_id, hours)
    return build_candidates(
        city,
        attribution,
        read_wards(city.id),
        pd.DataFrame(),           # unused: wind is supplied ready-made
        get_settings().now(),
        city_aqi=city_aqi(city.id),
        wind=city_wind(city.id),
    )


def signal_ts(city_id: str, ward_id: str) -> pd.Timestamp | None:
    """When the system first flagged this ward — the PRD E2 stopwatch's zero.

    TRD §7 defines signal_ts as the threshold event time, which is the forecast
    run that predicted the crossing. Not the moment the leaderboard happened to
    be computed: that is just when someone opened the page, and using it made the
    stopwatch read 0m 0s by construction — it measured nothing.
    """
    with read_conn() as con:
        row = con.execute(
            """SELECT max(run_ts) FROM forecasts
               WHERE city = ? AND ward_id = ? AND aqi_p50 > 200""",
            [city_id, ward_id],
        ).fetchone()
    return pd.Timestamp(row[0]) if row and row[0] is not None else None


def worst_wards(city_id: str, limit: int = CITY_SWEEP_WARDS) -> list[str]:
    """Ward ids ranked by forecast severity at t+24h."""
    from .scoring import current_run_ts

    run_ts = current_run_ts(city_id)
    if run_ts is None:
        return []
    with read_conn() as con:
        df = con.execute(
            """SELECT ward_id FROM forecasts
               WHERE city = ? AND horizon_h = 24 AND run_ts = ?
               ORDER BY aqi_p50 DESC LIMIT ?""",
            [city_id, run_ts, limit],
        ).df()
    return df["ward_id"].tolist() if not df.empty else []


def city_leaderboard(city: CityConfig, limit: int = CITY_SWEEP_WARDS) -> tuple[Leaderboard, dict]:
    """Merged leaderboard across the worst-forecast wards.

    Returns (leaderboard, meta) where meta reports how much of the city was
    actually evaluated — a truncated sweep that doesn't say so reads as
    "these are all the options in the city", which would be a lie.
    """
    ids = worst_wards(city.id, limit)
    wards = read_wards(city.id)
    wind = city_wind(city.id)
    aqi = city_aqi(city.id)
    at = get_settings().now()

    candidates, advisories = [], []
    for wid in ids:
        try:
            attribution, _, _ = ward_attribution(city, wid)
        except HTTPException:
            continue
        board = build_candidates(
            city, attribution, wards, pd.DataFrame(), at, city_aqi=aqi, wind=wind
        )
        candidates.extend(board.candidates)
        advisories.extend(board.advisories)

    # Deduplicate by physical target. Every ward downwind of the same industrial
    # estate raises its own candidate against it, so the city view listed one
    # estate three times as three separate orders — dispatching all three would
    # send three teams to one gate. Keep the strongest; it names the ward with
    # most at stake. ~200 m grid, since two orders that close are one site.
    best: dict[tuple, object] = {}
    for c in candidates:
        key = (c.action_type, round(c.source_lat, 3), round(c.source_lon, 3))
        if key not in best or c.roi_score > best[key].roi_score:
            best[key] = c
    deduped = list(best.values())
    if len(deduped) < len(candidates):
        logger.info(
            f"[{city.id}] merged {len(candidates) - len(deduped)} duplicate orders "
            f"against shared sources"
        )
    candidates = deduped
    candidates.sort(key=lambda c: (-c.roi_score, c.effort_units))

    # One advisory per (kind, category): the same Punjab smoke reaching 12 wards
    # is one fact, not twelve.
    seen, merged = set(), []
    for a in advisories:
        k = (a.kind, a.category)
        if k not in seen:
            seen.add(k)
            merged.append(a)

    meta = {
        "wards_evaluated": len(ids),
        "wards_total": int(len(wards)),
        "selection": "worst forecast AQI at t+24h",
        "city_aqi": aqi,
    }
    logger.info(
        f"[{city.id}] city leaderboard: {len(candidates)} candidates, "
        f"{len(merged)} advisories over {len(ids)}/{len(wards)} wards"
    )
    return Leaderboard(candidates=candidates, advisories=merged), meta
