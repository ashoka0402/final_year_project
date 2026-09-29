"""Citizen surface (PRD Epic D) — the forecast, said plainly, in the reader's
language. Backed by the Herald (vayu_core/herald.py)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from vayu_core import herald
from vayu_core.config import get_settings
from vayu_core.db import read_conn
from vayu_core.observations import snapshot

from ..deps import get_city, read_measurements, read_stations, read_wards

router = APIRouter(prefix="/cities", tags=["citizen"])


def _forecast_anchors(city_id: str, ward_id: str) -> dict[int, float]:
    """The ward's AQI forecast at each stored horizon, for the app's current run."""
    from ..scoring import current_run_ts

    run_ts = current_run_ts(city_id)
    if run_ts is None:
        return {}
    with read_conn() as con:
        df = con.execute(
            """SELECT horizon_h, aqi_p50 FROM forecasts
               WHERE city = ? AND ward_id = ? AND run_ts = ?""",
            [city_id, ward_id, run_ts],
        ).df()
    return {int(r.horizon_h): float(r.aqi_p50) for r in df.itertuples()} if not df.empty else {}


@router.get("/{city_id}/citizen/{ward_id}")
def get_citizen_brief(
    city_id: str,
    ward_id: str,
    lang: str = Query("en", description="en | hi | pa"),
) -> dict:
    """AQI now, the 48h clean-hours strip, and audience advisories (PRD D)."""
    city = get_city(city_id)
    at = get_settings().now()

    ward = read_wards(city.id)
    row = ward[ward["ward_id"] == ward_id]
    if row.empty:
        raise HTTPException(404, f"unknown ward '{ward_id}' in {city.id}")
    w = row.iloc[0]

    # Current AQI: the same IDW snapshot the Command Center shows, so the citizen
    # number matches the commissioner's.
    snap = snapshot(city, read_stations(city.id), read_measurements(city.id, at), ward, at)
    now = next((wr for wr in snap.wards if wr.ward_id == ward_id), None)
    now_aqi = now.aqi if now else None

    b = herald.brief(
        ward_id=ward_id, ward_name=str(w["name"]), now_aqi=now_aqi,
        anchors=_forecast_anchors(city.id, ward_id), at=at, tz=city.timezone, language=lang,
    )
    out = b.to_dict()
    out["low_confidence"] = bool(now.low_confidence) if now else True
    out["languages"] = [{"code": c, "label": herald.LANGUAGE_LABEL[c]} for c in herald.LANGUAGES]
    return out


@router.get("/{city_id}/citizen")
def citizen_default(city_id: str, lang: str = Query("en")) -> dict:
    """Brief for the worst-forecast ward — a sensible landing without a ward pick."""
    from .. import compute

    city = get_city(city_id)
    worst = compute.worst_wards(city.id, limit=1)
    if not worst:
        raise HTTPException(404, "no forecast available — run `make seed`")
    return get_citizen_brief(city_id, worst[0], lang)
