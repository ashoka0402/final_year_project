"""GRAP Autopilot API (PRD C4). Draft on a forecast stage crossing; human approves.

See vayu_core/interventions/grap.py for why the approval is always human.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from loguru import logger

from vayu_core.audit import record as audit_record
from vayu_core.config import get_settings
from vayu_core.db import read_conn, write_conn
from vayu_core.interventions.grap import (
    GrapDraft,
    GrapMeasure,
    build_draft,
    stage_for_aqi,
    stage_label,
)
from vayu_core.observations import snapshot

from ..deps import get_city, read_measurements, read_stations, read_wards

router = APIRouter(prefix="/cities", tags=["grap"])
approve_router = APIRouter(tags=["grap"])


def _city_forecast_series(city_id: str) -> list[tuple[int, float]]:
    """Population-weighted city AQI per horizon (TRD 5.7).

    Wards are delimited to equal population, so the population weighting is a
    plain mean across ward p50 forecasts.
    """
    from ..scoring import current_run_ts

    run_ts = current_run_ts(city_id)
    if run_ts is None:
        return []
    with read_conn() as con:
        df = con.execute(
            """SELECT horizon_h, avg(aqi_p50) AS city_aqi FROM forecasts
               WHERE city = ? AND run_ts = ?
               GROUP BY horizon_h ORDER BY horizon_h""",
            [city_id, run_ts],
        ).df()
    return [(int(r.horizon_h), float(r.city_aqi)) for r in df.itertuples()]


def _observed_city_aqi(city_id: str) -> int | None:
    city = get_city(city_id)
    at = get_settings().now()
    snap = snapshot(city, read_stations(city_id), read_measurements(city_id, at), read_wards(city_id), at)
    return snap.city_aqi


def _stored_draft(city_id: str) -> GrapDraft | None:
    """A persisted draft (e.g. the seeded demo record), if any."""
    with read_conn() as con:
        row = con.execute(
            """SELECT id, city, stage, trigger_forecast_ts, measures_json, status
               FROM grap_drafts WHERE city = ? ORDER BY trigger_forecast_ts DESC LIMIT 1""",
            [city_id],
        ).fetchone()
    if not row:
        return None
    measures = [GrapMeasure(**m) for m in json.loads(row[4])]
    stage = int(row[2])
    return GrapDraft(
        id=row[0], city=row[1], current_stage=stage - 1,
        current_stage_label=stage_label(stage - 1), forecast_stage=stage,
        forecast_stage_label=stage_label(stage), forecast_aqi=0,
        trigger_forecast_ts=row[3], crossing_eta_h=None, status=row[5], measures=measures,
    )


@router.get("/{city_id}/grap")
def get_grap(city_id: str) -> dict:
    """The autopilot draft for this city, plus the current in-force stage.

    `draft` is null when no stage crossing is forecast within 48h — the honest
    result, and the common one. A persisted (seeded/approved) draft is returned
    even without a live crossing so the human-approval flow is demonstrable.
    """
    city = get_city(city_id)
    observed = _observed_city_aqi(city.id)
    current_stage = stage_for_aqi(observed)
    series = _city_forecast_series(city.id)
    at = get_settings().now()

    live = build_draft(city, observed, series, at)
    stored = _stored_draft(city.id)

    # Prefer a persisted draft the operator may already be acting on; else the
    # freshly-computed one.
    draft = stored or live

    return {
        "city": city.id,
        "current_stage": current_stage,
        "current_stage_label": stage_label(current_stage),
        "observed_city_aqi": observed,
        "crossing_forecast": live is not None,
        "forecast_series": [{"horizon_h": h, "city_aqi": round(a)} for h, a in series],
        "draft": draft.to_dict() if draft else None,
    }


@approve_router.post("/grap/{draft_id}/approve")
def approve_grap(draft_id: str) -> dict:
    """Human approval — the only way a draft becomes active (PRD C4)."""
    return _transition(draft_id, "approved", "Approved GRAP measures")


@approve_router.post("/grap/{draft_id}/dismiss")
def dismiss_grap(draft_id: str) -> dict:
    return _transition(draft_id, "dismissed", "Dismissed GRAP draft")


def _transition(draft_id: str, status: str, decision: str) -> dict:
    with read_conn() as con:
        row = con.execute(
            "SELECT city, stage, status FROM grap_drafts WHERE id = ?", [draft_id]
        ).fetchone()
    if not row:
        raise HTTPException(404, f"unknown GRAP draft '{draft_id}'")
    if row[2] in ("approved", "dismissed"):
        return {"id": draft_id, "status": row[2], "note": "already resolved"}

    with write_conn() as con:
        con.execute("UPDATE grap_drafts SET status = ? WHERE id = ?", [status, draft_id])
    audit_record(
        "enforcer", f"{decision} · {stage_label(int(row[1]))} · {row[0]}",
        trigger=f"grap_{status}:{draft_id}",
        reasoning="Human-in-the-loop: a person approved these stage measures; the "
                  "autopilot only drafted them.",
        confidence=None,
    )
    logger.info(f"GRAP {draft_id} -> {status}")
    return {"id": draft_id, "status": status}
