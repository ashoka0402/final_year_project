"""Interventions: the ROI leaderboard, dispatch, and the inspector's order list.

This is where VAYU stops describing and starts acting. The state machine is the
contract (App Flow §4):

    candidate --dispatch--> dispatched --execute--> executed --verify--> verified

A candidate is computed live from current evidence and does not exist in the
database until someone dispatches it — persisting speculative rows would let a
stale ranking be actioned hours later against evidence that has moved on.
Dispatch is the moment a recommendation becomes a record: that is when the
dossier is rendered, the order is written, and the audit entry is made.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from vayu_core.config import get_settings
from vayu_core.db import read_conn, write_conn
from vayu_core.interventions.dossier import generate
from vayu_core.interventions.roi import Candidate

from .. import compute
from ..deps import get_city, read_wards

router = APIRouter(tags=["interventions"])

STATUSES = ("candidate", "dispatched", "executed", "verified")


class DispatchRequest(BaseModel):
    candidate: dict = Field(..., description="The Candidate object from the leaderboard")
    signal_ts: datetime | None = Field(
        None, description="When the threshold event fired; drives the PRD E2 stopwatch"
    )


class ExecuteRequest(BaseModel):
    note: str = Field(..., min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# leaderboard (PRD C1)
# ---------------------------------------------------------------------------

@router.get("/cities/{city_id}/interventions")
def get_leaderboard(
    city_id: str,
    ward_id: str | None = Query(None, description="Restrict to one ward"),
    action_type: str | None = Query(None),
) -> dict:
    """Ranked actions for a ward, or for the worst-forecast wards city-wide."""
    city = get_city(city_id)

    if ward_id:
        board = compute.ward_leaderboard(city, ward_id)
        meta = {"wards_evaluated": 1, "wards_total": None, "selection": f"ward {ward_id}",
                "city_aqi": compute.city_aqi(city.id)}
    else:
        board, meta = compute.city_leaderboard(city)

    candidates = board.candidates
    if action_type:
        candidates = [c for c in candidates if c.action_type == action_type]

    return {
        "city": city.id,
        "ward_id": ward_id,
        "computed_ts": get_settings().now().isoformat(),
        "candidates": [c.to_dict() for c in candidates],
        # Not decoration. When the leaderboard is empty because the sources are
        # 250 km away in another state, "no candidates" alone reads as "nothing
        # to do" — the opposite of the truth.
        "advisories": [a.to_dict() for a in board.advisories],
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# dispatch (PRD C2/C3)
# ---------------------------------------------------------------------------

def _row_to_order(r) -> dict:
    d = dict(r)
    for k in ("created_ts", "signal_ts", "dispatched_ts", "executed_ts"):
        v = d.get(k)
        d[k] = v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) and pd.notna(v) else None
    d["has_dossier"] = bool(d.get("dossier_path"))
    d.pop("dossier_path", None)   # a server path is not the client's business
    # DuckDB -> pandas hands back numpy scalars, which pydantic cannot serialize
    # ("Unable to serialize unknown type: numpy.int32"). Coerce to natives.
    for k, v in d.items():
        if isinstance(v, np.generic):
            d[k] = v.item()
        elif v is not None and isinstance(v, float) and pd.isna(v):
            d[k] = None
    return d


@router.post("/interventions/dispatch", status_code=201)
def dispatch(req: DispatchRequest = Body(...)) -> dict:
    """Render the dossier, persist the order, hand it to the inspector."""
    try:
        cand = Candidate(**{**req.candidate, "evidence": []})
    except TypeError as e:
        raise HTTPException(422, f"malformed candidate: {e}") from None

    # Rebuild evidence objects (they arrive as plain dicts over the wire).
    from vayu_core.attribution.fusion import Evidence

    ev = []
    for item in req.candidate.get("evidence", []):
        try:
            ev.append(Evidence(**item))
        except TypeError:
            continue
    cand.evidence = ev

    t0 = time.perf_counter()
    city = get_city(cand.city)
    now = get_settings().now()
    # Provenance, not page-load time: the signal is the forecast run that flagged
    # this ward (TRD §7). The client may override it, but it must never default
    # to "now" — that measures nothing.
    signal_ts = req.signal_ts or compute.signal_ts(city.id, cand.ward_id) or now

    with read_conn() as con:
        existing = con.execute(
            "SELECT status FROM interventions WHERE id = ?", [cand.id]
        ).fetchone()
    if existing:
        # Idempotent: a double-click must not produce two orders for one site.
        logger.info(f"{cand.id} already dispatched (status={existing[0]})")
        return get_order(cand.id)

    wards = read_wards(city.id, with_geom=True)
    row = wards[wards["ward_id"] == cand.ward_id]
    if row.empty:
        raise HTTPException(404, f"unknown ward '{cand.ward_id}'")
    w = row.iloc[0]

    wind = compute.city_wind(city.id)
    wdir = None
    if not wind.empty:
        recent = wind[wind["ts"] <= now]
        if not recent.empty:
            wdir = float(recent.iloc[-1]["wind_dir_deg"])

    dossier = generate(
        cand, city, w["geom_geojson"], float(w.centroid_lat), float(w.centroid_lon),
        signal_ts=signal_ts, wind_dir_from_deg=wdir,
    )
    pipeline_ms = int((time.perf_counter() - t0) * 1000)

    with write_conn() as con:
        con.execute(
            """INSERT INTO interventions
               (id, city, ward_id, created_ts, action_type, title, source_lat, source_lon,
                predicted_ugm3_averted, population_protected, effort_units, confidence,
                roi_score, status, dossier_path, signal_ts, dispatched_ts, seeded)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [cand.id, cand.city, cand.ward_id, now, cand.action_type, cand.title,
             cand.source_lat, cand.source_lon, cand.predicted_ugm3_averted,
             cand.population_protected, cand.effort_units, cand.confidence, cand.roi_score,
             "dispatched", str(dossier.path), signal_ts, now, False],
        )
        con.execute(
            """INSERT INTO audit_log (ts, agent, trigger, inputs_hash, decision, reasoning,
                                      confidence, duration_ms)
               VALUES (?,?,?,?,?,?,?,?)""",
            [now, "enforcer", f"dispatch:{cand.ward_id}", cand.id,
             f"Dispatched {cand.action_type} — {cand.title}", cand.rationale,
             cand.confidence, pipeline_ms],
        )

    elapsed = (now - pd.Timestamp(signal_ts)).total_seconds()
    logger.info(f"dispatched {cand.id} · signal->dossier {elapsed:.0f}s · pipeline {pipeline_ms}ms")
    out = get_order(cand.id)
    # Two different clocks, both reported because either alone misleads.
    # signal_to_dossier_s is the PRD E2 metric on the application clock; in a
    # replayed demo the forecast run and "now" are the same instant, so it is
    # legitimately 0 and would look broken on its own. pipeline_ms is the real
    # wall-clock cost of turning the signal into a signed-ready PDF.
    out["signal_to_dossier_s"] = int(elapsed)
    out["pipeline_ms"] = pipeline_ms
    return out


# ---------------------------------------------------------------------------
# inspector (PRD C3/C5)
# ---------------------------------------------------------------------------

@router.get("/interventions")
def list_orders(
    city_id: str | None = Query(None),
    status: str | None = Query(None),
) -> dict:
    """The inspector's list, newest first."""
    if status and status not in STATUSES:
        raise HTTPException(400, f"status must be one of {list(STATUSES)}")

    sql = "SELECT * FROM interventions WHERE 1=1"
    params: list = []
    if city_id:
        sql += " AND city = ?"
        params.append(city_id)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY created_ts DESC"

    with read_conn() as con:
        df = con.execute(sql, params).df()
    return {"orders": [_row_to_order(r) for _, r in df.iterrows()], "count": len(df)}


@router.get("/interventions/{order_id}")
def get_order(order_id: str) -> dict:
    with read_conn() as con:
        df = con.execute("SELECT * FROM interventions WHERE id = ?", [order_id]).df()
    if df.empty:
        raise HTTPException(404, f"unknown order '{order_id}'")
    return _row_to_order(df.iloc[0])


@router.get("/interventions/{order_id}/dossier")
def get_dossier(order_id: str) -> FileResponse:
    """The PDF an inspector carries (PRD C2: downloadable)."""
    with read_conn() as con:
        row = con.execute(
            "SELECT dossier_path FROM interventions WHERE id = ?", [order_id]
        ).fetchone()
    if not row:
        raise HTTPException(404, f"unknown order '{order_id}'")
    from pathlib import Path

    p = Path(row[0]) if row[0] else None
    if not p or not p.exists():
        raise HTTPException(404, "dossier file missing — re-dispatch to regenerate")
    return FileResponse(p, media_type="application/pdf", filename=f"{order_id}.pdf")


@router.post("/interventions/{order_id}/execute")
def execute(order_id: str, req: ExecuteRequest = Body(...)) -> dict:
    """Inspector marks the order done; verification tracking starts (PRD C5)."""
    order = get_order(order_id)
    if order["status"] == "executed":
        return order
    if order["status"] != "dispatched":
        raise HTTPException(
            409,
            f"order is '{order['status']}' — only a dispatched order can be executed",
        )

    now = get_settings().now()
    with write_conn() as con:
        con.execute(
            "UPDATE interventions SET status = 'executed', executed_ts = ? WHERE id = ?",
            [now, order_id],
        )
        con.execute(
            """INSERT INTO audit_log (ts, agent, trigger, inputs_hash, decision, reasoning,
                                      confidence, duration_ms)
               VALUES (?,?,?,?,?,?,?,?)""",
            [now, "inspector", f"execute:{order_id}", order_id,
             "Order marked executed", req.note, None, None],
        )
    logger.info(f"{order_id} executed: {req.note[:60]}")
    return get_order(order_id)
