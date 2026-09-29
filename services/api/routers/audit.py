"""Agent Activity audit trail (PRD F1) — list + live SSE stream.

Backs the right-edge drawer that streams every automated step with its reasoning
and confidence. Reads the single audit_log written by vayu_core/audit.py, so the
drawer and the dossier never disagree about what an agent did.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from vayu_core.db import read_conn

router = APIRouter(tags=["audit"])

# SSE poll interval. The cascade is written at seed time and on live dispatch, so
# the log is quiet between actions; a slow poll keeps a demo idle-connection cheap
# while still surfacing a dispatch within ~2s of it happening.
_POLL_S = 2.0
_MAX_STREAM_S = 300.0  # don't hold a connection open forever


def _rows(after_id: int = 0, limit: int = 100) -> list[dict]:
    with read_conn() as con:
        df = con.execute(
            """SELECT id, ts, agent, trigger, decision, reasoning, confidence, duration_ms
               FROM audit_log WHERE id > ? ORDER BY id ASC LIMIT ?""",
            [after_id, limit],
        ).df()
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": int(r["id"]),
            "ts": r["ts"].isoformat() if r["ts"] is not None else None,
            "agent": r["agent"],
            "trigger": r["trigger"],
            "decision": r["decision"],
            "reasoning": r["reasoning"],
            "confidence": None if pd.isna(r["confidence"]) else float(r["confidence"]),
            "duration_ms": None if pd.isna(r["duration_ms"]) else int(r["duration_ms"]),
        })
    return out


@router.get("/audit")
def list_audit(limit: int = Query(100, le=500)) -> dict:
    """Most recent entries, newest first (the drawer's initial fill)."""
    entries = sorted(_rows(0, 10_000), key=lambda e: e["id"], reverse=True)[:limit]
    return {"entries": entries, "count": len(entries)}


@router.get("/audit/stream")
async def stream_audit(request: Request) -> StreamingResponse:
    """SSE stream: recent history, then new entries as they are written."""

    async def gen():
        # Replay a short tail so a freshly-opened drawer isn't empty.
        seen = 0
        history = _rows(0, 10_000)
        tail = history[-30:]
        for e in tail:
            seen = max(seen, e["id"])
            yield f"data: {json.dumps(e)}\n\n"
        yield ": connected\n\n"

        waited = 0.0
        while waited < _MAX_STREAM_S:
            if await request.is_disconnected():
                break
            fresh = _rows(seen)
            for e in fresh:
                seen = e["id"]
                yield f"data: {json.dumps(e)}\n\n"
            if not fresh:
                yield ": ping\n\n"  # keep the connection warm
            await asyncio.sleep(_POLL_S)
            waited += _POLL_S

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
