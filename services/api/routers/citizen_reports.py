"""Citizen report submission + review API.

`POST /citizen/report/photo` is the endpoint a phone hits. It runs Gemini
vision, cross-checks the result against the satellite record for that grid cell,
and returns the verdict — including when the verdict is "we do not believe this
yet", which the client is expected to show the reporter honestly.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from vayu_core.citizen.ingest import ingest_photo_report, ingest_sensor_report
from vayu_core.config import get_settings, load_region
from vayu_core.db import read_conn

router = APIRouter(prefix="/citizen", tags=["citizen"])

# Phone cameras produce large files and Gemini is billed on input; 8MB is far
# above any reasonable compressed photo and well below a memory risk.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/heic"}


@router.post("/report/photo")
async def submit_photo(
    lat: float = Form(...),
    lon: float = Form(...),
    region_id: str = Form("india"),
    note: str = Form(""),
    when: str = Form(""),
    photo: UploadFile = File(...),
) -> dict:
    """Submit a geotagged photograph for AI analysis + satellite cross-check."""
    settings = get_settings()
    if not settings.google_ai_available:
        # Explicit, not a fake reading: the whole point of this endpoint is the
        # vision analysis, and inventing one would corrupt the record.
        raise HTTPException(
            503,
            "Photo analysis needs Google AI. Set GOOGLE_API_KEY "
            "(free key at aistudio.google.com/apikey).",
        )

    if photo.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported image type {photo.content_type!r}")

    data = await photo.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image too large ({len(data)/1e6:.1f} MB, max 8 MB)")

    try:
        region = load_region(region_id)
    except KeyError:
        raise HTTPException(404, f"Unknown region {region_id!r}") from None

    w, s, e, n = region.bbox
    if not (s <= lat <= n and w <= lon <= e):
        raise HTTPException(
            422, f"({lat}, {lon}) is outside {region.name}'s bounds — cannot cross-check it."
        )

    ts = _parse_when(when)
    from vayu_core.google_ai import GeminiUnavailable

    try:
        stored = ingest_photo_report(
            region, lat=lat, lon=lon, image_bytes=data,
            mime_type=photo.content_type or "image/jpeg", when=ts, note=note,
        )
    except GeminiUnavailable as exc:
        raise HTTPException(503, f"Vision analysis unavailable: {exc}") from exc

    return stored.to_dict()


@router.post("/report/sensor")
def submit_sensor(
    lat: float = Form(...),
    lon: float = Form(...),
    pm25: float = Form(...),
    region_id: str = Form("india"),
    note: str = Form(""),
    when: str = Form(""),
) -> dict:
    """Submit a low-cost sensor reading (no Google AI needed for this path)."""
    try:
        region = load_region(region_id)
    except KeyError:
        raise HTTPException(404, f"Unknown region {region_id!r}") from None
    if pm25 < 0 or pm25 > 2000:
        raise HTTPException(422, "pm25 must be between 0 and 2000 ug/m3")

    return ingest_sensor_report(
        region, lat=lat, lon=lon, pm25=pm25, when=_parse_when(when), note=note
    ).to_dict()


@router.get("/reports")
def list_reports(
    region_id: str = "india",
    verdict: str = Query("all", description="all | corroborated | unsupported | contradicted | unusable"),
    limit: int = 200,
) -> dict:
    """Review queue. Rejected reports are listed too — hiding them would make a
    broken filter indistinguishable from a working one."""
    where, params = ["region = ?"], [region_id]
    if verdict != "all":
        where.append("verdict = ?")
        params.append(verdict)

    with read_conn() as con:
        df = con.execute(
            f"""SELECT id, lat, lon, grid_lat, grid_lon, date, reported_ts, kind,
                       haze_severity, source_type, visible_smoke, ai_confidence,
                       ai_reasoning, ai_model, usable, pm25,
                       verdict, hcho_z, fire_count, verdict_detail, may_influence, note
                FROM citizen_reports WHERE {' AND '.join(where)}
                ORDER BY reported_ts DESC LIMIT ?""",
            [*params, limit],
        ).df()

    counts = {}
    with read_conn() as con:
        for v, c in con.execute(
            "SELECT verdict, count(*) FROM citizen_reports WHERE region = ? GROUP BY 1",
            [region_id],
        ).fetchall():
            counts[v] = int(c)

    items = df.where(df.notna(), None).to_dict("records")
    for it in items:
        for k in ("date", "reported_ts"):
            if it.get(k) is not None:
                it[k] = str(it[k])
    return {
        "region": region_id,
        "count": len(items),
        "by_verdict": counts,
        "google_ai_enabled": get_settings().google_ai_available,
        "items": items,
    }


def _parse_when(raw: str) -> dt.datetime | None:
    if not raw:
        return None
    try:
        d = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(422, f"Could not parse timestamp {raw!r}") from None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
