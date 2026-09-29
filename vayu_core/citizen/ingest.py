"""Turn a citizen submission into a verified, grid-joined observation.

The pipeline, end to end:

    photo bytes + (lat, lon)
      -> Gemini vision            structured observation
      -> snap to analysis grid    joinable with satellite rasters
      -> look up that cell/day    HCHO anomaly z, VIIRS fire count
      -> cross-check              corroborated | unsupported | contradicted
      -> persist                  including rejected reports, as the audit trail

Rejected reports are stored rather than dropped. They are the evidence that the
filter is doing its job, and a reviewer who cannot see what was rejected has no
way to tell a working filter from a silent one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from vayu_core.config import RegionConfig
from vayu_core.db import read_conn, upsert_df, write_conn

from .crosscheck import corroborate

# HCHO baselines need history to be meaningful; this is the window used to score
# a report's cell, matching the hotspot module's baseline requirement.
BASELINE_DAYS = 60


@dataclass
class StoredReport:
    id: str
    verdict: str
    may_influence: bool
    usable: bool
    detail: str
    haze_severity: str | None = None
    source_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "verdict": self.verdict,
            "may_influence": self.may_influence,
            "usable": self.usable,
            "detail": self.detail,
            "haze_severity": self.haze_severity,
            "source_type": self.source_type,
        }


def _report_id(lat: float, lon: float, when: dt.datetime, salt: str = "") -> str:
    h = hashlib.sha1(f"{lat:.4f}|{lon:.4f}|{when.isoformat()}|{salt}".encode()).hexdigest()[:16]
    return f"cr-{h}"


def satellite_context(
    region: RegionConfig, grid_lat: float, grid_lon: float, day: dt.date
) -> tuple[float | None, int]:
    """(HCHO z-score, fire count) for one cell on one day.

    The z-score is computed the same way the hotspot detector computes it — the
    cell's own robust baseline — so a report cannot be called "corroborated" by
    an anomaly the hotspot layer would not itself consider anomalous.

    Returns (None, 0) when the satellite had no valid retrieval there, which is
    a statement about coverage, not about the citizen.
    """
    lo = day - dt.timedelta(days=BASELINE_DAYS)
    with read_conn() as con:
        hcho = con.execute(
            """SELECT date, value FROM satellite_grid
               WHERE region = ? AND product = 'hcho'
                 AND grid_lat = ? AND grid_lon = ? AND date BETWEEN ? AND ?""",
            [region.id, grid_lat, grid_lon, lo, day],
        ).df()
        fires = con.execute(
            """SELECT coalesce(sum(fire_count), 0) FROM fire_grid
               WHERE region = ? AND grid_lat = ? AND grid_lon = ? AND date = ?""",
            [region.id, grid_lat, grid_lon, day],
        ).fetchone()

    fire_count = int(fires[0] or 0)
    if hcho.empty:
        return None, fire_count

    today = hcho[pd.to_datetime(hcho["date"]).dt.date == day]["value"]
    if today.empty:
        return None, fire_count

    from vayu_core.national.hotspots import MAD_TO_SIGMA, MIN_SPREAD_FRAC

    v = hcho["value"]
    med = float(v.median())
    mad = float((v - med).abs().median()) * MAD_TO_SIGMA
    spread = max(mad, abs(med) * MIN_SPREAD_FRAC)
    if spread <= 0:
        return None, fire_count
    return float((float(today.iloc[0]) - med) / spread), fire_count


def ingest_photo_report(
    region: RegionConfig,
    *,
    lat: float,
    lon: float,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    when: dt.datetime | None = None,
    note: str = "",
    photo_path: str = "",
) -> StoredReport:
    """Full pipeline for one photo submission.

    Raises `GeminiUnavailable` upward: if the vision model cannot be reached we
    must say so, not store a default reading that would later look like evidence.
    """
    from vayu_core.google_ai import analyse_photo

    when = when or dt.datetime.now(dt.timezone.utc)
    day = when.date()
    obs = analyse_photo(image_bytes, mime_type)

    glat, glon = region.snap(lat, lon)
    z, fires = satellite_context(region, glat, glon, day)

    # An unusable photo (indoor, or too low confidence) is stored for the audit
    # trail but never cross-checked — there is no claim to corroborate.
    if obs.usable:
        c = corroborate(
            haze_rank=obs.haze_rank,
            source_type=obs.source_type,
            visible_smoke=obs.visible_smoke,
            hcho_z=z,
            fire_count=fires,
        )
        verdict, detail, may = c.verdict, c.detail, c.may_influence_hotspots
    else:
        verdict, may = "unusable", False
        detail = (
            "Not an outdoor air-quality photograph, or too ambiguous to read "
            f"(confidence {obs.confidence:.2f}) — excluded from analysis."
        )

    rid = _report_id(lat, lon, when, obs.reasoning[:24])
    row = {
        "id": rid, "region": region.id,
        "lat": lat, "lon": lon, "grid_lat": glat, "grid_lon": glon,
        "date": day, "reported_ts": when, "kind": "photo",
        "haze_severity": obs.haze_severity, "haze_rank": obs.haze_rank,
        "source_type": obs.source_type, "visible_smoke": obs.visible_smoke,
        "ai_confidence": obs.confidence, "ai_reasoning": obs.reasoning,
        "ai_model": obs.model, "usable": obs.usable,
        "pm25": None,
        "verdict": verdict, "hcho_z": z, "fire_count": fires,
        "verdict_detail": detail, "may_influence": may,
        "photo_path": photo_path, "note": note[:500],
    }
    with write_conn() as con:
        upsert_df(con, "citizen_reports", pd.DataFrame([row]), ["id"])

    logger.info(f"citizen report {rid}: {obs.haze_severity}/{obs.source_type} -> {verdict}")
    return StoredReport(
        id=rid, verdict=verdict, may_influence=may, usable=obs.usable,
        detail=detail, haze_severity=obs.haze_severity, source_type=obs.source_type,
    )


def ingest_sensor_report(
    region: RegionConfig,
    *,
    lat: float,
    lon: float,
    pm25: float,
    when: dt.datetime | None = None,
    note: str = "",
) -> StoredReport:
    """A low-cost sensor reading from the public.

    Cross-checked the same way, but on a different claim: a sensor asserts a
    concentration, so "does the satellite agree something unusual was
    happening?" is the same question with a numeric trigger instead of a
    visual one. The CPCB 'Poor' boundary (PM2.5 > 90 ug/m3) is the threshold
    at which a reading is claiming genuinely degraded air.
    """
    when = when or dt.datetime.now(dt.timezone.utc)
    day = when.date()
    glat, glon = region.snap(lat, lon)
    z, fires = satellite_context(region, glat, glon, day)

    claims_pollution = pm25 > 90
    c = corroborate(
        # Map the numeric claim onto the same ordinal scale the cross-check uses.
        haze_rank=3 if claims_pollution else 0,
        source_type="none_visible",
        visible_smoke=claims_pollution,
        hcho_z=z,
        fire_count=fires,
    )

    rid = _report_id(lat, lon, when, f"sensor{pm25:.0f}")
    row = {
        "id": rid, "region": region.id,
        "lat": lat, "lon": lon, "grid_lat": glat, "grid_lon": glon,
        "date": day, "reported_ts": when, "kind": "sensor",
        "haze_severity": None, "haze_rank": None,
        "source_type": None, "visible_smoke": None,
        "ai_confidence": None, "ai_reasoning": None, "ai_model": None,
        "usable": True,
        "pm25": pm25,
        "verdict": c.verdict, "hcho_z": z, "fire_count": fires,
        "verdict_detail": c.detail, "may_influence": c.may_influence_hotspots,
        "photo_path": "", "note": note[:500],
    }
    with write_conn() as con:
        upsert_df(con, "citizen_reports", pd.DataFrame([row]), ["id"])
    return StoredReport(
        id=rid, verdict=c.verdict, may_influence=c.may_influence_hotspots,
        usable=True, detail=c.detail,
    )
