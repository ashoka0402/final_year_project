"""GRAP Autopilot (TRD 5.7, PRD C4) — draft, never decide.

When the city's air is forecast to cross a GRAP stage boundary within 48h, this
drafts the measures that stage mandates, each with its clause citation, and
stops. The draft is inert until a human approves it (POST /grap/{id}/approve).

That human-in-the-loop gate is the entire point, not a formality: GRAP Stage III
bans construction across the NCR and Stage IV halts trucks and industry. Those
are decisions with real economic weight, and an automated system proposing them
is useful only if a person is unambiguously the one who pulls the trigger. The
"human-in-the-loop" badge on the card says so, and nothing here writes an
approval on its own.

Stage thresholds (CAQM): I ≥ 201, II ≥ 301, III ≥ 401, IV ≥ 451.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

from vayu_core.config import CityConfig
from vayu_core.interventions.roi import load_corpus

# (stage number, lower AQI bound, label).
STAGE_THRESHOLDS = [
    (1, 201, "Stage I — Poor"),
    (2, 301, "Stage II — Very Poor"),
    (3, 401, "Stage III — Severe"),
    (4, 451, "Stage IV — Severe+"),
]

FORECAST_WINDOW_H = 48  # a crossing must be within this horizon to draft (TRD 5.7)


def stage_for_aqi(aqi: float | None) -> int:
    """The GRAP stage in force at an AQI (0 = below Stage I)."""
    if aqi is None:
        return 0
    stage = 0
    for s, lo, _ in STAGE_THRESHOLDS:
        if aqi >= lo:
            stage = s
    return stage


def stage_label(stage: int) -> str:
    for s, _, label in STAGE_THRESHOLDS:
        if s == stage:
            return label
    return "Below Stage I"


@dataclass
class GrapMeasure:
    clause_id: str
    title: str
    text: str
    citation: str
    action_supported: str | None = None


@dataclass
class GrapDraft:
    id: str
    city: str
    current_stage: int
    current_stage_label: str
    forecast_stage: int
    forecast_stage_label: str
    forecast_aqi: int
    trigger_forecast_ts: datetime
    crossing_eta_h: int | None
    status: str                       # draft | approved | dismissed
    measures: list[GrapMeasure] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trigger_forecast_ts"] = self.trigger_forecast_ts.isoformat()
        return d


def measures_for_stage(stage: int) -> list[GrapMeasure]:
    """All corpus clauses that come into force AT this stage.

    Only clauses whose own stage equals the crossing target — a Stage III draft
    lists the measures Stage III newly adds, not everything from Stage I upward,
    which are already in force and would clutter the approval.
    """
    out: list[GrapMeasure] = []
    for c in load_corpus():
        if c.get("stage") == stage:
            out.append(GrapMeasure(
                clause_id=c["id"], title=c["title"], text=c["text"],
                citation=c["citation"], action_supported=c.get("action_supported"),
            ))
    return out


def build_draft(
    city: CityConfig,
    current_aqi: int | None,
    forecast_series: list[tuple[int, float]],
    at: datetime,
) -> GrapDraft | None:
    """Draft measures if a stage crossing is forecast within 48h.

    `forecast_series` is [(horizon_h, city_aqi_p50), ...]. Returns None when no
    crossing is coming — the autopilot stays silent rather than manufacturing a
    reason to escalate.
    """
    current_stage = stage_for_aqi(current_aqi)

    within = [(h, a) for h, a in forecast_series if 0 < h <= FORECAST_WINDOW_H]
    if not within:
        return None

    peak_h, peak_aqi = max(within, key=lambda x: x[1])
    forecast_stage = stage_for_aqi(peak_aqi)

    # Only a crossing UP into a higher stage triggers a draft.
    if forecast_stage <= current_stage:
        return None

    # ETA: the first horizon at which the higher stage is reached.
    crossing_eta = next(
        (h for h, a in sorted(within) if stage_for_aqi(a) >= forecast_stage), peak_h
    )

    measures = measures_for_stage(forecast_stage)
    if not measures:
        return None

    sid = hashlib.sha256(
        f"grap{city.id}{forecast_stage}{at.date()}".encode()
    ).hexdigest()[:6].upper()

    return GrapDraft(
        id=f"GRAP-{city.id[:2].upper()}-{sid}",
        city=city.id,
        current_stage=current_stage,
        current_stage_label=stage_label(current_stage),
        forecast_stage=forecast_stage,
        forecast_stage_label=stage_label(forecast_stage),
        forecast_aqi=int(round(peak_aqi)),
        trigger_forecast_ts=at,
        crossing_eta_h=int(crossing_eta),
        status="draft",
        measures=measures,
    )


def serialize_for_db(draft: GrapDraft) -> tuple:
    """Row for the grap_drafts table."""
    return (
        draft.id, draft.city, draft.forecast_stage, draft.trigger_forecast_ts,
        json.dumps([asdict(m) for m in draft.measures]),
        json.dumps([m.citation for m in draft.measures]),
        draft.status,
    )
