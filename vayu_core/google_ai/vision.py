"""Citizen photograph -> structured pollution observation, via Gemini vision.

**The gap this fills.** India runs ~900 CAAQMS stations for 1.4 billion people;
most of the population lives further from a monitor than the monitor's readings
can meaningfully represent. Satellites cover everywhere but see a *column*
through the whole atmosphere once a day, through cloud, at 5-10 km. Neither sees
the burning field at the edge of a village at 7 a.m. A citizen with a phone does.

**What we ask the model, and what we refuse to ask it.**
A vision model cannot read micrograms per cubic metre off a photograph, and any
system claiming otherwise is fitting to a demo rather than to physics. So the
schema captures only what is defensible from an image:

  * `haze_severity`  — an ordinal visibility class, which is genuinely what
                       atmospheric extinction does to a photograph;
  * `source_type`    — the visible combustion/dust source, if any;
  * `confidence`     — the model's own, used to weight rather than to gate.

The numeric estimate stays with the instruments. What the photo contributes is
*location, timing and source attribution* — exactly what satellites and sparse
ground stations are worst at.

**Why this is not just a classifier.** The observation is only half the value.
Every report is cross-checked against the satellite HCHO anomaly and FIRMS fire
count for the same grid cell and day (see `vayu_core.citizen.crosscheck`), so a
report is not "trusted" or "untrusted" by reputation — it is *corroborated by
independent physics*, or it is flagged as standing alone.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, asdict

from loguru import logger

from .client import GeminiUnavailable, generate_json

# Ordinal, because visibility degradation is ordinal. The numbers are used for
# ranking and for agreement-checking against satellite AOD/HCHO — never
# converted into a concentration.
HAZE_CLASSES: tuple[str, ...] = ("clear", "slight", "moderate", "heavy", "severe")
HAZE_RANK = {c: i for i, c in enumerate(HAZE_CLASSES)}

# The sources a photograph can actually distinguish, mapped onto the same
# attribution vocabulary the rest of VAYU uses so a citizen report can be
# compared with a satellite/fire-derived attribution directly.
SOURCE_TYPES: tuple[str, ...] = (
    "crop_burning",
    "garbage_burning",
    "industrial_plume",
    "construction_dust",
    "vehicle_exhaust",
    "brick_kiln",
    "dust_storm",
    "none_visible",
)

_SYSTEM = (
    "You are an air-quality analyst examining a photograph submitted by a member "
    "of the public in India. Report only what is visually evident. You are NOT "
    "able to determine pollutant concentrations from an image and must never "
    "estimate one. If the image does not show outdoor air, say so via "
    "is_outdoor=false. Prefer 'none_visible' over speculation. Reply with JSON only."
)

_SCHEMA_PROMPT = f"""Return exactly this JSON object and nothing else:

{{
  "is_outdoor": boolean,           // false for indoor/selfie/screenshot/unrelated
  "haze_severity": one of {list(HAZE_CLASSES)},
  "source_type": one of {list(SOURCE_TYPES)},
  "visible_smoke": boolean,
  "confidence": number between 0 and 1,
  "reasoning": "one short sentence citing what in the image supports this"
}}

Guidance:
- haze_severity describes how much distant detail is lost to atmospheric haze:
  "clear" = horizon sharp, "severe" = objects tens of metres away are obscured.
- Judge haze from DISTANT objects, not from camera blur, fog at dawn, or a dirty lens.
- source_type is what is visibly emitting, not what you infer might be nearby.
- confidence should be low when the image is dark, blurred, or ambiguous."""


@dataclass
class PhotoObservation:
    """A structured, machine-comparable reading of a citizen photograph."""

    is_outdoor: bool
    haze_severity: str
    source_type: str
    visible_smoke: bool
    confidence: float
    reasoning: str = ""
    model: str = ""
    # Populated later by the corroboration step, not by the model.
    corroboration: dict = field(default_factory=dict)

    @property
    def haze_rank(self) -> int:
        return HAZE_RANK.get(self.haze_severity, 0)

    @property
    def usable(self) -> bool:
        """Whether this observation should be allowed to influence anything.

        An indoor photo carries no air-quality information, and a very
        low-confidence reading is noise. Both are kept for the audit trail but
        excluded from aggregation.
        """
        return self.is_outdoor and self.confidence >= 0.35

    def to_dict(self) -> dict:
        d = asdict(self)
        d["haze_rank"] = self.haze_rank
        d["usable"] = self.usable
        return d


def _coerce(raw: dict, model: str) -> PhotoObservation:
    """Clamp the model's reply onto the schema.

    An LLM will occasionally return a class outside the enum or a confidence of
    1.5. Coercing here means every downstream consumer can rely on the contract
    without re-validating.
    """
    haze = str(raw.get("haze_severity", "")).lower().strip()
    if haze not in HAZE_RANK:
        haze = "clear"

    src = str(raw.get("source_type", "")).lower().strip()
    if src not in SOURCE_TYPES:
        src = "none_visible"

    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0

    return PhotoObservation(
        is_outdoor=bool(raw.get("is_outdoor", True)),
        haze_severity=haze,
        source_type=src,
        visible_smoke=bool(raw.get("visible_smoke", False)),
        confidence=min(max(conf, 0.0), 1.0),
        reasoning=str(raw.get("reasoning", ""))[:280],
        model=model,
    )


def analyse_photo(image_bytes: bytes, mime_type: str = "image/jpeg") -> PhotoObservation:
    """Run Gemini vision over one photograph.

    Raises `GeminiUnavailable` when no key is configured or the call fails —
    callers must surface that honestly rather than substituting a default
    reading. A guessed pollution observation would propagate into hotspot
    detection, which is precisely the failure this project refuses.
    """
    from vayu_core.config import get_settings

    model = get_settings().gemini_model
    parts = [
        {"text": _SCHEMA_PROMPT},
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(image_bytes).decode(),
            }
        },
    ]
    raw = generate_json(parts, system=_SYSTEM, temperature=0.0, max_tokens=2048)
    if not isinstance(raw, dict):
        raise GeminiUnavailable("vision reply was not a JSON object")

    obs = _coerce(raw, model)
    logger.info(
        f"citizen photo -> haze={obs.haze_severity} source={obs.source_type} "
        f"conf={obs.confidence:.2f} usable={obs.usable}"
    )
    return obs
