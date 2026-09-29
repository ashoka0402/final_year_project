"""Cross-check a citizen report against independent satellite physics.

**The problem with citizen data, stated plainly.** Crowd-sourced reports are
the only way to see pollution where there are no instruments, and they are also
trivially wrong: a foggy winter dawn looks like heavy smog, a dirty lens looks
like haze, and a motivated reporter can simply lie. Systems that solve this with
user reputation end up trusting whoever reports most, which is not the same as
whoever is right.

**What we do instead.** Every report is checked against two measurements that
were made by different instruments, for different reasons, and that no reporter
can influence:

  * the **S5P/TROPOMI HCHO anomaly** for that grid cell and day — how unusual
    formaldehyde was, scored against that cell's own robust baseline;
  * the **VIIRS/FIRMS fire count** in that cell — whether anything was actually
    burning.

A report claiming heavy smoke from crop burning in a cell where the satellite
saw a 4-sigma HCHO anomaly and eleven active fires is corroborated by physics.
The same report in a cell with a flat HCHO baseline and no detections is not
rejected — it is *flagged as standing alone*, which is a different and more
honest statement. Both are shown; only corroborated reports are allowed to
influence hotspot detection.

This is also what makes the citizen layer scientifically defensible rather than
decorative: it inherits the credibility of the satellite record.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# HCHO z-score at or above which the satellite is considered to independently
# agree that something unusual was happening. Matches the hotspot threshold in
# `vayu_core.national.hotspots` so the two layers cannot disagree by definition.
CORROBORATING_Z = 2.5

# Sources whose signature is combustion, and which should therefore leave a
# trace in the fire record and/or the HCHO column. Dust and vehicle exhaust do
# not produce HCHO the way biomass burning does, so a missing HCHO anomaly is
# not evidence against them — the verdict logic accounts for that below.
COMBUSTION_SOURCES = {"crop_burning", "garbage_burning", "brick_kiln", "industrial_plume"}

# Verdicts. Deliberately three-valued: absence of evidence is not evidence of
# absence, and collapsing "unsupported" into "contradicted" would let cloud
# cover or a satellite gap look like a lying citizen.
CORROBORATED = "corroborated"
UNSUPPORTED = "unsupported"
CONTRADICTED = "contradicted"
NO_SATELLITE_DATA = "no_satellite_data"


@dataclass
class Corroboration:
    verdict: str
    hcho_z: float | None
    fire_count: int | None
    detail: str

    @property
    def may_influence_hotspots(self) -> bool:
        """Only physics-backed reports are allowed to move the science."""
        return self.verdict == CORROBORATED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["may_influence_hotspots"] = self.may_influence_hotspots
        return d


def corroborate(
    *,
    haze_rank: int,
    source_type: str,
    visible_smoke: bool,
    hcho_z: float | None,
    fire_count: int | None,
    z_threshold: float = CORROBORATING_Z,
) -> Corroboration:
    """Compare one citizen observation with the satellite record for its cell.

    `hcho_z` is None when the satellite had no valid pixel there that day —
    cloud, swath gap, or a masked retrieval. That is a *data* gap, and it must
    not be read as the citizen being wrong.
    """
    # A satellite gap is a statement about the satellite, not the citizen.
    if hcho_z is None and not fire_count:
        return Corroboration(
            NO_SATELLITE_DATA, hcho_z, fire_count,
            "No valid satellite retrieval for this cell and day — report stands unverified.",
        )

    fires = int(fire_count or 0)
    z = hcho_z
    claims_pollution = haze_rank >= HAZE_MODERATE or visible_smoke
    combustion_claim = source_type in COMBUSTION_SOURCES

    # Strongest case: the citizen reports burning, and both independent
    # instruments agree something was burning and the chemistry followed.
    if combustion_claim and fires > 0 and z is not None and z >= z_threshold:
        return Corroboration(
            CORROBORATED, z, fires,
            f"{fires} active fire detection(s) and an HCHO anomaly of {z:.1f}σ "
            f"in the same cell — independent satellite agreement.",
        )

    # Fire detections alone still corroborate a burning claim: HCHO may be
    # missing to cloud, or the plume may have drifted before the overpass.
    if combustion_claim and fires > 0:
        return Corroboration(
            CORROBORATED, z, fires,
            f"{fires} active fire detection(s) in the same cell corroborate the "
            f"reported burning" + (f"; HCHO anomaly {z:.1f}σ." if z is not None else "."),
        )

    # A strong HCHO anomaly supports a smoke/haze claim regardless of whether
    # the reporter could name a source. Note the fire clause is built from the
    # actual count: writing "no fire pixel" unconditionally here produced
    # explanations that said "no fire pixel" for a cell with 37 detections,
    # which is exactly the kind of confidently-wrong evidence an official must
    # never be handed.
    if claims_pollution and z is not None and z >= z_threshold:
        if fires > 0:
            why = (
                f" — {fires} active fire detection(s) in the same cell, though the "
                f"reporter did not identify a visible source."
            )
        else:
            why = " (no fire pixel — may be a small fire, or a plume that drifted in)."
        return Corroboration(
            CORROBORATED, z, fires,
            f"HCHO anomaly of {z:.1f}σ in the same cell supports the reported conditions{why}",
        )

    # A confident burning claim where the satellite saw a genuinely normal
    # atmosphere AND no fire is the one case worth flagging as contradicted.
    if combustion_claim and fires == 0 and z is not None and z < 0.5:
        return Corroboration(
            CONTRADICTED, z, fires,
            f"No fire detected and HCHO was normal ({z:.1f}σ) — the reported "
            f"burning is not visible to the satellite. Could be very small, "
            f"indoors, after the overpass, or mistaken.",
        )

    return Corroboration(
        UNSUPPORTED, z, fires,
        "Satellite neither confirms nor contradicts this report"
        + (f" (HCHO {z:.1f}σ, {fires} fires)." if z is not None else f" ({fires} fires)."),
    )


# "moderate" is the point at which a photograph is claiming meaningfully
# degraded air rather than ordinary atmospheric depth.
HAZE_MODERATE = 2
