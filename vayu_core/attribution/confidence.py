"""Attribution confidence (TRD 5.3).

    confidence = σ( w0 + w1·evidence + w2·wind_stability + w3·station_agreement )

mapped to [0, 1] through a logistic. The inputs are the three things that
actually determine whether an attribution should be believed:

  * **evidence count** — one fire pixel in the cone is a hint; twelve is a case.
    Saturating (log-scaled), because the 30th pixel adds little.
  * **wind stability** — the whole attribution rests on the back-trajectory. If
    the wind was fast and steady the cone means something; if it wandered or
    barely moved, the cone is a guess and confidence must fall accordingly.
  * **station agreement** — do nearby monitors corroborate the ward's level? If
    stations disagree wildly, the ward's own value is uncertain and so is any
    attribution of it.

Weights are chosen, not fitted — there is no labelled ground truth for "what
share of this ward's PM2.5 came from burning", so claiming a fitted model would
be dishonest. They are documented here and reproduced on the Methodology page,
and the resulting number is presented as a confidence *ring*, not a probability.

Per-category priors exist because the categories are not equally observable: a
VIIRS fire pixel is a direct satellite observation of a specific event, whereas
"traffic" is inferred from road length and time of day, which is a much weaker
claim about a specific ward-hour.
"""

from __future__ import annotations

import math

from .trajectory import Trajectory

# Logistic weights (documented judgement, not fitted).
W_INTERCEPT = -1.2
W_EVIDENCE = 1.1
W_WIND = 1.6
W_AGREEMENT = 0.9

# How directly observable each category is. A fire pixel is measured; a traffic
# share is inferred from a proxy.
CATEGORY_PRIOR = {
    "open_burning": 0.35,       # direct satellite observation of the event
    "regional_transport": 0.10,  # trajectory geometry is solid, the PM split is not
    "industry": -0.15,           # OSM landuse says a site exists, not that it emitted
    "construction": -0.30,       # sample permit layer; compliance flag is curated
    "traffic": -0.35,            # proxy only — no vehicle counts exist to check against
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def wind_stability(traj: Trajectory) -> float:
    """0-1: how much the trajectory geometry can be trusted.

    Two failure modes, both real:
      * too slow — the parcel barely moved, so "upwind" is meaningless;
      * too fast/short-lived — not penalised here, but a very long path is
        implicitly less certain because the cone widens with distance, which the
        cone geometry already encodes.
    """
    if not traj.polyline or traj.stagnant:
        return 0.0

    # Speed: below ~3 km/h attribution is meaningless; by ~12 km/h transport is
    # coherent and further speed adds little confidence.
    speed = min(max((traj.mean_speed_kmh - 3.0) / 9.0, 0.0), 1.0)

    # Directional coherence: compare the straight-line distance from the ward to
    # the path's end against the distance actually travelled. A parcel that
    # looped around scores low; a clean run scores near 1.
    if traj.length_km <= 0.1:
        return 0.0
    from vayu_core.geo import haversine_km

    lon0, lat0 = float(traj.polyline[0][0]), float(traj.polyline[0][1])
    lon1, lat1 = float(traj.polyline[-1][0]), float(traj.polyline[-1][1])
    straight = haversine_km(lat0, lon0, lat1, lon1)
    directness = min(straight / traj.length_km, 1.0)

    return round(0.5 * speed + 0.5 * directness, 3)


def category_confidence(
    category: str,
    evidence_count: int,
    traj: Trajectory,
    station_agreement: float = 0.5,
) -> float:
    """Confidence in [0, 1] for one category's share."""
    # Saturating evidence term: 0 pixels -> 0, ~10 -> ~1.
    ev = min(math.log1p(max(evidence_count, 0)) / math.log(11.0), 1.0)
    wind = wind_stability(traj)
    agree = min(max(station_agreement, 0.0), 1.0)

    z = (
        W_INTERCEPT
        + W_EVIDENCE * ev
        + W_WIND * wind
        + W_AGREEMENT * agree
        + CATEGORY_PRIOR.get(category, 0.0)
    )
    return round(_sigmoid(z), 2)


def station_agreement_score(values: list[float]) -> float:
    """0-1 agreement among nearby station readings.

    Coefficient of variation, inverted: identical readings -> 1, wildly
    scattered -> 0. A single station cannot corroborate itself, so it returns a
    neutral 0.5 rather than a falsely confident 1.
    """
    vals = [v for v in values if v is not None and v > 0]
    if len(vals) < 2:
        return 0.5
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return 0.5
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    cv = math.sqrt(var) / mean
    # CV of 0 -> 1.0; CV of 0.6+ -> ~0.
    return round(max(0.0, 1.0 - cv / 0.6), 3)
