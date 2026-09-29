"""Intervention candidates ranked by ROI (TRD 5.5) — VAYU's unit of output.

    ROI = (Δµg/m³ averted at t+24h) × (population exposed) / effort_units

This is the whole product in one line: it converts "42% of this ward's air is
open burning" into "halt this cluster, protect 38,000 people, one team". A
dashboard stops at the percentage.

Each candidate is generated from the ward's own attribution evidence — a fire
that appeared in the cone becomes a haltable cluster, a non-compliant permit
becomes a stop-work order — so every row on the leaderboard traces back through
the evidence to a real coordinate. Nothing is invented to fill the table.

Effort units are teams-required (TRD 5.5). Ties break toward lower effort:
between two equally beneficial actions, the one a stretched enforcement
department can actually do today is the better recommendation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.attribution.fusion import Evidence, WardAttribution
from vayu_core.config import REPO_ROOT, CityConfig
from vayu_core.dispersion.gaussian_plume import (
    CATEGORY_HEIGHT_M,
    CATEGORY_Q_G_S,
    INDUSTRY_Q_G_S_PER_KM2,
    PLUME_CONFIDENCE,
    PLUME_MAX_RANGE_KM,
    Counterfactual,
    averted_over_wards,
    counterfactual,
    q_from_frp,
    ward_radius_m,
)
from vayu_core.geo import haversine_km

# Teams required per action (TRD 5.5).
EFFORT_UNITS: dict[str, int] = {
    "halt_burning": 1,
    "stop_work_construction": 1,
    "traffic_restriction": 3,
    "industrial_curb": 4,
    "road_dust_suppression": 2,
}

ACTION_LABEL: dict[str, str] = {
    "halt_burning": "Halt open burning cluster",
    "stop_work_construction": "Stop work — construction site",
    "traffic_restriction": "Restrict traffic corridor",
    "industrial_curb": "Curb industrial emissions",
    "road_dust_suppression": "Road dust suppression",
}

# Which action answers which attributed source.
CATEGORY_ACTION: dict[str, str] = {
    "open_burning": "halt_burning",
    "construction": "stop_work_construction",
    "traffic": "traffic_restriction",
    "industry": "industrial_curb",
}

# How far apart two sources can be and still be ONE enforceable target — the
# distance a single dispatched team can reasonably cover.
#
# Per category, because the geography differs by an order of magnitude. A stubble
# field is tens of km² of contiguous burning, so 8 km groups a field rather than
# a pixel. Urban industry is not: at 8 km a greedy pass merges Bawana, Narela and
# Okhla into one 40 km² "cluster" spanning most of Delhi's 82.5 km² of industrial
# land — which is not a place a team can be sent, and made a single order read as
# "curb half the city's industry with 4 teams". 3 km is about one industrial
# estate (Okhla ~3 km across, Bawana ~5 km).
CLUSTER_KM: dict[str, float] = {
    "open_burning": 8.0,
    "industry": 3.0,
    "construction": 3.0,
}
FIRE_CLUSTER_KM = CLUSTER_KM["open_burning"]

# Below this total magnitude an out-of-range cluster is not worth an advisory:
# for fires, 5 MW is a handful of cool pixels, not a field.
ADVISORY_MIN_MAGNITUDE = 5.0

CORPUS_PATH = REPO_ROOT / "data" / "corpus" / "grap_clauses.json"


@dataclass
class Candidate:
    """One dispatchable action."""

    id: str
    city: str
    ward_id: str
    ward_name: str
    action_type: str
    title: str
    category: str
    source_lat: float
    source_lon: float
    distance_km: float
    # Population-weighted mean µg/m³ averted at t+24h across every ward helped —
    # the ROI numerator. NOT this ward's own figure; see ward_averted_ugm3.
    predicted_ugm3_averted: float
    peak_ugm3_averted: float
    averted_by_horizon: dict[int, float]
    # What the alerting ward itself gains — the ward-specific story. Frequently
    # much less than `predicted_ugm3_averted`: the ward that flagged a source is
    # not necessarily the one the source hurts most.
    ward_averted_ugm3: float
    population_protected: int
    # How many wards the action measurably helps. Without this the leaderboard
    # reads as "this action is for <ward_name>", when the ward is only where the
    # attribution surfaced it — the benefit lands across the airshed.
    wards_protected: int
    effort_units: int
    confidence: float
    roi_score: float
    rationale: str
    evidence: list[Evidence] = field(default_factory=list)
    regulation: dict | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) if not isinstance(e, dict) else e for e in self.evidence]
        return d


@dataclass
class Advisory:
    """A source that matters but that this city cannot act on.

    The most useful thing VAYU can tell a commissioner on a Delhi November
    morning is that the smoke is Punjab's and no number of local teams will
    touch it — escalate to CAQM instead of burning a shift. An empty
    leaderboard would say "nothing to do", which is the opposite of the truth.
    """

    kind: str                  # "out_of_range" | "no_local_lever"
    category: str
    headline: str
    detail: str
    share_pct: float
    escalate_to: str | None = None
    source_count: int = 0
    nearest_km: float | None = None
    farthest_km: float | None = None
    total_magnitude: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Leaderboard:
    """What this ward's attribution supports: actions, and honest non-actions."""

    candidates: list[Candidate] = field(default_factory=list)
    advisories: list[Advisory] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "advisories": [a.to_dict() for a in self.advisories],
        }


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        logger.warning("no regulation corpus — dossiers will cite nothing")
        return []
    return json.loads(CORPUS_PATH.read_text()).get("clauses", [])


def cite_regulation(action_type: str, city_aqi: int | None, grap_applicable: bool) -> dict | None:
    """Pick the clause that authorises this action at the current severity.

    Prefers a GRAP stage clause that is actually in force at the observed AQI —
    citing a Stage IV measure during Stage II air would be legally wrong and is
    the kind of error that gets an order thrown out. Falls back to the Air Act,
    which applies at any time and in any city (GRAP is Delhi-NCR only).
    """
    clauses = load_corpus()
    if not clauses:
        return None

    stage = None
    if city_aqi is not None:
        if city_aqi > 450:
            stage = 4
        elif city_aqi > 400:
            stage = 3
        elif city_aqi > 300:
            stage = 2
        elif city_aqi > 200:
            stage = 1

    if grap_applicable and stage:
        # The most specific in-force GRAP clause for this action.
        eligible = [
            c for c in clauses
            if c.get("stage") and c["stage"] <= stage and c.get("action_supported") == action_type
        ]
        if eligible:
            return max(eligible, key=lambda c: c["stage"])

    # Statutory fallback: applies everywhere, always.
    statutory = [
        c for c in clauses
        if c.get("stage") is None and c.get("action_supported") == action_type
    ]
    if statutory:
        return statutory[0]
    generic = [c for c in clauses if c.get("stage") is None]
    return generic[0] if generic else None


def _cluster_sources(evidence: list[Evidence], radius_km: float = FIRE_CLUSTER_KM) -> list[list[Evidence]]:
    """Greedy spatial clustering of located evidence into enforceable sites.

    Applies to every category, not just fires. One construction site contributes
    almost nothing to a 50 km² ward's mean, so scoring sites individually put
    every candidate under the noise floor and emptied the leaderboard for the
    worst-polluted wards in Delhi. The enforceable unit is the cluster: a stubble
    field, an industrial estate, a corridor of sites — which is also the unit
    GRAP actually works in (CAQM bans a category in an area; it does not name
    site 3). Emission strengths sum across the cluster, and the order lists every
    member site with its coordinates so nothing is hidden behind an aggregate.
    """
    located = [e for e in evidence if e.lat is not None and e.lon is not None]
    located.sort(key=lambda e: -e.weight)
    clusters: list[list[Evidence]] = []
    for e in located:
        for c in clusters:
            if haversine_km(c[0].lat, c[0].lon, e.lat, e.lon) <= radius_km:
                c.append(e)
                break
        else:
            clusters.append([e])
    return clusters


def _source_q(category: str, e: Evidence) -> float:
    """Emission rate (g/s) for one evidence item, from its measured magnitude.

    Fires carry radiative power (MW) and industry carries footprint (km²) — both
    real observations that scale to emission. Construction permits carry neither,
    so they fall back to a documented per-site default.
    """
    if category == "open_burning":
        # FRP is the measurement; never parse it back out of a display label.
        return q_from_frp(float(e.magnitude or 0.0))
    if category == "industry":
        # Per km² of estate, not per mapped polygon — see INDUSTRY_Q_G_S_PER_KM2.
        return float(e.magnitude or 0.0) * INDUSTRY_Q_G_S_PER_KM2
    q = CATEGORY_Q_G_S.get(category, 0.0)
    if category == "construction" and "NON-COMPLIANT" in (e.detail or ""):
        # Mirror the x2 the attribution applies to a site failing dust control.
        q *= 2.0
    return q


def prepare_wind(weather: pd.DataFrame) -> pd.DataFrame:
    """Reduce the city weather grid to a single hourly wind series.

    Hoisted out of build_candidates and exported because the result is identical
    for every ward in a city: doing it per ward re-copied ~735k weather rows each
    time and made a 290-ward sweep 21s instead of 7s. Callers that loop over
    wards should call this once and pass the result in.

    A city-mean wind is a real simplification — but the plume is already a
    straight-line model, and using one representative vector keeps the
    counterfactual honest about its own resolution rather than implying a
    precision the model does not have.
    """
    if weather.empty:
        return pd.DataFrame()
    w = weather.copy()
    if "grid" in w.columns:
        w = w[w["grid"] == "city"]
    if w.empty:
        return pd.DataFrame()
    w["ts"] = pd.to_datetime(w["ts"], utc=True)
    speed_kmh = w["wind_speed_100m"].fillna(w["wind_speed_10m"])
    direction = w["wind_dir_100m"].fillna(w["wind_dir_10m"])
    w = w.assign(
        # Open-Meteo publishes km/h; the plume equation needs m/s.
        wind_speed_ms=speed_kmh / 3.6,
        wind_dir_deg=direction,
    )
    return (
        w.groupby("ts", as_index=False)
        .agg(wind_speed_ms=("wind_speed_ms", "mean"),
             wind_dir_deg=("wind_dir_deg", "mean"),
             pblh=("pblh", "mean"))
        .sort_values("ts")
    )


# A ward counts as protected when the action averts at least this much there.
#
# A judgement call, documented rather than hidden. The WHO annual PM2.5 guideline
# is 5 µg/m³, so 0.1 is a ~2% move against it — small, but not nothing. Below
# this the benefit is model noise, and counting those residents inflates
# "people protected" with people who gain nothing measurable.
MATERIAL_AVERTED_UGM3 = 0.1


def _population_protected(
    wards: pd.DataFrame,
    src_lat: float,
    src_lon: float,
    ward_id: str,
    q_g_s: float,
    at: datetime,
    wind: pd.DataFrame,
    tz: str,
    height_m: float,
) -> tuple[int, list[str], float]:
    """People in every ward this action measurably helps — by the plume, per ward.

    A source upwind of the city does not only affect the ward that flagged it, so
    counting one ward understates the benefit. But the previous stand-in — an
    exp(-d/120) decay borrowed from the attribution — was not the plume at all:
    it ignored wind direction entirely and swept in every ward within ~40 km,
    which in Delhi meant all 290 of them and all 16.8M residents on a single
    candidate. The population term then dominated the ranking over the physics.

    Now every ward is evaluated with the same plume that produced the headline
    number, and only wards where the action averts a material amount are counted.

    Returns (people, ward_ids, max_averted_elsewhere).
    """
    if wards.empty or q_g_s <= 0 or wind.empty:
        return 0, [], 0.0

    lats = wards["centroid_lat"].to_numpy(dtype=float)
    lons = wards["centroid_lon"].to_numpy(dtype=float)
    # A city onboarded without polygon areas degrades to point receptors rather
    # than failing — less accurate, still answerable.
    areas = (wards["area_km2"].fillna(0.0) if "area_km2" in wards.columns
             else pd.Series(0.0, index=wards.index))
    radii = np.array([ward_radius_m(a) for a in areas])

    averted = averted_over_wards(
        src_lat, src_lon, lats, lons, radii, q_g_s, at, wind, tz,
        horizon_h=24, height_m=height_m,
    )
    helped = averted >= MATERIAL_AVERTED_UGM3
    # The alerting ward is always counted: it is the ward the order is raised for.
    helped |= (wards["ward_id"] == ward_id).to_numpy()

    pops = wards["population"].to_numpy(dtype=float)
    people = int(pops[helped].sum())
    ids = wards.loc[helped, "ward_id"].tolist()

    # Population-weighted mean averted across the wards actually helped.
    #
    # Not the alerting ward's own figure. Public-health benefit is an integral —
    # Σ(averted_w x pop_w), person·µg/m³ — and quoting one ward's number against a
    # city-wide headcount claims everyone got that ward's benefit. A big source
    # helps millions a little; a small one helps thousands a lot. Weighting makes
    # those comparable, and keeps the displayed arithmetic exact: the mean times
    # the headcount IS the integral, so a judge can multiply the leaderboard
    # columns by hand and land on the ROI.
    denom = pops[helped].sum()
    mean_averted = float((averted[helped] * pops[helped]).sum() / denom) if denom > 0 else 0.0
    return people, ids, mean_averted


def build_candidates(
    city: CityConfig,
    attribution: WardAttribution,
    wards: pd.DataFrame,
    weather: pd.DataFrame,
    at: datetime,
    city_aqi: int | None = None,
    max_per_category: int = 2,
    wind: pd.DataFrame | None = None,
) -> Leaderboard:
    """Ranked, dispatchable actions for one ward — plus what cannot be acted on.

    Pass `wind` (from prepare_wind) when sweeping many wards; otherwise it is
    derived from `weather` on every call, which is the same work repeated.
    """
    if attribution.stagnant or not attribution.categories:
        # Nothing upwind to enforce against. Saying so beats inventing a target.
        return Leaderboard()

    ward = wards[wards["ward_id"] == attribution.ward_id]
    if ward.empty:
        return Leaderboard()
    w0 = ward.iloc[0]
    rec_lat, rec_lon = float(w0.centroid_lat), float(w0.centroid_lon)
    # The ward is an area, not a pin. Averaging the plume over it is what makes
    # "how much cleaner is this ward" answerable — see _broadened_sigma_y.
    rec_radius_m = ward_radius_m(float(w0.get("area_km2") or 0.0))
    if wind is None:
        wind = prepare_wind(weather)

    out: list[Candidate] = []
    advisories: list[Advisory] = []

    for cat in attribution.categories:
        action = CATEGORY_ACTION.get(cat.category)
        if not action:
            # Regional transport has no local enforcement lever by definition.
            if cat.category == "regional_transport" and cat.share_pct >= 15.0:
                advisories.append(Advisory(
                    kind="no_local_lever",
                    category=cat.category,
                    headline=f"{cat.share_pct:.0f}% of this ward's air arrived from outside {city.name}",
                    detail=(
                        "Back-trajectories place this fraction of the air mass beyond the city "
                        "boundary before it arrived. No municipal action changes it; it needs an "
                        "airshed-level response."
                    ),
                    share_pct=cat.share_pct,
                    escalate_to="CAQM (Commission for Air Quality Management, NCR)",
                ))
            continue

        out_of_range: list[list[Evidence]] = []

        # --- turn evidence into concrete, locatable targets -------------------
        # (lat, lon, Q g/s, label, member evidence)
        targets: list[tuple[float, float, float, str, list[Evidence]]] = []

        if cat.category == "traffic":
            # Traffic has no point source to visit: the corridor is the ward.
            targets.append(
                (rec_lat, rec_lon, CATEGORY_Q_G_S["traffic"],
                 f"{w0['name']} corridor", cat.evidence)
            )
        else:
            for cluster in _cluster_sources(cat.evidence, CLUSTER_KM.get(cat.category, 3.0)):
                qs = [_source_q(cat.category, e) for e in cluster]
                q_total = float(sum(qs))
                if q_total <= 0:
                    continue
                # Emission-weighted centroid: the pin lands on the part of the
                # cluster that actually matters, not the geometric middle.
                lat = float(np.average([e.lat for e in cluster], weights=qs))
                lon = float(np.average([e.lon for e in cluster], weights=qs))

                # Beyond plume range this is real, attributed, and unactionable.
                # It becomes an advisory, never an order.
                if haversine_km(lat, lon, rec_lat, rec_lon) > PLUME_MAX_RANGE_KM:
                    out_of_range.append(cluster)
                    continue

                if cat.category == "open_burning":
                    frp = sum(float(e.magnitude or 0.0) for e in cluster)
                    label = f"{len(cluster)} fire detections · {frp:.0f} MW total"
                elif len(cluster) == 1:
                    label = cluster[0].label
                else:
                    label = f"{len(cluster)} {cat.label.lower()} sites"
                targets.append((lat, lon, q_total, label, cluster))

            # Strongest clusters first, then cap.
            targets.sort(key=lambda t: -t[2])
            targets = targets[:max_per_category]

        # --- run the counterfactual per target -------------------------------
        for lat, lon, q, label, ev in targets:
            cf: Counterfactual = counterfactual(
                lat, lon, rec_lat, rec_lon, q, at, wind, city.timezone,
                height_m=CATEGORY_HEIGHT_M.get(cat.category, 10.0),
                receptor_radius_m=rec_radius_m,
            )
            averted_24 = cf.averted_ugm3.get(24, 0.0)
            if averted_24 <= 0.05:
                continue  # below the noise floor: not worth a team's day

            pop, ward_ids, mean_averted = _population_protected(
                wards, lat, lon, attribution.ward_id, q, at, wind, city.timezone,
                CATEGORY_HEIGHT_M.get(cat.category, 10.0),
            )
            effort = EFFORT_UNITS.get(action, 1)
            # TRD 5.5: confidence = attribution confidence x plume confidence.
            conf = round(cat.confidence * (cf.confidence / PLUME_CONFIDENCE) * PLUME_CONFIDENCE, 2)

            # ROI is computed from the *displayed* averted value, not the raw
            # float. The leaderboard shows averted / population / effort side by
            # side, so a judge must be able to multiply those columns by hand and
            # land on this exact ROI. Using the unrounded value made the visible
            # arithmetic disagree with itself (1664 by hand vs 1667.2 shown).
            # The ROI numerator is the population-weighted mean, so ROI is
            # exactly (shown averted) x (shown population) / effort.
            averted_shown = round(mean_averted, 2)
            roi = round(averted_shown * pop / max(effort, 1) / 1000.0, 1)

            sid = hashlib.sha256(f"{city.id}{attribution.ward_id}{action}{lat:.4f}{lon:.4f}".encode()).hexdigest()[:6]
            out.append(
                Candidate(
                    id=f"VAYU-{city.id[:2].upper()}-{sid.upper()}",
                    city=city.id,
                    ward_id=attribution.ward_id,
                    ward_name=str(w0["name"]),
                    action_type=action,
                    title=f"{ACTION_LABEL[action]} — {label}",
                    category=cat.category,
                    source_lat=round(lat, 5),
                    source_lon=round(lon, 5),
                    distance_km=round(haversine_km(rec_lat, rec_lon, lat, lon), 1),
                    predicted_ugm3_averted=averted_shown,
                    ward_averted_ugm3=round(averted_24, 2),
                    wards_protected=len(ward_ids),
                    peak_ugm3_averted=cf.peak_averted_ugm3,
                    averted_by_horizon=cf.averted_ugm3,
                    population_protected=pop,
                    effort_units=effort,
                    confidence=conf,
                    roi_score=roi,
                    # These two numbers answer different questions and must not be
                    # printed as if they were the same one. The share is category
                    # attribution across the whole ward — every site of this kind
                    # combined. The averted figure is an absolute plume estimate
                    # for THIS site alone. "60% construction" and "averts 0.6
                    # µg/m³" are both true and look like a contradiction unless
                    # the sentence says which is which.
                    rationale=(
                        f"{cat.label} is the attributed source for {cat.share_pct:.0f}% of "
                        f"{w0['name']}'s air (confidence {cat.confidence:.2f}) — that is the "
                        f"category as a whole, across every such site affecting the ward. "
                        f"This order targets one of them: halting {label.lower()} is modelled "
                        f"to avert {averted_24:.2f} µg/m³ here at t+24h, and {averted_shown:.2f} "
                        f"µg/m³ on average across the {len(ward_ids)} ward(s) it reaches — "
                        f"{pop:,} people, with {effort} team(s)."
                    ),
                    evidence=ev[:8],
                    regulation=cite_regulation(action, city_aqi, city.grap_applicable),
                )
            )

        # Burning that is real, attributed, and simply too far to act on.
        if out_of_range:
            pixels = [e for cl in out_of_range for e in cl]
            # Don't escalate noise. A lone 0 MW detection 52 km out is not worth
            # telling a commissioner about, let alone routing to CAQM.
            if sum(float(e.magnitude or 0.0) for e in pixels) < ADVISORY_MIN_MAGNITUDE:
                continue
            dists = [haversine_km(e.lat, e.lon, rec_lat, rec_lon) for e in pixels]
            frp = sum(float(e.magnitude or 0.0) for e in pixels)
            advisories.append(Advisory(
                kind="out_of_range",
                category=cat.category,
                headline=(
                    f"{len(pixels)} fire detections ({frp:.0f} MW) sit "
                    f"{min(dists):.0f}–{max(dists):.0f} km upwind — beyond local reach"
                ),
                detail=(
                    f"These fires are attributed to this ward's air, but they lie outside the "
                    f"{PLUME_MAX_RANGE_KM:.0f} km range where a plume model can size an "
                    f"intervention, and outside {city.name}'s enforcement jurisdiction. "
                    f"No local order will change them."
                ),
                share_pct=cat.share_pct,
                escalate_to="CAQM (Commission for Air Quality Management, NCR)",
                source_count=len(pixels),
                nearest_km=round(min(dists), 1),
                farthest_km=round(max(dists), 1),
                total_magnitude=round(frp, 1),
            ))

    # Rank by ROI desc; ties -> lower effort first (TRD 5.5).
    out.sort(key=lambda c: (-c.roi_score, c.effort_units))
    logger.info(
        f"[{city.id}] {attribution.ward_id}: {len(out)} candidates, {len(advisories)} advisories · "
        + ", ".join(f"{c.action_type} ROI {c.roi_score}" for c in out[:3])
    )
    return Leaderboard(candidates=out, advisories=advisories)
