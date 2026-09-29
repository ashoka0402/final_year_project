"""ROI ranking tests (TRD §10: "ranking monotonic in averted µg/m³ and
population; ties broken by effort").

The leaderboard's #1 row becomes a dispatched order against a real location.
These tests pin the properties that make that defensible: the ranking follows
the physics, an upwind source is never recommended, and the cited regulation is
actually in force at the observed AQI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vayu_core.attribution.fusion import CategoryAttribution, Evidence, WardAttribution
from vayu_core.config import load_city
from vayu_core.interventions.roi import (
    EFFORT_UNITS,
    Candidate,
    build_candidates,
    cite_regulation,
    load_corpus,
)

CITY = load_city("delhi")
AT = pd.Timestamp("2025-11-03T06:00Z")

# Ward sits SE of the sources; wind from the NW carries smoke onto it.
WARD_LAT, WARD_LON = 28.60, 77.10
# ~30 km NW of the ward: inside PLUME_MAX_RANGE_KM, so the plume may size it.
# (Delhi's real November stubble sits 200-300 km out and is deliberately NOT
# representable here — see test_distant_stubble_is_an_advisory_not_an_order.)
FIRE_LAT, FIRE_LON = 28.80, 76.90


def _wards() -> pd.DataFrame:
    # area_km2 is real ward data and drives the area-receptor averaging; ~12 km²
    # is a typical Delhi ward.
    return pd.DataFrame([
        {"ward_id": "W1", "name": "Test Ward", "centroid_lat": WARD_LAT,
         "centroid_lon": WARD_LON, "population": 50_000, "area_km2": 12.0},
        {"ward_id": "W2", "name": "Neighbour", "centroid_lat": WARD_LAT + 0.05,
         "centroid_lon": WARD_LON + 0.05, "population": 30_000, "area_km2": 12.0},
    ])


def _weather(speed_kmh=14.4, direction=315.0, hours=60) -> pd.DataFrame:
    ts = pd.date_range("2025-11-03T00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "city": "delhi", "grid": "city", "grid_i": 0, "grid_j": 0, "ts": ts,
        "wind_speed_100m": speed_kmh, "wind_dir_100m": direction,
        "wind_speed_10m": speed_kmh, "wind_dir_10m": direction,
        "pblh": 400.0, "kind": "hist",
    })


def _fire_evidence(frp: float, n: int = 3) -> list[Evidence]:
    return [
        Evidence(
            type="fire",
            label=f"VIIRS fire detection · {frp / n:.1f} MW",
            lat=FIRE_LAT + i * 0.005, lon=FIRE_LON + i * 0.005,
            distance_km=40.0, timestamp=(AT - pd.Timedelta(hours=2)).isoformat(),
            detail="2h ago", source="NASA FIRMS VIIRS", weight=frp / n,
            magnitude=frp / n,
        )
        for i in range(n)
    ]


def _attribution(categories: list[CategoryAttribution]) -> WardAttribution:
    return WardAttribution(
        city="delhi", ward_id="W1", computed_ts=AT, window_h=24,
        categories=categories, trajectory_hours=12,
    )


def _burning(share=42.0, conf=0.85, frp=90.0) -> CategoryAttribution:
    return CategoryAttribution(
        category="open_burning", label="Open burning", share_pct=share,
        confidence=conf, raw_score=1.0, evidence=_fire_evidence(frp),
    )


def _board(cats, weather=None, wards=None, city_aqi=380):
    return build_candidates(
        CITY, _attribution(cats), wards if wards is not None else _wards(),
        weather if weather is not None else _weather(), AT, city_aqi=city_aqi,
    )


def _build(cats, **kw):
    """Just the dispatchable candidates."""
    return _board(cats, **kw).candidates


# ---- candidate generation ---------------------------------------------------

def test_a_burning_cluster_becomes_a_dispatchable_candidate():
    cands = _build([_burning()])
    assert cands, "a 42% burning attribution with fires upwind must yield an action"
    c = cands[0]
    assert c.action_type == "halt_burning"
    assert c.predicted_ugm3_averted > 0
    assert c.population_protected > 0
    assert c.effort_units == EFFORT_UNITS["halt_burning"]
    assert c.evidence, "a candidate with no evidence cannot be defended"
    assert c.id.startswith("VAYU-DE-")


def test_fire_pixels_are_clustered_into_one_site_not_one_order_each():
    """Three pixels 500 m apart are one field, and one team's job."""
    cands = _build([_burning(frp=90.0)])
    burning = [c for c in cands if c.action_type == "halt_burning"]
    assert len(burning) == 1, "each pixel became its own order"


def test_upwind_source_is_never_recommended():
    """Wind from the SE: the fire NW of the ward is now downwind of it and
    cannot be responsible — dispatching a team there would be wrong."""
    cands = _build([_burning()], weather=_weather(direction=135.0))
    assert not [c for c in cands if c.action_type == "halt_burning"]


def test_stagnant_attribution_yields_no_candidates():
    stagnant = WardAttribution(
        city="delhi", ward_id="W1", computed_ts=AT, window_h=24,
        categories=[], stagnant=True, note="stagnant", trajectory_hours=12,
    )
    assert build_candidates(CITY, stagnant, _wards(), _weather(), AT).candidates == []


def test_regional_transport_produces_no_local_action():
    """There is no lever a municipal commissioner can pull on Punjab's air as
    'regional transport' — the burning clusters inside it are the lever."""
    cats = [CategoryAttribution("regional_transport", "Regional transport", 80.0, 0.8, 1.0, [])]
    board = _board(cats)
    assert board.candidates == []
    # But it must not stay silent: an empty leaderboard reads as "nothing to do",
    # when the truth is "not yours to fix — escalate".
    assert board.advisories, "80% regional transport must produce an advisory"
    a = board.advisories[0]
    assert a.kind == "no_local_lever"
    assert "CAQM" in (a.escalate_to or "")


# ---- ranking (TRD §10) ------------------------------------------------------

def test_ranking_is_monotonic_in_averted_concentration():
    weak = _build([_burning(frp=10.0)])
    strong = _build([_burning(frp=300.0)])
    assert strong[0].predicted_ugm3_averted > weak[0].predicted_ugm3_averted
    assert strong[0].roi_score > weak[0].roi_score


def test_ranking_is_monotonic_in_population():
    small = _build([_burning()], wards=_wards())
    big_wards = _wards().assign(population=[5_000_000, 30_000])
    big = _build([_burning()], wards=big_wards)
    assert big[0].population_protected > small[0].population_protected
    assert big[0].roi_score > small[0].roi_score


def test_ties_break_toward_lower_effort():
    """Two actions of equal benefit: recommend the one a stretched department
    can actually do today."""
    a = Candidate(
        id="A", city="delhi", ward_id="W1", ward_name="W", action_type="industrial_curb",
        title="", category="industry", source_lat=0, source_lon=0, distance_km=1,
        predicted_ugm3_averted=10, peak_ugm3_averted=10, averted_by_horizon={24: 10},
        ward_averted_ugm3=10, population_protected=1000, wards_protected=1, effort_units=4, confidence=0.5,
        roi_score=5.0, rationale="",
    )
    b = Candidate(**{**a.__dict__, "id": "B", "action_type": "halt_burning", "effort_units": 1})
    ranked = sorted([a, b], key=lambda c: (-c.roi_score, c.effort_units))
    assert ranked[0].id == "B", "equal ROI must prefer the 1-team action"


def test_roi_formula_matches_the_spec():
    """ROI = averted x population / effort (scaled by 1000 for readability)."""
    c = _build([_burning()])[0]
    expected = round(c.predicted_ugm3_averted * c.population_protected / c.effort_units / 1000.0, 1)
    assert c.roi_score == pytest.approx(expected, rel=1e-6)


def test_confidence_combines_attribution_and_plume():
    """TRD 5.5: confidence = attribution confidence x plume confidence (0.8)."""
    high = _build([_burning(conf=0.9)])[0]
    low = _build([_burning(conf=0.3)])[0]
    assert high.confidence > low.confidence
    assert 0 <= high.confidence <= 0.9 * 0.8 + 1e-9


def test_population_counts_every_ward_helped_not_just_the_flagged_one():
    cands = _build([_burning(frp=400.0)])
    # Both test wards are near each other and downwind; a big cluster should
    # measurably help both.
    assert cands[0].population_protected >= 50_000


# ---- regulation citation ----------------------------------------------------

def test_corpus_loads_and_every_clause_has_a_citation():
    clauses = load_corpus()
    assert clauses, "no regulation corpus — dossiers would cite nothing"
    for c in clauses:
        assert c["citation"] and c["title"] and c["text"]
        assert c.get("action_supported") or c.get("stage") is None


def test_citation_matches_the_stage_actually_in_force():
    """Citing a Stage IV measure during Stage II air would be legally wrong."""
    stage2 = cite_regulation("stop_work_construction", city_aqi=350, grap_applicable=True)
    assert stage2 and stage2["stage"] <= 2

    stage3 = cite_regulation("stop_work_construction", city_aqi=430, grap_applicable=True)
    assert stage3 and stage3["stage"] <= 3
    # The severe-air citation should be at least as strong as the very-poor one.
    assert stage3["stage"] >= stage2["stage"]


def test_non_grap_city_falls_back_to_the_air_act():
    """GRAP is Delhi-NCR only. Lucknow still needs a legal basis."""
    c = cite_regulation("industrial_curb", city_aqi=420, grap_applicable=False)
    assert c is not None
    assert c["stage"] is None
    assert "Air" in c["instrument"] or "Environment" in c["instrument"]


def test_every_candidate_carries_a_regulation():
    for c in _build([_burning()]):
        assert c.regulation is not None, "an order with no legal basis is unusable"
        assert c.regulation["citation"]


def test_burning_citation_is_about_burning():
    c = _build([_burning()], city_aqi=280)[0]
    assert c.regulation["action_supported"] == "halt_burning"
    assert "burning" in c.regulation["title"].lower()


# ---- model range & jurisdiction ---------------------------------------------

def test_distant_stubble_is_an_advisory_not_an_order():
    """The real Delhi November case, and the reason PLUME_MAX_RANGE_KM exists.

    Punjab's rice stubble burns 200-300 km upwind. It genuinely drives Delhi's
    smog, but a steady-state plume cannot size it and no Delhi team can reach it.
    Quoting "halt this field, avert 60 µg/m³" would be a fabricated number on a
    legal document. VAYU must instead say: not yours — escalate.
    """
    punjab = CategoryAttribution(
        category="open_burning", label="Open burning", share_pct=44.0, confidence=0.8,
        raw_score=1.0,
        evidence=[
            Evidence(type="fire", label="VIIRS fire detection · 40.0 MW",
                     lat=30.20 + i * 0.01, lon=75.20 + i * 0.01, distance_km=250.0,
                     timestamp=(AT - pd.Timedelta(hours=3)).isoformat(), detail="3h ago",
                     source="NASA FIRMS VIIRS", weight=0.9, magnitude=40.0)
            for i in range(6)
        ],
    )
    board = _board([punjab])
    assert not [c for c in board.candidates if c.action_type == "halt_burning"], (
        "a 250 km source must never become a dispatchable order"
    )
    adv = [a for a in board.advisories if a.kind == "out_of_range"]
    assert adv, "distant stubble must be reported, not silently dropped"
    a = adv[0]
    assert a.nearest_km > 200
    assert a.source_count == 6
    assert a.total_magnitude == pytest.approx(240.0)
    assert "CAQM" in (a.escalate_to or ""), "must escalate to the airshed authority"


def test_plume_declines_beyond_its_range_rather_than_returning_zero():
    """`in_range=False` is a refusal to answer, not an answer of zero — the
    distinction that stops a 250 km source being ranked as 'no benefit'."""
    from vayu_core.dispersion.gaussian_plume import PLUME_MAX_RANGE_KM, counterfactual

    wind = pd.DataFrame({
        "ts": pd.date_range("2025-11-03T00:00Z", periods=60, freq="h", tz="UTC"),
        "wind_speed_ms": 4.0, "wind_dir_deg": 315.0, "pblh": 400.0,
    })
    near = counterfactual(28.80, 76.90, 28.60, 77.10, 50.0, AT, wind, "Asia/Kolkata")
    far = counterfactual(30.20, 75.20, 28.60, 77.10, 50.0, AT, wind, "Asia/Kolkata")

    assert near.in_range is True
    assert near.peak_averted_ugm3 > 0
    assert far.in_range is False
    assert far.distance_km > PLUME_MAX_RANGE_KM
    assert far.confidence == 0.0


def test_emission_rate_comes_from_measured_frp_not_the_display_label():
    """FRP must be read from `magnitude` (the observation). Parsing it out of a
    human-readable label would silently zero every plume the day someone
    reformats the UI string."""
    relabelled = CategoryAttribution(
        category="open_burning", label="Open burning", share_pct=42.0, confidence=0.85,
        raw_score=1.0,
        evidence=[
            Evidence(type="fire", label="Fire (VIIRS, high confidence)",  # no MW in the text
                     lat=FIRE_LAT, lon=FIRE_LON, distance_km=30.0,
                     timestamp=(AT - pd.Timedelta(hours=2)).isoformat(), detail="2h ago",
                     source="NASA FIRMS VIIRS", weight=0.9, magnitude=90.0)
        ],
    )
    cands = _build([relabelled])
    assert cands and cands[0].predicted_ugm3_averted > 0, (
        "FRP was not recovered from magnitude — the plume got Q=0"
    )


# ---- population & exposure --------------------------------------------------

def test_population_protected_follows_the_plume_not_a_distance_ring():
    """The people term must obey the wind.

    It was a stand-in exp(-d/120) decay that ignored direction entirely and
    counted every ward within ~40 km — in Delhi, all 290 of them and all 16.8M
    residents on one candidate. The population term then outvoted the physics.
    A ward directly upwind gains nothing and must not be counted.
    """
    # Wind is FROM 315° (NW), so the air arrives from the north-west: a ward
    # NW of the fire (lat+, lon-) sits upwind of it and can never receive its
    # smoke. Place it well clear of the crosswind axis.
    upwind_ward = {"ward_id": "UP", "name": "Upwind", "centroid_lat": FIRE_LAT + 0.15,
                   "centroid_lon": FIRE_LON - 0.15, "population": 9_000_000, "area_km2": 12.0}
    wards = pd.concat([_wards(), pd.DataFrame([upwind_ward])], ignore_index=True)
    c = _build([_burning(frp=200.0)], wards=wards)[0]
    # The 9M upwind residents must not appear in the count.
    assert c.population_protected < 9_000_000, (
        "an upwind ward was counted as protected — the distance ring is back"
    )


def test_roi_arithmetic_is_verifiable_from_the_displayed_columns():
    """A judge must be able to multiply the leaderboard columns by hand."""
    c = _build([_burning()])[0]
    expected = round(
        c.predicted_ugm3_averted * c.population_protected / c.effort_units / 1000.0, 1
    )
    assert c.roi_score == pytest.approx(expected, rel=1e-6)


def test_ward_and_mean_averted_are_reported_separately():
    """Two different quantities: what this ward gains vs the population-weighted
    mean across everywhere the plume reaches. Conflating them claims every
    resident got the alerting ward's benefit."""
    c = _build([_burning()])[0]
    assert c.ward_averted_ugm3 > 0
    assert c.predicted_ugm3_averted > 0
    assert c.peak_ugm3_averted >= c.ward_averted_ugm3


# ---- clustering & emission scaling ------------------------------------------

def test_industry_emission_scales_with_area_not_site_count():
    """OSM landuse=industrial polygons are estates (0.01-9.6 km²), not factories.
    A flat per-feature rate made 54 polygons a 135 g/s source — bigger than a
    stubble field — and handed one order the whole city."""
    from vayu_core.dispersion.gaussian_plume import INDUSTRY_Q_G_S_PER_KM2
    from vayu_core.interventions.roi import _source_q

    big = Evidence(type="industry", label="Estate", lat=1, lon=1, magnitude=4.0)
    small = Evidence(type="industry", label="Workshop", lat=1, lon=1, magnitude=0.02)
    assert _source_q("industry", big) == pytest.approx(4.0 * INDUSTRY_Q_G_S_PER_KM2)
    assert _source_q("industry", big) > 100 * _source_q("industry", small)


def test_cluster_radius_is_per_category():
    """8 km groups a stubble field; in a city it merges Bawana, Narela and Okhla
    into one 40 km² 'target' no team can be sent to."""
    from vayu_core.interventions.roi import CLUSTER_KM

    assert CLUSTER_KM["open_burning"] > CLUSTER_KM["industry"]
    assert CLUSTER_KM["industry"] <= 3.0


def test_noise_does_not_become_an_advisory():
    """A lone 0 MW detection is not worth routing to CAQM."""
    faint = CategoryAttribution(
        category="open_burning", label="Open burning", share_pct=20.0, confidence=0.8,
        raw_score=1.0,
        evidence=[Evidence(type="fire", label="VIIRS fire detection · 0.1 MW",
                           lat=30.2, lon=75.2, distance_km=250.0,
                           timestamp=AT.isoformat(), detail="3h ago",
                           source="NASA FIRMS VIIRS", weight=0.1, magnitude=0.1)],
    )
    board = _board([faint])
    assert not [a for a in board.advisories if a.kind == "out_of_range"]
