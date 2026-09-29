"""Ward population estimation.

ROI = averted µg/m³ x POPULATION / effort. Population is a direct multiplier on
the ranking that decides where inspection teams get sent, so a systematic error
here silently mis-ranks the entire product while every number still looks
plausible on screen. This file pins the property that broke once already.
"""

from __future__ import annotations

import pandas as pd

from vayu_core.config import load_city
from services.pipeline.wards import _estimate_population


def _wards(areas: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ward_id": [f"W{i}" for i in range(len(areas))],
        "area_km2": areas,
    })


def test_population_does_not_track_polygon_area():
    """The regression that mattered.

    Apportioning by area gave Delhi's biggest ward (Chhawla, 78 km² of farmland)
    879,253 people and its smallest 1,598 — a 550x spread across wards that
    Delhi Municipal Corporation Act 1957 s.5 requires be drawn to equal
    population. It ranked farmland above dense neighbourhoods on a leaderboard
    that dispatches enforcement teams.
    """
    city = load_city("delhi")
    # One huge rural ward, two small dense ones — the real Delhi shape.
    pop = _estimate_population(_wards([78.0, 2.0, 1.5]), city)
    assert pop.nunique() == 1, "population must not vary with polygon area"


def test_population_sums_to_the_census_city_total():
    city = load_city("delhi")
    n = 290
    pop = _estimate_population(_wards([5.0] * n), city)
    # Equal split of an integer total rounds per ward; allow that drift only.
    assert abs(int(pop.sum()) - city.population.total) <= n


def test_per_ward_population_is_plausible_for_delhi():
    city = load_city("delhi")
    pop = _estimate_population(_wards([5.0] * 290), city)
    per = int(pop.iloc[0])
    # Real MCD wards run roughly 40k-90k people. A number outside this band means
    # the split or the city total is wrong.
    assert 30_000 < per < 100_000, f"implausible per-ward population {per:,}"


def test_lucknow_uses_its_own_total():
    delhi = _estimate_population(_wards([5.0] * 290), load_city("delhi"))
    lko = _estimate_population(_wards([5.0] * 112), load_city("lucknow"))
    assert int(lko.iloc[0]) < int(delhi.iloc[0])
    assert int(lko.iloc[0]) > 10_000


def test_provenance_states_the_method_not_just_the_source():
    """PRD F2: every number carries how it was derived. A reader must be able to
    tell this is an estimate from a delimitation principle, not a Census count."""
    for cid in ("delhi", "lucknow"):
        m = load_city(cid).population.method.lower()
        assert "equal" in m
        assert "estimate" in m
        assert "area" not in m, "stale provenance still claims area apportionment"
