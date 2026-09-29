"""Citizen report vs satellite corroboration.

The claim under test is the one the whole citizen layer rests on: a report is
believed because independent physics agrees with it, not because of who filed
it. The most important assertions here are the *negative* ones — that a
satellite gap never reads as a lying citizen, and that only corroborated
reports may move the science.
"""

from __future__ import annotations

from vayu_core.citizen import crosscheck as C


def test_burning_with_fires_and_hcho_anomaly_is_corroborated():
    r = C.corroborate(
        haze_rank=4, source_type="crop_burning", visible_smoke=True,
        hcho_z=4.2, fire_count=11,
    )
    assert r.verdict == C.CORROBORATED
    assert r.may_influence_hotspots
    assert "11 active fire" in r.detail and "4.2" in r.detail


def test_satellite_gap_is_not_treated_as_a_false_report():
    """The single most important guard. Cloud cover, a swath gap or a masked
    retrieval says nothing about the citizen — collapsing that into
    'contradicted' would punish people for the satellite's blind spots."""
    r = C.corroborate(
        haze_rank=4, source_type="crop_burning", visible_smoke=True,
        hcho_z=None, fire_count=0,
    )
    assert r.verdict == C.NO_SATELLITE_DATA
    assert not r.may_influence_hotspots, "unverified reports must not move the science"
    assert "unverified" in r.detail.lower()


def test_fire_detection_alone_corroborates_a_burning_claim():
    """HCHO can be missing to cloud, or the plume can drift before the 13:30
    overpass. A fire pixel in the same cell is already independent agreement."""
    r = C.corroborate(
        haze_rank=3, source_type="crop_burning", visible_smoke=True,
        hcho_z=None, fire_count=6,
    )
    assert r.verdict == C.CORROBORATED and r.may_influence_hotspots


def test_hcho_anomaly_alone_corroborates_a_smoke_claim():
    """VIIRS misses small or short-lived fires; a strong chemical anomaly is
    still evidence."""
    r = C.corroborate(
        haze_rank=4, source_type="none_visible", visible_smoke=True,
        hcho_z=3.1, fire_count=0,
    )
    assert r.verdict == C.CORROBORATED
    assert "drifted" in r.detail or "small fire" in r.detail


def test_explanation_never_claims_no_fire_when_fires_exist():
    """Regression for a real bug. A haze report with no named source fell into
    the HCHO-only branch, whose text hardcoded 'no fire pixel' — so a cell with
    37 detections was explained to an official as having none. Evidence that is
    confidently wrong is worse than evidence that is absent."""
    r = C.corroborate(
        haze_rank=2, source_type="none_visible", visible_smoke=False,
        hcho_z=3.2, fire_count=37,
    )
    assert r.verdict == C.CORROBORATED
    assert "no fire pixel" not in r.detail, f"contradicts fire_count=37: {r.detail}"
    assert "37 active fire" in r.detail


def test_confident_burning_claim_with_clean_satellite_is_contradicted():
    r = C.corroborate(
        haze_rank=4, source_type="crop_burning", visible_smoke=True,
        hcho_z=0.1, fire_count=0,
    )
    assert r.verdict == C.CONTRADICTED
    assert not r.may_influence_hotspots


def test_dust_without_hcho_is_unsupported_not_contradicted():
    """Dust and vehicle exhaust do not produce HCHO the way biomass burning
    does, so a flat HCHO column is NOT evidence against a dust report. Marking
    it 'contradicted' would be a chemistry error."""
    r = C.corroborate(
        haze_rank=3, source_type="dust_storm", visible_smoke=False,
        hcho_z=0.2, fire_count=0,
    )
    assert r.verdict == C.UNSUPPORTED, "non-combustion source must not be contradicted by flat HCHO"
    assert not r.may_influence_hotspots


def test_only_corroborated_reports_may_influence_hotspots():
    for verdict_kwargs in (
        dict(haze_rank=1, source_type="none_visible", visible_smoke=False, hcho_z=0.0, fire_count=0),
        dict(haze_rank=4, source_type="crop_burning", visible_smoke=True, hcho_z=0.1, fire_count=0),
        dict(haze_rank=4, source_type="crop_burning", visible_smoke=True, hcho_z=None, fire_count=0),
    ):
        r = C.corroborate(**verdict_kwargs)
        assert not r.may_influence_hotspots, f"{r.verdict} must not influence hotspots"


def test_threshold_matches_the_hotspot_module():
    """If these two drifted apart, a report could be 'corroborated' by an
    anomaly the hotspot layer does not consider a hotspot."""
    from vayu_core.national import hotspots as H

    assert C.CORROBORATING_Z == H.DEFAULT_Z


def test_to_dict_is_serialisable_and_carries_the_gate():
    d = C.corroborate(
        haze_rank=4, source_type="crop_burning", visible_smoke=True,
        hcho_z=4.0, fire_count=3,
    ).to_dict()
    assert d["verdict"] == C.CORROBORATED
    assert d["may_influence_hotspots"] is True
    assert set(d) >= {"verdict", "hcho_z", "fire_count", "detail"}
