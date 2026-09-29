"""Herald advisories + clean-hours (PRD Epic D).

These messages reach a parent deciding whether their child walks to school, so
the tests guard the ways an advisory could mislead: telling someone air is
"clean" when it is merely the day's least-bad, promising a good window that does
not exist, or silently dropping a language.
"""

from __future__ import annotations

import pandas as pd

from vayu_core.herald import (
    AUDIENCES,
    LANGUAGES,
    advisory,
    brief,
    clean_hours,
)

AT = pd.Timestamp("2025-11-03T06:00Z")
TZ = "Asia/Kolkata"


def test_every_language_and_audience_has_a_template():
    """A missing (language, bucket, audience) cell would show an English string
    to a Hindi reader, or crash. Exercise the whole grid."""
    for lang in LANGUAGES:
        for aqi in (60, 150, 260, 360, 460):
            for aud in AUDIENCES:
                a = advisory(aqi, aud, lang, "06:00–09:00", "06:00, 3 Nov")
                assert a.text and len(a.text) > 10
                assert "{best}" not in a.text, "placeholder left unfilled"


def test_advisories_stay_within_the_word_budget():
    """PRD: advisory content <=80 words."""
    for lang in LANGUAGES:
        for aqi in (60, 260, 460):
            for aud in AUDIENCES:
                a = advisory(aqi, aud, lang, "06:00–09:00", "t")
                assert len(a.text.split()) <= 80, f"{lang}/{aqi}/{aud} too long"


def test_severe_air_never_says_go_outside():
    """The one message that must never be wrong."""
    for lang in LANGUAGES:
        for aud in AUDIENCES:
            a = advisory(460, aud, lang, None, "t").text.lower()
            # No cheerful "normal outdoor activity" phrasing in a health emergency.
            assert "normal outdoor" not in a
            assert "ਆਮ ਵਾਂਗ ਬਾਹਰ" not in a  # pa: "outdoors as usual"


def test_clean_window_is_not_invented_when_air_is_uniformly_severe():
    """If the next 48h are all severe, there is no clean window and we must say
    so rather than green-light the least-bad hour."""
    anchors = {24: 430.0, 48: 445.0, 72: 440.0}
    ch = clean_hours(430.0, anchors, AT, TZ)
    assert ch.best_window is None, "offered a 'best window' during uniform severe air"
    assert not any(b.clean for b in ch.blocks), "marked a severe hour as clean"


def test_a_real_dip_is_found_and_offered():
    """A genuine clean window must be surfaced with a local time."""
    anchors = {24: 90.0, 48: 250.0, 72: 250.0}  # cleaner near t+24
    ch = clean_hours(180.0, anchors, AT, TZ)
    assert ch.best_window is not None
    assert ch.best_aqi is not None and ch.best_aqi < 180


def test_clean_blocks_require_breathable_air_not_just_the_daily_low():
    """Even the cleanest hour of a very-poor day must not be painted green if it
    is still above the breathable threshold."""
    anchors = {24: 320.0, 48: 340.0, 72: 330.0}
    ch = clean_hours(310.0, anchors, AT, TZ)
    assert not any(b.clean for b in ch.blocks)


def test_brief_covers_at_least_three_languages():
    """PRD success metric: >=3 languages on citizen advisories."""
    assert len(LANGUAGES) >= 3
    langs = {brief("W1", "Test", 260, {24: 280.0, 48: 300.0}, AT, TZ, lang).language
             for lang in LANGUAGES}
    assert langs == set(LANGUAGES)


def test_brief_handles_a_ward_with_no_reading():
    b = brief("W1", "Test", None, {}, AT, TZ, "en")
    assert b.now_aqi is None
    assert b.clean_hours.best_window is None
    # Still returns advisories rather than a blank page.
    assert len(b.advisories) == len(AUDIENCES)


def test_unknown_language_falls_back_to_english_not_a_crash():
    a = advisory(260, "general", "zz", "06:00–09:00", "t")
    assert a.text
    b = brief("W1", "Test", 260, {24: 280.0}, AT, TZ, "zz")
    assert b.language == "en"


def test_blocks_carry_local_time_and_colour():
    ch = clean_hours(180.0, {24: 90.0, 48: 250.0}, AT, TZ)
    assert ch.blocks
    b0 = ch.blocks[0]
    assert b0.color.startswith("#")
    assert "+05:30" in b0.ts, "blocks must be in the city's local time, not UTC"
