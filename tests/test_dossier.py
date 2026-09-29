"""Dossier PDF (PRD C2).

AC: "contains map snapshot, evidence table, regulation citation with source,
predicted impact, order ID, sign-off block; saved and downloadable."

This is the artifact a human carries into the field and signs. Every check here
is about a person being misled by a plausible-looking page: a missing watermark
makes a prototype look official, a wrong clock makes a stale order look fresh,
and a citation without its source cannot be checked by the person held to it.
"""

from __future__ import annotations

import fitz  # pymupdf
import pandas as pd
import pytest

from vayu_core.attribution.fusion import Evidence
from vayu_core.config import load_city
from vayu_core.interventions.dossier import generate, render_map
from vayu_core.interventions.roi import Candidate

CITY = load_city("delhi")
AT = pd.Timestamp("2025-11-03T06:00Z")

WARD_GEOM = (
    '{"type":"Polygon","coordinates":[[[77.05,28.55],[77.15,28.55],'
    '[77.15,28.65],[77.05,28.65],[77.05,28.55]]]}'
)


def _candidate(**over) -> Candidate:
    base = dict(
        id="VAYU-DE-TEST01", city="delhi", ward_id="W1", ward_name="Kalkaji",
        action_type="halt_burning", title="Halt open burning cluster — 3 fire detections · 90 MW total",
        category="open_burning", source_lat=28.80, source_lon=76.90, distance_km=29.4,
        predicted_ugm3_averted=12.4, peak_ugm3_averted=31.2,
        averted_by_horizon={12: 9.1, 24: 12.4, 48: 6.0},
        ward_averted_ugm3=18.7,
        population_protected=57_889, wards_protected=1, effort_units=1, confidence=0.61, roi_score=717.8,
        rationale="Open burning is the attributed source for 42% of Kalkaji's air.",
        evidence=[
            Evidence(type="fire", label="VIIRS fire detection · 30.0 MW", lat=28.80 + i * 0.01,
                     lon=76.90 + i * 0.01, distance_km=29.0,
                     timestamp=(AT - pd.Timedelta(hours=2)).isoformat(), detail="2h ago",
                     source="NASA FIRMS VIIRS", weight=0.9, magnitude=30.0)
            for i in range(3)
        ],
        regulation={
            "id": "GRAP-I-OPEN-BURNING",
            "instrument": "CAQM Graded Response Action Plan (Delhi-NCR)",
            "stage": 1, "clause": "Stage I — Poor (AQI 201-300)",
            "title": "Prohibition on open burning of solid waste and biomass",
            "text": "Ensure a strict ban on open burning of solid waste, garbage and biomass.",
            "citation": "CAQM GRAP Schedule, Stage I — measure on open burning",
            "penalty_reference": "Environmental compensation under CAQM directions",
        },
    )
    base.update(over)
    return Candidate(**base)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("dossiers")
    d = generate(
        _candidate(), CITY, WARD_GEOM, 28.60, 77.10,
        signal_ts=AT - pd.Timedelta(minutes=3, seconds=42),
        wind_dir_from_deg=290.0, out_dir=out,
    )
    doc = fitz.open(d.path)
    text = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    return d, doc, text


# ---- PRD C2 acceptance criteria ---------------------------------------------

def test_the_pdf_is_saved_and_readable(built):
    d, doc, _ = built
    assert d.path.exists() and d.path.suffix == ".pdf"
    assert d.path.stat().st_size > 5_000
    assert doc.page_count >= 1


def test_contains_the_order_id(built):
    _, _, text = built
    assert "VAYU-DE-TEST01" in text


def test_contains_a_map_snapshot(built):
    _, doc, _ = built
    images = [img for i in range(doc.page_count) for img in doc[i].get_images()]
    assert images, "no map image embedded — an inspector cannot find the site"


def test_contains_the_evidence_table_with_sources(built):
    _, _, text = built
    assert "Evidence" in text
    assert "NASA FIRMS VIIRS" in text, "evidence must name its source to be checkable"
    assert "28.80" in text or "28.8" in text, "evidence coordinates missing"


def test_contains_the_regulation_citation_and_its_instrument(built):
    _, _, text = built
    assert "CAQM GRAP Schedule" in text
    assert "Prohibition on open burning" in text
    assert "Graded Response Action Plan" in text


def test_contains_predicted_impact(built):
    _, _, text = built
    assert "12.4" in text          # averted at t+24h
    assert "57,889" in text        # population protected
    assert "µg/m³" in text or "g/m" in text


def test_contains_a_sign_off_block(built):
    _, _, text = built
    low = text.lower()
    assert "authorised by" in low
    assert "executed by" in low
    assert "signature" in low


# ---- honesty guarantees -----------------------------------------------------

def test_every_page_is_watermarked_as_a_prototype(built):
    """A page that looks official but is not is the worst failure this file can
    catch. The watermark once rendered *underneath* the content and survived
    only in the whitespace of the final page."""
    _, doc, _ = built
    for i in range(doc.page_count):
        page_text = doc[i].get_text()
        assert "PROTOTYPE" in page_text, f"page {i + 1} is missing the watermark"


def test_carries_the_not_an_official_document_disclaimer(built):
    _, _, text = built
    assert "not an official document" in text
    assert "not an official order" in text
    assert "caqm.nic.in" in text, "must tell the reader where to verify the clause"


def test_states_that_impact_is_modelled_not_measured(built):
    _, _, text = built
    low = text.lower()
    assert "gaussian plume" in low
    assert "not a measurement" in low


def test_names_its_scientific_provenance(built):
    """The emission factors are the load-bearing assumption behind every averted
    µg/m³. They must be attributable on the page, not buried in source."""
    _, _, text = built
    assert "Wooster" in text
    assert "Andreae" in text
    assert "Briggs" in text


# ---- the clock --------------------------------------------------------------

def test_the_stopwatch_runs_on_the_application_clock(built):
    """PRD E2: signal -> dossier under 5 minutes.

    This read "368840m 7s" (256 days) because the signal came from the demo
    timeline while the generation stamp came from the real wall clock. Any
    demo-mode dossier would have shown a nonsense elapsed time.
    """
    _, _, text = built
    assert "3m 42s" in text
    assert "368840" not in text


def test_generated_timestamp_is_the_demo_instant_not_today(built):
    _, _, text = built
    assert "03 Nov 2025" in text


# ---- map rendering ----------------------------------------------------------

def test_map_renders_without_ward_geometry():
    """A city onboarded without polygons must still produce a usable order."""
    png = render_map(_candidate(), None, 28.60, 77.10, 290.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 3_000


def test_map_renders_with_no_located_evidence():
    png = render_map(_candidate(evidence=[]), WARD_GEOM, 28.60, 77.10, None)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_title_is_not_duplicated(built):
    """`title` already carries the action label; prefixing it again produced
    "Stop work - construction site - Stop work - construction site - ..."."""
    _, _, text = built
    assert text.count("Halt open burning cluster") == 1
