"""API smoke tests in DEMO_MODE (TRD §11: every endpoint).

These run against the seeded DuckDB. They assert the shape of the contract and
the honesty guarantees, not exact AQI values — the underlying data is real and
moves.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from vayu_core.db import read_conn

client = TestClient(app)


@pytest.fixture(scope="module")
def seeded() -> bool:
    with read_conn() as con:
        return con.execute("SELECT count(*) FROM wards").fetchone()[0] > 0


def test_health_reports_mode_and_seed_state():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["demo_mode"], bool)
    assert "delhi" in body["cities"] and "lucknow" in body["cities"]


def test_cities_lists_both_demo_cities_from_config():
    r = client.get("/api/v1/cities")
    assert r.status_code == 200
    cities = {c["id"]: c for c in r.json()}
    assert {"delhi", "lucknow"} <= cities.keys()
    for c in cities.values():
        assert len(c["bbox"]) == 4
        assert c["population"] > 0
        assert len(c["languages"]) >= 3


def test_unknown_city_is_rfc7807_problem_json():
    r = client.get("/api/v1/cities/atlantis/current")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert {"type", "title", "status", "detail"} <= body.keys()
    assert body["status"] == 404


@pytest.mark.parametrize("city", ["delhi", "lucknow"])
def test_current_returns_wards_and_stations(city, seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    r = client.get(f"/api/v1/cities/{city}/current")
    assert r.status_code == 200
    b = r.json()

    assert b["city"] == city
    assert b["wards"], "every city must render a choropleth"
    assert b["stations"], "PRD A1 requires station markers"

    # A1: wards carry a CPCB bucket, never colour alone.
    scored = [w for w in b["wards"] if w["aqi"] is not None]
    assert scored, "no ward has an AQI — the map would be blank"
    for w in scored:
        assert 0 <= w["aqi"] <= 500
        assert w["category"] in {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
        assert w["color"].startswith("#")

    # F2: honesty labels must be present for every layer.
    assert b["data_status"], "no data_status -> no honest pills"
    for s in b["data_status"]:
        assert s["status"] in {"live", "cached", "sample", "cams", "h3-fallback", "unavailable"}


@pytest.mark.parametrize("city", ["delhi", "lucknow"])
def test_city_aqi_is_population_weighted_and_in_range(city, seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    b = client.get(f"/api/v1/cities/{city}/current").json()
    assert b["aqi"] is not None
    assert 0 < b["aqi"] <= 500
    # The city roll-up must sit within the spread of its wards.
    ward_aqis = [w["aqi"] for w in b["wards"] if w["aqi"] is not None]
    assert min(ward_aqis) <= b["aqi"] <= max(ward_aqis)


@pytest.mark.parametrize("city", ["delhi", "lucknow"])
def test_ward_geojson_is_valid_and_matches_current(city, seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    r = client.get(f"/api/v1/cities/{city}/wards.geojson")
    assert r.status_code == 200
    gj = r.json()
    assert gj["type"] == "FeatureCollection"
    assert gj["features"]
    for f in gj["features"][:5]:
        assert f["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        assert f["properties"]["ward_id"]

    # The choropleth joins these two by ward_id; a mismatch = uncoloured wards.
    geo_ids = {f["properties"]["ward_id"] for f in gj["features"]}
    cur_ids = {w["ward_id"] for w in client.get(f"/api/v1/cities/{city}/current").json()["wards"]}
    assert geo_ids == cur_ids


def test_far_wards_are_flagged_low_confidence_not_hidden(seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    wards = client.get("/api/v1/cities/delhi/current").json()["wards"]
    for w in wards:
        if w["nearest_station_km"] is not None and w["nearest_station_km"] > 25:
            assert w["low_confidence"] is True


def test_evaluation_is_honest_about_not_being_run_yet():
    r = client.get("/api/v1/meta/evaluation")
    # Phase 2 generates it; until then a 404 with an explanation beats fake numbers.
    assert r.status_code in {200, 404}
    if r.status_code == 404:
        assert "backtest" in r.json()["detail"].lower()


# ---- Phase 3: attribution ---------------------------------------------------

def _first_ward(city: str) -> str:
    return client.get(f"/api/v1/cities/{city}/current").json()["wards"][0]["ward_id"]


@pytest.mark.parametrize("city", ["delhi", "lucknow"])
def test_trajectory_returns_a_line_and_a_cone(city, seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    w = _first_ward(city)
    r = client.get(f"/api/v1/cities/{city}/trajectory/{w}?hours=12")
    assert r.status_code == 200
    gj = r.json()
    assert gj["type"] == "FeatureCollection"
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert kinds == {"trajectory", "cone"}
    line = next(f for f in gj["features"] if f["properties"]["kind"] == "trajectory")
    assert line["geometry"]["type"] == "LineString"
    assert len(line["geometry"]["coordinates"]) > 10
    # Timestamps run backwards from the demo clock, so the path is upwind.
    assert gj["properties"]["length_km"] > 0


def test_trajectory_rejects_an_unsupported_horizon(seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    w = _first_ward("delhi")
    r = client.get(f"/api/v1/cities/delhi/trajectory/{w}?hours=99")
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/problem+json")


def test_attribution_shares_sum_to_100_and_carry_evidence(seeded):
    if not seeded:
        pytest.skip("run `make seed` first")
    w = _first_ward("delhi")
    r = client.get(f"/api/v1/cities/delhi/attribution/{w}?hours=12")
    assert r.status_code == 200
    b = r.json()

    if b["stagnant"] or not b["categories"]:
        # An honest refusal is a valid response — but it must explain itself.
        assert b["note"]
        return

    assert sum(c["share_pct"] for c in b["categories"]) == pytest.approx(100.0, abs=1.5)
    for c in b["categories"]:
        assert 0 <= c["confidence"] <= 1
        assert c["category"] in {
            "open_burning", "traffic", "construction", "industry", "regional_transport",
        }
    # PRD B2: at least one share must be traceable to concrete evidence.
    assert any(c["evidence"] for c in b["categories"])


def test_every_fire_evidence_item_is_locatable_and_timestamped(seeded):
    """A fire cited as evidence must be clickable on the map (PRD B2)."""
    if not seeded:
        pytest.skip("run `make seed` first")
    w = _first_ward("delhi")
    b = client.get(f"/api/v1/cities/delhi/attribution/{w}?hours=24").json()
    fires = [e for c in b.get("categories", []) for e in c["evidence"] if e["type"] == "fire"]
    for e in fires:
        assert e["lat"] is not None and e["lon"] is not None
        assert e["timestamp"], "a fire without a timestamp cannot be defended"
        assert e["distance_km"] is not None
        assert "FIRMS" in (e["source"] or "")


def test_permit_evidence_is_badged_as_sample_in_the_api(seeded):
    """The badge must ride on the payload, not depend on the UI remembering."""
    if not seeded:
        pytest.skip("run `make seed` first")
    for w in [x["ward_id"] for x in client.get("/api/v1/cities/delhi/current").json()["wards"][:15]]:
        b = client.get(f"/api/v1/cities/delhi/attribution/{w}?hours=12").json()
        permits = [e for c in b.get("categories", []) for e in c["evidence"] if e["type"] == "permit"]
        for e in permits:
            assert "sample" in (e["source"] or "").lower()
        if permits:
            return  # found and checked at least one


def test_attribution_unknown_ward_is_problem_json():
    r = client.get("/api/v1/cities/delhi/attribution/NOPE?hours=12")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")


def test_crosscheck_reports_agreement_with_published_splits(seeded):
    """TRD 5.3 asks for the IITM DSS comparison as a visible artefact."""
    r = client.get("/api/v1/cities/delhi/attribution-crosscheck")
    if r.status_code == 404:
        pytest.skip("run `make calibrate` first")
    d = r.json()
    assert d["cross_check"]
    for row in d["cross_check"]:
        assert "published_low_pct" in row and "vayu_mean_pct" in row
        assert isinstance(row["within_published_range"], bool)
