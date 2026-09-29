"""Interventions API — leaderboard, dispatch, inspector (PRD C1/C2/C3/C5).

These endpoints move a recommendation into the real world: dispatch renders a
legal-looking document and creates a record an inspector will act on. The tests
concentrate on the ways that can go wrong quietly — a double-click producing two
teams at one gate, an order advancing from a state it was never in, an empty
leaderboard that means "not your problem" being read as "nothing to do".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from vayu_core.db import write_conn

CITY = "delhi"
WARD = "W196"


@pytest.fixture(autouse=True)
def clean_orders():
    """Orders are real rows; never let one test's dispatch leak into another."""
    with write_conn() as con:
        con.execute("DELETE FROM interventions WHERE seeded = FALSE")
        con.execute("DELETE FROM audit_log WHERE agent IN ('enforcer', 'inspector')")
    yield
    with write_conn() as con:
        con.execute("DELETE FROM interventions WHERE seeded = FALSE")
        con.execute("DELETE FROM audit_log WHERE agent IN ('enforcer', 'inspector')")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _first_candidate(client) -> dict:
    r = client.get(f"/api/v1/cities/{CITY}/interventions?ward_id={WARD}")
    assert r.status_code == 200
    cands = r.json()["candidates"]
    if not cands:
        pytest.skip("no candidates on the seeded demo data for this ward")
    return cands[0]


# ---- leaderboard (C1) -------------------------------------------------------

def test_ward_leaderboard_returns_ranked_candidates(client):
    r = client.get(f"/api/v1/cities/{CITY}/interventions?ward_id={WARD}")
    assert r.status_code == 200
    d = r.json()
    rois = [c["roi_score"] for c in d["candidates"]]
    assert rois == sorted(rois, reverse=True), "leaderboard is not ranked"


def test_every_candidate_carries_what_an_order_needs(client):
    c = _first_candidate(client)
    for k in ("id", "ward_id", "action_type", "source_lat", "source_lon",
              "predicted_ugm3_averted", "ward_averted_ugm3", "population_protected",
              "effort_units", "confidence", "roi_score", "rationale", "regulation"):
        assert k in c, f"candidate is missing {k}"
    assert c["regulation"] is not None, "an order with no legal basis is unusable"


def test_city_leaderboard_says_how_much_of_the_city_it_looked_at(client):
    """A truncated sweep that doesn't say so reads as "these are all the options
    in the city" — which would be a lie."""
    r = client.get(f"/api/v1/cities/{CITY}/interventions")
    assert r.status_code == 200
    meta = r.json()["meta"]
    assert meta["wards_evaluated"] >= 1
    assert meta["wards_total"] > meta["wards_evaluated"]
    assert meta["selection"]


def test_leaderboard_carries_advisories_alongside_candidates(client):
    """Delhi in November: the burning is 200-300 km away in Punjab. The API must
    say so rather than return a bare empty list."""
    r = client.get(f"/api/v1/cities/{CITY}/interventions")
    d = r.json()
    assert "advisories" in d
    if not d["candidates"]:
        assert d["advisories"], "empty leaderboard with no explanation"


def test_action_type_filter(client):
    r = client.get(f"/api/v1/cities/{CITY}/interventions?action_type=halt_burning")
    assert r.status_code == 200
    assert all(c["action_type"] == "halt_burning" for c in r.json()["candidates"])


def test_unknown_city_is_404(client):
    assert client.get("/api/v1/cities/atlantis/interventions").status_code == 404


def test_unknown_ward_is_404(client):
    assert client.get(f"/api/v1/cities/{CITY}/interventions?ward_id=W99999").status_code == 404


# ---- dispatch (C2/C3) -------------------------------------------------------

def test_dispatch_creates_an_order_with_a_dossier(client):
    cand = _first_candidate(client)
    r = client.post("/api/v1/interventions/dispatch",
                    json={"candidate": cand, "signal_ts": "2025-11-03T05:56:18Z"})
    assert r.status_code == 201
    o = r.json()
    assert o["status"] == "dispatched"
    assert o["has_dossier"] is True
    assert o["id"] == cand["id"]


def test_dispatch_reports_the_stopwatch_under_the_prd_target(client):
    """PRD E2: signal -> dossier in under 5 minutes, shown as a live stopwatch."""
    cand = _first_candidate(client)
    r = client.post("/api/v1/interventions/dispatch",
                    json={"candidate": cand, "signal_ts": "2025-11-03T05:56:18Z"})
    elapsed = r.json()["signal_to_dossier_s"]
    assert 0 <= elapsed < 300, f"signal->dossier {elapsed}s misses the <5min target"


def test_dispatch_is_idempotent(client):
    """A double-click must not send two teams to one gate."""
    cand = _first_candidate(client)
    a = client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    b = client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    assert a.json()["id"] == b.json()["id"]
    # Exactly one order for this candidate — a seeded demo record may also exist,
    # so count this id specifically rather than the whole table.
    orders = client.get("/api/v1/interventions").json()["orders"]
    assert sum(1 for o in orders if o["id"] == cand["id"]) == 1


def test_dispatch_writes_an_audit_entry(client):
    """PRD F1: every automated step is auditable with its reasoning."""
    from vayu_core.db import read_conn

    cand = _first_candidate(client)
    client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    with read_conn() as con:
        rows = con.execute(
            "SELECT agent, decision, reasoning, confidence FROM audit_log WHERE inputs_hash = ?",
            [cand["id"]],
        ).df()
    assert len(rows) == 1
    assert rows.iloc[0]["agent"] == "enforcer"
    assert rows.iloc[0]["reasoning"], "an audit entry without reasoning proves nothing"


def test_malformed_candidate_is_rejected(client):
    r = client.post("/api/v1/interventions/dispatch", json={"candidate": {"id": "X"}})
    assert r.status_code == 422


def test_dossier_downloads_as_a_pdf(client):
    cand = _first_candidate(client)
    client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    r = client.get(f"/api/v1/interventions/{cand['id']}/dossier")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


# ---- inspector (C3/C5) ------------------------------------------------------

def test_order_appears_in_the_inspector_list(client):
    cand = _first_candidate(client)
    client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    d = client.get("/api/v1/interventions?status=dispatched").json()
    assert cand["id"] in [o["id"] for o in d["orders"]]


def test_execute_moves_the_order_and_records_the_note(client):
    from vayu_core.db import read_conn

    cand = _first_candidate(client)
    client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    note = "burning extinguished 14:20, 3 violations issued"
    r = client.post(f"/api/v1/interventions/{cand['id']}/execute", json={"note": note})
    assert r.status_code == 200
    assert r.json()["status"] == "executed"
    assert r.json()["executed_ts"] is not None
    with read_conn() as con:
        rows = con.execute(
            "SELECT reasoning FROM audit_log WHERE agent = 'inspector'"
        ).df()
    assert note in rows.iloc[0]["reasoning"]


def test_cannot_execute_an_order_that_was_never_dispatched(client):
    """The state machine is the contract: candidate -> dispatched -> executed."""
    r = client.post("/api/v1/interventions/GHOST-1/execute", json={"note": "done"})
    assert r.status_code == 404


def test_execute_requires_a_note(client):
    cand = _first_candidate(client)
    client.post("/api/v1/interventions/dispatch", json={"candidate": cand})
    r = client.post(f"/api/v1/interventions/{cand['id']}/execute", json={"note": ""})
    assert r.status_code == 422, "an executed order with no account of what was done"


def test_bad_status_filter_is_rejected(client):
    assert client.get("/api/v1/interventions?status=bogus").status_code == 400


def test_dossier_path_is_not_leaked_to_the_client(client):
    cand = _first_candidate(client)
    o = client.post("/api/v1/interventions/dispatch", json={"candidate": cand}).json()
    assert "dossier_path" not in o, "server filesystem path exposed to the client"
    assert o["has_dossier"] is True
