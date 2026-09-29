"""Agent Activity audit trail (PRD F1)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from vayu_core import audit


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def cascade():
    """Record one entry per cascade agent so the test does not depend on ambient
    seed state that other test files' cleanup may have wiped."""
    for agent in ("forecaster", "attributor", "enforcer", "herald"):
        audit.record(agent, f"{agent} step", reasoning="cascade", confidence=0.8)


def test_audit_list_returns_the_cascade(client, cascade):
    r = client.get("/api/v1/audit")
    assert r.status_code == 200
    d = r.json()
    assert d["count"] > 0, "no audit history — the Agent drawer would be empty"
    agents = {e["agent"] for e in d["entries"]}
    # The automated cascade must be representable (TRD 7).
    assert {"forecaster", "attributor", "enforcer"} <= agents


def test_entries_carry_reasoning_and_confidence(client):
    e = client.get("/api/v1/audit").json()["entries"][0]
    for k in ("id", "ts", "agent", "decision", "reasoning", "confidence"):
        assert k in e
    assert e["decision"], "an audit entry with no decision explains nothing"


def test_entries_are_newest_first(client):
    ids = [e["id"] for e in client.get("/api/v1/audit").json()["entries"]]
    assert ids == sorted(ids, reverse=True)


def test_record_is_written_and_readable(client):
    before = client.get("/api/v1/audit").json()["count"]
    audit.record("enforcer", "unit-test decision", reasoning="testing", confidence=0.5)
    after = client.get("/api/v1/audit").json()["count"]
    assert after == before + 1
    top = client.get("/api/v1/audit").json()["entries"][0]
    assert top["decision"] == "unit-test decision"
    assert top["confidence"] == 0.5


def test_a_failed_audit_write_does_not_raise(monkeypatch):
    """A broken audit write must never sink the pipeline step it records."""
    import vayu_core.audit as a

    def boom(*_, **__):
        raise RuntimeError("db down")

    monkeypatch.setattr(a, "write_conn", boom)
    a.record("enforcer", "should not raise")  # must swallow
