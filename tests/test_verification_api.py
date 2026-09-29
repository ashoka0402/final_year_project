"""Verification API + ward-hourly series (PRD E1).

The endpoint marks VAYU's predictions against real CPCB readings. The tests
guard the two ways it could mislead: reporting a verdict before enough data
exists, and letting the seeded demo record pretend to be a real dispatched order.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from vayu_core.verification.series import ward_hourly_pm25

CITY = "delhi"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_verifications_endpoint_returns_the_seeded_record(client):
    r = client.get(f"/api/v1/verifications?city_id={CITY}")
    assert r.status_code == 200
    vs = r.json()["verifications"]
    assert vs, "the seeded demo record should always be verifiable on demo data"
    seeded = [v for v in vs if v["order"].get("seeded")]
    assert seeded, "no seeded record found — /verify would be empty on a clean clone"


def test_seeded_record_is_flagged_not_disguised(client):
    """PRD E1: the demo record must be badged. Presenting a fabricated dispatch
    as a real one would undermine the whole honesty pitch."""
    vs = client.get(f"/api/v1/verifications?city_id={CITY}").json()["verifications"]
    v = next(v for v in vs if v["order"].get("seeded"))
    assert v["order"]["seeded"] is True


def test_a_verified_record_carries_its_controls_and_ci(client):
    vs = client.get(f"/api/v1/verifications?city_id={CITY}").json()["verifications"]
    verified = [v for v in vs if v.get("status") == "verified"]
    if not verified:
        pytest.skip("seeded record still pending on this data window")
    v = verified[0]
    assert v["control_wards"], "a verdict with no controls cannot be defended"
    assert "ci_low" in v and "ci_high" in v
    assert "significant" in v
    # The honest field: a verdict that spans zero must not be called significant.
    if v["ci_low"] <= 0 <= v["ci_high"]:
        assert v["significant"] is False


def test_verifying_a_non_executed_order_is_rejected(client):
    r = client.get("/api/v1/verifications/NOPE-1")
    assert r.status_code == 404


# ---- ward-hourly series -----------------------------------------------------

def test_series_interpolates_stations_onto_wards():
    ts = pd.date_range("2025-10-30T00:00Z", periods=6, freq="h", tz="UTC")
    stations = pd.DataFrame([
        {"station_id": "S1", "lat": 28.60, "lon": 77.10},
        {"station_id": "S2", "lat": 28.70, "lon": 77.20},
    ])
    meas = pd.DataFrame([
        {"station_id": s, "param": "pm25", "ts": t, "value": 150.0 + i}
        for i, t in enumerate(ts) for s in ("S1", "S2")
    ])
    wards = pd.DataFrame([
        {"ward_id": "W1", "name": "W1", "centroid_lat": 28.62, "centroid_lon": 77.12},
        {"ward_id": "W2", "name": "W2", "centroid_lat": 28.68, "centroid_lon": 77.18},
    ])
    out = ward_hourly_pm25(meas, stations, wards,
                           pd.Timestamp("2025-10-30T00:00Z"), pd.Timestamp("2025-10-30T06:00Z"))
    assert not out.empty
    assert set(out["ward_id"]) == {"W1", "W2"}
    assert out["ts"].nunique() == 6
    # IDW of values near 150 must land near 150, never wildly outside.
    assert out["pm25"].between(140, 165).all()


def test_series_is_empty_when_no_pm25_present():
    out = ward_hourly_pm25(
        pd.DataFrame(columns=["station_id", "param", "ts", "value"]),
        pd.DataFrame(columns=["station_id", "lat", "lon"]),
        pd.DataFrame(columns=["ward_id", "name", "centroid_lat", "centroid_lon"]),
        pd.Timestamp("2025-10-30T00:00Z"), pd.Timestamp("2025-10-31T00:00Z"),
    )
    assert out.empty
