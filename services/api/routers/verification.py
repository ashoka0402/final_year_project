"""Verification: did the order actually work? (PRD E1)

The last arrow of the loop and the only place VAYU marks its own homework. See
vayu_core/verification/did.py for why difference-in-differences rather than a
before/after comparison.
"""

from __future__ import annotations

from datetime import timedelta
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from vayu_core.config import get_settings
from vayu_core.db import read_conn, write_conn
from vayu_core.verification.did import PRE_DAYS, Pending, pick_controls, verify
from vayu_core.verification.series import ward_hourly_pm25

from ..deps import read_wards

router = APIRouter(tags=["verification"])


@lru_cache(maxsize=8)
def _ward_hourly(city_id: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """Cached: an IDW pass over ~200 hours is not free, and the window is fixed
    for a given intervention."""
    start, end = pd.Timestamp(start_iso), pd.Timestamp(end_iso)
    with read_conn() as con:
        meas = con.execute(
            """SELECT station_id, param, ts, value FROM measurements
               WHERE city = ? AND param = 'pm25' AND ts >= ? AND ts < ?""",
            [city_id, start, end],
        ).df()
        stations = con.execute(
            "SELECT station_id, lat, lon FROM stations WHERE city = ?", [city_id]
        ).df()
    return ward_hourly_pm25(meas, stations, read_wards(city_id), start, end)


def _verify_one(order: dict) -> dict:
    """Run (or re-run) the verdict for one executed order."""
    now = get_settings().now()
    executed = pd.Timestamp(order["executed_ts"])
    start = executed - timedelta(days=PRE_DAYS)
    end = min(executed + timedelta(hours=48), now)

    hourly = _ward_hourly(order["city"], start.isoformat(), end.isoformat())
    wards = read_wards(order["city"])

    controls = pick_controls(
        order["ward_id"], wards, hourly, executed,
        float(order["source_lat"]), float(order["source_lon"]),
    )
    result = verify(
        order["id"], order["ward_id"], controls, hourly, executed,
        float(order["predicted_ugm3_averted"]), now,
    )

    if isinstance(result, Pending):
        out = result.to_dict()
        out["order"] = order
        return out

    # Persist so a verdict is a record, not a recomputation that could drift.
    with write_conn() as con:
        con.execute("DELETE FROM verifications WHERE intervention_id = ?", [order["id"]])
        con.execute(
            """INSERT INTO verifications
               (intervention_id, method, control_wards, predicted_reduction,
                observed_reduction, ci_low, ci_high, pct_realized, computed_ts)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [result.intervention_id, result.method, ",".join(result.control_wards),
             result.predicted_reduction, _nan_to_none(result.observed_reduction),
             _nan_to_none(result.ci_low), _nan_to_none(result.ci_high),
             _nan_to_none(result.pct_realized), result.computed_ts],
        )
        if order["status"] != "verified" and not pd.isna(result.observed_reduction):
            con.execute("UPDATE interventions SET status = 'verified' WHERE id = ?", [order["id"]])

    out = result.to_dict()
    out["status"] = "verified"
    out["order"] = order
    # Names, not ids: "W094" means nothing to a reader checking our controls.
    name_of = dict(zip(wards["ward_id"], wards["name"]))
    out["control_ward_names"] = [name_of.get(w, w) for w in result.control_wards]
    out["ward_name"] = name_of.get(order["ward_id"], order["ward_id"])
    return out


def _nan_to_none(x: float) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _orders(city_id: str | None) -> list[dict]:
    from .interventions import _row_to_order

    sql = "SELECT * FROM interventions WHERE status IN ('executed', 'verified')"
    params: list = []
    if city_id:
        sql += " AND city = ?"
        params.append(city_id)
    sql += " ORDER BY executed_ts DESC"
    with read_conn() as con:
        df = con.execute(sql, params).df()
    return [_row_to_order(r) for _, r in df.iterrows()]


@router.get("/verifications")
def list_verifications(city_id: str | None = Query(None)) -> dict:
    """Every executed order with its verdict, or its countdown."""
    results = []
    for o in _orders(city_id):
        try:
            results.append(_verify_one(o))
        except Exception as e:  # noqa: BLE001
            # One bad order must not blank the whole page.
            logger.warning(f"verification failed for {o['id']}: {e}")
            results.append({"intervention_id": o["id"], "status": "error",
                            "detail": str(e), "order": o})
    return {"verifications": results, "count": len(results)}


@router.get("/verifications/{order_id}")
def get_verification(order_id: str) -> dict:
    from .interventions import get_order

    order = get_order(order_id)
    if order["status"] not in ("executed", "verified"):
        raise HTTPException(
            409,
            f"order is '{order['status']}' — only an executed order can be verified",
        )
    return _verify_one(order)
