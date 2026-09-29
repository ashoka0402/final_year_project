"""Forecast surfaces: per-ward p10/p50/p90, alerts, SHAP explanation."""

from __future__ import annotations


import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from vayu_core.aqi import category_for
from vayu_core.config import get_settings
from vayu_core.db import read_conn
from vayu_core.forecast.model import HORIZONS

from ..deps import get_city, read_stations, read_wards
from ..scoring import current_run_ts
from ..schemas import ExplainOut, ForecastOut, ForecastWard, HazardAlert

router = APIRouter(prefix="/cities", tags=["forecast"])


def _latest_run(city_id: str) -> pd.Timestamp | None:
    """The run for the app's current hour (clock-aware), scored on demand if it
    doesn't exist yet. Replaces the old `max(run_ts)` so the date picker / live
    clock select the right run instead of always the most recent one written."""
    run_ts = current_run_ts(city_id)
    return None if run_ts is None else pd.Timestamp(run_ts)


@router.get("/{city_id}/forecast", response_model=ForecastOut)
def get_forecast(city_id: str, h: int = Query(48, description="Horizon in hours")) -> ForecastOut:
    city = get_city(city_id)
    if h not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {sorted(HORIZONS)}")

    run_ts = _latest_run(city.id)
    if run_ts is None:
        raise HTTPException(
            404,
            "No forecast has been produced yet — run `make seed` (which trains and scores) "
            "or POST /agents/run-cycle.",
        )

    with read_conn() as con:
        df = con.execute(
            """SELECT f.ward_id, w.name, w.population, f.target_ts, f.p10, f.p50, f.p90,
                      f.aqi_p50, f.model_ver
               FROM forecasts f JOIN wards w ON w.city = f.city AND w.ward_id = f.ward_id
               WHERE f.city = ? AND f.run_ts = ? AND f.horizon_h = ?
               ORDER BY f.aqi_p50 DESC""",
            [city.id, run_ts.to_pydatetime(), h],
        ).df()

    wards = [
        ForecastWard(
            ward_id=r.ward_id,
            name=r.name,
            population=int(r.population),
            p10=float(r.p10),
            p50=float(r.p50),
            p90=float(r.p90),
            aqi_p50=int(r.aqi_p50),
            category=category_for(int(r.aqi_p50))[0],
            color=category_for(int(r.aqi_p50))[1],
        )
        for r in df.itertuples()
    ]

    return ForecastOut(
        city=city.id,
        run_ts=run_ts.to_pydatetime(),
        horizon_h=h,
        model_ver=str(df["model_ver"].iloc[0]) if not df.empty else "n/a",
        target_ts=pd.Timestamp(df["target_ts"].iloc[0]).to_pydatetime() if not df.empty else None,
        wards=wards,
    )


@router.get("/{city_id}/alerts", response_model=list[HazardAlert])
def get_alerts(city_id: str, threshold: int = 300) -> list[HazardAlert]:
    """Wards predicted to cross `threshold` within the forecast window (PRD A3)."""
    city = get_city(city_id)
    run_ts = _latest_run(city.id)
    if run_ts is None:
        return []

    with read_conn() as con:
        df = con.execute(
            """SELECT f.ward_id, w.name, w.population, f.horizon_h, f.target_ts,
                      f.p10, f.p50, f.p90, f.aqi_p50
               FROM forecasts f JOIN wards w ON w.city = f.city AND w.ward_id = f.ward_id
               WHERE f.city = ? AND f.run_ts = ?
               ORDER BY f.ward_id, f.horizon_h""",
            [city.id, run_ts.to_pydatetime()],
        ).df()
    if df.empty:
        return []

    from vayu_core.forecast.run import crossing_alerts

    wards_df = read_wards(city.id)
    alerts = crossing_alerts(df, wards_df, threshold=threshold)
    return [
        HazardAlert(
            ward_id=a["ward_id"],
            name=a["name"],
            population=a["population"],
            eta_h=a["eta_h"],
            target_ts=pd.Timestamp(a["target_ts"]).to_pydatetime(),
            aqi_p50=a["aqi_p50"],
            pm25_p50=round(a["pm25_p50"], 1),
            confidence=a["confidence"],
        )
        for a in alerts
    ]


@router.get("/{city_id}/forecast/explain/{ward_id}", response_model=ExplainOut)
def explain(city_id: str, ward_id: str, h: int = Query(48)) -> ExplainOut:
    """Top-6 SHAP contributions behind a ward's forecast (PRD A4).

    Explains via the station nearest the ward centroid: the ward value is an IDW
    blend, so attributing it to one feature vector would be a fiction. We say
    which station is doing the explaining rather than implying otherwise.
    """
    city = get_city(city_id)
    settings = get_settings()

    wards = read_wards(city.id)
    ward = wards[wards["ward_id"] == ward_id]
    if ward.empty:
        raise HTTPException(404, f"unknown ward '{ward_id}' in {city.id}")

    from vayu_core.forecast.features import build_features, model_frame
    from vayu_core.forecast.model import Forecaster, add_city_code
    from vayu_core.geo import haversine_km

    fc = Forecaster()
    if not fc.available:
        raise HTTPException(404, "No trained model yet — run `make seed`.")

    stations = read_stations(city.id)
    if stations.empty:
        raise HTTPException(404, "no stations")

    wlat, wlon = float(ward.iloc[0]["centroid_lat"]), float(ward.iloc[0]["centroid_lon"])
    stations = stations.assign(_d=[haversine_km(wlat, wlon, s.lat, s.lon) for s in stations.itertuples()])
    nearest = stations.sort_values("_d").iloc[0]

    at = settings.now()
    with read_conn() as con:
        meas = con.execute(
            """SELECT city, station_id, param, ts, value FROM measurements
               WHERE city = ? AND station_id = ? AND ts <= ? AND ts >= ?""",
            [city.id, nearest.station_id, at, at - pd.Timedelta(days=5)],
        ).df()
        wx = con.execute(
            "SELECT * FROM weather_hourly WHERE city = ? AND ts <= ? AND ts >= ?",
            [city.id, at, at - pd.Timedelta(days=5)],
        ).df()
    if meas.empty:
        raise HTTPException(404, "no recent measurements to explain from")
    meas["ts"] = pd.to_datetime(meas["ts"], utc=True)
    if not wx.empty:
        wx["ts"] = pd.to_datetime(wx["ts"], utc=True)

    feats = build_features(city, meas, wx, stations[stations.station_id == nearest.station_id])
    feats = feats[feats["ts"] <= at].dropna(subset=["pm25_lag1"])
    if feats.empty:
        raise HTTPException(404, "not enough history at the nearest station to explain")
    # The model's inputs include the horizon's fx_* weather; explaining off the
    # base frame alone raises a KeyError for every fx_ column.
    row = model_frame(feats.sort_values("ts").tail(1), wx, h, with_target=False)
    row = add_city_code(row, city.id)

    return ExplainOut(
        city=city.id,
        ward_id=ward_id,
        horizon_h=h,
        explained_via_station=str(nearest["name"]),
        station_distance_km=round(float(nearest["_d"]), 1),
        features=fc.explain(row, h),
    )
