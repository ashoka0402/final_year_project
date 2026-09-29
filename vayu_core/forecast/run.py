"""Produce ward-level forecasts and persist them (TRD §10).

The API must not run LightGBM per request — the forecast endpoint has a 300ms
warm budget and reads from the `forecasts` table. This module is what fills it:

    station features at `at`  ->  p10/p50/p90 per station per horizon
                              ->  IDW to ward centroids (p=2, k=5)
                              ->  AQI + crossing ETA per ward
                              ->  forecasts table

IDW is applied to each quantile independently. That is deliberate: interpolating
the band edges preserves "uncertainty is wider where stations disagree", which is
exactly what a ward sitting between a clean and a filthy station should show.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.aqi import aqi_from_pm25
from vayu_core.config import CityConfig
from vayu_core.geo import idw

from .features import SCORING_WINDOW_DAYS, build_features, model_frame
from .model import HORIZONS, Forecaster, add_city_code

CROSSING_AQI = 300


def _latest_station_rows(feats: pd.DataFrame, at: datetime) -> pd.DataFrame:
    """The most recent feature row per station at or before `at`.

    Only stations that reported within SCORING_WINDOW_DAYS of `at` count. Without
    this, a station whose last reading is years old (e.g. the historical-winter
    training stations, last seen 2016/2018) was treated as a *current* source and
    fed into the IDW nowcast — a decade-old value driving today's forecast. The
    cutoff fixes that and, not by coincidence, makes the windowed feature build
    (which can't see beyond the window) produce identical forecasts to the full
    build: both use exactly the stations with recent data.
    """
    cutoff = at - timedelta(days=SCORING_WINDOW_DAYS)
    d = feats[(feats["ts"] <= at) & (feats["ts"] >= cutoff)]
    if d.empty:
        return d
    d = d.dropna(subset=["pm25_lag1"])
    return d.sort_values("ts").groupby("station_id", as_index=False).last()


def run_forecast(
    city: CityConfig,
    measurements: pd.DataFrame,
    weather: pd.DataFrame,
    stations: pd.DataFrame,
    wards: pd.DataFrame,
    fires: pd.DataFrame | None,
    at: datetime,
    forecaster: Forecaster | None = None,
    windowed: bool = True,
) -> pd.DataFrame:
    """Rows ready for the `forecasts` table, one per (ward, horizon).

    `windowed=True` (the default) builds features only over the ~10 days before
    `at` — all the lag/fire windows need — instead of all history. This is what
    makes scoring fast enough for live/on-demand use; the output is identical
    because scoring only reads the latest row per station. Pass windowed=False
    to force a full-history build (e.g. when debugging feature parity).
    """
    fc = forecaster or Forecaster()
    if not fc.available:
        logger.warning(f"[{city.id}] no trained models — skipping forecast")
        return pd.DataFrame()

    since = at - timedelta(days=SCORING_WINDOW_DAYS) if windowed else None
    feats = build_features(city, measurements, weather, stations, fires, since=since)
    if feats.empty:
        return pd.DataFrame()

    latest = _latest_station_rows(feats, at)
    if latest.empty:
        logger.warning(f"[{city.id}] no station has recent enough data at {at:%Y-%m-%d %H:%M}")
        return pd.DataFrame()
    latest = add_city_code(latest, city.id)

    coords = {s.station_id: (s.lat, s.lon) for s in stations.itertuples()}
    src_pts = [coords[s] for s in latest["station_id"] if s in coords]
    keep = [s in coords for s in latest["station_id"]]
    latest = latest[keep]

    targets = [(float(w.centroid_lat), float(w.centroid_lon)) for w in wards.itertuples()]
    rows: list[dict] = []

    # Ward AQI trajectory across horizons, so we can report a crossing ETA.
    ward_p50: dict[str, dict[int, float]] = {w.ward_id: {} for w in wards.itertuples()}

    for horizon in HORIZONS:
        # Attach the weather forecast valid at t+horizon before predicting —
        # without it the model is asked about tomorrow while shown only today.
        rows_h = model_frame(latest, weather, horizon, with_target=False)
        preds = fc.predict(rows_h, horizon)
        target_ts = at + timedelta(hours=horizon)

        band: dict[str, np.ndarray] = {}
        for q in ("p10", "p50", "p90"):
            vals, nearest = idw(targets, src_pts, preds[q].to_numpy(), power=2.0, k=5)
            band[q] = vals
            band["_near"] = nearest

        for i, w in enumerate(wards.itertuples()):
            p50 = band["p50"][i]
            if np.isnan(p50):
                continue
            # Enforce the band ordering after interpolation too — IDW on three
            # independent surfaces can reorder them at a given point.
            lo, mid, hi = sorted([band["p10"][i], p50, band["p90"][i]])
            aqi = aqi_from_pm25(mid)
            ward_p50[w.ward_id][horizon] = float(mid)
            rows.append(
                {
                    "city": city.id,
                    "ward_id": w.ward_id,
                    "run_ts": at,
                    "target_ts": target_ts,
                    "horizon_h": horizon,
                    "p10": round(float(lo), 1),
                    "p50": round(float(mid), 1),
                    "p90": round(float(hi), 1),
                    "aqi_p50": aqi,
                    "model_ver": fc.version,
                }
            )

    df = pd.DataFrame(rows)
    logger.info(f"[{city.id}] forecast: {len(df):,} rows ({len(wards)} wards x {len(HORIZONS)} horizons) at {at:%Y-%m-%d %H:%M}")
    return df


def crossing_alerts(forecasts: pd.DataFrame, wards: pd.DataFrame, threshold: int = CROSSING_AQI) -> list[dict]:
    """Wards predicted to cross `threshold` within the forecast window (PRD A3).

    ETA is the earliest horizon whose p50 crosses. Confidence is the share of
    the band above the line — a ward whose p90 alone crosses is a *maybe*, and
    the alert card should say so rather than shout.
    """
    if forecasts.empty:
        return []

    names = {w.ward_id: w.name for w in wards.itertuples()}
    pops = {w.ward_id: int(w.population) for w in wards.itertuples()}
    out: list[dict] = []

    for ward_id, grp in forecasts.groupby("ward_id"):
        g = grp.sort_values("horizon_h")
        crossing = g[g["aqi_p50"] >= threshold]
        if crossing.empty:
            continue
        first = crossing.iloc[0]

        # How much of the p10–p90 band sits above the threshold?
        lo_aqi = aqi_from_pm25(first["p10"]) or 0
        hi_aqi = aqi_from_pm25(first["p90"]) or 0
        if hi_aqi <= lo_aqi:
            conf = 1.0 if lo_aqi >= threshold else 0.0
        else:
            conf = float(np.clip((hi_aqi - threshold) / (hi_aqi - lo_aqi), 0.0, 1.0))

        out.append(
            {
                "ward_id": ward_id,
                "name": names.get(ward_id, ward_id),
                "population": pops.get(ward_id, 0),
                "eta_h": int(first["horizon_h"]),
                "target_ts": first["target_ts"],
                "aqi_p50": int(first["aqi_p50"]),
                "pm25_p50": float(first["p50"]),
                "confidence": round(conf, 2),
            }
        )

    out.sort(key=lambda a: (a["eta_h"], -a["aqi_p50"]))
    return out
