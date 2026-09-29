"""Current conditions: station readings -> ward-level AQI.

This is the read path behind `GET /cities/{id}/current` and the Command Center
choropleth (PRD A1). It is deliberately in vayu_core rather than the API layer:
the station->ward step is the same IDW the Forecaster uses to place its
predictions (TRD 5.1), and the two must not drift apart.

Two honesty rules are enforced here rather than left to the UI:
  * a ward far from every station is flagged low-confidence, with the distance
    attached (App Flow §7: ">25 km -> low-confidence watermark").
  * a reading older than the staleness window is not presented as current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vayu_core.aqi import aqi_from_pm25, category_for
from vayu_core.config import CityConfig
from vayu_core.geo import idw

# How far back a reading may be and still count as "now".
STALENESS = timedelta(hours=3)

# Beyond this, a ward's interpolated value is an extrapolation worth flagging.
LOW_CONFIDENCE_KM = 25.0


@dataclass
class StationReading:
    station_id: str
    name: str
    lat: float
    lon: float
    provider: str
    ts: datetime | None
    pm25: float | None
    aqi: int | None
    category: str | None
    color: str | None
    source: str | None


@dataclass
class WardReading:
    ward_id: str
    name: str
    pm25: float | None
    aqi: int | None
    category: str | None
    color: str | None
    population: int
    nearest_station_km: float | None
    low_confidence: bool


@dataclass
class CitySnapshot:
    as_of: datetime
    stations: list[StationReading]
    wards: list[WardReading]
    city_aqi: int | None = None
    city_category: str | None = None
    city_color: str | None = None
    observed_param: str = "pm25"
    sources: list[str] = field(default_factory=list)


def _latest_per_station(measurements: pd.DataFrame, at: datetime) -> pd.DataFrame:
    """Most recent reading per (station, param) at or before `at`, within STALENESS."""
    if measurements.empty:
        return measurements
    df = measurements[(measurements["ts"] <= at) & (measurements["ts"] >= at - STALENESS)]
    if df.empty:
        return df
    return (
        df.sort_values("ts")
        .groupby(["station_id", "param"], as_index=False)
        .last()
    )


def snapshot(
    city: CityConfig,
    stations: pd.DataFrame,
    measurements: pd.DataFrame,
    wards: pd.DataFrame,
    at: datetime,
) -> CitySnapshot:
    """Build the current-conditions snapshot for a city at time `at`."""
    latest = _latest_per_station(measurements, at)

    # ---- stations -----------------------------------------------------------
    station_rows: list[StationReading] = []
    pm_by_station: dict[str, float] = {}
    sources: set[str] = set()

    by_station = (
        latest.pivot_table(index="station_id", columns="param", values="value", aggfunc="last")
        if not latest.empty
        else pd.DataFrame()
    )
    ts_by_station = (
        latest.groupby("station_id")["ts"].max() if not latest.empty else pd.Series(dtype="datetime64[ns, UTC]")
    )
    src_by_station = (
        latest.groupby("station_id")["source"].last() if not latest.empty else pd.Series(dtype=object)
    )

    for st in stations.itertuples():
        pm25 = None
        if not by_station.empty and st.station_id in by_station.index and "pm25" in by_station.columns:
            v = by_station.at[st.station_id, "pm25"]
            pm25 = None if pd.isna(v) else float(v)

        aqi = aqi_from_pm25(pm25)
        label, color = category_for(aqi) if aqi is not None else (None, None)
        src = src_by_station.get(st.station_id) if len(src_by_station) else None
        if src:
            sources.add(str(src))
        if pm25 is not None:
            pm_by_station[st.station_id] = pm25

        ts = ts_by_station.get(st.station_id) if len(ts_by_station) else None
        station_rows.append(
            StationReading(
                station_id=st.station_id,
                name=st.name,
                lat=float(st.lat),
                lon=float(st.lon),
                provider=str(st.provider),
                ts=None if ts is None or pd.isna(ts) else ts.to_pydatetime(),
                pm25=pm25,
                aqi=aqi,
                category=label,
                color=color,
                source=str(src) if src else None,
            )
        )

    # ---- wards (IDW from stations that actually reported) --------------------
    reporting = [s for s in station_rows if s.station_id in pm_by_station]
    ward_rows: list[WardReading] = []

    if reporting and not wards.empty:
        src_pts = [(s.lat, s.lon) for s in reporting]
        src_vals = [pm_by_station[s.station_id] for s in reporting]
        targets = [(float(w.centroid_lat), float(w.centroid_lon)) for w in wards.itertuples()]
        # No max_km here: a distant ward still gets an estimate, but carries the
        # distance so the UI can mark it low-confidence rather than pretend.
        values, nearest = idw(targets, src_pts, src_vals, power=2.0, k=5)

        for w, pm25, near in zip(wards.itertuples(), values, nearest):
            val = None if np.isnan(pm25) else round(float(pm25), 1)
            aqi = aqi_from_pm25(val)
            label, color = category_for(aqi) if aqi is not None else (None, None)
            ward_rows.append(
                WardReading(
                    ward_id=w.ward_id,
                    name=w.name,
                    pm25=val,
                    aqi=aqi,
                    category=label,
                    color=color,
                    population=int(w.population),
                    nearest_station_km=None if np.isinf(near) else round(float(near), 1),
                    low_confidence=bool(np.isinf(near) or near > LOW_CONFIDENCE_KM),
                )
            )
    else:
        # Designed empty state rather than a blank map: wards render in grey.
        for w in wards.itertuples():
            ward_rows.append(
                WardReading(
                    ward_id=w.ward_id,
                    name=w.name,
                    pm25=None,
                    aqi=None,
                    category=None,
                    color=None,
                    population=int(w.population),
                    nearest_station_km=None,
                    low_confidence=True,
                )
            )

    # ---- city roll-up -------------------------------------------------------
    # Population-weighted, matching the GRAP stage check in TRD 5.7 — a city AQI
    # that ignored where people live would trigger the wrong stage.
    city_aqi = city_label = city_color = None
    scored = [w for w in ward_rows if w.pm25 is not None and w.population > 0]
    if scored:
        weights = np.array([w.population for w in scored], dtype=float)
        vals = np.array([w.pm25 for w in scored], dtype=float)
        city_pm = float(np.sum(weights * vals) / np.sum(weights))
        city_aqi = aqi_from_pm25(city_pm)
        if city_aqi is not None:
            city_label, city_color = category_for(city_aqi)

    return CitySnapshot(
        as_of=at,
        stations=station_rows,
        wards=ward_rows,
        city_aqi=city_aqi,
        city_category=city_label,
        city_color=city_color,
        sources=sorted(sources),
    )
