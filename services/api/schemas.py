"""Pydantic response models — the contract the web app generates types from."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

DataStatusValue = Literal["live", "cached", "sample", "cams", "h3-fallback", "unavailable"]


class Problem(BaseModel):
    """RFC7807 problem detail (TRD §6)."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None


class CitySummary(BaseModel):
    id: str
    name: str
    timezone: str
    bbox: list[float]
    map_center: list[float]
    map_zoom: float
    languages: list[str]
    grap_applicable: bool
    ward_count: int = 0
    station_count: int = 0
    population: int = 0
    population_source: str = ""


class DataStatus(BaseModel):
    source: str
    status: DataStatusValue
    detail: str = ""
    rows_loaded: int = 0
    fetched_ts: datetime | None = None


class StationOut(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    provider: str
    ts: datetime | None = None
    pm25: float | None = None
    aqi: int | None = None
    category: str | None = None
    color: str | None = None
    source: str | None = None


class WardOut(BaseModel):
    ward_id: str
    name: str
    pm25: float | None = None
    aqi: int | None = None
    category: str | None = None
    color: str | None = None
    population: int = 0
    nearest_station_km: float | None = None
    low_confidence: bool = False


class CurrentOut(BaseModel):
    city: str
    as_of: datetime
    demo_mode: bool
    aqi: int | None = Field(None, description="Population-weighted city AQI")
    category: str | None = None
    color: str | None = None
    dominant_param: str = "pm25"
    sources: list[str] = []
    wards: list[WardOut] = []
    stations: list[StationOut] = []
    data_status: list[DataStatus] = []


class ForecastWard(BaseModel):
    ward_id: str
    name: str
    population: int = 0
    p10: float
    p50: float
    p90: float
    aqi_p50: int
    category: str
    color: str


class ForecastOut(BaseModel):
    city: str
    run_ts: datetime
    horizon_h: int
    model_ver: str
    target_ts: datetime | None = None
    wards: list[ForecastWard] = []


class HazardAlert(BaseModel):
    ward_id: str
    name: str
    population: int
    eta_h: int
    target_ts: datetime
    aqi_p50: int
    pm25_p50: float
    confidence: float = Field(description="Share of the p10–p90 band above the threshold")


class ExplainFeature(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: str
    value: float | None = None


class ExplainOut(BaseModel):
    city: str
    ward_id: str
    horizon_h: int
    explained_via_station: str
    station_distance_km: float
    features: list[ExplainFeature] = []


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    demo_mode: bool
    now: datetime
    cities: list[str]
    seeded: bool
    detail: str = ""


class GeoJSON(BaseModel):
    type: str = "FeatureCollection"
    features: list[dict[str, Any]] = []
