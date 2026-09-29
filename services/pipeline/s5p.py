"""Sentinel-5P NO2 column density via Google Earth Engine. Fully optional.

Strengthens the `industry` and `traffic` attribution categories with a satellite
NO2 anomaly when GEE credentials are present (TRD 5.3: the S5P term multiplies
the industry score, and defaults to 1.0 — i.e. no effect — when absent).

Without credentials this reports `unavailable` and the UI hides the layer
entirely (PRD B4: "layer hidden gracefully when absent"). It never blocks a
demo, and no other module may hard-depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from vayu_core.config import REPO_ROOT, CityConfig, get_settings

COLLECTION = "COPERNICUS/S5P/NRTI/L3_NO2"
BAND = "tropospheric_NO2_column_number_density"


@dataclass
class S5PLayer:
    status: str  # 'live' | 'unavailable'
    png_path: str | None = None
    bounds: list[float] | None = None
    detail: str = ""


def available() -> bool:
    creds = get_settings().gee_service_account_json
    return bool(creds) and Path(creds).exists()


def fetch_no2(city: CityConfig) -> S5PLayer:
    """Weekly-mean NO2 column over the city bbox as a PNG overlay + bounds."""
    if not available():
        return S5PLayer(status="unavailable", detail="GEE_SERVICE_ACCOUNT_JSON not set — S5P layer hidden")

    try:
        import ee  # type: ignore

        settings = get_settings()
        credentials = ee.ServiceAccountCredentials(None, settings.gee_service_account_json)
        ee.Initialize(credentials)

        w, s, e, n = city.bbox
        region = ee.Geometry.Rectangle([w, s, e, n])
        img = (
            ee.ImageCollection(COLLECTION)
            .select(BAND)
            .filterDate(ee.Date(ee.Date.now()).advance(-7, "day"), ee.Date.now())
            .filterBounds(region)
            .mean()
        )
        url = img.getThumbURL(
            {
                "region": region,
                "dimensions": 1024,
                "format": "png",
                "min": 0,
                "max": 0.0002,  # mol/m^2, typical urban NO2 column range
                "palette": ["000004", "3b0f70", "8c2981", "de4968", "fe9f6d", "fcfdbf"],
            }
        )
        out = REPO_ROOT / "data" / "cache" / f"s5p_{city.id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)


        import httpx
        import certifi

        with httpx.Client(verify=certifi.where(), timeout=120) as c:
            out.write_bytes(c.get(url).content)

        logger.info(f"[{city.id}] S5P NO2 overlay written to {out.name}")
        return S5PLayer(status="live", png_path=str(out), bounds=[w, s, e, n], detail="Sentinel-5P NRTI L3 NO2, 7-day mean")
    except Exception as exc:  # noqa: BLE001 - GEE failures must never break a seed
        logger.warning(f"[{city.id}] S5P unavailable: {type(exc).__name__}: {exc}")
        return S5PLayer(status="unavailable", detail=f"{type(exc).__name__}")
