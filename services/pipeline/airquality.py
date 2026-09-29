"""Historical hourly air quality from Open-Meteo's CAMS reanalysis.

Why this source exists in VAYU:
India's measured station archive (OpenAQ v3, CPCB's own portal) is key-gated,
and the CPCB real-time feed is a *snapshot* — it has no history at all. But the
Forecaster needs a long hourly series per station to learn from, and the demo
must run offline. Open-Meteo's air-quality API serves the ECMWF CAMS global
reanalysis hourly, at any coordinate, with no key and no signup.

So VAYU samples CAMS **at the real CPCB station coordinates**. That gives a real,
physically-modelled hourly series anchored to the actual monitoring network.

Honesty (master prompt §2): this is MODELLED reanalysis, not a measurement.
It is stored as source='cams-reanalysis', surfaced as a 'sample' pill, and
documented in docs/DATA_PROVENANCE.md. CAMS is known to be biased low over
Indian cities (coarse ~40 km grid, uncertain emission inventories), which is
stated in the Methodology limitations rather than hidden. Supplying an
OPENAQ_API_KEY replaces this series with measured values.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from loguru import logger

from vayu_core.config import CityConfig
from vayu_core.db import SOURCE_CAMS
from vayu_core.qc import screen

from .http import fetch_json

ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo variable -> our param name
VARS = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
    "ozone": "o3",
}

# Coordinates per request. Open-Meteo accepts comma-separated lat/lon lists and
# returns one object per location; batching keeps a 50-station seed to ~5 calls.
BATCH = 10


def fetch_history(
    city: CityConfig,
    stations: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Hourly CAMS series for every station, [start, end] inclusive (UTC days)."""
    if stations.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    rows = stations.reset_index(drop=True)

    for b in range(0, len(rows), BATCH):
        chunk = rows.iloc[b : b + BATCH]
        payload = fetch_json(
            ENDPOINT,
            params={
                "latitude": ",".join(f"{v:.4f}" for v in chunk["lat"]),
                "longitude": ",".join(f"{v:.4f}" for v in chunk["lon"]),
                "hourly": ",".join(VARS.keys()),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "timezone": "UTC",
            },
            # Reanalysis of a past window never changes; cache for a month.
            ttl=timedelta(days=30),
            timeout=120.0,
        )
        # A single-location request returns a dict; multi-location returns a list.
        blocks = payload if isinstance(payload, list) else [payload]
        if len(blocks) != len(chunk):
            logger.warning(
                f"[{city.id}] CAMS returned {len(blocks)} blocks for {len(chunk)} stations — "
                "pairing by order"
            )

        for station, block in zip(chunk.itertuples(), blocks):
            hourly = block.get("hourly") or {}
            times = hourly.get("time") or []
            if not times:
                continue
            ts = pd.to_datetime(pd.Series(times), utc=True)
            for var, param in VARS.items():
                series = hourly.get(var)
                if not series:
                    continue
                df = pd.DataFrame({"ts": ts, "value": pd.Series(series, dtype="float64")}).dropna()
                if df.empty:
                    continue
                # Open-Meteo reports CO in ug/m3; CPCB's index is defined on
                # mg/m3. Convert here so `measurements` has one unit per param.
                if param == "co":
                    df["value"] = df["value"] / 1000.0
                df["city"] = city.id
                df["station_id"] = station.station_id
                df["param"] = param
                df["unit"] = "mg/m3" if param == "co" else "ug/m3"
                df["source"] = SOURCE_CAMS
                frames.append(df)

        logger.info(f"[{city.id}] CAMS history: {min(b + BATCH, len(rows))}/{len(rows)} stations")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out[["city", "station_id", "param", "ts", "value", "unit", "source"]]
    out = out.drop_duplicates(subset=["city", "station_id", "param", "ts"])
    out = screen(out, f"[{city.id}] CAMS")
    logger.info(f"[{city.id}] CAMS history: {len(out):,} rows {start} → {end}")
    return out


def historical_winters() -> list[tuple[date, date]]:
    """Extra training windows covering past Delhi stubble seasons.

    The measured record has a hole: OpenAQ's Delhi stations ran 2016-2018 on one
    set of instruments, went quiet, and resumed in Feb 2025 on new sensor ids.
    The demo window therefore contains exactly ONE November — the holdout — so a
    model trained on it has never seen a stubble season and predicts mean
    reversion into the worst air of the year (measured: a -28 ug/m3 bias, and a
    53% RMSE loss to persistence).

    Five stations still expose the retired sensors, including Anand Vihar, and
    November 2016 there measured a 492 ug/m3 monthly mean. Pulling those winters
    in gives the model real examples of sustained severe pollution. Lags never
    cross the gap: features.py splits each station's record into contiguous
    blocks.
    """
    return [
        (date(2016, 10, 1), date(2017, 2, 28)),
        (date(2017, 10, 1), date(2018, 2, 21)),
    ]


def demo_window(now: datetime, history_days: int = 255, forward_days: int = 21) -> tuple[date, date]:
    """The bundled data window around a pinned DEMO_NOW.

    Reaches back far enough to train on and forward past DEMO_NOW so the demo's
    "forecast" horizon has ground truth to be verified against — which is what
    lets the backtest (Phase 2) and outcome verification (Phase 5) run offline.

    Why 255 days: the window must span more than one pollution regime. A short
    Jul->Nov window trains the model on monsoon (PM2.5 ~40-60) and then asks it
    to predict the stubble season (250-500) — a distribution shift it has never
    seen, which showed up as a ~83 ug/m3 validation MAE. Reaching back to late
    February picks up the tail of the previous winter (Delhi Feb runs AQI
    200-350), so polluted conditions are represented in training.

    It does not reach a *full* previous stubble season: OpenAQ's currently-live
    Delhi sensors were registered around 2025-02-18 (the 2016-2018 instruments
    are retired and leave a multi-year gap), so ~255 days is the deepest
    continuous measured history available. Stated in docs/evaluation.md.
    """
    return (now - timedelta(days=history_days)).date(), (now + timedelta(days=forward_days)).date()
