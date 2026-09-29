"""Feature engineering for the Forecaster (TRD 5.1).

Builds one row per (station, hour) with everything the model may look at, and
nothing it may not: every feature here is computable at prediction time from
data that would genuinely be available then. The single most dangerous bug in a
forecaster is leakage — a feature that quietly encodes the future — so the
construction rules are:

  * pm2.5 lags/rollings are shifted so row `t` only ever sees `t` and earlier.
  * upwind features use the upwind station's value at time `t`, not the target.

Weather appears TWICE, and the distinction matters more than anything else here:

  * `<var>`     — conditions at issue time `t`. Describes the air being handed over.
  * `fx_<var>`  — conditions at the TARGET hour `t + horizon`. This is what
                  actually decides the answer: a Delhi build-up is made by the
                  wind and mixing height *tomorrow*, not today. Using only
                  issue-time weather asks the model "what will PM2.5 be in 48h?"
                  while hiding the 48h weather from it.

This is legitimate, not leakage: a real deployment genuinely has this. Open-Meteo
publishes a 4-day forecast, so at issue time we really do know the predicted wind
at t+72h. The backtest uses the same field, so the reported skill is the skill an
operator would actually get — including the weather model's own error.

Targets and forecast-time weather are attached together by
`add_target_and_forecast_weather`, because both depend on the horizon.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.config import CityConfig
from vayu_core.geo import bearing_deg, haversine_km

PM_LAGS = (1, 3, 6, 12, 24, 48)
ROLL_WINDOWS = (6, 24)

# Upwind search cone (TRD 5.1): nearest station within +/-45 deg of the upwind
# bearing, between 5 and 50 km.
UPWIND_HALF_ANGLE = 45.0
UPWIND_MIN_KM = 5.0
UPWIND_MAX_KM = 50.0

# Local fire window, per TRD 5.1: FRP within 50 km upwind over 24h. This catches
# landfill and municipal waste burning inside the city.
FIRE_LOCAL_KM = 50.0
FIRE_LOCAL_HOURS = 24

# Regional fire window — an addition to TRD 5.1, and the reason for it:
# the 50 km rule cannot see the event VAYU exists to catch. Punjab/Haryana
# stubble fires sit 150-300 km upwind of Delhi; measured on the bundled window
# the 50 km feature peaks at ~29 MW while the region is burning ~19,000 MW. At
# typical winter wind speeds that smoke takes 12-24h to arrive, so a 48h window
# out to 300 km is the physically right envelope, and a narrower cone (+/-30 deg)
# reflects that long-range transport is more directional than local mixing.
FIRE_REGIONAL_MIN_KM = 50.0
FIRE_REGIONAL_MAX_KM = 300.0
FIRE_REGIONAL_HOURS = 48
FIRE_REGIONAL_HALF_ANGLE = 30.0

# Caps the (hours x nearby_fires) broadcast in `_add_fires` to roughly this
# many elements per allocation, regardless of how many hours of history or how
# many in-range fires a station has — see that function's docstring for why
# this bound exists.
_FIRE_ROW_CHUNK_ELEMENTS = 5_000_000

WEATHER_COLUMNS = [
    "u",
    "v",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "pblh",
    "rh",
    "temp_c",
    "precip",
    "pressure",
    "vent_index",
]

# Weather at the target hour — the physics that actually decides the forecast.
FORECAST_WEATHER_COLUMNS = [f"fx_{c}" for c in WEATHER_COLUMNS]

# Change in conditions between issue and target: "the wind is about to drop" is
# a different, stronger signal than either endpoint alone.
DELTA_WEATHER_COLUMNS = ["fx_d_wind_speed", "fx_d_pblh", "fx_d_vent_index"]

BASE_COLUMNS = [
    *[f"pm25_lag{h}" for h in PM_LAGS],
    *[f"pm25_roll{w}" for w in ROLL_WINDOWS],
    "pm25_delta_24h",
    "pm10_lag1",
    "no2_lag1",
    "hour_sin",
    "hour_cos",
    "dow",
    "month",
    "is_holiday",
    *WEATHER_COLUMNS,
    "upwind_pm25",
    "upwind_fire_frp_24h",
    "upwind_fire_frp_regional_48h",
    "upwind_fire_count_regional_48h",
]

FEATURE_COLUMNS = [*BASE_COLUMNS, *FORECAST_WEATHER_COLUMNS, *DELTA_WEATHER_COLUMNS]

# Plain-English labels for the "Why this forecast?" panel (PRD A4). The judge
# reads "Low wind speed +38", not "wind_speed".
FEATURE_LABELS: dict[str, str] = {
    "pm25_lag1": "PM2.5 an hour ago",
    "pm25_lag3": "PM2.5 3h ago",
    "pm25_lag6": "PM2.5 6h ago",
    "pm25_lag12": "PM2.5 12h ago",
    "pm25_lag24": "PM2.5 yesterday",
    "pm25_lag48": "PM2.5 two days ago",
    "pm25_roll6": "6h average PM2.5",
    "pm25_roll24": "24h average PM2.5",
    "pm25_delta_24h": "24h trend",
    "pm10_lag1": "PM10 an hour ago",
    "no2_lag1": "NO₂ an hour ago (traffic)",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day",
    "dow": "Day of week",
    "month": "Season",
    "is_holiday": "Festival / holiday",
    "u": "Wind (east–west)",
    "v": "Wind (north–south)",
    "wind_speed": "Wind speed",
    "wind_dir_sin": "Wind direction",
    "wind_dir_cos": "Wind direction",
    "pblh": "Mixing height",
    "rh": "Humidity",
    "temp_c": "Temperature",
    "precip": "Rain",
    "pressure": "Surface pressure",
    "vent_index": "Ventilation (wind × mixing height)",
    "upwind_pm25": "Upwind station PM2.5",
    "upwind_fire_frp_24h": "Local fires upwind (50 km, 24h)",
    "upwind_fire_frp_regional_48h": "Stubble fires upwind (300 km, 48h)",
    "upwind_fire_count_regional_48h": "Upwind fire count (300 km)",
    # Forecast-time weather — labelled "forecast" so the panel never implies we
    # measured it.
    "fx_u": "Forecast wind (east–west)",
    "fx_v": "Forecast wind (north–south)",
    "fx_wind_speed": "Forecast wind speed",
    "fx_wind_dir_sin": "Forecast wind direction",
    "fx_wind_dir_cos": "Forecast wind direction",
    "fx_pblh": "Forecast mixing height",
    "fx_rh": "Forecast humidity",
    "fx_temp_c": "Forecast temperature",
    "fx_precip": "Forecast rain",
    "fx_pressure": "Forecast pressure",
    "fx_vent_index": "Forecast ventilation",
    "fx_d_wind_speed": "Wind speed change",
    "fx_d_pblh": "Mixing height change",
    "fx_d_vent_index": "Ventilation change",
}


def _pivot_params(measurements: pd.DataFrame) -> pd.DataFrame:
    """Long measurements -> wide (station_id, ts) x param, hourly."""
    df = measurements.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("h")
    wide = (
        df.pivot_table(index=["station_id", "ts"], columns="param", values="value", aggfunc="mean")
        .reset_index()
    )
    for p in ("pm25", "pm10", "no2"):
        if p not in wide.columns:
            wide[p] = np.nan
    return wide[["station_id", "ts", "pm25", "pm10", "no2"]]


# A station's record can contain multi-year holes — Delhi's OpenAQ history runs
# 2016-2018 on one instrument, then resumes 2025 on another. Reindexing straight
# across that gap would materialise ~60k empty hours per station (millions of
# rows) to no purpose, so the series is split into contiguous blocks first. Lags
# must not span the seam either: yesterday's PM2.5 is not a value from 2018.
MAX_GAP_HOURS = 24 * 7


def _regular_hourly_index(wide: pd.DataFrame) -> pd.DataFrame:
    """Reindex each station onto a gapless hourly grid, per contiguous block.

    Station feeds drop hours. Without an explicit grid, `shift(24)` means "24
    rows back", not "24 hours back" — so a gap silently turns a lag into a lie.
    """
    out = []
    for sid, grp in wide.groupby("station_id", sort=False):
        g = grp.set_index("ts").sort_index()
        # Break the record wherever it goes quiet for longer than MAX_GAP_HOURS.
        gap = g.index.to_series().diff() > pd.Timedelta(hours=MAX_GAP_HOURS)
        block_id = gap.cumsum()
        for bid, block in g.groupby(block_id):
            full = pd.date_range(block.index.min(), block.index.max(), freq="h", tz="UTC")
            b = block.reindex(full)
            b["station_id"] = sid
            # `block` travels with the row so lags can be grouped by it. Splitting
            # the reindex alone is not enough: shift() is row-based, so without
            # this a 2025 row's "yesterday" silently reaches back to 2018.
            b["block"] = f"{sid}#{bid}"
            b.index.name = "ts"
            out.append(b.reset_index())
    return pd.concat(out, ignore_index=True) if out else wide


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    # Group by contiguous block, not just station: shift() counts rows, so a
    # station whose record has a multi-year hole would otherwise borrow its
    # "yesterday" from the far side of the gap.
    key = "block" if "block" in df.columns else "station_id"
    g = df.groupby(key, sort=False)
    for h in PM_LAGS:
        df[f"pm25_lag{h}"] = g["pm25"].shift(h)
    for w in ROLL_WINDOWS:
        # shift(1) first: a rolling mean that includes the current hour would
        # leak the value we are trying to predict from.
        df[f"pm25_roll{w}"] = g["pm25"].transform(lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 3)).mean())
    df["pm25_delta_24h"] = df["pm25_lag1"] - df["pm25_lag24"]
    df["pm10_lag1"] = g["pm10"].shift(1)
    df["no2_lag1"] = g["no2"].shift(1)
    return df


def _add_time(df: pd.DataFrame, city: CityConfig) -> pd.DataFrame:
    # Local time: rush hours and cooking peaks follow IST, not UTC.
    local = df["ts"].dt.tz_convert(city.timezone)
    hour = local.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow"] = local.dt.dayofweek
    df["month"] = local.dt.month

    try:
        import holidays as _holidays

        subdiv = {"delhi": "DL", "lucknow": "UP"}.get(city.id)
        years = sorted(local.dt.year.unique().tolist())
        cal = _holidays.India(years=years, subdiv=subdiv)
        dates = local.dt.date
        df["is_holiday"] = dates.map(lambda d: 1 if d in cal else 0).astype(int)
    except Exception as exc:  # noqa: BLE001 - a missing calendar must not kill a forecast
        logger.warning(f"[{city.id}] holiday calendar unavailable ({exc}) — is_holiday=0")
        df["is_holiday"] = 0
    return df


def _nearest_grid(stations: pd.DataFrame, city: CityConfig) -> dict[str, tuple[int, int]]:
    pts = city.grid_points()
    out: dict[str, tuple[int, int]] = {}
    for st in stations.itertuples():
        i, j, _, _ = min(pts, key=lambda p: haversine_km(st.lat, st.lon, p[2], p[3]))
        out[st.station_id] = (i, j)
    return out


def prepare_weather(weather: pd.DataFrame, prefer: str = "hist") -> pd.DataFrame:
    """Derive u/v/ventilation once, from one weather `kind`.

    `prefer` decides which field wins where both exist, and the choice is not
    cosmetic:

      * 'hist'     — reanalysis. Correct for issue-time features, which describe
                     conditions we have already observed.
      * 'forecast' — what the forecast said. Correct for `fx_*` (target-hour)
                     features. Using reanalysis there would give the model
                     perfect knowledge of tomorrow's wind, which no operator has,
                     and would inflate every accuracy number in evaluation.md.
    """
    wx = weather.copy()
    # Station features use the fine city grid; the coarse airshed grid exists for
    # trajectories and would otherwise duplicate every (grid_i, grid_j) key.
    if "grid" in wx.columns:
        wx = wx[wx["grid"] == "city"]
    if wx.empty:
        return pd.DataFrame(columns=["grid_i", "grid_j", "ts", *WEATHER_COLUMNS])
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True).dt.floor("h")

    if "kind" in wx.columns:
        # Rank so the preferred kind sorts first, then drop the duplicate hour.
        wx["_rank"] = (wx["kind"] != prefer).astype(int)
        wx = wx.sort_values("_rank").drop_duplicates(subset=["grid_i", "grid_j", "ts"], keep="first")
        wx = wx.drop(columns=["_rank"])

    speed = wx["wind_speed_100m"].fillna(wx["wind_speed_10m"])
    direction = wx["wind_dir_100m"].fillna(wx["wind_dir_10m"])
    rad = np.radians(direction)
    # Meteorological convention: direction is where wind comes FROM.
    wx["u"] = -speed * np.sin(rad)
    wx["v"] = -speed * np.cos(rad)
    wx["wind_speed"] = speed
    wx["wind_dir_sin"] = np.sin(rad)
    wx["wind_dir_cos"] = np.cos(rad)
    # Ventilation index: the single best physical predictor of a Delhi winter
    # build-up — stagnant air under a low inversion traps everything emitted.
    wx["vent_index"] = speed * wx["pblh"]
    return wx[["grid_i", "grid_j", "ts", *WEATHER_COLUMNS]]


def _add_weather(df: pd.DataFrame, weather: pd.DataFrame, stations: pd.DataFrame, city: CityConfig) -> pd.DataFrame:
    """Weather at issue time, plus the station->grid mapping later joins reuse."""
    grid = _nearest_grid(stations, city)
    df["grid_i"] = df["station_id"].map(lambda s: grid.get(s, (0, 0))[0])
    df["grid_j"] = df["station_id"].map(lambda s: grid.get(s, (0, 0))[1])

    if weather.empty:
        for c in WEATHER_COLUMNS:
            df[c] = np.nan
        return df

    wx = prepare_weather(weather)
    return df.merge(wx, on=["grid_i", "grid_j", "ts"], how="left")


def _add_upwind(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Nearest upwind station's PM2.5 (TRD 5.1).

    This is the feature that should let the model beat persistence on a
    transport event: a plume arriving from the north-west shows up at Narela
    before it shows up at ITO.
    """
    st = {s.station_id: (s.lat, s.lon) for s in stations.itertuples()}
    ids = list(st)
    if len(ids) < 2:
        df["upwind_pm25"] = np.nan
        return df

    # Precompute pairwise distance + bearing once: O(n^2) over ~100 stations.
    pair: dict[str, list[tuple[str, float, float]]] = {}
    for a in ids:
        lat_a, lon_a = st[a]
        cands = []
        for b in ids:
            if a == b:
                continue
            lat_b, lon_b = st[b]
            d = haversine_km(lat_a, lon_a, lat_b, lon_b)
            if UPWIND_MIN_KM <= d <= UPWIND_MAX_KM:
                cands.append((b, d, bearing_deg(lat_a, lon_a, lat_b, lon_b)))
        pair[a] = sorted(cands, key=lambda x: x[1])

    # pm2.5 as a (ts x station) matrix so a candidate's value at a given hour is
    # an array lookup rather than a dict probe per row.
    wide = df.pivot_table(index="ts", columns="station_id", values="pm25", aggfunc="last")
    ts_pos = {t: i for i, t in enumerate(wide.index)}
    col_pos = {c: i for i, c in enumerate(wide.columns)}
    pm_mat = wide.to_numpy()

    # Upwind bearing = direction the wind comes from = atan2(-u, -v).
    upwind_bearing = ((np.degrees(np.arctan2(-df["u"].to_numpy(), -df["v"].to_numpy())) + 360) % 360)
    row_ts_idx = np.array([ts_pos.get(t, -1) for t in df["ts"]])

    out = np.full(len(df), np.nan)
    for sid, cands in pair.items():
        if not cands:
            continue
        idx = np.flatnonzero((df["station_id"] == sid).to_numpy())
        if idx.size == 0:
            continue
        bear = upwind_bearing[idx]
        tpos = row_ts_idx[idx]
        remaining = np.ones(idx.size, dtype=bool) & ~np.isnan(bear) & (tpos >= 0)

        # Candidates are distance-sorted, so the first qualifying one wins.
        for other, _d, brg in cands:
            if not remaining.any():
                break
            cpos = col_pos.get(other)
            if cpos is None:
                continue
            diff = np.abs((brg - bear + 180.0) % 360.0 - 180.0)
            hit = remaining & (diff <= UPWIND_HALF_ANGLE)
            if not hit.any():
                continue
            vals = np.full(idx.size, np.nan)
            vals[hit] = pm_mat[tpos[hit], cpos]
            got = hit & ~np.isnan(vals)
            out[idx[got]] = vals[got]
            remaining &= ~got

    df["upwind_pm25"] = out
    return df


def _add_fires(df: pd.DataFrame, fires: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Summed FRP of fires in the upwind cone within 50 km over the last 24h.

    Zero is a real, meaningful value here (no fires upwind) — distinct from
    "we have no fire data", which is why the fires layer reports `unavailable`
    rather than silently contributing zeros when there is no FIRMS key.

    Vectorised per station. The naive row x fire loop is 197k station-hours x
    4.4k detections ~ 864M pure-Python iterations, which effectively hangs the
    seed. Here the geometry (distance, bearing) is computed once per station
    against all fires, the far ones are dropped, and the time/bearing test runs
    as a numpy (hours x nearby_fires) broadcast.

    That (hours x nearby_fires) broadcast is itself chunked along the hours
    axis (`_FIRE_ROW_CHUNK`). Dropping far fires bounds the fires axis, but
    nothing originally bounded the hours axis — fine when this was written
    against a few months of history and ~4.4k detections, but `measurements`
    has since grown to 9 years of hourly rows per station and `fires` to
    ~21k detections. windowed=False (only ever exercised by the full-vs-
    windowed equivalence test, never by live scoring) then tries to build one
    dense array of tens of billions of elements per station — not slow, an
    out-of-memory swap-thrashing hang. Found by testing (a "hung" 2+ hour test
    run with zero growing CPU time — i.e. blocked on memory, not looping), not
    by reading the code. Chunking makes no numerical difference; it only caps
    how much of the same computation happens per allocation.
    """
    fire_cols = [
        "upwind_fire_frp_24h",
        "upwind_fire_frp_regional_48h",
        "upwind_fire_count_regional_48h",
    ]
    if fires is None or fires.empty:
        for c in fire_cols:
            df[c] = 0.0
        return df

    f = fires.copy()
    f["acq_ts"] = pd.to_datetime(f["acq_ts"], utc=True)
    f = f[["lat", "lon", "frp", "acq_ts"]].dropna()
    if f.empty:
        for c in fire_cols:
            df[c] = 0.0
        return df

    f_lat = f["lat"].to_numpy(dtype=float)
    f_lon = f["lon"].to_numpy(dtype=float)
    f_frp = f["frp"].to_numpy(dtype=float)
    # Drop the tz before going to numpy (which has no tz concept) rather than
    # letting .astype() strip it with a warning.
    f_ts = f["acq_ts"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]")

    upwind_bearing = ((np.degrees(np.arctan2(-df["u"].to_numpy(), -df["v"].to_numpy())) + 360) % 360)
    row_ts = pd.to_datetime(df["ts"]).dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    local = np.zeros(len(df))
    regional = np.zeros(len(df))
    regional_n = np.zeros(len(df))

    for st in stations.itertuples():
        idx = np.flatnonzero((df["station_id"] == st.station_id).to_numpy())
        if idx.size == 0:
            continue

        # Geometry once per station, against every fire.
        dlat = (f_lat - st.lat) * 110.574
        dlon = (f_lon - st.lon) * 111.320 * math.cos(math.radians(st.lat))
        dist = np.hypot(dlat, dlon)
        keep = dist <= FIRE_REGIONAL_MAX_KM
        if not keep.any():
            continue

        d_k = dist[keep]
        brg = (np.degrees(np.arctan2(dlon[keep], dlat[keep])) + 360) % 360
        frp_k = f_frp[keep]
        ts_k = f_ts[keep]
        decay = np.exp(-d_k / 150.0)[None, :]

        # Bound the per-allocation (hours x fires) array to roughly
        # _FIRE_ROW_CHUNK_ELEMENTS regardless of how many hours of history or
        # how many nearby fires this station has.
        rows_per_chunk = max(1, _FIRE_ROW_CHUNK_ELEMENTS // d_k.size)
        for start in range(0, idx.size, rows_per_chunk):
            chunk = idx[start : start + rows_per_chunk]

            # (hours x fires) broadcasts, one bounded chunk at a time.
            age_h = (row_ts[chunk][:, None] - ts_k[None, :]) / np.timedelta64(1, "h")
            bear = upwind_bearing[chunk][:, None]
            diff = np.abs((brg[None, :] - bear + 180.0) % 360.0 - 180.0)

            local_hit = (
                (age_h >= 0) & (age_h <= FIRE_LOCAL_HOURS)
                & (d_k[None, :] <= FIRE_LOCAL_KM)
                & (diff <= UPWIND_HALF_ANGLE)
            )
            local[chunk] = np.where(local_hit, frp_k[None, :], 0.0).sum(axis=1)

            reg_hit = (
                (age_h >= 0) & (age_h <= FIRE_REGIONAL_HOURS)
                & (d_k[None, :] > FIRE_REGIONAL_MIN_KM)
                & (d_k[None, :] <= FIRE_REGIONAL_MAX_KM)
                & (diff <= FIRE_REGIONAL_HALF_ANGLE)
            )
            # Distance-decayed FRP: 250 km of atmosphere dilutes a plume, so a
            # fire at the far edge cannot count the same as one at 60 km.
            # e-folding at 150 km is the same decay scale the attribution
            # fusion uses.
            regional[chunk] = np.where(reg_hit, frp_k[None, :] * decay, 0.0).sum(axis=1)
            regional_n[chunk] = reg_hit.sum(axis=1)

    df["upwind_fire_frp_24h"] = local
    df["upwind_fire_frp_regional_48h"] = regional
    df["upwind_fire_count_regional_48h"] = regional_n
    return df


# Longest look-back any feature needs before the row it describes: pm25_lag48
# and the regional 48h fire window. A scoring window must include at least this
# much history before the earliest hour it wants to score, plus slack for gaps.
MAX_LOOKBACK_HOURS = 48
SCORING_WINDOW_DAYS = 10  # 48h lookback + generous slack for missing hours


def build_features(
    city: CityConfig,
    measurements: pd.DataFrame,
    weather: pd.DataFrame,
    stations: pd.DataFrame,
    fires: pd.DataFrame | None = None,
    since: datetime | None = None,
) -> pd.DataFrame:
    """One row per (station, hour) with every model input. No target column.

    `since` bounds the measurement/fire history the features are built over. It
    is the difference between training and scoring:

      * Training passes `since=None` and builds over all history — the model must
        see every regime it will be asked about.
      * Scoring for one instant `at` only needs the ~48h before `at` (the longest
        lag/fire window), so `run_forecast` passes `since = at - 10 days`. That
        cuts ~1M station-hours to a few thousand and takes scoring from ~80s to
        ~2s per city — the change that makes live forecasting and the date picker
        viable. Weather is NOT trimmed here: the forecast-weather features need
        data out to `at + 72h`, and weather is already cheap to join.
    """
    if measurements.empty or stations.empty:
        return pd.DataFrame()

    if since is not None:
        m = measurements.copy()
        m["ts"] = pd.to_datetime(m["ts"], utc=True)
        measurements = m[m["ts"] >= since]
        if fires is not None and not fires.empty and "acq_ts" in fires.columns:
            f = fires.copy()
            f["acq_ts"] = pd.to_datetime(f["acq_ts"], utc=True)
            fires = f[f["acq_ts"] >= since - timedelta(hours=MAX_LOOKBACK_HOURS)]
        if measurements.empty:
            return pd.DataFrame()

    wide = _pivot_params(measurements)
    wide = _regular_hourly_index(wide)
    wide = wide.sort_values(["station_id", "ts"]).reset_index(drop=True)

    wide = _add_lags(wide)
    wide = _add_time(wide, city)
    wide = _add_weather(wide, weather, stations, city)
    wide = _add_upwind(wide, stations)
    wide = _add_fires(wide, fires, stations)

    logger.info(f"[{city.id}] features: {len(wide):,} station-hours (base)")
    return wide


def add_target(df: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    """Attach `y` = PM2.5 `horizon_h` hours after each row's timestamp."""
    out = df.copy()
    key = "block" if "block" in out.columns else "station_id"
    out["y"] = out.groupby(key, sort=False)["pm25"].shift(-horizon_h)
    return out


def add_forecast_weather(
    df: pd.DataFrame, weather: pd.DataFrame, horizon_h: int
) -> pd.DataFrame:
    """Join the weather valid at `t + horizon_h` as `fx_*`, plus its deltas.

    Requires the grid_i/grid_j columns that `_add_weather` leaves on the frame.
    """
    out = df.copy()
    if weather.empty or "grid_i" not in out.columns:
        for c in FORECAST_WEATHER_COLUMNS + DELTA_WEATHER_COLUMNS:
            out[c] = np.nan
        return out

    # prefer='forecast': fx_* must be what the forecast SAID at the target hour.
    wx = prepare_weather(weather, prefer="forecast").rename(
        columns={c: f"fx_{c}" for c in WEATHER_COLUMNS}
    )
    # Join on the target hour rather than the issue hour.
    out["_target_ts"] = out["ts"] + pd.Timedelta(hours=horizon_h)
    out = out.merge(
        wx.rename(columns={"ts": "_target_ts"}),
        on=["grid_i", "grid_j", "_target_ts"],
        how="left",
    ).drop(columns=["_target_ts"])

    for c in ("wind_speed", "pblh", "vent_index"):
        out[f"fx_d_{c}"] = out[f"fx_{c}"] - out[c]
    return out


def model_frame(
    df: pd.DataFrame, weather: pd.DataFrame, horizon_h: int, with_target: bool = True
) -> pd.DataFrame:
    """Rows ready for LightGBM at one horizon: base + fx_* (+ y)."""
    out = add_forecast_weather(df, weather, horizon_h)
    if with_target:
        out = add_target(out, horizon_h)
    return out


def training_frame(
    df: pd.DataFrame, weather: pd.DataFrame, horizon_h: int
) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) with rows that have no target or no recent history dropped."""
    d = model_frame(df, weather, horizon_h).dropna(subset=["y", "pm25_lag1"])
    return d[FEATURE_COLUMNS], d["y"]
