"""Surface AQI from satellite data via a CNN-LSTM.

**What this is validated on, stated plainly.** The satellite layer (HCHO, NO2,
SO2, CO, O3, AOD) is genuinely national — 15,360 cells over India. Ground-truth
CPCB readings and meteorological reanalysis are NOT: this project has real,
matched daily data only for the Delhi-NCR + Lucknow corridor (188 stations, the
same region the rest of VAYU already models). Training and reporting RMSE/MAE/R
against stations outside that corridor would be a claim this repository cannot
back up, so the model is trained and evaluated there — and architected to extend
nationally the moment national CPCB history and ERA5/IMDAA reanalysis access
exist. That is one new region config and a re-run, not a rewrite; every other
national layer in this codebase (satellite_grid, fire_grid, hotspots, corridors)
already works that way.

**Architecture, and why this shape.**

    per (station, day):
      CNN   3x3 patch of satellite cells around the station, 6 channels
            -> a spatial embedding for that day
      LSTM  5-day sequence of [spatial embedding, meteorology, yesterday's PM2.5]
            -> predicted PM2.5 today

A CNN over a full-India image and an LSTM over years of history is the
textbook approach, but a raw pixel-grid CNN needs far more
labelled area than 188 point stations provide. Centering the convolution on
each station's local neighbourhood keeps the spatial-context idea genuine (a
station's air is not explained by its own cell alone; upwind cells matter) while
keeping the parameter count small enough for a few thousand station-days to
actually train, rather than overfitting a large image model to a handful of
labelled points.

Persistence (yesterday's PM2.5) is in the input for the same reason the
LightGBM forecaster predicts a residual rather than a level: pollution is
strongly autocorrelated day to day, and a model given no memory of that has to
relearn autocorrelation from scratch on a small dataset instead of spending its
capacity on what the satellite actually adds.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.aqi import aqi_from_pm25
from vayu_core.config import CityConfig, RegionConfig

# Which cities' station+weather data back the validated corridor. Extending
# this list (plus the matching national ground truth) is the entire path to
# scaling the claim beyond the corridor — no architecture change needed.
GROUND_TRUTH_CITIES = ("delhi", "delhi_ncr", "lucknow")

PATCH = 1          # cells either side of the station's own cell -> 3x3
LOOKBACK_DAYS = 5  # LSTM sequence length
# The full six-pollutant set this national layer targets. Not every one is
# necessarily ingested for a given region at a given time — `available_channels`
# checks the database rather than assuming, so a not-yet-ingested product fails
# loudly (a clear log line) instead of silently corrupting every sample the way
# an early version of this module did (every patch returned None, 0 samples,
# no error — found by testing against real data, not by reading the code).
ALL_SAT_CHANNELS = ("hcho", "no2", "so2", "co", "o3", "aod")
MET_COLS = ("temp_c", "rh", "wind_speed_10m", "wind_dir_10m", "pblh", "precip", "pressure")

MODEL_VERSION = "cnn-lstm-v1"


def available_channels(con, region_id: str, wanted: tuple[str, ...] = ALL_SAT_CHANNELS) -> tuple[str, ...]:
    have = {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT product FROM satellite_grid WHERE region = ?", [region_id]
        ).fetchall()
    }
    missing = [c for c in wanted if c not in have]
    if missing:
        logger.warning(
            f"[{region_id}] surface-AQI training: {missing} not ingested yet, "
            f"training on {[c for c in wanted if c in have]} only"
        )
    return tuple(c for c in wanted if c in have)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def _station_daily_pm25(con, cities: tuple[str, ...]) -> pd.DataFrame:
    """One row per (station_id, date): the day's mean PM2.5, deduped across
    overlapping city configs (a station inside both delhi and delhi_ncr's bbox
    is the same physical sensor, not two samples)."""
    df = con.execute(
        f"""SELECT m.station_id, s.lat, s.lon, date_trunc('day', m.ts) AS date,
                   avg(m.value) AS pm25
            FROM measurements m JOIN stations s USING (station_id)
            WHERE m.param = 'pm25' AND m.city IN ({",".join("?" * len(cities))})
            GROUP BY 1, 2, 3, 4""",
        list(cities),
    ).df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.drop_duplicates(subset=["station_id", "date"])


def _satellite_patch(
    con, region: RegionConfig, lat: float, lon: float, day: dt.date, channels: tuple[str, ...]
) -> np.ndarray | None:
    """6-channel, (2*PATCH+1)^2-cell patch around (lat, lon) on `day`.

    Missing cells (no valid retrieval that day, or edge of the national grid)
    are filled with that channel's grid-wide mean for the day rather than zero
    — a zero would read as "no formaldehyde", which is a physically different
    and much stronger claim than "not observed here today".
    """
    glat, glon = region.snap(lat, lon)
    d = region.grid_deg
    lat_win = [round(glat + k * d, 4) for k in range(-PATCH, PATCH + 1)]
    lon_win = [round(glon + k * d, 4) for k in range(-PATCH, PATCH + 1)]

    rows = con.execute(
        f"""SELECT product, grid_lat, grid_lon, value FROM satellite_grid
            WHERE region = ? AND date = ? AND product IN ({",".join("?" * len(channels))})
              AND grid_lat = ANY(?) AND grid_lon = ANY(?)""",
        [region.id, day, *channels, lat_win, lon_win],
    ).df()

    day_means = con.execute(
        f"""SELECT product, avg(value) AS m FROM satellite_grid
            WHERE region = ? AND date = ? AND product IN ({",".join("?" * len(channels))})
            GROUP BY 1""",
        [region.id, day, *channels],
    ).df()
    fallback = dict(zip(day_means["product"], day_means["m"])) if not day_means.empty else {}
    if not fallback:
        return None  # nothing observed for this region-day at all

    side = 2 * PATCH + 1
    patch = np.full((len(channels), side, side), np.nan)
    lat_idx = {v: i for i, v in enumerate(lat_win)}
    lon_idx = {v: i for i, v in enumerate(lon_win)}
    for r in rows.itertuples():
        c = channels.index(r.product)
        patch[c, lat_idx[round(r.grid_lat, 4)], lon_idx[round(r.grid_lon, 4)]] = r.value

    for c, prod in enumerate(channels):
        nan_mask = np.isnan(patch[c])
        if nan_mask.all() and prod not in fallback:
            return None
        patch[c][nan_mask] = fallback.get(prod, 0.0)
    return patch.astype("float32")


def _city_daily_weather(con, city: CityConfig) -> pd.DataFrame:
    df = con.execute(
        f"""SELECT date_trunc('day', ts) AS date, {",".join(f"avg({c}) AS {c}" for c in MET_COLS)}
            FROM weather_hourly WHERE city = ? AND grid = 'city' AND kind = 'hist'
            GROUP BY 1""",
        [city.id],
    ).df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


@dataclass
class Dataset:
    X_sat: np.ndarray   # (N, LOOKBACK, C, side, side)
    X_met: np.ndarray   # (N, LOOKBACK, len(MET_COLS) + 1)  (+1 = lagged pm25)
    y: np.ndarray        # (N,)  PM2.5 target
    dates: list[dt.date]  # for the time-based split
    station_ids: list[str]
    channels: tuple[str, ...] = ()  # which satellite products actually went in — carried
                                     # alongside the tensors so train() never has to assume.


def build_dataset(
    region: RegionConfig,
    cities: tuple[str, ...] = GROUND_TRUTH_CITIES,
    channels: tuple[str, ...] | None = None,
) -> Dataset:
    """Assemble the (station, day) training set for the validated corridor.

    Every sample needs LOOKBACK_DAYS of consecutive satellite+PM2.5 history
    ending on the target day — matching the anti-leakage discipline the rest of
    this codebase already follows (the LightGBM forecaster's windowed scoring):
    a sample is built only from data at or before the day it describes.
    """
    from vayu_core.config import load_city
    from vayu_core.db import read_conn

    with read_conn() as con:
        pm25 = _station_daily_pm25(con, cities)
        if pm25.empty:
            return Dataset(np.empty(0), np.empty(0), np.empty(0), [], [], ())
        channels = channels or available_channels(con, region.id)
        if not channels:
            raise ValueError(f"no satellite products ingested for region {region.id!r} yet")

        weather_by_city = {c: _city_daily_weather(con, load_city(c)) for c in cities}

        # A station's readings can be tagged to more than one city config
        # (delhi + delhi_ncr overlap); take whichever city has weather for it.
        X_sat, X_met, y, dates, sids = [], [], [], [], []
        for station_id, g in pm25.groupby("station_id"):
            g = g.sort_values("date").reset_index(drop=True)
            lat, lon = float(g["lat"].iloc[0]), float(g["lon"].iloc[0])

            wx = None
            for c in cities:
                wdf = weather_by_city[c]
                if not wdf.empty:
                    wx = wdf
                    break
            if wx is None:
                continue

            merged = g.merge(wx, on="date", how="left")
            merged["pm25_lag1"] = merged["pm25"].shift(1)

            for i in range(LOOKBACK_DAYS, len(merged)):
                window = merged.iloc[i - LOOKBACK_DAYS + 1 : i + 1]
                if window[["pm25_lag1", *MET_COLS]].isna().any().any():
                    continue
                day_seq_sat, day_seq_met, ok = [], [], True
                for _, row in window.iterrows():
                    patch = _satellite_patch(con, region, lat, lon, row["date"], channels)
                    if patch is None:
                        ok = False
                        break
                    day_seq_sat.append(patch)
                    day_seq_met.append([row[c] for c in MET_COLS] + [row["pm25_lag1"]])
                if not ok:
                    continue

                X_sat.append(np.stack(day_seq_sat))
                X_met.append(np.asarray(day_seq_met, dtype="float32"))
                y.append(float(merged.iloc[i]["pm25"]))
                dates.append(merged.iloc[i]["date"])
                sids.append(station_id)

    if not y:
        return Dataset(np.empty(0), np.empty(0), np.empty(0), [], [], ())
    return Dataset(
        X_sat=np.stack(X_sat), X_met=np.stack(X_met), y=np.asarray(y, dtype="float32"),
        dates=dates, station_ids=sids, channels=channels,
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def _import_torch():
    """Deferred import, single-threaded: torch is a training/offline-scoring
    dependency only, NOT imported by the API's request path — the live
    surfaces read pre-computed `aqi_grid` rows, the same pattern the LightGBM
    forecaster already uses (the API must not run a model per request). Keeping
    torch out of that path keeps the deployed container's per-request memory
    where the 2Gi floor already assumes it to be.

    Pinning intra-op threads to 1 is load-bearing, not a performance tweak:
    this codebase also imports scikit-learn/LightGBM in the same process
    (the forecaster tests do), and each ships its own bundled OpenMP runtime.
    Two live OpenMP runtimes contending over the same process's threads on
    macOS deadlocked `loss.backward()` on a dataset of 40 rows — reproduced by
    running the sklearn-heavy forecast tests and the CNN-LSTM tests together,
    absent when either ran alone. These models are tiny (a few thousand
    station-days); losing intra-op parallelism costs nothing measurable.
    """
    import torch

    torch.set_num_threads(1)
    return torch


def _build_model():
    torch = _import_torch()
    from torch import nn

    class CNNLSTM(nn.Module):
        def __init__(self, n_channels: int, patch_side: int, n_met: int, hidden: int = 32):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(n_channels, 16, kernel_size=2), nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=2), nn.ReLU(),
                nn.Flatten(),
            )
            cnn_out = 16 * (patch_side - 2) * (patch_side - 2)
            self.lstm = nn.LSTM(input_size=cnn_out + n_met, hidden_size=hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x_sat: "torch.Tensor", x_met: "torch.Tensor") -> "torch.Tensor":
            b, t, c, h, w = x_sat.shape
            emb = self.cnn(x_sat.reshape(b * t, c, h, w)).reshape(b, t, -1)
            seq = torch.cat([emb, x_met], dim=-1)
            _, (hn, _) = self.lstm(seq)
            return self.head(hn[-1]).squeeze(-1)

    return CNNLSTM


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #


@dataclass
class TrainReport:
    n_train: int
    n_holdout: int
    holdout_start: dt.date
    rmse: float
    mae: float
    r: float
    baseline_rmse: float  # persistence (yesterday's PM2.5) — the honesty check
    epochs: int


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std, mean, std


def train(
    ds: Dataset,
    holdout_days: int = 10,
    epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 30,
    artifact_dir=None,
) -> tuple[object, dict, TrainReport]:
    """Time-based train/holdout split — never random.

    A random split would let the model see satellite conditions from the same
    week it is being tested on, which is exactly the kind of leakage a burning
    season's autocorrelated pollution makes easy to hide and easy to be
    flattered by. The holdout is the last `holdout_days` of the season, matching
    the rolling-origin discipline the LightGBM forecaster's backtest already
    uses for the same reason.

    `epochs` is a ceiling, not a target: training stops early via a second
    split carved out of the TRAIN period alone (a random ~20% of TRAIN *days*,
    not the most recent ones — see below) once that inner-validation RMSE
    stops improving for `patience` epochs. Deciding "how long to train" by
    watching the real holdout would be the same leakage the outer split above
    exists to prevent.

    Two bugs found by testing against the real, non-stationary season (not by
    reading the code) before this shape was settled on:

    1. An unnormalised PM2.5 target (mean ~150 ug/m3, up to ~490 in the
       stubble-burning peak) fed to a near-zero-initialised head made
       full-batch Adam crawl: loss barely moved across 150 fixed epochs, and
       RMSE against a holdout that never exceeds ~490 came out at 210 — a
       scale bug, not a weak model. Standardising the target the same way
       every other input already is fixes it.
    2. The first version of this inner split carved off the LAST `holdout_days`
       of TRAIN as validation (mirroring the outer split). That is wrong here
       specifically because Delhi-NCR's PM2.5 roughly triples from early
       October to the mid-November stubble-burning peak: a trailing-days split
       leaves FIT holding only the calm early season (mean ~95) while VAL and
       the real HOLDOUT both land in the high-pollution tail (mean ~215-220) —
       the model never trains on a single example resembling what it is
       evaluated on. Holdout R went to -0.51. Sampling validation as a random
       20% of TRAIN days (interleaved with FIT, not appended after it) fixes
       this: both FIT and the inner VAL set span the full range of pollution
       regimes actually seen this season, which is what lets early stopping on
       VAL say something true about HOLDOUT generalisation. This is still not
       leakage against the true holdout — every VAL day remains strictly
       within TRAIN, i.e. strictly before `cutoff`.
    """
    torch = _import_torch()

    torch.manual_seed(seed)
    CNNLSTM = _build_model()

    dates = np.array(ds.dates)
    cutoff = max(dates) - dt.timedelta(days=holdout_days)
    train_mask, hold_mask = dates <= cutoff, dates > cutoff
    if train_mask.sum() < 20 or hold_mask.sum() < 5:
        raise ValueError(
            f"not enough data for a time-based split (train={train_mask.sum()}, "
            f"holdout={hold_mask.sum()}) — need more days or stations"
        )
    train_dates = sorted(set(dates[train_mask]))
    n_val_days = max(1, round(len(train_dates) * 0.2))
    if len(train_dates) - n_val_days < 5:
        fit_mask, val_mask = train_mask, train_mask  # too few distinct days to carve a third split; fall back to fixed epochs
    else:
        rng = np.random.default_rng(seed)
        val_dates = set(rng.choice(train_dates, size=n_val_days, replace=False).tolist())
        is_val_date = np.isin(dates, list(val_dates))
        fit_mask, val_mask = train_mask & ~is_val_date, train_mask & is_val_date

    # Standardise on TRAIN statistics only, then apply to every split —
    # fitting on the full dataset (including the holdout) would leak its
    # distribution into training.
    met_flat = ds.X_met.reshape(-1, ds.X_met.shape[-1])
    _, met_mean, met_std = _standardise(met_flat[np.repeat(train_mask, LOOKBACK_DAYS)])
    X_met_n = (ds.X_met - met_mean.reshape(1, 1, -1)) / met_std.reshape(1, 1, -1)

    sat_flat = ds.X_sat.reshape(ds.X_sat.shape[0], LOOKBACK_DAYS, len(ds.channels), -1)
    sat_mean = sat_flat[train_mask].mean(axis=(0, 1, 3), keepdims=True)
    sat_std = sat_flat[train_mask].std(axis=(0, 1, 3), keepdims=True) + 1e-6
    X_sat_n = (ds.X_sat - sat_mean.reshape(1, 1, -1, 1, 1)) / sat_std.reshape(1, 1, -1, 1, 1)

    # The target itself is standardised too. Left on its raw ug/m3 scale
    # (mean ~150), the model's near-zero-initialised head had to be dragged up
    # to that scale by gradient descent alone — full-batch Adam did this at a
    # crawl (loss barely moved across 150 epochs; RMSE=210 against a target
    # that never exceeds ~490). Standardising puts the optimisation problem on
    # the same well-conditioned footing every other input already gets.
    y_mean, y_std = float(ds.y[fit_mask].mean()), float(ds.y[fit_mask].std()) + 1e-6
    y_n = (ds.y - y_mean) / y_std

    def _t(a):
        return torch.tensor(a, dtype=torch.float32)

    Xs_fit, Xm_fit, y_fit = _t(X_sat_n[fit_mask]), _t(X_met_n[fit_mask]), _t(y_n[fit_mask])
    Xs_val, Xm_val = _t(X_sat_n[val_mask]), _t(X_met_n[val_mask])
    y_val_raw = ds.y[val_mask]
    Xs_ho, Xm_ho = _t(X_sat_n[hold_mask]), _t(X_met_n[hold_mask])
    actual_ho = ds.y[hold_mask]

    model = CNNLSTM(n_channels=len(ds.channels), patch_side=2 * PATCH + 1, n_met=X_met_n.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse, best_state, bad_epochs, actual_epochs = float("inf"), None, 0, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xs_fit, Xm_fit), y_fit)
        loss.backward()
        opt.step()
        actual_epochs = epoch + 1

        model.eval()
        with torch.no_grad():
            val_pred = model(Xs_val, Xm_val).numpy() * y_std + y_mean
        val_rmse = float(np.sqrt(np.mean((val_pred - y_val_raw) ** 2)))
        if val_rmse < best_val_rmse - 1e-3:
            best_val_rmse, best_state, bad_epochs = val_rmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_ho = model(Xs_ho, Xm_ho).numpy() * y_std + y_mean

    rmse = float(np.sqrt(np.mean((pred_ho - actual_ho) ** 2)))
    mae = float(np.mean(np.abs(pred_ho - actual_ho)))
    r = float(np.corrcoef(pred_ho, actual_ho)[0, 1]) if len(actual_ho) > 2 else float("nan")

    # Persistence baseline (yesterday's PM2.5, already in X_met's last column)
    # — the same honesty check the LightGBM forecaster reports against. A
    # model that cannot beat "today looks like yesterday" is not adding value.
    persistence_pred = X_met_n[hold_mask][:, -1, -1] * met_std[0, -1] + met_mean[0, -1]
    baseline_rmse = float(np.sqrt(np.mean((persistence_pred - actual_ho) ** 2)))

    report = TrainReport(
        n_train=int(fit_mask.sum()), n_holdout=int(hold_mask.sum()),
        holdout_start=cutoff + dt.timedelta(days=1),
        rmse=round(rmse, 2), mae=round(mae, 2), r=round(r, 3),
        baseline_rmse=round(baseline_rmse, 2), epochs=actual_epochs,
    )
    norm = {
        "met_mean": met_mean.tolist(), "met_std": met_std.tolist(),
        "sat_mean": sat_mean.tolist(), "sat_std": sat_std.tolist(),
        "y_mean": y_mean, "y_std": y_std,
    }
    if artifact_dir:
        import json
        import pathlib

        artifact_dir = pathlib.Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_dir / "cnn_lstm.pt")
        (artifact_dir / "norm.json").write_text(json.dumps(norm))
    return model, norm, report


# --------------------------------------------------------------------------- #
# Grid inference — the actual "spatial maps of surface AQI" deliverable
# --------------------------------------------------------------------------- #


def score_grid(
    model, norm: dict, region: RegionConfig, ds: Dataset, day: dt.date
) -> pd.DataFrame:
    """Predicted PM2.5/AQI at every STATION location for `day`.

    Deliberately does not extrapolate to grid cells with no station nearby: the
    model was trained on station-centred patches with real meteorology, and
    meteorology only exists for the validated corridor's cities. Producing a
    number for, say, a cell in Kerala would silently imply a corridor-trained
    model generalises to weather regimes it has never seen — exactly the
    unsupported-claim failure this module's docstring commits to avoiding.
    Widening this to a true national raster is one line here (iterate every
    region grid cell instead of every station) once national meteorology exists.
    """
    torch = _import_torch()

    day_idx = [i for i, d in enumerate(ds.dates) if d == day]
    if not day_idx:
        return pd.DataFrame()

    met_mean = np.asarray(norm["met_mean"])
    met_std = np.asarray(norm["met_std"])
    sat_mean = np.asarray(norm["sat_mean"])
    sat_std = np.asarray(norm["sat_std"])
    y_mean, y_std = norm["y_mean"], norm["y_std"]

    rows = []
    model.eval()
    with torch.no_grad():
        for i in day_idx:
            x_sat = (ds.X_sat[i : i + 1] - sat_mean.reshape(1, 1, -1, 1, 1)) / sat_std.reshape(1, 1, -1, 1, 1)
            x_met = (ds.X_met[i : i + 1] - met_mean.reshape(1, 1, -1)) / met_std.reshape(1, 1, -1)
            pred_n = float(
                model(torch.tensor(x_sat, dtype=torch.float32), torch.tensor(x_met, dtype=torch.float32))
                .numpy()[0]
            )
            pred_pm25 = pred_n * y_std + y_mean
            aqi = aqi_from_pm25(pred_pm25)
            rows.append(
                {
                    "region": region.id, "date": day, "station_id": ds.station_ids[i],
                    "pm25": round(pred_pm25, 1), "aqi": aqi, "model_ver": MODEL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def write_aqi_grid(region: RegionConfig, rows: pd.DataFrame) -> int:
    """Persist predictions keyed by the station's own snapped grid cell —
    `aqi_grid`'s schema is cell-keyed, so a station's prediction is written at
    the cell it actually sits in rather than invented for cells around it."""
    from vayu_core.aqi import category_for
    from vayu_core.db import read_conn, upsert_df

    if rows.empty:
        return 0
    with read_conn() as con:
        stations = con.execute(
            "SELECT DISTINCT station_id, lat, lon FROM stations WHERE station_id = ANY(?)",
            [rows["station_id"].unique().tolist()],
        ).df()
    rows = rows.merge(stations, on="station_id", how="left").dropna(subset=["lat", "lon"])
    snapped = rows.apply(lambda r: region.snap(r["lat"], r["lon"]), axis=1)
    rows["grid_lat"] = [s[0] for s in snapped]
    rows["grid_lon"] = [s[1] for s in snapped]
    cats = rows["aqi"].apply(lambda a: category_for(a)[0] if a is not None else None)
    rows["category"] = cats
    out = rows[["region", "grid_lat", "grid_lon", "date", "model_ver", "pm25", "aqi", "category"]]
    # Multiple stations can snap to the same cell; average rather than pick one
    # arbitrarily, consistent with `to_grid`'s cell-mean convention elsewhere.
    out = out.groupby(["region", "grid_lat", "grid_lon", "date", "model_ver"], as_index=False).agg(
        pm25=("pm25", "mean"), aqi=("aqi", "mean")
    )
    out["aqi"] = out["aqi"].round().astype(int)
    out["category"] = out["aqi"].apply(lambda a: category_for(a)[0])
    from vayu_core.db import write_conn

    with write_conn() as con:
        return upsert_df(con, "aqi_grid", out, ["region", "grid_lat", "grid_lon", "date", "model_ver"])
