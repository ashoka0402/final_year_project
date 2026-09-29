"""Ward-level hourly PM2.5 — the observed side of verification.

Stations measure points; interventions are judged over wards. This bridges the
two by IDW-interpolating each hour's station readings onto ward centroids, the
same p=2/k=5 scheme the Command Center snapshot uses (TRD 5.1), so the number a
verdict is drawn from is the number the map showed.

Honesty: this is interpolation, not measurement. Delhi has ~52 stations for 290
wards, so most wards have no monitor and their series is inferred from
neighbours. That smoothing is precisely why difference-in-differences needs
control wards — an interpolated target and an interpolated control share the same
smoothing bias, and differencing cancels most of it.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from vayu_core.geo import idw


def ward_hourly_pm25(
    measurements: pd.DataFrame,
    stations: pd.DataFrame,
    wards: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Hourly PM2.5 per ward over [start, end). Columns: ward_id, ts, pm25.

    One IDW pass per hour. Vectorised over wards because the naive form is
    (hours x wards x stations) haversines — ~200 hours x 290 wards x 52 stations
    is 3M distance calculations per verification, and the station geometry never
    moves between hours.
    """
    if measurements.empty or stations.empty or wards.empty:
        return pd.DataFrame(columns=["ward_id", "ts", "pm25"])

    m = measurements[measurements["param"] == "pm25"].copy()
    if m.empty:
        return pd.DataFrame(columns=["ward_id", "ts", "pm25"])

    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    m = m[(m["ts"] >= start) & (m["ts"] < end)]
    if m.empty:
        return pd.DataFrame(columns=["ward_id", "ts", "pm25"])

    m["hour"] = m["ts"].dt.floor("h")
    # One value per station-hour: a station reporting twice in an hour must not
    # get double weight in the interpolation.
    hourly = m.groupby(["hour", "station_id"], as_index=False)["value"].mean()

    st = stations.set_index("station_id")
    targets = list(zip(wards["centroid_lat"].astype(float), wards["centroid_lon"].astype(float)))
    ward_ids = wards["ward_id"].tolist()

    rows: list[pd.DataFrame] = []
    for hour, grp in hourly.groupby("hour"):
        known = grp[grp["station_id"].isin(st.index)]
        if known.empty:
            continue
        sources = [
            (float(st.loc[s, "lat"]), float(st.loc[s, "lon"])) for s in known["station_id"]
        ]
        vals, _ = idw(targets, sources, known["value"].tolist())
        rows.append(pd.DataFrame({"ward_id": ward_ids, "ts": hour, "pm25": vals}))

    if not rows:
        return pd.DataFrame(columns=["ward_id", "ts", "pm25"])

    out = pd.concat(rows, ignore_index=True)
    out = out[~out["pm25"].isna()]
    logger.info(
        f"ward-hourly pm25: {len(out):,} rows over {out['ts'].nunique()} hours "
        f"x {out['ward_id'].nunique()} wards"
    )
    return out
