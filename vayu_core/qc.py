"""Plausibility screening for incoming measurements.

Real monitoring networks emit impossible values. In the bundled Delhi window,
**1.3% of measured PM2.5 readings are zero or negative** — physically impossible
for a mass concentration, and a sensor fault rather than clean air. Training on
them teaches the model that Delhi is occasionally spotless, and they drag the
climatology baseline down too.

The bounds are deliberately loose. The job here is to remove the *impossible*,
not the *inconvenient*: Delhi genuinely exceeds 500 µg/m³ PM2.5 during severe
episodes (0.32% of readings do), and clipping those would erase exactly the
events VAYU exists to forecast. Anything dropped is counted and logged, never
silently discarded.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

# (min_exclusive, max_inclusive) in the param's stored unit.
# Upper bounds are set above the highest credible Indian reading, so they only
# catch instrument faults and stuck registers.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 1000.0),   # world-record urban hourly values sit near 1000
    "pm10": (0.0, 3000.0),   # dust storms genuinely reach four figures
    "no2": (0.0, 1000.0),
    "so2": (0.0, 2000.0),
    "co": (0.0, 100.0),      # mg/m3
    "o3": (0.0, 800.0),
    "nh3": (0.0, 3000.0),
}


def screen(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Drop physically impossible measurements, logging what went.

    Expects long-format columns: param, value.
    """
    if df.empty or "param" not in df.columns or "value" not in df.columns:
        return df

    keep = pd.Series(True, index=df.index)
    reasons: dict[str, int] = {}

    for param, (lo, hi) in PLAUSIBLE.items():
        m = df["param"] == param
        if not m.any():
            continue
        bad_low = m & (df["value"] <= lo)
        bad_high = m & (df["value"] > hi)
        if bad_low.any():
            reasons[f"{param}<=0"] = int(bad_low.sum())
        if bad_high.any():
            reasons[f"{param}>{hi:g}"] = int(bad_high.sum())
        keep &= ~(bad_low | bad_high)

    # A NaN value carries no information and breaks lag arithmetic downstream.
    nan_mask = df["value"].isna()
    if nan_mask.any():
        reasons["null"] = int(nan_mask.sum())
        keep &= ~nan_mask

    dropped = int((~keep).sum())
    if dropped:
        pct = dropped / len(df) * 100
        logger.info(
            f"{label} QC: dropped {dropped:,} of {len(df):,} readings ({pct:.2f}%) — "
            + ", ".join(f"{k}: {v:,}" for k, v in sorted(reasons.items()))
        )
    return df[keep]
