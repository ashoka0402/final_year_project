"""CPCB National Air Quality Index conversion.

Implements the Central Pollution Control Board sub-index scheme used by SAMEER
and every Indian AQI bulletin. VAYU reports AQI in exactly the buckets an Indian
commissioner already argues with, so nothing has to be mentally re-based.

Reference: CPCB, "National Air Quality Index" (2014), Table 2 — sub-index
breakpoints. Sub-index is piecewise-linear within a band:

    I = I_lo + (I_hi - I_lo) * (C - C_lo) / (C_hi - C_lo)

The overall AQI of a station is the MAXIMUM of its available sub-indices (CPCB
uses worst-pollutant-wins), and requires at least one of PM2.5/PM10 to be valid.

Breakpoints per TRD 5.1. PM2.5 (24h avg, ug/m3):
    0-30 -> 0-50, 31-60 -> 51-100, 61-90 -> 101-200,
    91-120 -> 201-300, 121-250 -> 301-400, >250 -> 401-500
"""

from __future__ import annotations

from dataclasses import dataclass

# (C_lo, C_hi, I_lo, I_hi) per pollutant, in the pollutant's reporting unit.
# The top band is open-ended upward; we clamp the index at 500 (CPCB does too).
BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    # PM2.5, 24-hour average, ug/m3
    "pm25": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    # PM10, 24-hour average, ug/m3
    "pm10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    # NO2, 24-hour average, ug/m3
    "no2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 1000, 401, 500),
    ],
    # SO2, 24-hour average, ug/m3
    "so2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2400, 401, 500),
    ],
    # CO, 8-hour average, mg/m3
    "co": [
        (0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, 50, 401, 500),
    ],
    # O3, 8-hour average, ug/m3
    "o3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500),
    ],
}

# CPCB category bands with the official colours. AQI is never conveyed by colour
# alone in the UI (PRD non-functional: accessibility) — label always travels with it.
CATEGORIES: list[tuple[int, int, str, str]] = [
    (0, 50, "Good", "#009865"),
    (51, 100, "Satisfactory", "#A3C853"),
    (101, 200, "Moderate", "#FFF833"),
    (201, 300, "Poor", "#F29C33"),
    (301, 400, "Very Poor", "#E93F33"),
    (401, 500, "Severe", "#AF2D24"),
]


@dataclass(frozen=True)
class AqiResult:
    """An AQI value plus the pollutant that drove it (CPCB: worst wins)."""

    aqi: int
    category: str
    color: str
    dominant_param: str | None


def sub_index(param: str, concentration: float | None) -> float | None:
    """CPCB sub-index for one pollutant concentration.

    Returns None when the pollutant is unknown or the value is missing/negative,
    so callers can distinguish "no data" from "clean air" (an AQI of 0 is a
    claim; None is an absence — the UI renders them very differently).
    """
    if concentration is None or param not in BREAKPOINTS:
        return None
    c = float(concentration)
    if c < 0:
        return None

    bands = BREAKPOINTS[param]
    for c_lo, c_hi, i_lo, i_hi in bands:
        if c <= c_hi:
            # Linear interpolation inside the band.
            return i_lo + (i_hi - i_lo) * (c - c_lo) / (c_hi - c_lo)

    # Above the top breakpoint CPCB caps the index at 500 rather than
    # extrapolating; a 900 ug/m3 hour is "Severe", not AQI 900.
    return 500.0


def concentration_from_sub_index(param: str, index: float | None) -> float | None:
    """Inverse of `sub_index`: recover a concentration from a published CPCB index.

    Why this exists: India's official real-time feed (CPCB CAAQMS via
    data.gov.in) publishes *sub-indices*, not concentrations — a fact the field
    names ("avg_value") hide. Reading those numbers as ug/m3 puts every Delhi
    station at "AQI 500 Severe" during monsoon. The giveaway is CO: a published
    value of ~52 is impossible as mg/m3 (severe poisoning) and impossible as
    ug/m3 (below ambient), but is exactly 1.05 mg/m3 read as a sub-index.

    The breakpoint map is piecewise-linear and strictly monotonic, so the
    inverse is exact up to the integer rounding of the published index
    (about +/-1-3 ug/m3 across the PM2.5 range).

    IMPORTANT semantics: CPCB sub-indices are computed on 24-hour averages for
    PM2.5/PM10/NO2/SO2/NH3 and 8-hour maxima for CO/O3. What comes back is
    therefore a 24h-average concentration, NOT an instantaneous hourly value.
    Callers that need hourly resolution must not treat these interchangeably.
    """
    if index is None or param not in BREAKPOINTS:
        return None
    i = float(index)
    if i < 0:
        return None

    bands = BREAKPOINTS[param]
    for c_lo, c_hi, i_lo, i_hi in bands:
        if i <= i_hi:
            if i_hi == i_lo:
                return c_lo
            return c_lo + (c_hi - c_lo) * (i - i_lo) / (i_hi - i_lo)

    # Index above 500 shouldn't occur (CPCB clamps); return the top concentration.
    return bands[-1][1]


def aqi_from_sub_indices(indices: dict[str, float | None]) -> AqiResult | None:
    """Overall AQI from already-computed CPCB sub-indices (worst pollutant wins).

    Used on the CPCB live path, where inverting to concentration and re-deriving
    the index would only add rounding error to a number CPCB already published.
    """
    subs = {
        p: float(v)
        for p, v in indices.items()
        if v is not None and p in BREAKPOINTS and float(v) >= 0
    }
    if not subs or not ({"pm25", "pm10"} & subs.keys()):
        return None
    dominant = max(subs, key=lambda p: subs[p])
    aqi = int(round(min(subs[dominant], 500)))
    label, color = category_for(aqi)
    return AqiResult(aqi=aqi, category=label, color=color, dominant_param=dominant)


def category_for(aqi: float) -> tuple[str, str]:
    """(label, hex colour) for an AQI value."""
    a = int(round(aqi))
    for lo, hi, label, color in CATEGORIES:
        if lo <= a <= hi:
            return label, color
    return ("Severe", "#AF2D24") if a > 500 else ("Good", "#009865")


def aqi_from_concentrations(values: dict[str, float | None]) -> AqiResult | None:
    """Overall CPCB AQI from a {param: concentration} mapping.

    CPCB requires PM2.5 or PM10 to be present for a valid station AQI — an index
    built only from, say, CO would understate a smoke event. Returns None if
    neither particulate is available, rather than quietly reporting a partial AQI.
    """
    subs: dict[str, float] = {}
    for param, conc in values.items():
        s = sub_index(param, conc)
        if s is not None:
            subs[param] = s

    if not subs or not ({"pm25", "pm10"} & subs.keys()):
        return None

    dominant = max(subs, key=lambda p: subs[p])
    aqi = int(round(subs[dominant]))
    label, color = category_for(aqi)
    return AqiResult(aqi=aqi, category=label, color=color, dominant_param=dominant)


def aqi_from_pm25(pm25: float | None) -> int | None:
    """Convenience: AQI from PM2.5 alone (the forecast target)."""
    s = sub_index("pm25", pm25)
    return None if s is None else int(round(s))
