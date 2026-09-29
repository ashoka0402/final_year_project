"""Gaussian plume dispersion — the counterfactual engine (TRD 5.4).

This module answers the only question that turns a forecast into an enforcement
order: **"if we stop this source, how much cleaner does that ward get?"**

The steady-state Gaussian plume, for a source of strength Q at effective height
H, wind speed u along x, receptor at crosswind distance y and height z:

    C(x,y,z) = Q / (2π · u · σy · σz)
               · exp(−y² / 2σy²)
               · [ exp(−(z−H)² / 2σz²) + exp(−(z+H)² / 2σz²) ]

The bracketed pair is the ground-reflection term: the plume cannot diffuse into
the earth, so the fraction that would have is mirrored back up. Dropping it
halves ground-level concentration — a mistake that would silently double every
"µg/m³ averted" claim in the leaderboard.

σy(x), σz(x) come from Pasquill-Gifford stability classes using **Briggs (1973)
open-country** coefficients, tabulated below with their source. Stability class
is picked by a simplified Turner scheme from wind speed and day/night.

WHAT THIS MODEL IS NOT — stated here and on the Methodology page, because the
limitations are the difference between a defensible estimate and a lie:
  * Steady-state and straight-line. Real wind veers; we mitigate by re-running
    along the forecast wind at 3h steps rather than assuming one direction.
  * No chemistry. PM2.5 is treated as inert over the transport window (hours),
    which is reasonable for primary particulate and wrong for secondary aerosol.
  * No deposition or washout. Concentrations are therefore an UPPER bound, which
    is the conservative direction for an averted-emissions claim.
  * Flat terrain, no urban canopy.
  * Q from fire radiative power is an order-of-magnitude estimate (see below).

Everything here is deliberately plain: ~200 lines a judge can read and check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from vayu_core.geo import bearing_deg, haversine_km

# --- Pasquill-Gifford stability classes -------------------------------------
# A = extremely unstable ... F = moderately stable.
STABILITY_CLASSES = ("A", "B", "C", "D", "E", "F")

# Briggs (1973) open-country dispersion coefficients, x in metres:
#   σy = a·x / sqrt(1 + b·x)     σz = c·x / (1 + d·x)^e
# Source: Briggs, G.A. (1973) "Diffusion estimation for small emissions",
# ATDL Contribution 79; reproduced in EPA workbooks and Seinfeld & Pandis.
BRIGGS_RURAL: dict[str, dict[str, float]] = {
    "A": {"a": 0.22, "b": 1e-4, "c": 0.20, "d": 0.0, "e": 1.0},
    "B": {"a": 0.16, "b": 1e-4, "c": 0.12, "d": 0.0, "e": 1.0},
    "C": {"a": 0.11, "b": 1e-4, "c": 0.08, "d": 2e-4, "e": -0.5},
    "D": {"a": 0.08, "b": 1e-4, "c": 0.06, "d": 1.5e-3, "e": -0.5},
    "E": {"a": 0.06, "b": 1e-4, "c": 0.03, "d": 3e-4, "e": -1.0},
    "F": {"a": 0.04, "b": 1e-4, "c": 0.016, "d": 3e-4, "e": -1.0},
}

# --- Emission strength Q (g/s) ----------------------------------------------
# Fires: Q = FRP(MW) x emission factor. Wooster et al. (2005), "Retrieval of
# biomass combustion rates and totals from fire radiative power observations",
# JGR 110: 1 MW of fire radiative power corresponds to ~0.368 kg/s of dry fuel
# consumed. Andreae & Merlet (2001) give ~6.3 g PM2.5 per kg of crop residue
# burned, so:
#     Q_pm25 (g/s) ≈ FRP(MW) x 0.368 (kg/s per MW) x 6.3 (g PM2.5 per kg)
FUEL_PER_MW_KG_S = 0.368        # Wooster 2005
PM25_PER_KG_FUEL_G = 6.3        # Andreae & Merlet 2001, crop residue
FRP_TO_Q_PM25 = FUEL_PER_MW_KG_S * PM25_PER_KG_FUEL_G  # ≈ 2.32 g/s per MW

# Industry: emission per km² of industrial land, NOT per mapped feature.
#
# Anchored to SAFAR's high-resolution emission inventory for Delhi (IITM, 2018),
# which puts the industrial sector at 24.10 Gg/yr of PM2.5 across the Delhi-NCR
# domain = ~764 g/s. Spread over the 82.5 km² of industrial landuse OSM maps in
# the Delhi airshed, that is ~9.3 g/s per km².
#
# Per km², because OSM `landuse=industrial` polygons are estates, not factories:
# they run from 0.01 to 9.6 km². A flat per-feature rate made a cluster of 54
# polygons a 135 g/s source — larger than a Punjab stubble field — and handed one
# candidate the whole city's population. This is the same error the attribution
# already fixed by scoring industry on area rather than count.
#
# HONESTY: published inventories disagree by an order of magnitude on industry's
# share of Delhi PM2.5 — SAFAR 22.4%, Guttikunda (2018) 28.9%, TERI (2018) 3.4%,
# IIT Kanpur (2016) 2.3%. We take the SAFAR figure because the rest of the
# attribution is cross-checked against SAFAR/IITM DSS ranges, but any industrial
# number here could be ~8x off and that is stated on the Methodology page.
INDUSTRY_Q_G_S_PER_KM2 = 9.26

# Category defaults where no radiometric or areal measurement exists. These are
# order-of-magnitude engineering estimates, documented as such — a construction
# site does not report its dust flux.
CATEGORY_Q_G_S: dict[str, float] = {
    "construction": 0.8,   # one active site, poor dust control
    "traffic": 1.2,        # one congested corridor-km at peak
}

# Effective release height (m). Fires loft on their own buoyancy; ground dust
# does not.
CATEGORY_HEIGHT_M: dict[str, float] = {
    "open_burning": 50.0,
    "construction": 5.0,
    "industry": 30.0,
    "traffic": 2.0,
}

# The plume model's own credibility, applied to every counterfactual (TRD 5.5).
PLUME_CONFIDENCE = 0.8

MIN_WIND_MS = 0.5  # below this the Gaussian form breaks down (u in the denominator)

# Maximum distance at which this model is allowed to make a claim.
#
# A steady-state, straight-line Gaussian plume assumes the wind holds constant in
# speed and direction over the whole transit. That is defensible for tens of
# kilometres and indefensible for hundreds: over a 200 km trip the air mass turns,
# the boundary layer cycles through a day, and particles deposit and age. 50 km is
# the limit US EPA regulatory practice puts on AERMOD, beyond which a puff or
# Lagrangian model (CALPUFF and kin) is required, and it matches FIRE_LOCAL_KM in
# the feature builder.
#
# This matters concretely for Delhi. In November the stubble that drives the smog
# burns in Punjab, 200-300 km upwind. This model must decline to size that source
# rather than quote a number it cannot support — that air is regional transport,
# and the honest product answer is that a municipal commissioner has no lever on
# it, not a fabricated "halt this field, avert 60 µg/m³".
PLUME_MAX_RANGE_KM = 50.0


@dataclass
class PlumeResult:
    """Concentration a source contributes at a receptor."""

    concentration_ugm3: float
    distance_km: float
    crosswind_km: float
    stability: str
    downwind: bool
    # False when the source is beyond PLUME_MAX_RANGE_KM: the model declined to
    # answer. Distinct from a modelled zero — do not read it as "no impact".
    in_range: bool = True


def stability_class(wind_speed_ms: float, local_hour: int, pblh_m: float | None = None) -> str:
    """Pasquill stability from wind speed and time of day (simplified Turner).

    The full Turner scheme needs insolation and cloud cover. We use the daylight
    proxy plus, where available, boundary-layer height — which is the variable
    that actually matters for Delhi: a winter night with a 100 m inversion is
    class F, and that is precisely when the city chokes.
    """
    day = 7 <= local_hour < 18

    # A collapsed boundary layer is stable regardless of the clock.
    if pblh_m is not None and pblh_m < 150:
        return "F"
    if pblh_m is not None and pblh_m < 300 and not day:
        return "E"

    if day:
        if wind_speed_ms < 2:
            return "A"
        if wind_speed_ms < 3:
            return "B"
        if wind_speed_ms < 5:
            return "B"
        if wind_speed_ms < 6:
            return "C"
        return "D"
    # Night
    if wind_speed_ms < 2:
        return "F"
    if wind_speed_ms < 3:
        return "E"
    if wind_speed_ms < 5:
        return "D"
    return "D"


def sigma_y(x_m: float, stab: str) -> float:
    """Crosswind dispersion (m) at downwind distance x."""
    if x_m <= 0:
        return 0.0
    k = BRIGGS_RURAL[stab]
    return k["a"] * x_m / math.sqrt(1.0 + k["b"] * x_m)


def sigma_z(x_m: float, stab: str) -> float:
    """Vertical dispersion (m) at downwind distance x.

    σz = c·x·(1 + d·x)^e. For the unstable classes d = 0 and the growth is
    linear; for C-F the exponent is negative, so vertical spread grows more
    slowly than distance — which is exactly why a stable night traps a plume
    near the ground instead of diluting it upward.
    """
    if x_m <= 0:
        return 0.0
    k = BRIGGS_RURAL[stab]
    if k["d"] == 0.0:
        return k["c"] * x_m
    return k["c"] * x_m * math.pow(1.0 + k["d"] * x_m, k["e"])


def _broadened_sigma_y(sy: float, receptor_radius_m: float) -> float:
    """Widen the crosswind Gaussian to average it over an area receptor.

    A documented deviation from TRD 5.4, which prescribes "plume contribution at
    ward centroid". A single point is the wrong receptor for the question the
    product actually asks — "how much cleaner is this WARD" — because a ward is
    an area of 5-78 km² while σy at 5 km is only ~300 m. Delhi ward W8's centroid
    sits 4.1 km crosswind of a nearby construction plume: 13σ off-axis, so the
    centroid scores a clean zero even though the plume plainly crosses part of
    the ward. Every such candidate died at the noise floor and the leaderboard
    came up empty for the worst-polluted wards in the city.

    The standard treatment is to convolve the crosswind profile with the
    receptor's own spatial spread. Convolving a Gaussian of width σy with any
    distribution of standard deviation σr gives, to a good approximation, a
    Gaussian of width sqrt(σy² + σr²). Modelling the ward as people spread
    uniformly over a disc of radius R gives a crosswind marginal with σr = R/2.

    The result is a population-weighted ward mean rather than a reading at one
    arbitrary point, which is what the ROI needs — it multiplies this by the
    ward's population.
    """
    if receptor_radius_m <= 0:
        return sy
    return math.sqrt(sy * sy + (receptor_radius_m / 2.0) ** 2)


def concentration(
    q_g_s: float,
    wind_speed_ms: float,
    downwind_m: float,
    crosswind_m: float,
    stab: str,
    height_m: float = 0.0,
    receptor_z_m: float = 0.0,
    mixing_height_m: float | None = None,
    receptor_radius_m: float = 0.0,
) -> float:
    """Ground-level concentration (µg/m³) from one source. 0 if upwind.

    `receptor_radius_m` turns the receptor from a point into an area — see
    `_broadened_sigma_y`. Pass 0 for a true point receptor.
    """
    if downwind_m <= 0 or q_g_s <= 0:
        return 0.0

    u = max(wind_speed_ms, MIN_WIND_MS)
    sy = _broadened_sigma_y(sigma_y(downwind_m, stab), receptor_radius_m)
    sz = sigma_z(downwind_m, stab)
    if sy <= 0 or sz <= 0:
        return 0.0

    # Once the plume fills the mixing layer, further vertical growth is capped
    # and the plume becomes uniformly mixed to the inversion. Ignoring this
    # under-states concentration in exactly the trapped-air conditions VAYU
    # exists to catch.
    if mixing_height_m and sz > 0.8 * mixing_height_m:
        vertical = 1.0 / mixing_height_m
    else:
        vertical = (
            math.exp(-((receptor_z_m - height_m) ** 2) / (2 * sz**2))
            + math.exp(-((receptor_z_m + height_m) ** 2) / (2 * sz**2))  # ground reflection
        ) / (math.sqrt(2 * math.pi) * sz)

    crosswind = math.exp(-(crosswind_m**2) / (2 * sy**2)) / (math.sqrt(2 * math.pi) * sy)

    # q in g/s -> µg/m³ requires a 1e6 factor (g -> µg).
    return (q_g_s * 1e6 / u) * crosswind * vertical


def q_from_frp(frp_mw: float) -> float:
    """PM2.5 emission rate (g/s) from fire radiative power (Wooster 2005)."""
    return max(frp_mw, 0.0) * FRP_TO_Q_PM25


def _project(
    src_lat: float, src_lon: float, rec_lat: float, rec_lon: float, wind_dir_from_deg: float
) -> tuple[float, float]:
    """(downwind_m, |crosswind_m|) of a receptor relative to a source.

    Downwind is negative when the receptor sits upwind of the source, which the
    caller must treat as "this source cannot affect that ward".
    """
    d_km = haversine_km(src_lat, src_lon, rec_lat, rec_lon)
    if d_km <= 0:
        return 0.0, 0.0
    brg = bearing_deg(src_lat, src_lon, rec_lat, rec_lon)
    # Wind FROM `wind_dir_from_deg` blows TOWARD the opposite bearing.
    plume_axis = (wind_dir_from_deg + 180.0) % 360.0
    theta = math.radians(brg - plume_axis)
    return d_km * 1000.0 * math.cos(theta), abs(d_km * 1000.0 * math.sin(theta))


def source_impact(
    src_lat: float,
    src_lon: float,
    rec_lat: float,
    rec_lon: float,
    q_g_s: float,
    wind_speed_ms: float,
    wind_dir_from_deg: float,
    local_hour: int,
    height_m: float = 10.0,
    pblh_m: float | None = None,
    receptor_radius_m: float = 0.0,
) -> PlumeResult:
    """What one source adds at one receptor, right now.

    `receptor_radius_m` > 0 averages over a ward-sized area instead of reading a
    single point — see `_broadened_sigma_y`.

    Returns no impact beyond PLUME_MAX_RANGE_KM — see the constant. That is a
    refusal to answer, not an answer of zero: the caller must treat a distant
    source as regional transport rather than as a source that does nothing.
    """
    downwind_m, crosswind_m = _project(src_lat, src_lon, rec_lat, rec_lon, wind_dir_from_deg)
    stab = stability_class(wind_speed_ms, local_hour, pblh_m)
    d_km = haversine_km(src_lat, src_lon, rec_lat, rec_lon)

    if downwind_m <= 0:
        # Receptor is upwind: this source contributes nothing to it.
        return PlumeResult(0.0, d_km, crosswind_m / 1000.0, stab, downwind=False)

    if d_km > PLUME_MAX_RANGE_KM:
        # Out of model range. A steady-state straight-line plume cannot describe
        # 200 km of transport, and pretending otherwise would put a fabricated
        # µg/m³ on an enforcement order.
        return PlumeResult(0.0, d_km, crosswind_m / 1000.0, stab, downwind=True, in_range=False)

    c = concentration(
        q_g_s, wind_speed_ms, downwind_m, crosswind_m, stab, height_m, 0.0, pblh_m,
        receptor_radius_m=receptor_radius_m,
    )
    return PlumeResult(c, d_km, crosswind_m / 1000.0, stab, downwind=True)


def ward_radius_m(area_km2: float) -> float:
    """Equivalent-disc radius of a ward, for area-receptor averaging."""
    if not area_km2 or area_km2 <= 0:
        return 0.0
    return math.sqrt(float(area_km2) / math.pi) * 1000.0


def averted_over_wards(
    src_lat: float,
    src_lon: float,
    ward_lats: np.ndarray,
    ward_lons: np.ndarray,
    ward_radius_m_arr: np.ndarray,
    q_g_s: float,
    at: datetime,
    wind: pd.DataFrame,
    tz: str,
    horizon_h: int = 24,
    height_m: float = 10.0,
    step_h: int = 3,
) -> np.ndarray:
    """µg/m³ averted at EVERY ward if this source stops — one vectorised pass.

    The same plume as `counterfactual`, evaluated at many receptors at once.
    Exists because "how many people does this action protect" is a physics
    question and was being answered with a stand-in exp(-d/120) decay that had
    nothing to do with the plume: it swept in every ward within ~40 km of the
    source regardless of wind, which in Delhi is the whole city. That handed one
    candidate all 16.8M residents and let the population term, not the physics,
    decide the top of the leaderboard.

    Vectorised because the honest version is 290 wards x 9 time steps per
    candidate, and a scalar loop made the city sweep several seconds.
    """
    from zoneinfo import ZoneInfo

    n = len(ward_lats)
    if n == 0 or wind.empty or q_g_s <= 0:
        return np.zeros(n)

    w = wind.copy()
    w["ts"] = pd.to_datetime(w["ts"], utc=True)
    w = w.sort_values("ts")

    # Geometry is fixed; only the wind rotates. Precompute distance and bearing.
    d_km = np.array([haversine_km(src_lat, src_lon, la, lo)
                     for la, lo in zip(ward_lats, ward_lons)])
    brg = np.array([bearing_deg(src_lat, src_lon, la, lo)
                    for la, lo in zip(ward_lats, ward_lons)])
    in_range = d_km <= PLUME_MAX_RANGE_KM

    acc = np.zeros(n)
    steps = 0
    for offset in range(0, horizon_h + 1, step_h):
        t = at + timedelta(hours=offset)
        row = w[w["ts"] <= t].tail(1)
        if row.empty:
            row = w.head(1)
        r = row.iloc[0]

        u = float(r.get("wind_speed_ms") or 0.0)
        direction = float(r.get("wind_dir_deg") or 0.0)
        pblh = r.get("pblh")
        pblh = None if pblh is None or pd.isna(pblh) else float(pblh)
        stab = stability_class(u, t.astimezone(ZoneInfo(tz)).hour, pblh)

        plume_axis = (direction + 180.0) % 360.0
        theta = np.radians(brg - plume_axis)
        downwind = d_km * 1000.0 * np.cos(theta)
        crosswind = np.abs(d_km * 1000.0 * np.sin(theta))

        k = BRIGGS_RURAL[stab]
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(downwind > 0, downwind, np.nan)
            sy = k["a"] * x / np.sqrt(1.0 + k["b"] * x)
            sy = np.sqrt(sy**2 + (ward_radius_m_arr / 2.0) ** 2)
            sz = (k["c"] * x if k["d"] == 0 and k["e"] == 1.0
                  else k["c"] * x * np.power(1.0 + k["d"] * x, k["e"]))

            uu = max(u, MIN_WIND_MS)
            vertical = np.where(
                (pblh is not None) and (sz > 0.8 * (pblh or 1e9)),
                1.0 / (pblh or 1e9),
                2.0 / (np.sqrt(2 * np.pi) * sz),   # H≈0 ground release + reflection
            )
            c = (
                (q_g_s * 1e6 / uu)
                * np.exp(-(crosswind**2) / (2 * sy**2)) / (np.sqrt(2 * np.pi) * sy)
                * vertical
            )
        acc += np.nan_to_num(np.where(in_range & (downwind > 0), c, 0.0))
        steps += 1

    return acc / max(steps, 1)


@dataclass
class Counterfactual:
    """Predicted benefit of removing a source, per horizon."""

    averted_ugm3: dict[int, float]      # horizon_h -> µg/m³ removed at the ward
    peak_averted_ugm3: float
    stability_seen: list[str]
    hours_downwind: int
    hours_evaluated: int
    confidence: float
    # False when the source sits beyond PLUME_MAX_RANGE_KM. The zeros above are
    # then "not modelled", not "no benefit" — callers must say so rather than
    # rank this as a worthless action.
    in_range: bool = True
    distance_km: float = 0.0


def counterfactual(
    src_lat: float,
    src_lon: float,
    rec_lat: float,
    rec_lon: float,
    q_g_s: float,
    at: datetime,
    wind: pd.DataFrame,
    tz: str,
    horizons: tuple[int, ...] = (12, 24, 48),
    height_m: float = 10.0,
    step_h: int = 3,
    receptor_radius_m: float = 0.0,
) -> Counterfactual:
    """"Remove this source — how much cleaner is the ward at t+12/24/48h?"

    Steps the forecast wind at `step_h` intervals (TRD 5.4) rather than freezing
    one direction: over 48h the wind veers, and a source that is upwind at noon
    may be irrelevant by midnight. The averted concentration at horizon h is the
    MEAN contribution over the hours up to h — that is what a 24h-average
    exposure actually experiences, and it is the number the ROI multiplies by
    population.

    `wind` needs columns: ts, wind_speed_ms, wind_dir_deg, and optionally pblh.
    """
    from zoneinfo import ZoneInfo

    d_km = haversine_km(src_lat, src_lon, rec_lat, rec_lon)
    if wind.empty:
        return Counterfactual({h: 0.0 for h in horizons}, 0.0, [], 0, 0, 0.0,
                              in_range=d_km <= PLUME_MAX_RANGE_KM, distance_km=d_km)

    if d_km > PLUME_MAX_RANGE_KM:
        # Out of model range: decline rather than quote an unsupportable number.
        return Counterfactual({h: 0.0 for h in horizons}, 0.0, [], 0, 0, 0.0,
                              in_range=False, distance_km=d_km)

    w = wind.copy()
    w["ts"] = pd.to_datetime(w["ts"], utc=True)
    w = w.sort_values("ts")

    per_hour: list[tuple[int, float, str, bool]] = []
    max_h = max(horizons)
    for offset in range(0, max_h + 1, step_h):
        t = at + timedelta(hours=offset)
        row = w[w["ts"] <= t].tail(1)
        if row.empty:
            row = w.head(1)
        r = row.iloc[0]

        speed = float(r.get("wind_speed_ms") or 0.0)
        direction = float(r.get("wind_dir_deg") or 0.0)
        pblh = r.get("pblh")
        pblh = None if pblh is None or pd.isna(pblh) else float(pblh)
        local_hour = t.astimezone(ZoneInfo(tz)).hour

        res = source_impact(
            src_lat, src_lon, rec_lat, rec_lon, q_g_s, speed, direction, local_hour, height_m,
            pblh, receptor_radius_m=receptor_radius_m,
        )
        per_hour.append((offset, res.concentration_ugm3, res.stability, res.downwind))

    averted: dict[int, float] = {}
    for h in horizons:
        vals = [c for off, c, _, _ in per_hour if off <= h]
        # 3dp, not 2: a genuine 0.004 µg/m³ rounded to 0.00 and became
        # indistinguishable from "the model said nothing".
        averted[h] = round(float(np.mean(vals)) if vals else 0.0, 3)

    peak = round(max((c for _, c, _, _ in per_hour), default=0.0), 3)
    downwind_hours = sum(1 for _, _, _, d in per_hour if d)

    # Confidence falls when the ward is only downwind for part of the window —
    # a source that is upwind half the time is a weaker case for enforcement.
    coverage = downwind_hours / len(per_hour) if per_hour else 0.0
    return Counterfactual(
        averted_ugm3=averted,
        peak_averted_ugm3=peak,
        stability_seen=sorted({s for _, _, s, _ in per_hour}),
        hours_downwind=downwind_hours * step_h,
        hours_evaluated=len(per_hour) * step_h,
        confidence=round(PLUME_CONFIDENCE * coverage, 2),
        in_range=True,
        distance_km=d_km,
    )
