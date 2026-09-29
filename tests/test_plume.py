"""Gaussian plume tests (TRD §10: "concentration decreases downwind; mass
conservation sanity; stability class selection").

The plume decides how many µg/m³ an enforcement action averts, which is the
numerator of every ROI ranking. A factor-of-two error here silently reorders the
leaderboard and sends inspectors to the wrong site — and would look completely
plausible on screen. So the physics is pinned against analytic truth.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vayu_core.dispersion.gaussian_plume import (
    BRIGGS_RURAL,
    FRP_TO_Q_PM25,
    PLUME_CONFIDENCE,
    STABILITY_CLASSES,
    concentration,
    counterfactual,
    q_from_frp,
    sigma_y,
    sigma_z,
    source_impact,
    stability_class,
)

Q = 10.0        # g/s
U = 3.0         # m/s


# ---- dispersion coefficients ------------------------------------------------

@pytest.mark.parametrize("stab", STABILITY_CLASSES)
def test_sigmas_grow_with_distance(stab):
    for x in (100, 500, 2000, 10000):
        assert sigma_y(x, stab) > sigma_y(x / 2, stab)
        assert sigma_z(x, stab) > sigma_z(x / 2, stab)


@pytest.mark.parametrize("stab", STABILITY_CLASSES)
def test_sigmas_are_zero_at_the_source_and_never_negative(stab):
    assert sigma_y(0, stab) == 0.0
    assert sigma_z(0, stab) == 0.0
    assert sigma_y(-5, stab) == 0.0


def test_unstable_air_disperses_faster_than_stable_air():
    """Class A (hot, convective) must spread a plume far more than class F
    (a winter inversion). If this inverts, every stagnation episode is
    under-predicted — the exact case VAYU exists for."""
    x = 2000.0
    assert sigma_z(x, "A") > sigma_z(x, "D") > sigma_z(x, "F")
    assert sigma_y(x, "A") > sigma_y(x, "D") > sigma_y(x, "F")


# ---- the plume equation -----------------------------------------------------

def test_concentration_decreases_downwind():
    """TRD §10's headline requirement."""
    prev = float("inf")
    for x in (200, 500, 1000, 2000, 5000, 10000):
        c = concentration(Q, U, x, 0.0, "D", height_m=0.0)
        assert c < prev, f"concentration rose at {x} m"
        prev = c
    assert prev > 0


def test_concentration_decreases_crosswind():
    on_axis = concentration(Q, U, 1000, 0.0, "D")
    off_axis = concentration(Q, U, 1000, 200.0, "D")
    far_off = concentration(Q, U, 1000, 800.0, "D")
    assert on_axis > off_axis > far_off


def test_upwind_and_zero_source_contribute_nothing():
    assert concentration(Q, U, -500, 0.0, "D") == 0.0
    assert concentration(0.0, U, 500, 0.0, "D") == 0.0


def test_stronger_wind_dilutes():
    """C ∝ 1/u — doubling the wind must halve the concentration."""
    slow = concentration(Q, 2.0, 1000, 0.0, "D")
    fast = concentration(Q, 4.0, 1000, 0.0, "D")
    assert fast == pytest.approx(slow / 2, rel=1e-6)


def test_concentration_is_linear_in_emission_rate():
    """Doubling Q doubles C — the property the counterfactual relies on."""
    a = concentration(Q, U, 1000, 0.0, "D")
    b = concentration(2 * Q, U, 1000, 0.0, "D")
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_ground_reflection_doubles_a_ground_level_source():
    """A source at H=0 measured at z=0 gets both the direct and the reflected
    term, which coincide. Dropping reflection would halve every averted-µg/m³
    claim in the leaderboard."""
    x, stab = 1000.0, "D"
    sy, sz = sigma_y(x, stab), sigma_z(x, stab)
    analytic_no_reflection = (
        (Q * 1e6 / U)
        * (1.0 / (math.sqrt(2 * math.pi) * sy))
        * (1.0 / (math.sqrt(2 * math.pi) * sz))
    )
    got = concentration(Q, U, x, 0.0, stab, height_m=0.0, receptor_z_m=0.0)
    assert got == pytest.approx(2 * analytic_no_reflection, rel=1e-6)


def test_matches_the_closed_form_gaussian():
    """Full analytic check of C(x,y,z) for an elevated source."""
    x, y, H, stab = 1500.0, 120.0, 40.0, "C"
    sy, sz = sigma_y(x, stab), sigma_z(x, stab)
    expected = (
        (Q * 1e6 / U)
        * math.exp(-(y**2) / (2 * sy**2)) / (math.sqrt(2 * math.pi) * sy)
        * (
            math.exp(-((0 - H) ** 2) / (2 * sz**2))
            + math.exp(-((0 + H) ** 2) / (2 * sz**2))
        ) / (math.sqrt(2 * math.pi) * sz)
    )
    got = concentration(Q, U, x, y, stab, height_m=H, receptor_z_m=0.0)
    assert got == pytest.approx(expected, rel=1e-9)


def test_elevated_source_is_cleaner_at_the_ground_nearby():
    """A tall stack puts less at ground level close in than a ground release —
    the reason fires (buoyant, lofted) and construction dust are modelled at
    different heights."""
    ground = concentration(Q, U, 500, 0.0, "D", height_m=0.0)
    elevated = concentration(Q, U, 500, 0.0, "D", height_m=80.0)
    assert elevated < ground


def test_mass_conservation_across_the_crosswind_profile():
    """Sanity check from TRD §10.

    Integrating C·u over the crosswind plane at any downwind distance must
    return the emission rate Q — the plume moves mass, it does not create or
    destroy it. Checked at two distances: if σ scaling were wrong, the recovered
    Q would drift with x.
    """
    stab = "D"
    for x in (1000.0, 4000.0):
        ys = np.linspace(-4000, 4000, 4001)
        zs = np.linspace(0, 3000, 1501)
        # Integrate the full plane; ground reflection means we integrate z>=0
        # only and the reflected image accounts for the rest.
        total = 0.0
        for z in zs:
            c = np.array([
                concentration(Q, U, x, float(y), stab, height_m=0.0, receptor_z_m=float(z))
                for y in ys[::40]
            ])
            total += np.trapezoid(c, ys[::40]) * (zs[1] - zs[0])
        recovered_q_g_s = total * U / 1e6  # µg/m³ -> g
        assert recovered_q_g_s == pytest.approx(Q, rel=0.05), (
            f"mass not conserved at {x} m: recovered {recovered_q_g_s:.2f} g/s vs {Q}"
        )


def test_mixing_height_cap_traps_the_plume():
    """Once σz fills the boundary layer the plume is uniformly mixed to the
    inversion — ignoring this under-states exactly the trapped-air episodes."""
    free = concentration(Q, U, 20000, 0.0, "D", mixing_height_m=None)
    capped = concentration(Q, U, 20000, 0.0, "D", mixing_height_m=120.0)
    assert capped > free, "a low inversion must concentrate, not dilute"


# ---- stability classification ----------------------------------------------

def test_stability_class_selection():
    # Calm sunny afternoon -> strongly unstable.
    assert stability_class(1.0, 13) == "A"
    # Calm clear night -> strongly stable.
    assert stability_class(1.0, 2) == "F"
    # Windy -> neutral, day or night.
    assert stability_class(8.0, 13) == "D"
    assert stability_class(8.0, 2) == "D"


def test_a_collapsed_boundary_layer_forces_stable_class():
    """Delhi's winter mornings: 100 m inversion at 10am. The clock says 'day',
    the physics says 'trapped'. PBLH must win."""
    assert stability_class(1.5, 10, pblh_m=90) == "F"
    assert stability_class(1.5, 10, pblh_m=2000) == "A"


@pytest.mark.parametrize("hour", range(0, 24, 3))
def test_stability_is_always_a_valid_class(hour):
    for u in (0.2, 1.5, 4.0, 12.0):
        assert stability_class(u, hour) in BRIGGS_RURAL


# ---- emission strength ------------------------------------------------------

def test_q_from_frp_uses_the_published_factors():
    """Wooster 2005 (0.368 kg/s per MW) x Andreae & Merlet (6.3 g PM2.5/kg)."""
    assert q_from_frp(1.0) == pytest.approx(FRP_TO_Q_PM25, rel=1e-9)
    assert q_from_frp(100.0) == pytest.approx(100 * FRP_TO_Q_PM25, rel=1e-9)
    assert q_from_frp(0.0) == 0.0
    assert q_from_frp(-5.0) == 0.0  # a negative FRP is bad data, not a sink


def test_frp_factor_is_physically_sensible():
    # A 50 MW stubble cluster should emit on the order of 100 g/s PM2.5.
    q = q_from_frp(50.0)
    assert 50 < q < 250, f"implausible emission rate {q:.0f} g/s"


# ---- geometry ---------------------------------------------------------------

def test_source_impact_is_zero_when_the_ward_is_upwind():
    """Wind FROM the north-west: a ward to the NORTH-WEST of a fire is upwind of
    it and cannot be affected."""
    # fire at 28.6/77.0; ward north-west of it
    res = source_impact(
        src_lat=28.60, src_lon=77.00, rec_lat=28.90, rec_lon=76.70,
        q_g_s=Q, wind_speed_ms=4.0, wind_dir_from_deg=315.0, local_hour=12,
    )
    assert res.downwind is False
    assert res.concentration_ugm3 == 0.0


def test_source_impact_is_positive_when_the_ward_is_downwind():
    """Same wind, ward to the SOUTH-EAST of the fire — directly downwind."""
    res = source_impact(
        src_lat=28.90, src_lon=76.70, rec_lat=28.60, rec_lon=77.00,
        q_g_s=Q, wind_speed_ms=4.0, wind_dir_from_deg=315.0, local_hour=12,
    )
    assert res.downwind is True
    assert res.concentration_ugm3 > 0


# ---- counterfactual ---------------------------------------------------------

def _wind(speed=4.0, direction=315.0, hours=60, pblh=800.0) -> pd.DataFrame:
    ts = pd.date_range("2025-11-03T00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {"ts": ts, "wind_speed_ms": speed, "wind_dir_deg": direction, "pblh": pblh}
    )


def test_counterfactual_reports_every_horizon_and_a_peak():
    at = pd.Timestamp("2025-11-03T06:00Z")
    cf = counterfactual(
        src_lat=28.90, src_lon=76.70, rec_lat=28.60, rec_lon=77.00,
        q_g_s=Q, at=at, wind=_wind(), tz="Asia/Kolkata",
    )
    assert set(cf.averted_ugm3) == {12, 24, 48}
    assert all(v >= 0 for v in cf.averted_ugm3.values())
    assert cf.peak_averted_ugm3 >= max(cf.averted_ugm3.values())
    assert 0 <= cf.confidence <= PLUME_CONFIDENCE


def test_counterfactual_is_zero_for_a_permanently_upwind_source():
    at = pd.Timestamp("2025-11-03T06:00Z")
    cf = counterfactual(
        src_lat=28.60, src_lon=77.00, rec_lat=28.90, rec_lon=76.70,  # ward upwind
        q_g_s=Q, at=at, wind=_wind(), tz="Asia/Kolkata",
    )
    assert cf.peak_averted_ugm3 == 0.0
    assert cf.hours_downwind == 0
    assert cf.confidence == 0.0, "an upwind source must not be recommended for enforcement"


def test_confidence_scales_with_how_long_the_ward_stays_downwind():
    at = pd.Timestamp("2025-11-03T06:00Z")
    always = counterfactual(
        28.90, 76.70, 28.60, 77.00, Q, at, _wind(), "Asia/Kolkata"
    )
    assert always.hours_downwind > 0
    # Full coverage caps at the plume model's own confidence, never above.
    assert always.confidence <= PLUME_CONFIDENCE


def test_counterfactual_survives_an_empty_wind_frame():
    at = pd.Timestamp("2025-11-03T06:00Z")
    cf = counterfactual(28.9, 76.7, 28.6, 77.0, Q, at, pd.DataFrame(), "Asia/Kolkata")
    assert cf.peak_averted_ugm3 == 0.0 and cf.confidence == 0.0
