"""Difference-in-differences verification (TRD 5.6, PRD E1).

This is the module that marks VAYU's own homework in public. Every test here is
about a way it could quietly flatter itself: crediting the weather to the order,
reporting a confidence interval narrower than the data supports, or picking
controls that make the result look good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vayu_core.verification.did import (
    MIN_POST_HOURS,
    PCT_REALIZED_CLAMP,
    Pending,
    Verification,
    pick_controls,
    verify,
)

EXECUTED = pd.Timestamp("2025-11-03T06:00Z")
NOW = EXECUTED + pd.Timedelta(hours=48)


def _hourly(spec: dict[str, tuple[float, float]], noise: float = 0.0) -> pd.DataFrame:
    """Build ward-hourly PM2.5. spec: ward_id -> (pre_level, post_level)."""
    rng = np.random.default_rng(7)
    rows = []
    pre = pd.date_range(EXECUTED - pd.Timedelta(days=7), EXECUTED, freq="h", inclusive="left", tz="UTC")
    post = pd.date_range(EXECUTED, EXECUTED + pd.Timedelta(hours=48), freq="h", inclusive="left", tz="UTC")
    for wid, (a, b) in spec.items():
        for ts in pre:
            rows.append({"ward_id": wid, "ts": ts, "pm25": a + rng.normal(0, noise)})
        for ts in post:
            rows.append({"ward_id": wid, "ts": ts, "pm25": b + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def _wards(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"ward_id": w, "name": w, "centroid_lat": 28.60 + i * 0.02, "centroid_lon": 77.10 + i * 0.02,
         "population": 57_889, "area_km2": 12.0}
        for i, w in enumerate(ids)
    ])


# ---- the core estimate ------------------------------------------------------

def test_weather_alone_is_not_credited_to_the_order():
    """The whole reason DiD exists.

    Target and controls both fall 40 µg/m³ — the wind picked up city-wide. A
    before/after comparison would report a 40 µg/m³ win. DiD must report zero.
    """
    df = _hourly({"T": (200, 160), "C1": (200, 160), "C2": (200, 160), "C3": (200, 160)})
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=10.0, now=NOW)
    assert isinstance(v, Verification)
    assert v.observed_reduction == pytest.approx(0.0, abs=0.01), (
        "a city-wide weather improvement was credited to the intervention"
    )
    assert v.pct_realized == pytest.approx(0.0, abs=0.1)


def test_a_real_effect_is_recovered():
    """Controls fall 10, target falls 30 → the order bought the extra 20."""
    df = _hourly({"T": (200, 170), "C1": (200, 190), "C2": (200, 190), "C3": (200, 190)})
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=20.0, now=NOW)
    assert v.observed_reduction == pytest.approx(20.0, abs=0.01)
    assert v.pct_realized == pytest.approx(100.0, abs=0.5)


def test_a_reduction_is_reported_as_a_positive_number():
    """Sign convention: the whole pipeline says "µg/m³ averted", so an
    improvement is positive. A flipped sign would report every success as a
    failure and vice versa."""
    df = _hourly({"T": (200, 150), "C1": (200, 200), "C2": (200, 200), "C3": (200, 200)})
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=50.0, now=NOW)
    assert v.observed_reduction > 0
    assert v.target_post < v.target_pre


def test_a_worsening_ward_reports_a_negative_reduction():
    """An order that coincided with things getting worse must say so."""
    df = _hourly({"T": (200, 220), "C1": (200, 200), "C2": (200, 200), "C3": (200, 200)})
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=20.0, now=NOW)
    assert v.observed_reduction < 0
    assert v.pct_realized == PCT_REALIZED_CLAMP[0], "a negative result must clamp to 0%, not go negative"


def test_pct_realized_is_clamped_at_the_top():
    """Beyond 150% the ratio is describing a coincident wind shift, not the order."""
    df = _hourly({"T": (200, 100), "C1": (200, 200), "C2": (200, 200), "C3": (200, 200)})
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=1.0, now=NOW)
    assert v.pct_realized == PCT_REALIZED_CLAMP[1]


# ---- the confidence interval ------------------------------------------------

def test_ci_brackets_the_point_estimate():
    df = _hourly({"T": (200, 170), "C1": (200, 190), "C2": (200, 190), "C3": (200, 190)}, noise=8.0)
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=20.0, now=NOW)
    assert v.ci_low <= v.observed_reduction <= v.ci_high
    assert v.ci_low < v.ci_high, "bounds are inverted — the sign flip broke their order"


def test_a_null_effect_is_not_called_significant():
    """No effect + noise → the interval must span zero, and we must not claim a win."""
    df = _hourly({"T": (200, 200), "C1": (200, 200), "C2": (200, 200), "C3": (200, 200)}, noise=15.0)
    v = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, predicted_reduction=20.0, now=NOW)
    assert v.ci_low < 0 < v.ci_high
    assert v.significant is False


def test_noisier_data_widens_the_interval():
    """The block bootstrap must actually respond to noise. If it didn't, the CI
    would be decoration."""
    quiet = verify("X", "T", ["C1"], _hourly({"T": (200, 170), "C1": (200, 190)}, noise=2.0),
                   EXECUTED, 20.0, NOW)
    loud = verify("X", "T", ["C1"], _hourly({"T": (200, 170), "C1": (200, 190)}, noise=25.0),
                  EXECUTED, 20.0, NOW)
    assert (loud.ci_high - loud.ci_low) > (quiet.ci_high - quiet.ci_low)


def test_the_verdict_is_reproducible():
    """A verdict that moved between page loads would be indefensible."""
    df = _hourly({"T": (200, 170), "C1": (200, 190), "C2": (200, 190), "C3": (200, 190)}, noise=10.0)
    a = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, 20.0, NOW)
    b = verify("X", "T", ["C1", "C2", "C3"], df, EXECUTED, 20.0, NOW)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


# ---- refusing to answer -----------------------------------------------------

def test_too_early_to_judge_returns_pending():
    """App Flow §4.1: a verdict needs >=40h of post data. Six hours of readings
    would be a coin flip dressed as a measurement."""
    df = _hourly({"T": (200, 170), "C1": (200, 190)})
    p = verify("X", "T", ["C1"], df, EXECUTED, 20.0, EXECUTED + pd.Timedelta(hours=6))
    assert isinstance(p, Pending)
    assert p.hours_elapsed == 6
    assert p.hours_remaining == MIN_POST_HOURS - 6


def test_no_controls_means_no_verdict():
    """Without controls the order cannot be separated from the weather, and we
    say that rather than reporting the before/after difference."""
    df = _hourly({"T": (200, 170)})
    v = verify("X", "T", [], df, EXECUTED, 20.0, NOW)
    assert np.isnan(v.observed_reduction)
    assert v.note and "control" in v.note.lower()


def test_missing_readings_do_not_produce_a_confident_number():
    empty = pd.DataFrame(columns=["ward_id", "ts", "pm25"])
    v = verify("X", "T", ["C1"], empty, EXECUTED, 20.0, NOW)
    assert np.isnan(v.observed_reduction)
    assert v.note


# ---- control selection ------------------------------------------------------

def test_controls_are_chosen_on_pre_period_behaviour_only():
    """The failure that would make every number here worthless.

    C_BAD tracks the target badly before the intervention but would flatter the
    result after it. C_GOOD tracks it closely before. Selection must prefer
    C_GOOD — it cannot be allowed to see the post period.
    """
    df = _hourly({
        "T": (200, 170),
        "C_GOOD": (198, 500),   # great pre-match, absurd post — must still win
        "C_BAD": (50, 170),     # terrible pre-match, perfect post
    })
    wards = _wards(["T", "C_GOOD", "C_BAD"])
    picked = pick_controls("T", wards, df, EXECUTED, source_lat=29.5, source_lon=77.5, n=1)
    assert picked == ["C_GOOD"]


def test_controls_inside_the_plume_are_excluded():
    """A ward next to the source is treated by the same action; using it as a
    control would absorb the effect and bias the estimate toward zero."""
    df = _hourly({"T": (200, 170), "NEAR": (200, 190), "FAR": (200, 190)})
    wards = pd.DataFrame([
        {"ward_id": "T", "name": "T", "centroid_lat": 28.60, "centroid_lon": 77.10,
         "population": 57_889, "area_km2": 12.0},
        # 1 km from the source — inside the plume.
        {"ward_id": "NEAR", "name": "NEAR", "centroid_lat": 28.70, "centroid_lon": 77.10,
         "population": 57_889, "area_km2": 12.0},
        {"ward_id": "FAR", "name": "FAR", "centroid_lat": 28.55, "centroid_lon": 77.05,
         "population": 57_889, "area_km2": 12.0},
    ])
    picked = pick_controls("T", wards, df, EXECUTED, source_lat=28.705, source_lon=77.10, n=2)
    assert "NEAR" not in picked
    assert "FAR" in picked


def test_the_target_is_never_its_own_control():
    df = _hourly({"T": (200, 170), "C1": (200, 190)})
    picked = pick_controls("T", _wards(["T", "C1"]), df, EXECUTED, 29.5, 77.5, n=3)
    assert "T" not in picked
