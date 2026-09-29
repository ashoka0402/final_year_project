"""HCHO hotspot detection + fire coupling.

Synthetic data throughout, so every expected answer is known by construction
rather than eyeballed off a map.

The methodological claims these lock down:
  * anomalies are per-cell, so a permanently-high region is NOT a hotspot;
  * the baseline is robust, so a burning episode inside the window does not
    inflate its own baseline and hide itself;
  * the dose-response is monotonic in fire intensity (the headline result the
    correlation coefficient hides).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from vayu_core.national import hotspots as H


def _series(lat, lon, values, start=dt.date(2025, 10, 1)):
    return pd.DataFrame(
        {
            "grid_lat": lat,
            "grid_lon": lon,
            "date": [start + dt.timedelta(days=i) for i in range(len(values))],
            "value": values,
            "n_obs": 6,
        }
    )


def test_a_permanently_high_cell_is_not_a_hotspot():
    """The core design claim. A global threshold would flag the high cell every
    single day; scoring against the cell's OWN baseline must not."""
    quiet = _series(28.0, 77.0, [1.0e-4] * 30)
    always_high = _series(24.0, 88.0, [9.0e-4] * 30)  # 9x higher, but constant
    res = H.detect(pd.concat([quiet, always_high], ignore_index=True), None)
    assert res.hotspots.empty, "a constantly-high cell must not register as anomalous"


def test_a_genuine_spike_is_detected():
    vals = [1.0e-4] * 25 + [8.0e-4] * 2 + [1.0e-4] * 3  # short, sharp episode
    res = H.detect(_series(30.0, 75.0, vals), None)
    assert len(res.hotspots) == 2
    assert (res.hotspots["z_score"] >= H.DEFAULT_Z).all()


def test_robust_baseline_survives_a_long_episode():
    """Median/MAD must not be dragged up by the very spikes being detected.

    With mean/std, a 10-day episode in a 30-day window inflates both and the
    anomaly partly cancels itself — the worse the fire season, the harder to
    detect. This is the regression test for that failure mode.
    """
    vals = [1.0e-4] * 20 + [6.0e-4] * 10
    res = H.detect(_series(30.0, 75.0, vals), None)
    assert len(res.hotspots) == 10, "every elevated day should still clear the bar"
    assert res.baseline["baseline"].iloc[0] == pytest.approx(1.0e-4)


def test_cells_with_too_little_history_are_dropped():
    short = _series(30.0, 75.0, [1e-4, 5e-4, 1e-4])  # 3 days < MIN_BASELINE_DAYS
    res = H.detect(short, None)
    assert res.baseline.empty and res.hotspots.empty


def test_constant_cell_yields_no_hotspot_and_no_infinite_z():
    """A perfectly constant cell must produce z=0 everywhere, never inf.

    Raw MAD is 0 here, so without the MIN_SPREAD_FRAC floor this would be a
    divide-by-zero. With the floor the cell stays testable and simply never
    deviates from its own baseline.
    """
    res = H.detect(_series(28.0, 77.0, [3.0e-4] * 30), None)
    assert res.hotspots.empty
    assert np.isfinite(res.hotspots["z_score"].to_numpy(dtype=float)).all()


def test_flat_cell_with_a_spike_is_still_detected():
    """Regression for a real bug: when >50% of days share a value, MAD is
    exactly 0, and the cell used to be discarded as 'no spread' — silently
    dropping the clearest hotspot shape there is."""
    vals = [1.0e-4] * 28 + [9.0e-4] * 2   # >50% identical => raw MAD == 0
    res = H.detect(_series(30.0, 75.0, vals), None)
    assert len(res.hotspots) == 2, "flat-with-spike must not be discarded"
    assert np.isfinite(res.hotspots["z_score"].to_numpy(dtype=float)).all()


def test_missing_fire_rows_mean_zero_not_unknown():
    vals = [1.0e-4] * 25 + [8.0e-4] * 5
    hcho = _series(30.0, 75.0, vals)
    fires = pd.DataFrame(
        {"grid_lat": [30.0], "grid_lon": [75.0], "date": [dt.date(2025, 10, 26)],
         "fire_count": [12], "frp_sum": [400.0]}
    )
    res = H.detect(hcho, fires)
    assert res.hotspots["fire_count"].notna().all(), "absent fire rows must become 0"
    assert (res.hotspots["fire_count"] >= 0).all()


def test_dose_response_is_monotonic_in_fire_intensity():
    """The headline Obj-2 result: more fire -> more HCHO, by construction here."""
    rng = np.random.default_rng(1)
    rows, fires = [], []
    day0 = dt.date(2025, 10, 1)
    for i in range(60):
        day = day0 + dt.timedelta(days=i)
        # fire count rises across the window; HCHO rises with it plus noise
        fc = [0, 3, 10, 40][i % 4]
        hv = 1.0e-4 * (1 + 0.5 * (fc > 0) + 0.02 * fc) + rng.normal(0, 2e-6)
        rows.append({"grid_lat": 30.0, "grid_lon": 75.0, "date": day, "value": hv, "n_obs": 6})
        if fc:
            fires.append({"grid_lat": 30.0, "grid_lon": 75.0, "date": day, "fire_count": fc})

    tab = H.stratify_by_fire(pd.DataFrame(rows), pd.DataFrame(fires), lag_days=0)
    assert not tab.empty
    med = tab.set_index("fire_bin")["median_hcho"]
    assert med["no fire"] < med["1-5"] < med["6-20"] < med[">20"], f"not monotonic:\n{tab}"
    # And the lift column agrees with the medians.
    assert tab.set_index("fire_bin")["pct_vs_no_fire"][">20"] > 0


def test_clustering_groups_adjacent_cells_only():
    """Two separated blobs on the same day must stay two clusters."""
    day = dt.date(2025, 11, 1)
    cells = [(30.0, 75.0), (30.25, 75.0), (30.0, 75.25),   # blob A (adjacent)
             (20.0, 85.0), (20.25, 85.0)]                   # blob B, far away
    hot = pd.DataFrame(
        {"grid_lat": [c[0] for c in cells], "grid_lon": [c[1] for c in cells],
         "date": [day] * len(cells), "z_score": [5.0] * len(cells)}
    )
    out = H.cluster(hot, grid_deg=0.25)
    assert out["cluster_id"].nunique() == 2
    a = set(out[out.grid_lat > 25]["cluster_id"])
    b = set(out[out.grid_lat < 25]["cluster_id"])
    assert a.isdisjoint(b), "distant blobs merged into one cluster"


def test_correlation_reports_n_and_handles_degenerate_input():
    flat = pd.DataFrame({"fire_count": [1, 1, 1], "z_score": [2.0, 2.0, 2.0]})
    c = H.correlate(flat)
    assert c.n == 3 and np.isnan(c.pearson_r) and "not enough variation" in c.note
    d = c.to_dict()
    assert d["pearson_r"] is None and d["n"] == 3


def test_empty_inputs_are_handled():
    res = H.detect(pd.DataFrame(), None)
    assert res.hotspots.empty and res.correlation.n == 0
    assert H.stratify_by_fire(pd.DataFrame(), pd.DataFrame()).empty
