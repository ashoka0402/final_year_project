"""Re-derive the fusion SCALE constants and report the IITM DSS cross-check.

`make calibrate` runs this. It exists so the scale constants in fusion.py are an
auditable artefact rather than magic numbers: anyone can re-run it, see the raw
scores, and see exactly how the published ranges map onto them.

What this is and is not:
  * It IS a calibration of five unit-conversion constants against Delhi's
    published winter source-apportionment ranges (IITM DSS / SAFAR), which
    TRD 5.3 explicitly asks us to cross-check against.
  * It is NOT a fit to ground truth. No per-ward "true share of PM2.5 from
    burning" exists anywhere, for any city. Only the city-wide *balance* is
    anchored; all per-ward variation still comes from the evidence — which fires
    are in this ward's cone, how much road it has, which permits sit upwind.

Run it after changing an evidence layer (e.g. switching industry from point
counts to polygon areas, which moved industry from 65% to ~15%).
"""

from __future__ import annotations

import json
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.config import REPO_ROOT, get_settings, load_city
from vayu_core.db import read_conn

from . import fusion as F
from .trajectory import WindField, back_trajectory

N_WARDS = 60
TRAJECTORY_HOURS = 12


def _load(city_id: str):
    with read_conn() as c:
        wards = c.execute(
            "SELECT ward_id,name,centroid_lat,centroid_lon FROM wards WHERE city=?", [city_id]
        ).df()
        wx = c.execute(
            "SELECT * FROM weather_hourly WHERE city=? AND grid='airshed'", [city_id]
        ).df()
        fires = c.execute("SELECT * FROM fires WHERE city=?", [city_id]).df()
        permits = c.execute("SELECT * FROM permits WHERE city=?", [city_id]).df()
        rd = c.execute("SELECT ward_id, road_density FROM ward_roads WHERE city=?", [city_id]).df()
        no2 = c.execute(
            """SELECT avg(value) FROM measurements WHERE city=? AND param='no2'
               AND ts BETWEEN ?::TIMESTAMPTZ - INTERVAL 3 HOUR AND ?::TIMESTAMPTZ""",
            [city_id, get_settings().now(), get_settings().now()],
        ).fetchone()[0]
    if not wx.empty:
        wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    osm_path = REPO_ROOT / "data" / "samples" / f"osm_{city_id}.geojson"
    osm = json.loads(osm_path.read_text()) if osm_path.exists() else None
    return wards, wx, fires, permits, rd, no2, osm


def raw_scores(city_id: str = "delhi", n: int = N_WARDS) -> tuple[dict[str, list[float]], list[float]]:
    """Per-ward raw (unscaled) scores, plus the out-of-city fraction of burning.

    The second value is not optional bookkeeping: `attribute()` subtracts
    identified out-of-city fires from the regional term to stop double-counting
    Punjab smoke. Calibrating on raw scores *without* that guard optimises a
    different function than production runs, and the shares drift (measured:
    regional landed at 13.5% against a 25% target).
    """
    city = load_city(city_id)
    at = get_settings().now()
    wards, wx, fires, permits, rd, no2, osm = _load(city_id)
    if wards.empty or wx.empty:
        raise SystemExit("no data — run `make seed` first")

    field = WindField(city, wx)
    dens = dict(zip(rd.ward_id, rd.road_density)) if not rd.empty else {}
    hour = at.astimezone(ZoneInfo(city.timezone)).hour

    out: dict[str, list[float]] = {k: [] for k in F.CATEGORIES}
    burn_outside: list[float] = []
    for r in wards.sample(min(n, len(wards)), random_state=1).itertuples():
        t = back_trajectory(city, r.ward_id, r.centroid_lat, r.centroid_lon, at, TRAJECTORY_HOURS, field)
        if t.stagnant:
            continue
        cone = F._cone_polygon(t)
        b, _, out_frac = F._score_burning(fires, cone, r.centroid_lat, r.centroid_lon, at, city.bbox)
        i, _ = F._score_industry(osm, cone, r.centroid_lat, r.centroid_lon)
        c, _ = F._score_construction(permits, cone, r.centroid_lat, r.centroid_lon)
        tr, _ = F._score_traffic(dens.get(r.ward_id, 0.0), hour, no2, r.name)
        rg, _ = F._score_regional(t, city.bbox, 1.0)
        for k, v in (
            ("open_burning", b), ("industry", i), ("construction", c),
            ("traffic", tr), ("regional_transport", rg),
        ):
            out[k].append(v)
        burn_outside.append(out_frac)
    return out, burn_outside


def _mean_shares(
    raws: dict[str, list[float]], burn_outside: list[float], scales: dict[str, float]
) -> dict[str, float]:
    """Mean per-ward share under a given set of scales.

    Mirrors `attribute()` exactly, double-count guard included — otherwise the
    calibration optimises something production never computes.
    """
    n = len(next(iter(raws.values())))
    acc = {k: 0.0 for k in raws}
    used = 0
    for i in range(n):
        scaled = {k: raws[k][i] * scales[k] for k in raws}
        # Same guard as fusion.attribute(): imported smoke already named under
        # open_burning must not be counted again as regional transport.
        from_outside = scaled["open_burning"] * burn_outside[i]
        if from_outside > 0 and scaled["regional_transport"] > 0:
            scaled["regional_transport"] = max(0.0, scaled["regional_transport"] - from_outside)
        total = sum(scaled.values())
        if total <= 0:
            continue
        used += 1
        for k in raws:
            acc[k] += scaled[k] / total * 100.0
    return {k: (v / used if used else 0.0) for k, v in acc.items()}


def derive(iterations: int = 40) -> dict[str, float]:
    """Solve for scales whose MEAN PER-WARD SHARE hits the published midpoints.

    Iterative on purpose. Setting scale = target / mean(raw) in one pass does not
    converge, because a share is a *ratio* computed per ward and the mean of
    ratios is not the ratio of means (Jensen). One pass left traffic at 28% and
    construction at 25% against published 15-25% and 5-15%. Multiplicative
    updates on the realised mean share converge in a few dozen steps.
    """
    raws, burn_outside = raw_scores()
    targets = {k: (lo + hi) / 2 for k, (lo, hi) in F.PUBLISHED_DELHI_WINTER_RANGES.items()}

    # Seed from the one-pass estimate, then refine.
    means = {k: float(np.mean(v)) if v else 0.0 for k, v in raws.items()}
    scales = {k: (targets[k] / means[k] if means[k] > 0 else 1.0) for k in F.CATEGORIES}

    for _ in range(iterations):
        got = _mean_shares(raws, burn_outside, scales)
        for k in F.CATEGORIES:
            if got[k] > 1e-6:
                # Damped update: full steps oscillate.
                scales[k] *= (targets[k] / got[k]) ** 0.5

    norm = scales["construction"] / 5.0 if scales["construction"] else 1.0
    return {k: round(v / norm, 3) for k, v in scales.items()}


def cross_check(city_id: str = "delhi") -> list[dict]:
    """Where our shares actually land vs the published ranges (TRD 5.3)."""
    city = load_city(city_id)
    at = get_settings().now()
    wards, wx, fires, permits, rd, no2, osm = _load(city_id)
    field = WindField(city, wx)
    dens = dict(zip(rd.ward_id, rd.road_density)) if not rd.empty else {}

    # Average over EVERY attributed ward, scoring 0 where a category is absent.
    # Averaging only over wards where a category appears inflates it — fusion
    # drops shares below 0.5%, so construction reported 27.5% while its true
    # city-wide mean was ~10%. Published splits are city-wide means including
    # the wards that have none, so the comparison must be like-for-like.
    agg: dict[str, list[float]] = {k: [] for k in F.CATEGORIES}
    for r in wards.sample(min(N_WARDS, len(wards)), random_state=1).itertuples():
        t = back_trajectory(city, r.ward_id, r.centroid_lat, r.centroid_lon, at, TRAJECTORY_HOURS, field)
        a = F.attribute(
            city, r.ward_id, r.name, r.centroid_lat, r.centroid_lon, t, at,
            fires=fires, osm=osm, permits=permits,
            road_density=dens.get(r.ward_id, 0.0), no2=no2,
            regional_pm_proxy=1.0, station_agreement=0.7,
        )
        if not a.categories:
            continue  # stagnant / no evidence: not a zero, an absence
        got = {c.category: c.share_pct for c in a.categories}
        for k in F.CATEGORIES:
            agg[k].append(got.get(k, 0.0))

    rows = []
    for k, (lo, hi) in F.PUBLISHED_DELHI_WINTER_RANGES.items():
        vals = agg.get(k, [0.0])
        mean = float(np.mean(vals))
        rows.append(
            {
                "category": k,
                "vayu_mean_pct": round(mean, 1),
                "vayu_min_pct": round(float(np.min(vals)), 1),
                "vayu_max_pct": round(float(np.max(vals)), 1),
                "published_low_pct": lo,
                "published_high_pct": hi,
                "within_published_range": bool(lo <= mean <= hi),
            }
        )
    return rows


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{level: <7}</level> {message}", level="WARNING")

    print("── deriving SCALE from raw evidence scores ──")
    scales = derive()
    for k, v in scales.items():
        print(f'    "{k}": {v},')

    print("\n── cross-check vs published Delhi winter ranges (IITM DSS / SAFAR) ──")
    print(f"{'category':22s} {'VAYU mean':>10s} {'range':>12s} {'published':>12s}  ok")
    for r in cross_check():
        ok = "✓" if r["within_published_range"] else "✗"
        rng = f"{r['vayu_min_pct']:.0f}-{r['vayu_max_pct']:.0f}%"
        pub = f"{r['published_low_pct']:.0f}-{r['published_high_pct']:.0f}%"
        print(f"  {r['category']:20s} {r['vayu_mean_pct']:9.1f}% {rng:>12s} {pub:>12s}  {ok}")

    out = REPO_ROOT / "docs" / "attribution_crosscheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scales": scales, "cross_check": cross_check()}, indent=2))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
