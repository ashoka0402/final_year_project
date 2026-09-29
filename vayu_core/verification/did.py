"""Difference-in-differences verification (TRD 5.6) — the honesty engine.

The loop VAYU claims is READING → SOURCE → INTERVENTION → ORDER → **VERIFIED
OUTCOME**. This module is the last arrow, and it is the one that can embarrass
us: it is the only place where the product's own prediction gets marked against
reality, in public, after the fact.

The problem it solves: after an order is executed, the ward's PM2.5 falls. Did
the order do that, or did the wind simply pick up? Air quality moves for reasons
that have nothing to do with enforcement, and a before/after comparison would
credit the intervention for every one of them.

Difference-in-differences answers it by subtracting what would have happened
anyway, estimated from control wards that were exposed to the same weather but
not to the action:

    observed = (target_post − target_pre) − mean(control_post − control_pre)

Controls are chosen on their PRE-period behaviour only (TRD 5.6): the 3 wards
whose 7-day AQI tracked the target's most closely, that sit outside the plume,
and that have comparable density. Choosing them on anything after the
intervention would let us pick the controls that flatter the result — which is
exactly the failure this module exists to prevent.

A negative result is a real, publishable outcome here. `pct_realized` near zero
means the order did nothing, and the honest thing is to say so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from vayu_core.geo import haversine_km

# TRD 5.6 windows.
PRE_DAYS = 7                  # how much history the control match looks at
POST_HOURS = 48               # the window an intervention is judged over
MIN_POST_HOURS = 40           # below this the verdict stays pending (App Flow §4.1)
N_CONTROLS = 3
BOOTSTRAP_N = 500
BLOCK_HOURS = 6               # block bootstrap: PM2.5 is autocorrelated hour to hour

# Controls must be far enough away not to be treated by the same plume, but near
# enough to share the weather. Inside the plume they would absorb part of the
# effect and bias the estimate toward zero.
MIN_CONTROL_KM = 8.0
MAX_CONTROL_KM = 30.0

# pct_realized is clamped to this band (TRD 5.6). Beyond 150% the ratio is
# telling us about noise or a coincident wind shift, not about the order.
PCT_REALIZED_CLAMP = (0.0, 150.0)


@dataclass
class Verification:
    intervention_id: str
    method: str
    control_wards: list[str]
    predicted_reduction: float
    observed_reduction: float
    ci_low: float
    ci_high: float
    pct_realized: float
    computed_ts: datetime
    # Everything below is for the chart and the honesty labels, not the maths.
    target_pre: float = 0.0
    target_post: float = 0.0
    control_pre: float = 0.0
    control_post: float = 0.0
    post_hours: int = 0
    significant: bool = False
    note: str | None = None
    series: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["computed_ts"] = self.computed_ts.isoformat()
        return d


@dataclass
class Pending:
    """Not enough post-intervention data yet. A verdict now would be noise."""

    intervention_id: str
    hours_elapsed: int
    hours_required: int = MIN_POST_HOURS

    @property
    def hours_remaining(self) -> int:
        return max(self.hours_required - self.hours_elapsed, 0)

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "status": "pending",
            "hours_elapsed": self.hours_elapsed,
            "hours_required": self.hours_required,
            "hours_remaining": self.hours_remaining,
            "method": "did",
        }


def pick_controls(
    target_ward: str,
    wards: pd.DataFrame,
    ward_hourly: pd.DataFrame,
    executed_ts: datetime,
    source_lat: float,
    source_lon: float,
    n: int = N_CONTROLS,
) -> list[str]:
    """The n wards that best match the target's PRE-period behaviour.

    Selection uses only data from before the intervention. Ranking controls on
    post-period data would be choosing the answer we want — the one failure mode
    that would make every number this module produces worthless.

    `ward_hourly` needs columns: ward_id, ts, pm25.
    """
    if wards.empty or ward_hourly.empty:
        return []

    pre_start = executed_ts - timedelta(days=PRE_DAYS)
    pre = ward_hourly[(ward_hourly["ts"] >= pre_start) & (ward_hourly["ts"] < executed_ts)]
    if pre.empty:
        return []

    tgt = pre[pre["ward_id"] == target_ward].set_index("ts")["pm25"]
    if tgt.empty:
        return []

    trow = wards[wards["ward_id"] == target_ward]
    if trow.empty:
        return []
    t = trow.iloc[0]
    # Wards are delimited to equal population, so density is carried entirely by
    # area: a small ward is a dense one. Matching on it matches urban form.
    t_density = float(t["population"]) / max(float(t.get("area_km2") or 1.0), 0.01)

    scored: list[tuple[float, str]] = []
    for w in wards.itertuples():
        if w.ward_id == target_ward:
            continue

        # Outside the plume, but sharing the weather.
        d_src = haversine_km(source_lat, source_lon, float(w.centroid_lat), float(w.centroid_lon))
        if d_src < MIN_CONTROL_KM:
            continue
        d_tgt = haversine_km(
            float(t.centroid_lat), float(t.centroid_lon),
            float(w.centroid_lat), float(w.centroid_lon),
        )
        if d_tgt > MAX_CONTROL_KM:
            continue

        c = pre[pre["ward_id"] == w.ward_id].set_index("ts")["pm25"]
        joined = pd.concat([tgt, c], axis=1, join="inner").dropna()
        if len(joined) < 24:
            continue  # too little overlap to claim a match

        # Mean absolute pre-period gap: how closely this ward tracked the target.
        aqi_distance = float(np.abs(joined.iloc[:, 0] - joined.iloc[:, 1]).mean())

        area = float(getattr(w, "area_km2", 0.0) or 1.0)
        density = float(w.population) / max(area, 0.01)
        density_penalty = abs(density - t_density) / max(t_density, 1.0)

        # Tracking is what makes a control a control; density is a tiebreak.
        scored.append((aqi_distance + 5.0 * density_penalty, w.ward_id))

    scored.sort()
    return [wid for _, wid in scored[:n]]


def _mean(series: pd.DataFrame, ward_ids: list[str], start: datetime, end: datetime) -> float:
    w = series[
        series["ward_id"].isin(ward_ids) & (series["ts"] >= start) & (series["ts"] < end)
    ]
    return float(w["pm25"].mean()) if not w.empty else float("nan")


def _block_bootstrap_ci(
    target_resid: np.ndarray, control_resid: np.ndarray, n: int = BOOTSTRAP_N
) -> tuple[float, float]:
    """95% CI for the DiD estimate via block bootstrap (TRD 5.6).

    Blocks, not individual hours: PM2.5 is strongly autocorrelated, so resampling
    single hours would treat 48 readings as 48 independent observations and
    report a confidence interval several times too narrow — overstating our
    certainty about the one number we ask to be judged on.
    """
    if len(target_resid) == 0 or len(control_resid) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(42)  # fixed: a verdict must not move between runs

    def blocks(a: np.ndarray) -> list[np.ndarray]:
        return [a[i : i + BLOCK_HOURS] for i in range(0, len(a), BLOCK_HOURS) if len(a[i : i + BLOCK_HOURS])]

    tb, cb = blocks(target_resid), blocks(control_resid)
    if not tb or not cb:
        return float("nan"), float("nan")

    draws = np.empty(n)
    for i in range(n):
        t = np.concatenate([tb[j] for j in rng.integers(0, len(tb), len(tb))])
        c = np.concatenate([cb[j] for j in rng.integers(0, len(cb), len(cb))])
        draws[i] = t.mean() - c.mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def verify(
    intervention_id: str,
    target_ward: str,
    control_wards: list[str],
    ward_hourly: pd.DataFrame,
    executed_ts: datetime,
    predicted_reduction: float,
    now: datetime,
) -> Verification | Pending:
    """Did the order work? Diff-in-diff against the chosen controls.

    Returns Pending until MIN_POST_HOURS of data exist — a verdict drawn from six
    hours of readings would be a coin flip dressed as a measurement.
    """
    hours_elapsed = int((now - executed_ts).total_seconds() // 3600)
    if hours_elapsed < MIN_POST_HOURS:
        return Pending(intervention_id, max(hours_elapsed, 0))

    if not control_wards:
        return Verification(
            intervention_id=intervention_id, method="did", control_wards=[],
            predicted_reduction=predicted_reduction, observed_reduction=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"), pct_realized=float("nan"),
            computed_ts=now, post_hours=hours_elapsed,
            note="No comparable control wards — the effect cannot be separated from the weather.",
        )

    pre_start = executed_ts - timedelta(days=PRE_DAYS)
    post_end = min(executed_ts + timedelta(hours=POST_HOURS), now)

    t_pre = _mean(ward_hourly, [target_ward], pre_start, executed_ts)
    t_post = _mean(ward_hourly, [target_ward], executed_ts, post_end)
    c_pre = _mean(ward_hourly, control_wards, pre_start, executed_ts)
    c_post = _mean(ward_hourly, control_wards, executed_ts, post_end)

    if any(np.isnan(x) for x in (t_pre, t_post, c_pre, c_post)):
        return Verification(
            intervention_id=intervention_id, method="did", control_wards=control_wards,
            predicted_reduction=predicted_reduction, observed_reduction=float("nan"),
            ci_low=float("nan"), ci_high=float("nan"), pct_realized=float("nan"),
            computed_ts=now, post_hours=hours_elapsed,
            note="Insufficient ward readings in the pre or post window.",
        )

    # Sign convention: a REDUCTION is positive, so the whole pipeline speaks the
    # same language as "µg/m³ averted". The raw DiD is a change (negative when
    # things improve), so it is negated exactly once, here.
    did = (t_post - t_pre) - (c_post - c_pre)
    observed_reduction = -did

    post = ward_hourly[(ward_hourly["ts"] >= executed_ts) & (ward_hourly["ts"] < post_end)]
    t_series = post[post["ward_id"] == target_ward].sort_values("ts")["pm25"].to_numpy()
    c_series = (
        post[post["ward_id"].isin(control_wards)]
        .groupby("ts")["pm25"].mean().sort_index().to_numpy()
    )
    lo, hi = _block_bootstrap_ci(t_series - t_pre, c_series - c_pre)
    # Negated with the same convention as the point estimate, so the bounds keep
    # their order after the sign flip.
    ci_low, ci_high = (-hi, -lo) if not np.isnan(lo) else (float("nan"), float("nan"))

    pct = (
        float(np.clip(observed_reduction / predicted_reduction * 100.0, *PCT_REALIZED_CLAMP))
        if predicted_reduction > 0
        else float("nan")
    )
    # Significant only when the interval excludes zero — i.e. we can distinguish
    # the order from the weather.
    significant = bool(not np.isnan(ci_low) and (ci_low > 0 or ci_high < 0))

    v = Verification(
        intervention_id=intervention_id,
        method="did",
        control_wards=control_wards,
        predicted_reduction=round(predicted_reduction, 2),
        observed_reduction=round(observed_reduction, 2),
        ci_low=round(ci_low, 2) if not np.isnan(ci_low) else float("nan"),
        ci_high=round(ci_high, 2) if not np.isnan(ci_high) else float("nan"),
        pct_realized=round(pct, 1) if not np.isnan(pct) else float("nan"),
        computed_ts=now,
        target_pre=round(t_pre, 1),
        target_post=round(t_post, 1),
        control_pre=round(c_pre, 1),
        control_post=round(c_post, 1),
        post_hours=hours_elapsed,
        significant=significant,
        series=_series_for_chart(ward_hourly, target_ward, control_wards, pre_start, post_end),
    )
    logger.info(
        f"verified {intervention_id}: predicted {predicted_reduction:.2f}, "
        f"observed {observed_reduction:.2f} ({pct:.0f}% realized), "
        f"CI [{ci_low:.2f}, {ci_high:.2f}], significant={significant}"
    )
    return v


def _series_for_chart(
    ward_hourly: pd.DataFrame,
    target_ward: str,
    control_wards: list[str],
    start: datetime,
    end: datetime,
) -> dict:
    """Daily target vs synthetic-control series for the DiD chart."""
    w = ward_hourly[(ward_hourly["ts"] >= start) & (ward_hourly["ts"] < end)].copy()
    if w.empty:
        return {}
    w["day"] = pd.to_datetime(w["ts"], utc=True).dt.floor("D")
    t = w[w["ward_id"] == target_ward].groupby("day")["pm25"].mean()
    c = w[w["ward_id"].isin(control_wards)].groupby("day")["pm25"].mean()
    days = sorted(set(t.index) | set(c.index))
    return {
        "days": [d.isoformat() for d in days],
        "target": [None if d not in t.index or pd.isna(t[d]) else round(float(t[d]), 1) for d in days],
        "control": [None if d not in c.index or pd.isna(c[d]) else round(float(c[d]), 1) for d in days],
    }
