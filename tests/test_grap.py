"""GRAP Autopilot (TRD 5.7, PRD C4).

The autopilot proposes NCR-wide construction and truck bans. The tests guard the
two things that make that safe: it only fires on a genuine forecast crossing, and
it drafts the measures the incoming stage actually mandates — never a stage that
isn't in force, and never without a human still holding the trigger.
"""

from __future__ import annotations

import pandas as pd

from vayu_core.config import load_city
from vayu_core.interventions.grap import (
    FORECAST_WINDOW_H,
    build_draft,
    measures_for_stage,
    stage_for_aqi,
)

CITY = load_city("delhi")
AT = pd.Timestamp("2025-11-03T06:00Z")


def test_stage_thresholds_match_caqm():
    assert stage_for_aqi(150) == 0
    assert stage_for_aqi(250) == 1
    assert stage_for_aqi(350) == 2
    assert stage_for_aqi(420) == 3
    assert stage_for_aqi(470) == 4
    assert stage_for_aqi(None) == 0


def test_a_forecast_crossing_drafts_the_incoming_stage():
    """Air is Stage II now, forecast to reach Stage III within 48h → draft the
    Stage III measures."""
    d = build_draft(CITY, current_aqi=350, forecast_series=[(24, 380.0), (48, 420.0)], at=AT)
    assert d is not None
    assert d.current_stage == 2
    assert d.forecast_stage == 3
    assert d.measures, "a crossing draft with no measures is useless"
    assert all(m.citation for m in d.measures), "every measure needs a clause citation"
    assert d.status == "draft", "the autopilot must never self-approve"


def test_no_crossing_stays_silent():
    """Forecast stays within the current stage → no draft. The autopilot must not
    invent a reason to escalate."""
    d = build_draft(CITY, current_aqi=350, forecast_series=[(24, 360.0), (48, 390.0)], at=AT)
    assert d is None


def test_a_forecast_improvement_never_drafts():
    """Air improving from Stage III toward Stage II must not draft anything."""
    d = build_draft(CITY, current_aqi=420, forecast_series=[(24, 380.0), (48, 340.0)], at=AT)
    assert d is None


def test_crossings_beyond_48h_are_ignored():
    """A crossing only at t+72h is outside the action window (TRD 5.7)."""
    d = build_draft(CITY, current_aqi=350, forecast_series=[(24, 360.0), (48, 380.0), (72, 460.0)], at=AT)
    assert d is None


def test_drafted_measures_belong_to_the_crossed_stage_only():
    """A Stage III draft lists Stage III's new measures — not Stage IV's (not in
    force) and not Stage I's (already in force)."""
    d = build_draft(CITY, current_aqi=350, forecast_series=[(48, 430.0)], at=AT)
    assert d is not None
    stage3_ids = {m.clause_id for m in measures_for_stage(3)}
    assert {m.clause_id for m in d.measures} == stage3_ids
    # Stage IV truck ban must not appear in a Stage III draft.
    assert not any("TRUCK" in m.clause_id for m in d.measures)


def test_eta_is_within_the_window():
    d = build_draft(CITY, current_aqi=350, forecast_series=[(24, 410.0), (48, 430.0)], at=AT)
    assert d is not None
    assert d.crossing_eta_h is not None and 0 < d.crossing_eta_h <= FORECAST_WINDOW_H


def test_measures_exist_for_every_stage():
    for s in (1, 2, 3, 4):
        assert measures_for_stage(s), f"no corpus measures for stage {s}"
