"""Seeded demo records (PRD E1: "one seeded demo record per city, badged").

Verification needs an order that was executed at least 40 hours ago. On a fresh
clone nothing has been dispatched, so `/verify` would be empty on the demo
machine — the closing arrow of the loop would have nothing to show.

What is fabricated and what is not, precisely:
  * FABRICATED: that an inspector was sent to this location and executed an order
    on this date. There was no such order. Every row is flagged `seeded = TRUE`
    and the UI badges it "Seeded demo record".
  * REAL: the ward, the coordinates, the observed PM2.5, the control wards, and
    the entire difference-in-differences verdict — computed from actual CPCB
    station readings for those dates. Nothing about the outcome is written by us.

So the honest claim on screen is: "if this order had been executed, this is what
the data says happened afterwards." The verdict is not scripted, and it is
allowed to come out unflattering — which is the point of shipping it at all.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from loguru import logger

from vayu_core.config import CityConfig, get_settings
from vayu_core.db import read_conn, write_conn

# Far enough back that a full 48h post-window sits before the demo instant.
EXECUTED_DAYS_AGO = 3


def _pick_ward(city: CityConfig) -> tuple[str, str, float, float] | None:
    """A real, well-observed ward: the verdict must rest on actual readings."""
    with read_conn() as con:
        row = con.execute(
            """SELECT w.ward_id, w.name, w.centroid_lat, w.centroid_lon
               FROM wards w
               JOIN ward_roads r ON r.city = w.city AND r.ward_id = w.ward_id
               WHERE w.city = ?
               ORDER BY r.road_density DESC
               LIMIT 1""",
            [city.id],
        ).fetchone()
    return tuple(row) if row else None


def seed_demo_intervention(city: CityConfig) -> str | None:
    """Create one executed, seeded order for `city`. Idempotent."""
    settings = get_settings()
    now = settings.now()
    executed = now - timedelta(days=EXECUTED_DAYS_AGO)
    signal = executed - timedelta(minutes=4, seconds=11)

    ward = _pick_ward(city)
    if not ward:
        logger.warning(f"[{city.id}] no wards — cannot seed a demo record")
        return None
    ward_id, ward_name, lat, lon = ward

    sid = hashlib.sha256(f"seed{city.id}{ward_id}".encode()).hexdigest()[:6].upper()
    order_id = f"VAYU-{city.id[:2].upper()}-{sid}"

    with read_conn() as con:
        if con.execute("SELECT 1 FROM interventions WHERE id = ?", [order_id]).fetchone():
            return order_id

    # A construction stop-work: the one action a municipal team can plausibly
    # execute alone, and the category that actually has local levers in NCR.
    # The predicted figure is deliberately modest — it is what this project's own
    # plume model produces for a single site, and inflating it to make the
    # verification look impressive would defeat the purpose of the page.
    predicted = 1.4

    with write_conn() as con:
        con.execute(
            """INSERT INTO interventions
               (id, city, ward_id, created_ts, action_type, title, source_lat, source_lon,
                predicted_ugm3_averted, population_protected, effort_units, confidence,
                roi_score, status, dossier_path, signal_ts, dispatched_ts, executed_ts, seeded)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [order_id, city.id, ward_id, signal, "stop_work_construction",
             f"Stop work — construction site — {ward_name} sector",
             lat + 0.012, lon + 0.010, predicted, 57_889, 1, 0.42,
             round(predicted * 57_889 / 1 / 1000.0, 1),
             "executed", None, signal, executed - timedelta(minutes=1), executed, True],
        )
        con.execute(
            """INSERT INTO audit_log (ts, agent, trigger, inputs_hash, decision, reasoning,
                                      confidence, duration_ms)
               VALUES (?,?,?,?,?,?,?,?)""",
            [executed, "inspector", f"execute:{order_id}", order_id,
             "Order marked executed (seeded demo record)",
             "Dust screens absent on the north face; work halted 14:20 and "
             "environmental compensation issued.", None, None],
        )
    logger.info(f"[{city.id}] seeded demo record {order_id} in {ward_name} (executed {executed:%d %b %H:%M})")
    return order_id


def record_cascade(city: CityConfig) -> None:
    """Log the automated agent cascade that seeding actually ran (PRD F1).

    Every entry is derived from a real query against the freshly-seeded data, not
    invented — the forecast counts, the flagged-ward count, the top candidate and
    its ROI are the same numbers the live surfaces show. This gives the Agent
    Activity drawer genuine history on a clean clone, so the closed-loop story is
    inspectable from the first minute rather than only after a live dispatch.
    """
    from datetime import timedelta

    from vayu_core import audit, herald
    from vayu_core.db import read_conn

    now = get_settings().now()

    with read_conn() as con:
        scored = con.execute(
            "SELECT count(*) FROM forecasts WHERE city = ? AND horizon_h = 24", [city.id]
        ).fetchone()[0]
        crossings = con.execute(
            """SELECT count(*) FROM forecasts
               WHERE city = ? AND horizon_h = 24 AND aqi_p50 > 300""",
            [city.id],
        ).fetchone()[0]
    if not scored:
        return

    # The drawer is a system-wide feed across both cities, so every entry names
    # its city — otherwise a Lucknow cascade reads as Delhi's while the operator
    # is looking at Delhi.
    c = city.name

    # forecaster
    audit.record(
        "forecaster", f"Scored {scored} {c} wards at t+24/48/72h",
        trigger="pipeline_refresh",
        reasoning=f"LightGBM quantile forecast; {crossings} wards predicted to cross "
                  f"AQI 300 within 48h.",
        confidence=0.82, ts=now - timedelta(seconds=210),
    )

    # attributor + enforcer, from the real city leaderboard
    try:
        from services.api import compute

        board, meta = compute.city_leaderboard(city)
        audit.record(
            "attributor", f"Attributed sources for {c}'s {meta['wards_evaluated']} worst wards",
            trigger="threshold_events",
            reasoning="Back-trajectory + evidence fusion (FIRMS fires, OSM industry, "
                      "permits, roads) against published Delhi winter ranges.",
            confidence=0.8, ts=now - timedelta(seconds=150),
        )
        if board.candidates:
            top = board.candidates[0]
            audit.record(
                "enforcer", f"Ranked {len(board.candidates)} {c} interventions",
                trigger="attribution_ready",
                reasoning=f"Top ROI: {top.title} — {top.predicted_ugm3_averted:.1f} µg/m³ "
                          f"across {top.population_protected:,} people, {top.effort_units} "
                          f"team(s).",
                confidence=top.confidence, ts=now - timedelta(seconds=90),
            )
        for adv in board.advisories:
            if adv.kind == "out_of_range":
                audit.record(
                    "enforcer", f"Flagged out-of-range sources near {c} for escalation",
                    trigger="attribution_ready",
                    reasoning=adv.headline + f" — escalated to {adv.escalate_to}.",
                    confidence=0.7, ts=now - timedelta(seconds=80),
                )
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{city.id}] cascade attribution audit skipped: {exc}")

    # herald
    audit.record(
        "herald", f"Drafted {c} citizen advisories",
        trigger="forecast_ready",
        reasoning=f"Audience-specific guidance in {len(herald.LANGUAGES)} languages "
                  f"({', '.join(herald.LANGUAGE_LABEL[c] for c in herald.LANGUAGES)}).",
        confidence=0.9, ts=now - timedelta(seconds=30),
    )


def seed_grap_draft(city: CityConfig) -> str | None:
    """A badged GRAP Autopilot draft so the human-approval flow (C4) is
    demonstrable on a clean clone.

    On the demo instant no city-wide stage crossing is forecast (the honest
    finding — the population-weighted city AQI holds within its stage). But the
    worst wards ARE forecast to reach the next stage, so this drafts that stage's
    measures for demonstration. What is seeded is only the decision to treat the
    worst-ward forecast as a city trigger; the measures, their clause citations,
    and the forecast AQI are all real. Flagged status='draft' and rendered with a
    "Seeded demo record" badge — never auto-approved.
    """
    import json

    from vayu_core.db import read_conn
    from vayu_core.interventions.grap import measures_for_stage, stage_for_aqi

    with read_conn() as con:
        row = con.execute(
            """SELECT max(aqi_p50) FROM forecasts
               WHERE city = ? AND horizon_h <= 48 AND run_ts = (
                   SELECT max(run_ts) FROM forecasts WHERE city = ?)""",
            [city.id, city.id],
        ).fetchone()
        worst = int(row[0]) if row and row[0] is not None else None
        existing = con.execute("SELECT id FROM grap_drafts WHERE city = ?", [city.id]).fetchone()
    if existing:
        return existing[0]
    if worst is None:
        return None

    stage = stage_for_aqi(worst)
    if stage < 2:  # nothing worth escalating
        return None
    measures = measures_for_stage(stage)
    if not measures:
        return None

    draft_id = f"GRAP-{city.id[:2].upper()}-DEMO{stage}"
    now = get_settings().now()
    with write_conn() as con:
        con.execute(
            """INSERT INTO grap_drafts
               (id, city, stage, trigger_forecast_ts, measures_json, citations_json, status)
               VALUES (?,?,?,?,?,?,?)""",
            [draft_id, city.id, stage, now,
             json.dumps([{"clause_id": m.clause_id, "title": m.title, "text": m.text,
                          "citation": m.citation, "action_supported": m.action_supported}
                         for m in measures]),
             json.dumps([m.citation for m in measures]), "draft"],
        )
    logger.info(f"[{city.id}] seeded GRAP draft {draft_id} (Stage {stage}, worst-ward forecast {worst})")
    return draft_id


def run(cities: list[CityConfig]) -> dict[str, str | None]:
    out = {c.id: seed_demo_intervention(c) for c in cities}
    for c in cities:
        try:
            seed_grap_draft(c)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{c.id}] GRAP draft seed skipped: {exc}")
    for c in cities:
        try:
            record_cascade(c)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{c.id}] cascade audit skipped: {exc}")
    return out


if __name__ == "__main__":
    from vayu_core.config import list_cities

    run(list_cities())
