"""Audit log — the record behind the Agent Activity drawer (PRD F1).

VAYU's pitch is that a human can trust an automated enforcement recommendation.
That only holds if every automated step is inspectable after the fact: what each
agent decided, why, and how sure it was. This module is the single writer, so the
drawer, the SSE stream, and the dossier all read one consistent history.

Reasoning text is deterministic here (a factual template). TRD 8 allows an LLM to
summarise it when a key is present, but the audit trail must never depend on a
network call — an unexplained decision is worse than a plainly-worded one.
"""

from __future__ import annotations

from datetime import datetime

from loguru import logger

from vayu_core.config import get_settings
from vayu_core.db import write_conn

# The automated actors in the cascade (TRD 7). Kept as a set so the drawer can
# colour/filter by a known vocabulary rather than free text.
AGENTS = ("forecaster", "attributor", "enforcer", "herald", "inspector", "verifier")


def record(
    agent: str,
    decision: str,
    *,
    trigger: str = "",
    reasoning: str = "",
    confidence: float | None = None,
    duration_ms: int | None = None,
    inputs_hash: str = "",
    ts: datetime | None = None,
) -> None:
    """Append one entry. Never raises into the caller — a failed audit write must
    not sink the pipeline step it was recording."""
    at = ts or get_settings().now()
    try:
        with write_conn() as con:
            con.execute(
                """INSERT INTO audit_log (ts, agent, trigger, inputs_hash, decision,
                                          reasoning, confidence, duration_ms)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [at, agent, trigger, inputs_hash, decision, reasoning, confidence, duration_ms],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"audit write failed ({agent}: {decision[:40]}): {exc}")
