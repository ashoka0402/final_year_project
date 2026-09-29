"""VAYU API — FastAPI application.

All errors are RFC7807 problem+json (TRD §6). The app must start and answer
even with an unseeded database: a fresh clone gets designed empty states, not a
stack trace (PRD non-functional: zero unhandled exceptions in the golden flow).
"""

from __future__ import annotations

import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from vayu_core.config import get_settings

from .routers import (
    attribution, audit, citizen, citizen_reports, cities, corridors, forecast, grap,
    interventions, meta, verification,
)

logger.remove()
logger.add(sys.stderr, format="<level>{level: <7}</level> {message}", level="INFO")

settings = get_settings()


LIVE_REFRESH_MINUTES = 15


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Idempotent CREATE TABLE IF NOT EXISTS — brings older DBs up to the current
    # schema without a reseed.
    from vayu_core.db import init_db

    init_db()
    mode = "DEMO_MODE (bundled data, clock pinned)" if settings.demo_mode else "LIVE"
    logger.info(f"VAYU API up · {mode} · now={settings.now():%Y-%m-%d %H:%M %Z}")

    scheduler = None
    if settings.demo_mode:
        # APScheduler stays off in DEMO_MODE (TRD §4) so a background refresh
        # can never move the ground under a live demo.
        logger.info("scheduler disabled in DEMO_MODE")
    else:
        import datetime as dt

        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        from vayu_core.national.surface_aqi import GROUND_TRUTH_CITIES
        from services.pipeline.live import refresh_live_measurements

        # AsyncIOScheduler runs jobs on its own thread-pool executor, not the
        # event loop — calling refresh_live_measurements() directly here
        # instead would block the event loop (and Cloud Run's readiness
        # check) for as long as the CPCB feed takes, which the smoke test
        # showed can be tens of seconds under rate-limit backoff for
        # delhi_ncr's full-national-feed pagination. next_run_time=now fires
        # the first run through the same non-blocking path instead of
        # duplicating the call inline.
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            refresh_live_measurements, "interval",
            minutes=LIVE_REFRESH_MINUTES, args=[GROUND_TRUTH_CITIES], id="live_cpcb_refresh",
            next_run_time=dt.datetime.now(),
        )
        scheduler.start()
        logger.info(f"live CPCB refresh scheduled every {LIVE_REFRESH_MINUTES} min for {GROUND_TRUTH_CITIES}")

    yield

    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(
    lifespan=lifespan,
    title="VAYU API",
    version="0.1.0",
    description=(
        "Verifiable Airshed Intelligence & Enforcement — "
        "READING → RESPONSIBLE SOURCE → RANKED INTERVENTION → ENFORCEMENT ORDER → VERIFIED OUTCOME"
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Request id + timing on every response (TRD §10: structured logging)."""
    rid = uuid.uuid4().hex[:8]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"[{rid}] {request.method} {request.url.path} failed")
        raise
    took = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{took:.1f}"
    if request.url.path.startswith("/api"):
        logger.info(f"[{rid}] {request.method} {request.url.path} → {response.status_code} in {took:.0f}ms")
    return response


def _problem(status: int, title: str, detail: str | None, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "about:blank", "title": title, "status": status, "detail": detail, "instance": str(request.url.path)},
        media_type="application/problem+json",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    titles = {404: "Not Found", 400: "Bad Request", 422: "Unprocessable Entity", 429: "Too Many Requests"}
    return _problem(exc.status_code, titles.get(exc.status_code, "Error"), str(exc.detail), request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _problem(422, "Unprocessable Entity", str(exc.errors()), request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"unhandled error on {request.url.path}")
    return _problem(500, "Internal Server Error", f"{type(exc).__name__}", request)


app.include_router(meta.router, prefix="/api/v1")
app.include_router(cities.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(attribution.router, prefix="/api/v1")
app.include_router(interventions.router, prefix="/api/v1")
app.include_router(verification.router, prefix="/api/v1")
app.include_router(citizen.router, prefix="/api/v1")
app.include_router(citizen_reports.router, prefix="/api/v1")
app.include_router(corridors.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(grap.router, prefix="/api/v1")
app.include_router(grap.approve_router, prefix="/api/v1")
