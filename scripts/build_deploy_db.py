"""Build a slim DuckDB for deployment.

The working archive is ~540MB, most of it 2016-2025 hourly weather and station
measurements that exist to TRAIN the forecast model. The model is already
trained (models/artifacts is in git), so a deployed container needs none of that
history — only what the live surfaces actually read.

Keeping the full archive in the image would mean slow cold starts and a large
push for data no request touches. This keeps the satellite/fire/hotspot layers
whole (they are the demo) and trims the deep training history.

    python -m scripts.build_deploy_db
"""

from __future__ import annotations

import datetime as dt
import pathlib

import duckdb

from vayu_core.db import SCHEMA

SRC = "data/vayu.duckdb"
DST = "data/vayu_deploy.duckdb"

# The archive window the deployed app can be browsed within. The satellite and
# fire layers are fully covered here, so nothing the national views read is lost.
KEEP_FROM = dt.date(2025, 9, 1)


def main() -> int:
    src = pathlib.Path(SRC)
    if not src.exists():
        print(f"missing {SRC} — run `make seed` first")
        return 1

    pathlib.Path(DST).unlink(missing_ok=True)
    con = duckdb.connect(DST)
    con.execute(SCHEMA)
    con.execute(f"ATTACH '{SRC}' AS src (READ_ONLY)")

    # Time-bounded: the heavy training history is what we are dropping.
    for table, col in [
        ("measurements", "ts"),
        ("weather_hourly", "ts"),
        ("forecasts", "run_ts"),
    ]:
        con.execute(
            f"INSERT INTO {table} SELECT * FROM src.{table} WHERE {col} >= TIMESTAMPTZ '{KEEP_FROM}'"
        )

    # Copied whole: these ARE the national demo, and they are already compact.
    #
    # `fires` (city-scoped point detections behind the ward-level Evidence
    # layer and attribution cone) is distinct from `fire_grid` (the national
    # grid the corridor bulletins read) and was missing from this list —
    # every deploy silently shipped zero fire evidence for every city while
    # the source archive held 74k+ real rows, found by comparing row counts
    # between source and deploy directly, not by reading this list and
    # assuming it was complete.
    for table in [
        "satellite_grid", "fire_grid", "fires", "aqi_grid", "hcho_hotspots",
        "stations", "wards", "ward_roads", "permits", "attributions",
        "interventions", "verifications", "grap_drafts", "data_status",
        "citizen_reports",
    ]:
        try:
            con.execute(f"INSERT INTO {table} SELECT * FROM src.{table}")
        except duckdb.Error as exc:  # a table absent in an older DB is fine
            print(f"  skip {table}: {str(exc)[:70]}")

    cols = [r[1] for r in con.execute("PRAGMA table_info(audit_log)").fetchall()]
    con.execute(f"INSERT INTO audit_log ({','.join(cols)}) SELECT {','.join(cols)} FROM src.audit_log")

    con.execute("DETACH src")
    for t in ("measurements", "weather_hourly", "satellite_grid", "fire_grid", "citizen_reports"):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t:18} {n:>10,}")
    con.close()

    duckdb.connect(DST).execute("CHECKPOINT")  # compact
    mb = pathlib.Path(DST).stat().st_size / 1e6
    print(f"\n{DST}: {mb:.0f} MB (from {src.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
