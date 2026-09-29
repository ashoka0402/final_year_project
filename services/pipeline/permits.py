"""Construction permits — the ONE curated layer in VAYU.

There is no public machine-readable feed of Delhi/Lucknow construction permits
with dust-control compliance status. Master prompt §2 allows a curated sample
precisely here, on two conditions, both enforced in code:

  1. It is generated from REAL geography, not invented coordinates. Sites are
     placed on actual OSM construction/brownfield landuse polygons where those
     exist, so a dispatched inspector would arrive somewhere that is genuinely a
     building site.
  2. It is labelled `sample` at every layer — DB `source`, API `data_status`,
     and a visible "Sample data" badge in the UI. It is never presented as a
     municipal record.

The compliance flag is the fabricated part, and it is the reason this file is
careful: a non-compliant flag is an accusation. It is derived deterministically
from the site id so the demo is reproducible, and the dossier for a
construction action states in the document that the permit layer is sample data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from vayu_core.config import REPO_ROOT, CityConfig

from .http import FetchError, fetch_text

ENDPOINT = "https://overpass-api.de/api/interpreter"
N_SITES = 30

SITE_TYPES = [
    ("Metro corridor extension", 4),
    ("Commercial tower", 3),
    ("Residential complex", 3),
    ("Flyover works", 2),
    ("Road widening", 2),
    ("Redevelopment block", 2),
]


def sample_path(city: CityConfig) -> Path:
    return REPO_ROOT / "data" / "samples" / f"permits_{city.id}.csv"


def _real_construction_sites(city: CityConfig) -> list[tuple[float, float, str | None]]:
    """Actual OSM construction/brownfield landuse — real building sites."""
    w, s, e, n = city.bbox
    q = (
        f"[out:json][timeout:60];("
        f'way["landuse"~"^(construction|brownfield)$"]({s},{w},{n},{e});'
        f'way["building"="construction"]({s},{w},{n},{e});'
        f");out center tags;"
    )
    try:
        payload = json.loads(fetch_text(ENDPOINT, params={"data": q}, ttl=timedelta(days=30), timeout=120.0))
    except (FetchError, json.JSONDecodeError) as exc:
        logger.warning(f"[{city.id}] OSM construction landuse unavailable: {exc}")
        return []

    out = []
    for el in payload.get("elements", []) or []:
        c = el.get("center") or {}
        lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        out.append((float(lat), float(lon), (el.get("tags") or {}).get("name")))
    logger.info(f"[{city.id}] OSM construction/brownfield sites found: {len(out)}")
    return out


def _deterministic(seed: str, n: int) -> int:
    """Stable pseudo-random int — the demo must be identical on stage."""
    return int(hashlib.sha256(seed.encode()).hexdigest(), 16) % n


def build_permits(city: CityConfig, wards: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (permits, status='sample'). Always 'sample' — never 'live'."""
    p = sample_path(city)
    if not force and p.exists():
        try:
            return pd.read_csv(p), "sample"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{city.id}] bundled permits unreadable: {exc}")

    real = _real_construction_sites(city)
    rows: list[dict] = []
    now = datetime.now(timezone.utc)

    # Prefer real OSM construction polygons; fall back to ward centroids (still
    # real geography — an actual ward, not a random point in a field).
    if len(real) >= N_SITES:
        picks = [real[_deterministic(f"{city.id}-site-{i}", len(real))] for i in range(N_SITES)]
        origin = "OSM construction/brownfield landuse"
    elif not wards.empty:
        idx = [_deterministic(f"{city.id}-ward-{i}", len(wards)) for i in range(N_SITES)]
        picks = [(float(wards.iloc[j].centroid_lat), float(wards.iloc[j].centroid_lon), str(wards.iloc[j]["name"])) for j in idx]
        origin = "ward centroids (no OSM construction landuse found)"
    else:
        return pd.DataFrame(), "unavailable"

    types = [t for t, weight in SITE_TYPES for _ in range(weight)]
    for i, (lat, lon, name) in enumerate(picks):
        sid = f"{city.id.upper()[:2]}-CNS-{i + 1:03d}"
        kind = types[_deterministic(f"{sid}-type", len(types))]
        # ~1 in 3 non-compliant. Deterministic so the demo is rehearsable.
        compliant = _deterministic(f"{sid}-dust", 3) != 0
        rows.append(
            {
                "city": city.id,
                "permit_id": sid,
                "name": name or f"{kind} — site {i + 1}",
                "site_type": kind,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "status": "active",
                "dust_control_compliant": compliant,
                "last_inspected": (now - timedelta(days=_deterministic(f"{sid}-insp", 90))).date().isoformat(),
                "source": "sample",
            }
        )

    df = pd.DataFrame(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    n_bad = int((~df["dust_control_compliant"]).sum())
    logger.info(
        f"[{city.id}] permits: {len(df)} SAMPLE sites on {origin} "
        f"({n_bad} flagged non-compliant) — labelled sample everywhere"
    )
    return df, "sample"
