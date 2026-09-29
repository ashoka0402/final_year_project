"""Economic corridors as the unit of analysis, and the federated exchange format.

**Why a corridor instead of a city.** Pollution follows freight routes and wind,
not municipal boundaries. The Delhi-Mumbai corridor crosses five states; a
Delhi-only dashboard structurally cannot see a plume forming over Haryana. Making
the corridor the analysis unit means the cross-boundary case is the default
rather than an integration bolted on later — which is exactly the
"interoperable, states share models and coordinate resources" requirement.

**Why the federated payload is a plain, versioned dict.** Interoperability
between state agencies fails on formats, not on algorithms. Every field carries
its unit and its provenance, and any authority can consume a corridor bulletin
without adopting VAYU's database, its models, or its code — the exchange format
is the contract, not the implementation.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import math
from dataclasses import dataclass, field

import pandas as pd

from vayu_core.config import REPO_ROOT

CORRIDORS_DIR = REPO_ROOT / "config" / "corridors"

# The exchange format is versioned so a consuming agency can reject or adapt a
# payload it does not understand, instead of silently misreading a changed field.
SCHEMA_VERSION = "vayu.corridor.v1"


@dataclass
class Corridor:
    id: str
    name: str
    states: list[str]
    waypoints: list[tuple[float, float]]  # (lon, lat), centre-line
    buffer_deg: float = 0.75

    def contains(self, lat: float, lon: float) -> bool:
        """Is this point within `buffer_deg` of the corridor centre-line?

        Point-to-segment distance in degrees, with a cos(lat) correction on the
        longitude term. Degrees are not metres — at 30°N a degree of longitude is
        ~87 km against ~111 km for latitude — and without that correction a
        north-south corridor would sample a visibly wider strip than an
        east-west one at the same buffer.
        """
        for (x1, y1), (x2, y2) in zip(self.waypoints, self.waypoints[1:]):
            if _dist_to_segment(lon, lat, x1, y1, x2, y2) <= self.buffer_deg:
                return True
        return False

    def cells(self, region) -> list[tuple[float, float]]:
        """Grid cells whose centre falls inside the corridor."""
        lats, lons = region.grid_axes()
        return [(la, lo) for la in lats for lo in lons if self.contains(la, lo)]


def _dist_to_segment(px, py, x1, y1, x2, y2) -> float:
    """Perpendicular distance from a point to a segment, in corrected degrees."""
    k = math.cos(math.radians((y1 + y2) / 2))  # shrink longitude toward the pole
    px, x1, x2 = px * k, x1 * k, x2 * k
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


@functools.lru_cache(maxsize=4)
def load_corridors(region_id: str = "india") -> tuple[Corridor, ...]:
    path = CORRIDORS_DIR / f"{region_id}.json"
    if not path.exists():
        return ()
    raw = json.loads(path.read_text())
    return tuple(
        Corridor(
            id=c["id"], name=c["name"], states=c.get("states", []),
            waypoints=[tuple(p) for p in c["waypoints"]],
            buffer_deg=float(c.get("buffer_deg", 0.75)),
        )
        for c in raw.get("corridors", [])
    )


def get_corridor(corridor_id: str, region_id: str = "india") -> Corridor | None:
    return next((c for c in load_corridors(region_id) for _ in [0] if c.id == corridor_id), None)


@dataclass
class CorridorBulletin:
    """One corridor's state on one day, in the federated exchange format."""

    corridor: Corridor
    date: dt.date
    cells_total: int
    cells_observed: int
    mean_hcho: float | None
    max_z: float | None
    hotspot_cells: int
    fire_count: int
    citizen_reports: int
    citizen_corroborated: int
    top_hotspots: list[dict] = field(default_factory=list)

    def to_payload(self) -> dict:
        """The wire format other agencies consume.

        Deliberately self-describing: units and provenance travel WITH the
        numbers, because a bare float in a shared feed is how cross-agency
        pipelines silently disagree. `coverage_pct` is included so a consumer can
        tell a genuinely clean corridor from one the satellite could not see.
        """
        return {
            "schema": SCHEMA_VERSION,
            "issued_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "corridor": {
                "id": self.corridor.id,
                "name": self.corridor.name,
                "states": self.corridor.states,
            },
            "date": self.date.isoformat(),
            "coverage": {
                "cells_total": self.cells_total,
                "cells_observed": self.cells_observed,
                "coverage_pct": round(100 * self.cells_observed / self.cells_total, 1)
                if self.cells_total
                else 0.0,
            },
            "hcho": {
                "mean": self.mean_hcho,
                "unit": "mol m-2",
                "max_anomaly_sigma": self.max_z,
                "hotspot_cells": self.hotspot_cells,
                "source": "Sentinel-5P/TROPOMI L3 (DLR)",
            },
            "fire": {"count": self.fire_count, "source": "VIIRS/MODIS FIRMS (NASA)"},
            "citizen": {
                "reports": self.citizen_reports,
                "satellite_corroborated": self.citizen_corroborated,
                "note": "Only corroborated reports influence analysis.",
            },
            "top_hotspots": self.top_hotspots,
        }


def build_bulletin(
    corridor: Corridor,
    region,
    day: dt.date,
    hcho: pd.DataFrame,
    fires: pd.DataFrame,
    hotspots: pd.DataFrame | None = None,
    citizen: pd.DataFrame | None = None,
) -> CorridorBulletin:
    """Assemble one corridor's daily bulletin from the already-computed layers."""
    cells = set(corridor.cells(region))
    total = len(cells)

    def _in(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        m = [(a, b) in cells for a, b in zip(df["grid_lat"], df["grid_lon"])]
        return df[m]

    h = _in(hcho)
    h = h[pd.to_datetime(h["date"]).dt.date == day] if not h.empty else h
    f = _in(fires)
    f = f[pd.to_datetime(f["date"]).dt.date == day] if not f.empty else f
    hs = _in(hotspots) if hotspots is not None else pd.DataFrame()
    hs = hs[pd.to_datetime(hs["date"]).dt.date == day] if not hs.empty else hs
    cz = _in(citizen) if citizen is not None else pd.DataFrame()
    cz = cz[pd.to_datetime(cz["date"]).dt.date == day] if not cz.empty else cz

    top = []
    if not hs.empty:
        for r in hs.nlargest(min(5, len(hs)), "z_score").itertuples():
            top.append(
                {
                    "lat": float(r.grid_lat), "lon": float(r.grid_lon),
                    "anomaly_sigma": round(float(r.z_score), 2),
                    "fire_count": int(getattr(r, "fire_count", 0) or 0),
                    "source_region": region.source_region_for(r.grid_lat, r.grid_lon),
                }
            )

    return CorridorBulletin(
        corridor=corridor, date=day,
        cells_total=total, cells_observed=int(len(h)),
        mean_hcho=float(h["value"].mean()) if not h.empty else None,
        max_z=round(float(hs["z_score"].max()), 2) if not hs.empty else None,
        hotspot_cells=int(len(hs)),
        fire_count=int(f["fire_count"].sum()) if not f.empty else 0,
        citizen_reports=int(len(cz)),
        citizen_corroborated=int(cz["may_influence"].sum()) if not cz.empty and "may_influence" in cz else 0,
        top_hotspots=top,
    )
