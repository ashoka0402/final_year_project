"""Settings and city configuration.

Two rules encoded here:

1. Every API key is optional (master prompt §5). Absence is a normal, tested
   code path — never an error.
2. A city is a config file, not a code path (PRD G1). Nothing downstream may
   branch on `city == "delhi"`; everything it needs is in CityConfig.
"""

from __future__ import annotations

import functools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CITIES_DIR = REPO_ROOT / "config" / "cities"
REGIONS_DIR = REPO_ROOT / "config" / "regions"

# data.gov.in publishes this key in its own API docs as the open demo key, which
# is what lets VAYU show real CPCB stations with zero signup. It is rate-limited;
# set DATA_GOV_IN_API_KEY in .env for your own.
DATA_GOV_PUBLIC_DEMO_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", extra="ignore", case_sensitive=False
    )

    demo_mode: bool = True
    demo_now: datetime = datetime(2025, 11, 3, 6, 0, tzinfo=timezone.utc)

    data_gov_in_api_key: str = ""
    openaq_api_key: str = ""
    firms_api_key: str = ""
    anthropic_api_key: str = ""
    gee_service_account_json: str = ""

    # --- Google AI -----------------------------------------------------------
    # Gemini does two jobs that nothing else in VAYU can do: read a citizen's
    # photograph and turn it into a structured pollution observation, and write
    # the plain-language advisory in the reader's own language. Both degrade to
    # an explicit "unavailable" rather than a fabricated answer when unset —
    # a guessed pollution reading is worse than no reading.
    google_api_key: str = ""
    # Pinned to a concrete version, not the `gemini-flash-latest` alias. The
    # alias is a shared hot path and returned 503 "high demand" on first try
    # here; a pinned model is steadier, and model retirement is loud (404 with
    # the replacement named) rather than a silent behaviour change mid-demo.
    gemini_model: str = "gemini-3.6-flash"
    # Vertex path (optional): set both to route through a GCP project instead
    # of the AI Studio key. Useful when deploying on Cloud Run in the same
    # project, which is also what the "deployability" story wants.
    google_cloud_project: str = ""
    google_cloud_location: str = "asia-south1"
    # Open-Meteo needs NO key: VAYU's entire 1.47M-row weather archive was
    # fetched without one. This exists only for two edge cases:
    #   * the free tier is priced by data volume (locations x hours x variables),
    #     not request count, so a deep historical backfill can earn a 429 — we
    #     chunk and throttle around that, but a key removes the ceiling;
    #   * Open-Meteo's free tier is licensed for non-commercial use only, so a
    #     real deployment of VAYU would need a paid plan.
    # Absent (the default), VAYU uses the free open endpoints.
    open_meteo_api_key: str = ""

    vayu_db_path: str = "data/vayu.duckdb"
    duckdb_memory_limit: str = ""  # empty string = no artificial cap (unconstrained host RAM locally)
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


    @property
    def db_path(self) -> Path:
        p = Path(self.vayu_db_path)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_gov_key(self) -> str:
        """User key when supplied, else the portal's published demo key."""
        return self.data_gov_in_api_key or DATA_GOV_PUBLIC_DEMO_KEY

    @property
    def google_ai_available(self) -> bool:
        """True when Gemini can actually be called (AI Studio key or Vertex)."""
        return bool(self.google_api_key or self.google_cloud_project)

    def now(self) -> datetime:
        """The app's notion of "now".

        Precedence:
          1. A runtime `as_of` override (the date picker / `?as_of=`), if set.
             This lets an operator time-travel to any hour without a restart.
          2. In DEMO_MODE, DEMO_NOW — the timeline is deterministic and the demo
             is rehearsable (the same click produces the same forecast on stage
             as in rehearsal).
          3. Otherwise, live wall clock (UTC).
        """
        override = _clock_override()
        if override is not None:
            return override
        if self.demo_mode:
            d = self.demo_now
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)


# --- Runtime clock override (L1b: live clock + date picker / ?as_of=) ----------
# A process-global "as of" that Settings.now() honours above everything else.
# Set by the /clock endpoint or a per-request `?as_of=` param so the whole app —
# nowcast, forecast horizons, order ages, GRAP stage — moves together to the
# chosen instant. None means "follow demo_mode / wall clock" (the default).
_CLOCK_OVERRIDE: datetime | None = None


def _clock_override() -> datetime | None:
    return _CLOCK_OVERRIDE


def clock_override() -> datetime | None:
    """Public read of the pinned instant, or None if following demo/wall clock."""
    return _CLOCK_OVERRIDE


def set_demo_mode(enabled: bool) -> bool:
    """Flip demo/live at runtime (no restart) by mutating the cached Settings.

    Demo ON  → bundled past data, clock pinned to DEMO_NOW.
    Demo OFF → live wall clock + live-fetch ingest paths for the present day.

    Clears any clock override so the app snaps to the new mode's natural "now"
    (today for live, DEMO_NOW for demo). Everything reads `settings.demo_mode`
    and `settings.now()`, so this moves the whole app together.
    """
    s = get_settings()
    s.demo_mode = enabled
    set_clock(None)
    return s.demo_mode


def set_clock(at: datetime | None) -> datetime | None:
    """Pin the app clock to `at` (tz-aware UTC), or clear it with None.

    Returns the resulting effective `now()` so callers can echo it back.
    """
    global _CLOCK_OVERRIDE
    if at is not None and at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    _CLOCK_OVERRIDE = at.astimezone(timezone.utc) if at is not None else None
    return get_settings().now()


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class WardsConfig(BaseModel):
    geojson: str
    source_url: str | None = None
    attribution: str | None = None
    id_property: str
    name_property: str
    use_h3: bool = False
    h3_resolution: int = 7


class PopulationConfig(BaseModel):
    total: int
    source: str
    method: str


class WeatherGrid(BaseModel):
    nx: int = 5
    ny: int = 5


class AirshedGrid(BaseModel):
    """A wider, coarser wind field for back-trajectories.

    The city grid spans ~50 km; a 24h back-trajectory travels ~360 km. Without a
    wider field the parcel leaves the grid within two hours and the rest of the
    path is edge wind extrapolated hundreds of km — which is precisely the leg
    that reaches the Punjab stubble fires we intend to attribute to.

    2 degrees of pad (~220 km) matches the FIRMS fire search box, and ~0.3 deg
    spacing (~33 km) is at the resolution of the underlying global NWP, so a
    finer grid would only re-sample the same numbers.
    """

    pad_deg: float = 2.0
    nx: int = 15
    ny: int = 15


class CityConfig(BaseModel):
    id: str
    name: str
    timezone: str
    bbox: list[float] = Field(min_length=4, max_length=4)  # [w, s, e, n]
    map_center: list[float] = Field(min_length=2, max_length=2)
    map_zoom: float
    grap_applicable: bool = False
    languages: list[str] = ["en"]
    weather_grid: WeatherGrid = WeatherGrid()
    airshed_grid: AirshedGrid = AirshedGrid()
    wards: WardsConfig
    population: PopulationConfig
    sources: dict[str, Any] = {}

    @property
    def wards_path(self) -> Path:
        p = Path(self.wards.geojson)
        return p if p.is_absolute() else REPO_ROOT / p

    def bbox_str(self, order: Literal["wsen", "swne"] = "wsen") -> str:
        w, s, e, n = self.bbox
        return f"{w},{s},{e},{n}" if order == "wsen" else f"{s},{w},{n},{e}"

    def _grid(self, bbox: list[float], nx: int, ny: int) -> list[tuple[int, int, float, float]]:
        w, s, e, n = bbox
        pts = []
        for i in range(nx):
            for j in range(ny):
                lon = w + (e - w) * (i + 0.5) / nx
                lat = s + (n - s) * (j + 0.5) / ny
                pts.append((i, j, round(lat, 4), round(lon, 4)))
        return pts

    def grid_points(self) -> list[tuple[int, int, float, float]]:
        """(i, j, lat, lon) on the city weather grid — drives station features.

        Points sit at cell centres rather than on the bbox edge.
        """
        return self._grid(self.bbox, self.weather_grid.nx, self.weather_grid.ny)

    @property
    def airshed_bbox(self) -> list[float]:
        """City bbox widened by the airshed pad — the domain a trajectory can reach."""
        w, s, e, n = self.bbox
        p = self.airshed_grid.pad_deg
        return [w - p, s - p, e + p, n + p]

    def airshed_points(self) -> list[tuple[int, int, float, float]]:
        """(i, j, lat, lon) on the wide grid the back-trajectory integrates through."""
        return self._grid(self.airshed_bbox, self.airshed_grid.nx, self.airshed_grid.ny)


@functools.lru_cache(maxsize=8)
def load_city(city_id: str) -> CityConfig:
    path = CITIES_DIR / f"{city_id.lower()}.json"
    if not path.exists():
        raise KeyError(f"unknown city '{city_id}' (no {path.name} in config/cities/)")
    return CityConfig.model_validate(json.loads(path.read_text()))


@functools.lru_cache(maxsize=1)
def list_cities() -> list[CityConfig]:
    """Every city is discovered from disk — adding a JSON file adds a city."""
    return [
        CityConfig.model_validate(json.loads(p.read_text()))
        for p in sorted(CITIES_DIR.glob("*.json"))
    ]


# --- National layer -----------------------------------------------------
# A *region* is the national counterpart to a city: where CityConfig describes
# irregular ward polygons at city scale, RegionConfig describes a regular
# lat/lon raster grid at country scale. They deliberately coexist rather than
# one bending into the other — satellite science is gridded, enforcement is
# jurisdictional, and forcing either into the other's shape loses information.


class SourceRegion(BaseModel):
    """A named emission zone hotspots get attributed to."""

    id: str
    name: str
    bbox: list[float] = Field(min_length=4, max_length=4)  # [w, s, e, n]
    kind: Literal["agricultural_residue", "forest_fire", "mixed"] = "mixed"

    def contains(self, lat: float, lon: float) -> bool:
        w, s, e, n = self.bbox
        return w <= lon <= e and s <= lat <= n


class BurningSeason(BaseModel):
    """A biomass-burning window. Hotspot detection is scoped to these explicitly."""

    label: str
    months: list[int]
    note: str = ""


class SatelliteProduct(BaseModel):
    """One columnar product: where to get it and what the numbers mean."""

    collection: str
    band: str
    unit: str
    label: str
    qa_band: str | None = None
    scale: float = 1.0
    note: str = ""


class RegionConfig(BaseModel):
    id: str
    name: str
    timezone: str
    bbox: list[float] = Field(min_length=4, max_length=4)  # [w, s, e, n]
    map_center: list[float] = Field(min_length=2, max_length=2)
    map_zoom: float
    grid_deg: float = 0.25
    seasons: dict[str, BurningSeason] = {}
    source_regions: list[SourceRegion] = []
    products: dict[str, SatelliteProduct] = {}

    def grid_axes(self) -> tuple[list[float], list[float]]:
        """Cell-centre latitudes and longitudes of the analysis grid.

        Centres (not edges) so a cell's coordinate is the point its value is
        attributed to — which is what the DB stores and the map renders.
        """
        w, s, e, n = self.bbox
        d = self.grid_deg
        lons, lats = [], []
        x = w + d / 2
        while x < e:
            lons.append(round(x, 4))
            x += d
        y = s + d / 2
        while y < n:
            lats.append(round(y, 4))
            y += d
        return lats, lons

    def grid_size(self) -> tuple[int, int]:
        lats, lons = self.grid_axes()
        return len(lats), len(lons)

    def snap(self, lat: float, lon: float) -> tuple[float, float]:
        """Snap an arbitrary point to its grid-cell centre.

        Used to join point data (CPCB stations, FIRMS fire pixels) onto the
        raster without a spatial index — the grid is regular, so this is pure
        arithmetic and stays fast over millions of fire detections.
        """
        w, s, _, _ = self.bbox
        d = self.grid_deg
        gi = int((lat - s) // d)
        gj = int((lon - w) // d)
        return (round(s + gi * d + d / 2, 4), round(w + gj * d + d / 2, 4))

    def source_region_for(self, lat: float, lon: float) -> str | None:
        for sr in self.source_regions:
            if sr.contains(lat, lon):
                return sr.id
        return None

    def is_burning_season(self, month: int) -> str | None:
        for key, season in self.seasons.items():
            if month in season.months:
                return key
        return None


@functools.lru_cache(maxsize=8)
def load_region(region_id: str) -> RegionConfig:
    path = REGIONS_DIR / f"{region_id.lower()}.json"
    if not path.exists():
        raise KeyError(f"unknown region '{region_id}' (no {path.name} in config/regions/)")
    return RegionConfig.model_validate(json.loads(path.read_text()))


@functools.lru_cache(maxsize=1)
def list_regions() -> list[RegionConfig]:
    """Every region is discovered from disk — same rule as cities."""
    if not REGIONS_DIR.exists():
        return []
    return [
        RegionConfig.model_validate(json.loads(p.read_text()))
        for p in sorted(REGIONS_DIR.glob("*.json"))
    ]
