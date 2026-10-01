"""
Generate deterministic synthetic/demo data for the Pune Aeris/VAYU project.

Run from repository root:

    python generate_pune_synthetic.py

Optional:
    python generate_pune_synthetic.py --include-aq-demo

Creates:
    data/samples/
        osm_pune.geojson
        roads_pune.geojson
        fires_pune.csv
        permits_pune.csv

With --include-aq-demo:
        stations_pune.parquet
        measurements_pune.parquet

IMPORTANT
---------
- These datasets are SYNTHETIC / DEMO data.
- They must not be represented as CPCB, FIRMS, OSM, or municipal records.
- CPCB/data.gov.in integration is deliberately untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "samples"

DATA.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pune geography
# ---------------------------------------------------------------------------

# Matches the Pune city config / ward GeoJSON scope discussed for this project.
WEST = 73.73193
SOUTH = 18.38539
EAST = 74.01838
NORTH = 18.62185

CITY = "pune"

# Approximate city centre used only for deterministic synthetic generation.
CENTER_LAT = 18.505
CENTER_LON = 73.875


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_int(text: str, modulo: int = 10_000_000) -> int:
    """Deterministic integer from a string."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def rng_for(seed: str) -> random.Random:
    return random.Random(stable_int(seed))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def point(lon: float, lat: float) -> dict:
    return {
        "type": "Point",
        "coordinates": [round(lon, 6), round(lat, 6)],
    }


def polygon_from_box(
    west: float,
    south: float,
    east: float,
    north: float,
) -> dict:
    ring = [
        [round(west, 6), round(south, 6)],
        [round(east, 6), round(south, 6)],
        [round(east, 6), round(north, 6)],
        [round(west, 6), round(north, 6)],
        [round(west, 6), round(south, 6)],
    ]
    return {
        "type": "Polygon",
        "coordinates": [ring],
    }


def distance_factor(lat: float, lon: float) -> float:
    """
    Smooth deterministic spatial factor.

    Higher around the city core and selected transport/industrial corridors.
    This is purely synthetic and exists to make demo data spatially coherent.
    """
    dx = (lon - CENTER_LON) / (EAST - WEST)
    dy = (lat - CENTER_LAT) / (NORTH - SOUTH)

    core = math.exp(-((dx / 0.35) ** 2 + (dy / 0.35) ** 2))

    corridor_1 = math.exp(
        -(((lon - 73.90) / 0.035) ** 2 + ((lat - 18.54) / 0.05) ** 2)
    )

    corridor_2 = math.exp(
        -(((lon - 73.82) / 0.04) ** 2 + ((lat - 18.48) / 0.05) ** 2)
    )

    return core * 0.55 + corridor_1 * 0.30 + corridor_2 * 0.15


# ---------------------------------------------------------------------------
# 1. Synthetic OSM-context layer
# ---------------------------------------------------------------------------

def generate_osm() -> Path:
    """
    Generate schools, hospitals, industrial polygons and brick kilns.

    Format mirrors services/pipeline/osm.py:
        FeatureCollection
        properties.kind
        properties.name
        properties.lat
        properties.lon

    Status/provenance is explicitly synthetic.
    """

    features = []

    # ----------------------------
    # Schools
    # ----------------------------
    for i in range(75):
        r = rng_for(f"school-{i}")

        lat = SOUTH + (NORTH - SOUTH) * r.random()
        lon = WEST + (EAST - WEST) * r.random()

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": f"synthetic/school/{i+1:03d}",
                    "kind": "school",
                    "name": f"Synthetic Pune School {i+1:03d}",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "source": "synthetic",
                },
                "geometry": point(lon, lat),
            }
        )

    # ----------------------------
    # Hospitals
    # ----------------------------
    for i in range(25):
        r = rng_for(f"hospital-{i}")

        lat = SOUTH + (NORTH - SOUTH) * r.random()
        lon = WEST + (EAST - WEST) * r.random()

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": f"synthetic/hospital/{i+1:03d}",
                    "kind": "hospital",
                    "name": f"Synthetic Pune Hospital {i+1:03d}",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "source": "synthetic",
                },
                "geometry": point(lon, lat),
            }
        )

    # ----------------------------
    # Industrial areas
    # ----------------------------
    industrial_centres = [
        (18.566, 73.806),
        (18.602, 73.742),
        (18.560, 73.926),
        (18.518, 73.912),
        (18.476, 73.903),
        (18.640, 73.785),
        (18.455, 73.804),
        (18.536, 73.970),
        (18.398, 73.810),
        (18.587, 73.884),
    ]

    for i, (lat, lon) in enumerate(industrial_centres, start=1):
        r = rng_for(f"industrial-{i}")

        width = 0.006 + r.random() * 0.009
        height = 0.005 + r.random() * 0.008

        west = clamp(lon - width, WEST, EAST)
        east = clamp(lon + width, WEST, EAST)
        south = clamp(lat - height, SOUTH, NORTH)
        north = clamp(lat + height, SOUTH, NORTH)

        geom = polygon_from_box(west, south, east, north)

        area_km2 = abs((east - west) * 111 * (north - south) * 111)

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": f"synthetic/industrial/{i:03d}",
                    "kind": "industrial",
                    "name": f"Synthetic Pune Industrial Area {i:02d}",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "area_km2": round(area_km2, 4),
                    "source": "synthetic",
                },
                "geometry": geom,
            }
        )

    # ----------------------------
    # Brick kilns
    # ----------------------------
    for i in range(12):
        r = rng_for(f"kiln-{i}")

        lat = SOUTH + (NORTH - SOUTH) * r.random()
        lon = WEST + (EAST - WEST) * r.random()

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "osm_id": f"synthetic/brickyard/{i+1:03d}",
                    "kind": "brick_kiln",
                    "name": f"Synthetic Brick Kiln {i+1:02d}",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "source": "synthetic",
                },
                "geometry": point(lon, lat),
            }
        )

    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "city": CITY,
            "source": "synthetic",
            "description": (
                "Synthetic Pune OSM-context layer for demonstration/testing. "
                "Not derived from OpenStreetMap records."
            ),
        },
        "features": features,
    }

    path = DATA / "osm_pune.geojson"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] {path} ({len(features)} features)")
    return path


# ---------------------------------------------------------------------------
# 2. Synthetic road layer
# ---------------------------------------------------------------------------

def generate_roads() -> Path:
    """
    Generate major synthetic road LineStrings.

    services/pipeline/roads.py expects:
        data/samples/roads_{city}.geojson

    Properties contain:
        highway
        name
        source
    """

    features = []

    road_classes = [
        ("motorway", 3, 4.0),
        ("trunk", 6, 3.0),
        ("primary", 12, 2.0),
        ("secondary", 18, 1.0),
    ]

    road_id = 1

    # Horizontal roads
    latitudes = [
        18.40,
        18.44,
        18.48,
        18.52,
        18.56,
        18.60,
    ]

    for idx, lat in enumerate(latitudes):
        r = rng_for(f"horizontal-road-{idx}")

        highway, _, weight = road_classes[idx % len(road_classes)]

        # Slightly curved synthetic line
        coords = [
            [WEST, round(lat - 0.004 + r.random() * 0.002, 6)],
            [WEST + 0.22 * (EAST - WEST), round(lat + 0.003, 6)],
            [WEST + 0.50 * (EAST - WEST), round(lat - 0.002, 6)],
            [WEST + 0.78 * (EAST - WEST), round(lat + 0.004, 6)],
            [EAST, round(lat, 6)],
        ]

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "road_id": f"SYN-R{road_id:03d}",
                    "highway": highway,
                    "name": f"Synthetic Pune {highway.title()} Road {road_id:03d}",
                    "weight": weight,
                    "source": "synthetic",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            }
        )
        road_id += 1

    # Vertical roads
    longitudes = [
        73.75,
        73.79,
        73.83,
        73.87,
        73.91,
        73.95,
        73.99,
    ]

    for idx, lon in enumerate(longitudes):
        r = rng_for(f"vertical-road-{idx}")

        highway, _, weight = road_classes[(idx + 1) % len(road_classes)]

        coords = [
            [round(lon, 6), SOUTH],
            [round(lon - 0.003, 6), SOUTH + 0.25 * (NORTH - SOUTH)],
            [round(lon + 0.002, 6), SOUTH + 0.52 * (NORTH - SOUTH)],
            [round(lon - 0.002, 6), SOUTH + 0.78 * (NORTH - SOUTH)],
            [round(lon, 6), NORTH],
        ]

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "road_id": f"SYN-R{road_id:03d}",
                    "highway": highway,
                    "name": f"Synthetic Pune {highway.title()} Road {road_id:03d}",
                    "weight": weight,
                    "source": "synthetic",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            }
        )
        road_id += 1

    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "city": CITY,
            "source": "synthetic",
            "description": (
                "Synthetic major-road network for pipeline testing. "
                "Not derived from OpenStreetMap."
            ),
        },
        "features": features,
    }

    path = DATA / "roads_pune.geojson"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] {path} ({len(features)} roads)")
    return path


# ---------------------------------------------------------------------------
# 3. Synthetic fire detections
# ---------------------------------------------------------------------------

def generate_fires() -> Path:
    """
    Generate synthetic VIIRS-like fire records.

    Matches the normalized columns used by services/pipeline/firms.py.
    """

    rows = []

    start = datetime(2025, 11, 1, tzinfo=timezone.utc)

    for i in range(180):
        r = rng_for(f"fire-{i}")

        # Cluster more fires around selected zones while remaining bounded.
        cluster_points = [
            (18.40, 73.81),
            (18.47, 73.92),
            (18.55, 73.88),
            (18.60, 73.77),
        ]

        base_lat, base_lon = cluster_points[i % len(cluster_points)]

        lat = clamp(base_lat + (r.random() - 0.5) * 0.10, SOUTH, NORTH)
        lon = clamp(base_lon + (r.random() - 0.5) * 0.12, WEST, EAST)

        ts = start + timedelta(
            days=r.randint(0, 29),
            hours=r.randint(0, 23),
            minutes=r.choice([0, 15, 30, 45]),
        )

        frp = 5 + r.random() * 120

        confidence = r.choices(
            ["l", "n", "h"],
            weights=[15, 65, 20],
            k=1,
        )[0]

        rows.append(
            {
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "frp": round(frp, 2),
                "confidence": confidence,
                "acq_date": ts.date().isoformat(),
                "acq_time": ts.strftime("%H%M"),
                "instrument": "VIIRS-SYNTHETIC",
            }
        )

    df = pd.DataFrame(rows)

    path = DATA / "fires_pune.csv"
    df.to_csv(path, index=False)

    print(f"[OK] {path} ({len(df)} fire detections)")
    return path


# ---------------------------------------------------------------------------
# 4. Synthetic construction permits
# ---------------------------------------------------------------------------

def generate_permits() -> Path:
    """
    Generate synthetic permit records.

    Follows services/pipeline/permits.py schema.
    """

    rows = []

    site_types = [
        "Metro corridor extension",
        "Commercial tower",
        "Residential complex",
        "Flyover works",
        "Road widening",
        "Redevelopment block",
    ]

    reference_points = [
        (18.5204, 73.8567),
        (18.5314, 73.8446),
        (18.5074, 73.8077),
        (18.5826, 73.9197),
        (18.4701, 73.8900),
        (18.5960, 73.7410),
        (18.4513, 73.8040),
        (18.5650, 73.8020),
    ]

    for i in range(30):
        r = rng_for(f"permit-{i}")

        base_lat, base_lon = reference_points[i % len(reference_points)]

        lat = clamp(
            base_lat + (r.random() - 0.5) * 0.018,
            SOUTH,
            NORTH,
        )
        lon = clamp(
            base_lon + (r.random() - 0.5) * 0.022,
            WEST,
            EAST,
        )

        site_type = site_types[stable_int(f"type-{i}", len(site_types))]

        compliant = stable_int(f"compliance-{i}", 3) != 0

        inspection_date = (
            datetime.now(timezone.utc)
            - timedelta(days=stable_int(f"inspection-{i}", 90))
        ).date()

        rows.append(
            {
                "city": CITY,
                "permit_id": f"PU-CNS-{i+1:03d}",
                "name": f"Synthetic Pune Construction Site {i+1:03d}",
                "site_type": site_type,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "status": "active",
                "dust_control_compliant": compliant,
                "last_inspected": inspection_date.isoformat(),
                "source": "synthetic",
            }
        )

    df = pd.DataFrame(rows)

    path = DATA / "permits_pune.csv"
    df.to_csv(path, index=False)

    print(f"[OK] {path} ({len(df)} permits)")
    return path


# ---------------------------------------------------------------------------
# 5. Optional synthetic AQ station + historical measurement layer
# ---------------------------------------------------------------------------

def generate_demo_aq() -> tuple[Path, Path]:
    """
    Generate synthetic AQ data in the same broad shape as the project's
    station/measurement pipeline.

    This is OPTIONAL and deliberately labelled synthetic.

    It should NOT be confused with CPCB/data.gov.in.
    """

    station_points = [
        (18.5204, 73.8567),
        (18.5314, 73.8446),
        (18.5074, 73.8077),
        (18.5826, 73.9197),
        (18.4701, 73.8900),
        (18.5960, 73.7410),
        (18.4513, 73.8040),
        (18.5650, 73.8020),
        (18.5380, 73.9330),
        (18.4890, 73.8570),
    ]

    stations = []

    for i, (lat, lon) in enumerate(station_points, start=1):
        stations.append(
            {
                "city": CITY,
                "station_id": f"PU-SYN-{i:03d}",
                "name": f"Synthetic Pune AQ Station {i:02d}",
                "lat": lat,
                "lon": lon,
                "provider": "synthetic",
                "first_seen": "2025-11-01T00:00:00+00:00",
                "last_seen": "2025-11-30T23:00:00+00:00",
            }
        )

    stations_df = pd.DataFrame(stations)

    # 30 days × 24 hours × 6 parameters × 10 stations
    params = {
        "pm25": ("ug/m3", 35.0),
        "pm10": ("ug/m3", 75.0),
        "no2": ("ug/m3", 28.0),
        "so2": ("ug/m3", 8.0),
        "co": ("mg/m3", 0.8),
        "o3": ("ug/m3", 42.0),
    }

    measurement_rows = []

    start = datetime(2025, 11, 1, tzinfo=timezone.utc)

    for station in stations:
        station_factor = distance_factor(
            station["lat"],
            station["lon"],
        )

        for hour in range(30 * 24):
            ts = start + timedelta(hours=hour)

            # Synthetic diurnal signal.
            hour_angle = (ts.hour / 24.0) * 2.0 * math.pi
            traffic_signal = 1.0 + 0.25 * math.sin(hour_angle - 0.8)

            # Slight November stagnation factor.
            stagnation = 1.0 + 0.20 * (
                math.sin((hour / 24.0) * 2 * math.pi / 6)
            )

            for param, (unit, baseline) in params.items():
                r = rng_for(
                    f"{station['station_id']}-{param}-{ts.isoformat()}"
                )

                local_factor = 1.0 + station_factor * 0.9

                if param == "pm25":
                    value = baseline * local_factor * traffic_signal * stagnation
                    value += r.gauss(0, 8)

                elif param == "pm10":
                    value = baseline * local_factor * traffic_signal
                    value += r.gauss(0, 12)

                elif param == "no2":
                    value = baseline * local_factor * traffic_signal
                    value += r.gauss(0, 5)

                elif param == "so2":
                    value = baseline * local_factor
                    value += r.gauss(0, 2)

                elif param == "co":
                    value = baseline * local_factor * traffic_signal
                    value += r.gauss(0, 0.15)

                else:  # o3
                    daylight = max(0.0, math.sin(hour_angle))
                    value = baseline * (0.8 + 0.5 * daylight)
                    value += r.gauss(0, 5)

                value = max(value, 0.01)

                measurement_rows.append(
                    {
                        "city": CITY,
                        "station_id": station["station_id"],
                        "param": param,
                        "ts": ts.isoformat(),
                        "value": round(value, 3),
                        "unit": unit,
                        "source": "synthetic",
                    }
                )

    measurements_df = pd.DataFrame(measurement_rows)

    stations_path = DATA / "stations_pune.parquet"
    measurements_path = DATA / "measurements_pune.parquet"

    stations_df.to_parquet(stations_path, index=False)
    measurements_df.to_parquet(measurements_path, index=False)

    print(f"[OK] {stations_path} ({len(stations_df)} stations)")
    print(
        f"[OK] {measurements_path} "
        f"({len(measurements_df):,} measurements)"
    )

    return stations_path, measurements_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Pune Aeris/VAYU data."
    )

    parser.add_argument(
        "--include-aq-demo",
        action="store_true",
        help=(
            "Also generate synthetic station and historical AQ measurements. "
            "These are NOT CPCB data."
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 72)
    print("AERIS / VAYU — PUNE SYNTHETIC DATA GENERATOR")
    print("=" * 72)
    print("All generated records are explicitly SYNTHETIC / DEMO data.")
    print("CPCB/data.gov.in integration is NOT modified.")
    print()

    generate_osm()
    generate_roads()
    generate_fires()
    generate_permits()

    if args.include_aq_demo:
        generate_demo_aq()

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print("\nGenerated files:")

    for path in sorted(DATA.glob("*_pune.*")):
        print(f"  {path.relative_to(ROOT)}")

    print()
    print("Reminder:")
    print("  - OSM layer: synthetic")
    print("  - Roads: synthetic")
    print("  - Fires: synthetic")
    print("  - Permits: synthetic")
    if args.include_aq_demo:
        print("  - AQ stations: synthetic")
        print("  - AQ measurements: synthetic")
    print("  - CPCB/data.gov.in: untouched")


if __name__ == "__main__":
    main()