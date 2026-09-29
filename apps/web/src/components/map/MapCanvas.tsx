"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { type MapGeoJSONFeature, type MapMouseEvent } from "maplibre-gl";

import { AQI_BANDS } from "@/lib/aqi";
import type {
  AttributionEvidence,
  City,
  Current,
  TrajectoryCollection,
  Ward,
  WardCollection,
} from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";
import { WardTooltip } from "./WardTooltip";

/**
 * One <MapCanvas> with a declarative layer registry (TRD §9).
 *
 * Implementation note — MapLibre native layers rather than the deck.gl
 * MapboxOverlay that TRD §9 sketches. To be precise about why, because the
 * first version of this comment overstated the case: deck.gl's picking did not
 * fire during development, but that was investigated in an environment where
 * requestAnimationFrame was frozen (a backgrounded tab), which independently
 * breaks any pick that needs a render pass. So deck.gl is NOT proven broken.
 *
 * The choice stands on its own merits: for choropleths, points and lines,
 * MapLibre's native layers give reliable hit-testing, feature-state hover and
 * selection, GPU-side data-driven styling, and correct interleaving beneath
 * basemap labels — with no version coupling between two GL libraries. Clicking a
 * ward is golden-flow step 2 and PRD B2 requires every attribution to trace to
 * clickable evidence, so interaction should rest on the simplest thing that
 * works.
 *
 * The trajectory is animated with a line-dash offset driven by MapLibre's own
 * render loop, which achieves the TripsLayer effect the master prompt asks for
 * without adding a second renderer.
 */

const EVIDENCE_COLOR: Record<string, string> = {
  fire: "#EF4444",
  industry: "#22D3EE",
  permit: "#A78BFA",
  traffic: "#F59E0B",
  regional: "#64748B",
};

// Basemaps as RASTER layers inside one persistent style, switched by toggling
// visibility — never setStyle. setStyle wipes VAYU's layers and its completion
// depends on the render loop (which can hang), so a visibility toggle is the
// robust design: VAYU's vector layers never move, feature-state survives, and
// switching is instant with no race. All keyless (master prompt §3).
type Basemap = "dark" | "light" | "satellite";

// CARTO retired the keyless basemaps.cartocdn.com endpoint this code
// originally used (it now serves a "get an API key" placeholder tile
// instead of a 404, which is why a broken basemap can look identical to a
// working one until you actually look at a tile). Esri's Canvas services
// are the keyless replacement, served from the same arcgisonline.com host
// already used below for satellite imagery, so this stays a single trusted,
// no-key provider rather than adding a new dependency.
const RASTER_TILES: Record<Basemap, { tiles: string[]; attribution: string }> = {
  dark: {
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
    attribution: "© Esri © OpenStreetMap contributors",
  },
  light: {
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
    attribution: "© Esri © OpenStreetMap contributors",
  },
  satellite: {
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
  },
};

const BASEMAP_LAYER: Record<Basemap, string> = {
  dark: "basemap-dark",
  light: "basemap-light",
  satellite: "basemap-satellite",
};

// One style holding all three basemaps; only the active one is visible. VAYU's
// layers are added on top by the registration effect and stay put forever.
function baseStyle(active: Basemap): maplibregl.StyleSpecification {
  return {
    version: 8,
    sources: Object.fromEntries(
      (Object.keys(RASTER_TILES) as Basemap[]).map((k) => [
        BASEMAP_LAYER[k],
        { type: "raster", tiles: RASTER_TILES[k].tiles, tileSize: 256, attribution: RASTER_TILES[k].attribution },
      ]),
    ),
    layers: (Object.keys(RASTER_TILES) as Basemap[]).map((k) => ({
      id: BASEMAP_LAYER[k],
      type: "raster" as const,
      source: BASEMAP_LAYER[k],
      layout: { visibility: k === active ? "visible" : "none" },
    })),
  };
}

const SRC_WARDS = "vayu-wards";
const SRC_STATIONS = "vayu-stations";
const SRC_TRAJ = "vayu-traj";
const SRC_CONE = "vayu-cone";
const SRC_EVIDENCE = "vayu-evidence";
const LYR_WARD_FILL = "vayu-ward-fill";
const LYR_WARD_LINE = "vayu-ward-line";
const LYR_STATIONS = "vayu-stations";
const LYR_CONE = "vayu-cone-fill";
const LYR_TRAJ = "vayu-traj-line";
const LYR_TRAJ_HEAD = "vayu-traj-head";
const LYR_EVIDENCE = "vayu-evidence-pts";

/** CPCB bands as a MapLibre `step` expression over the ward's feature-state. */
const AQI_STEP_EXPRESSION: maplibregl.ExpressionSpecification = [
  "step",
  ["feature-state", "aqi"],
  AQI_BANDS[0].color,
  ...AQI_BANDS.slice(1).flatMap((b) => [b.min, b.color] as [number, string]),
] as unknown as maplibregl.ExpressionSpecification;

interface Props {
  city: City;
  current?: Current;
  wards?: WardCollection;
  /** Back-trajectory GeoJSON (LineString + cone Polygon) for the selected ward. */
  trajectory?: TrajectoryCollection;
  /** Evidence points to plot; the hovered one pulses. */
  evidence?: AttributionEvidence[];
  hoveredEvidence?: AttributionEvidence | null;
  flyTo?: { lon: number; lat: number } | null;
}

type HoverInfo = { x: number; y: number; ward: Ward } | null;

export function MapCanvas({
  city,
  current,
  wards,
  trajectory,
  evidence,
  hoveredEvidence,
  flyTo,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  // Monotonic epoch, not a boolean: every style load bumps it, so switching the
  // basemap (which wipes all sources/layers) produces a NEW value and forces the
  // registration + data effects below to re-run and rebuild VAYU's layers. A
  // boolean false->true would get batched by React and the layers would vanish.
  // 0 = no style loaded yet; any positive value = ready. `!styleReady` still
  // gates correctly (0 is falsy).
  const [styleReady, setStyleReady] = useState(0);
  const [mapError, setMapError] = useState<string | null>(null);
  const [hover, setHover] = useState<HoverInfo>(null);

  const layersOn = useCommandStore((s) => s.layers);
  const basemap = useCommandStore((s) => s.basemap);
  const selectWard = useCommandStore((s) => s.selectWard);
  const selectedWardId = useCommandStore((s) => s.selectedWardId);
  const hoverWard = useCommandStore((s) => s.hoverWard);

  // ward_id -> reading: the join between geometry (cached hard) and values.
  const readings = useMemo(() => {
    const m = new Map<string, Ward>();
    current?.wards.forEach((w) => m.set(w.ward_id, w));
    return m;
  }, [current]);
  const readingsRef = useRef(readings);
  readingsRef.current = readings;

  const stationGeoJSON = useMemo(
    () => ({
      type: "FeatureCollection" as const,
      features: (current?.stations ?? []).map((s) => ({
        type: "Feature" as const,
        properties: {
          station_id: s.station_id,
          name: s.name,
          aqi: s.aqi,
          // No reading -> slate; never a colour that implies clean air.
          color: s.color ?? "#334155",
        },
        geometry: { type: "Point" as const, coordinates: [s.lon, s.lat] },
      })),
    }),
    [current],
  );

  // ---- init map once ------------------------------------------------------
  useEffect(() => {
    if (!container.current || map.current) return;

    const m = new maplibregl.Map({
      container: container.current,
      style: baseStyle(basemap),
      center: city.map_center as [number, number],
      zoom: city.map_zoom,
      attributionControl: { compact: true },
      dragRotate: false,
      pitchWithRotate: false,
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    m.addControl(new maplibregl.ScaleControl({ maxWidth: 90, unit: "metric" }), "bottom-right");

    // MapLibre reports style/tile failures on this channel and is otherwise
    // silent — without it a blank basemap looks identical to "still loading".
    //
    // Only surface it BEFORE the style loads. Once the map is up, this channel
    // also carries transient noise (a single 404 tile, a query against a layer
    // that has not been added yet), and latching on those would leave a
    // permanent "Basemap unavailable" banner over a perfectly working map.
    m.on("error", (e) => {
      const msg = e.error?.message ?? String(e);
      if (!m.isStyleLoaded()) {
        console.error("[maplibre]", msg);
        setMapError(msg);
      } else if (process.env.NODE_ENV !== "production") {
        console.warn("[maplibre]", msg);
      }
    });
    m.on("load", () => {
      setStyleReady((n) => n + 1);
      setMapError(null); // a late success clears an early failure
    });
    map.current = m;

    if (process.env.NODE_ENV !== "production") {
      Object.assign(window, { __vayuMap: m });
    }

    return () => {
      m.remove();
      map.current = null;
      setStyleReady(0);
    };
    // Init-only: a city change flies the camera rather than rebuilding the GL
    // context, so Delhi -> Lucknow stays instant (PRD G1).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- basemap switch ------------------------------------------------------
  // Just toggle raster-layer visibility — no setStyle, so VAYU's layers and
  // feature-state are untouched and the switch is instant and race-free.
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    for (const k of Object.keys(BASEMAP_LAYER) as Basemap[]) {
      if (m.getLayer(BASEMAP_LAYER[k])) {
        m.setLayoutProperty(BASEMAP_LAYER[k], "visibility", k === basemap ? "visible" : "none");
      }
    }
  }, [basemap, styleReady]);

  // ---- register sources + layers once the style is ready -------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;

    if (!m.getSource(SRC_WARDS)) {
      m.addSource(SRC_WARDS, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        // promoteId makes ward_id the feature id, which is what setFeatureState
        // keys on — without it every ward would need a numeric index.
        promoteId: "ward_id",
      });
      m.addLayer({
        id: LYR_WARD_FILL,
        type: "fill",
        source: SRC_WARDS,
        paint: {
          "fill-color": [
            "case",
            ["==", ["feature-state", "aqi"], null],
            "#334155",
            AQI_STEP_EXPRESSION,
          ] as unknown as maplibregl.ExpressionSpecification,
          "fill-opacity": [
            "case",
            ["boolean", ["feature-state", "hover"], false],
            0.85,
            ["==", ["feature-state", "aqi"], null],
            0.35,
            0.68,
          ] as unknown as maplibregl.ExpressionSpecification,
        },
      });
      // 150ms hover fade, per the App Flow's motion budget (150-250ms).
      m.setPaintProperty(LYR_WARD_FILL, "fill-opacity-transition", { duration: 150, delay: 0 });
      m.addLayer({
        id: LYR_WARD_LINE,
        type: "line",
        source: SRC_WARDS,
        paint: {
          "line-color": [
            "case",
            ["boolean", ["feature-state", "selected"], false],
            "#22D3EE",
            "#1F2A44",
          ] as unknown as maplibregl.ExpressionSpecification,
          "line-width": [
            "case",
            ["boolean", ["feature-state", "selected"], false],
            2.5,
            0.6,
          ] as unknown as maplibregl.ExpressionSpecification,
        },
      });
    }

    // Cone sits UNDER the trajectory and stations so it never hides them.
    if (!m.getSource(SRC_CONE)) {
      m.addSource(SRC_CONE, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addLayer({
        id: LYR_CONE,
        type: "fill",
        source: SRC_CONE,
        paint: { "fill-color": "#22D3EE", "fill-opacity": 0.12, "fill-outline-color": "#22D3EE" },
      });
    }

    if (!m.getSource(SRC_TRAJ)) {
      m.addSource(SRC_TRAJ, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addLayer({
        id: LYR_TRAJ,
        type: "line",
        source: SRC_TRAJ,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": "#22D3EE",
          "line-width": ["interpolate", ["linear"], ["zoom"], 6, 1.5, 12, 3.5] as unknown as maplibregl.ExpressionSpecification,
          "line-opacity": 0.9,
          // The dash array is animated to make the air appear to flow toward
          // the ward — the TripsLayer effect, without a second renderer.
          "line-dasharray": [0, 2, 3],
        },
      });
    }

    if (!m.getSource(SRC_STATIONS)) {
      m.addSource(SRC_STATIONS, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      m.addLayer({
        id: LYR_STATIONS,
        type: "circle",
        source: SRC_STATIONS,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 3, 12, 6, 15, 9] as unknown as maplibregl.ExpressionSpecification,
          "circle-color": ["get", "color"] as unknown as maplibregl.ExpressionSpecification,
          // White ring reads as "instrument" against the choropleth beneath.
          "circle-stroke-color": "#F8FAFC",
          "circle-stroke-width": 1.2,
        },
      });
    }

    // Evidence on top of everything: it is the thing being pointed at.
    if (!m.getSource(SRC_EVIDENCE)) {
      m.addSource(SRC_EVIDENCE, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      m.addLayer({
        id: LYR_EVIDENCE,
        type: "circle",
        source: SRC_EVIDENCE,
        paint: {
          "circle-radius": [
            "case", ["boolean", ["get", "hovered"], false], 11, 6,
          ] as unknown as maplibregl.ExpressionSpecification,
          "circle-color": ["get", "color"] as unknown as maplibregl.ExpressionSpecification,
          "circle-opacity": 0.85,
          "circle-stroke-color": "#F8FAFC",
          "circle-stroke-width": [
            "case", ["boolean", ["get", "hovered"], false], 2.5, 1,
          ] as unknown as maplibregl.ExpressionSpecification,
        },
      });
    }

    // ---- interaction ------------------------------------------------------
    let hovered: string | number | undefined;

    const clearHover = () => {
      if (hovered !== undefined) {
        m.setFeatureState({ source: SRC_WARDS, id: hovered }, { hover: false });
        hovered = undefined;
      }
      setHover(null);
      hoverWard(null);
      m.getCanvas().style.cursor = "";
    };

    const onMove = (e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
      const f = e.features?.[0];
      if (!f) return clearHover();
      const wardId = f.properties?.ward_id as string;
      const ward = readingsRef.current.get(wardId);
      if (!ward) return clearHover();

      if (hovered !== undefined && hovered !== f.id) {
        m.setFeatureState({ source: SRC_WARDS, id: hovered }, { hover: false });
      }
      hovered = f.id;
      m.setFeatureState({ source: SRC_WARDS, id: f.id! }, { hover: true });
      m.getCanvas().style.cursor = "pointer";
      hoverWard(wardId);
      setHover({ x: e.point.x, y: e.point.y, ward });
    };

    const onClick = (e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
      const f = e.features?.[0];
      if (f?.properties?.ward_id) selectWard(f.properties.ward_id as string);
    };

    m.on("mousemove", LYR_WARD_FILL, onMove);
    m.on("mouseleave", LYR_WARD_FILL, clearHover);
    m.on("click", LYR_WARD_FILL, onClick);

    return () => {
      m.off("mousemove", LYR_WARD_FILL, onMove);
      m.off("mouseleave", LYR_WARD_FILL, clearHover);
      m.off("click", LYR_WARD_FILL, onClick);
    };
  }, [styleReady, hoverWard, selectWard]);

  // ---- push ward geometry -------------------------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    const src = m.getSource(SRC_WARDS) as maplibregl.GeoJSONSource | undefined;
    if (!src) return;
    src.setData(
      wards?.features?.length
        ? ({ type: "FeatureCollection", features: wards.features } as GeoJSON.FeatureCollection)
        : { type: "FeatureCollection", features: [] },
    );
  }, [wards, styleReady]);

  // ---- push ward values as feature-state ----------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady || !wards?.features?.length) return;
    // removeFeatureState clears stale values from the previous city before the
    // new ones land — otherwise a ward id present in both would keep its colour.
    m.removeFeatureState({ source: SRC_WARDS });
    readings.forEach((w, id) => {
      m.setFeatureState({ source: SRC_WARDS, id }, { aqi: w.aqi ?? null });
    });
  }, [readings, wards, styleReady]);

  // ---- selection outline --------------------------------------------------
  const prevSelected = useRef<string | null>(null);
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    if (prevSelected.current) {
      m.setFeatureState({ source: SRC_WARDS, id: prevSelected.current }, { selected: false });
    }
    if (selectedWardId) {
      m.setFeatureState({ source: SRC_WARDS, id: selectedWardId }, { selected: true });
    }
    prevSelected.current = selectedWardId;
  }, [selectedWardId, styleReady]);

  // ---- push stations ------------------------------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    const src = m.getSource(SRC_STATIONS) as maplibregl.GeoJSONSource | undefined;
    src?.setData(stationGeoJSON as GeoJSON.FeatureCollection);
  }, [stationGeoJSON, styleReady]);

  // ---- trajectory + cone --------------------------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    const line = m.getSource(SRC_TRAJ) as maplibregl.GeoJSONSource | undefined;
    const cone = m.getSource(SRC_CONE) as maplibregl.GeoJSONSource | undefined;
    const empty = { type: "FeatureCollection", features: [] } as GeoJSON.FeatureCollection;

    if (!trajectory?.features?.length) {
      line?.setData(empty);
      cone?.setData(empty);
      return;
    }
    const feats = trajectory.features.filter((f) => f.geometry);
    line?.setData({
      type: "FeatureCollection",
      features: feats.filter((f) => f.geometry?.type === "LineString"),
    } as unknown as GeoJSON.FeatureCollection);
    cone?.setData({
      type: "FeatureCollection",
      features: feats.filter((f) => f.geometry?.type === "Polygon"),
    } as unknown as GeoJSON.FeatureCollection);
  }, [trajectory, styleReady]);

  // Flowing dash: the signature "air moving toward the ward" animation.
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady || !trajectory?.features?.length) return;

    // Dash sequence cycles so the gaps march along the line.
    const steps = [
      [0, 4, 3], [0.5, 4, 2.5], [1, 4, 2], [1.5, 4, 1.5],
      [2, 4, 1], [2.5, 4, 0.5], [3, 4, 0], [0, 0.5, 3, 3.5],
      [0, 1, 3, 3], [0, 1.5, 3, 2.5], [0, 2, 3, 2], [0, 2.5, 3, 1.5],
      [0, 3, 3, 1], [0, 3.5, 3, 0.5],
    ];
    let i = 0;
    let raf = 0;
    let last = 0;
    const tick = (t: number) => {
      // ~60ms per frame: readable motion without strobing.
      if (t - last > 60) {
        last = t;
        i = (i + 1) % steps.length;
        if (m.getLayer(LYR_TRAJ)) {
          m.setPaintProperty(LYR_TRAJ, "line-dasharray", steps[i]);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [trajectory, styleReady]);

  // ---- evidence points ----------------------------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    const src = m.getSource(SRC_EVIDENCE) as maplibregl.GeoJSONSource | undefined;
    const pts = (evidence ?? []).filter((e) => e.lat != null && e.lon != null);
    src?.setData({
      type: "FeatureCollection",
      features: pts.map((e) => ({
        type: "Feature" as const,
        properties: {
          label: e.label,
          color: EVIDENCE_COLOR[e.type] ?? "#94A3B8",
          hovered:
            hoveredEvidence != null &&
            hoveredEvidence.lat === e.lat &&
            hoveredEvidence.lon === e.lon,
        },
        geometry: { type: "Point" as const, coordinates: [e.lon!, e.lat!] },
      })),
    } as GeoJSON.FeatureCollection);
  }, [evidence, hoveredEvidence, styleReady]);

  // ---- fly to a focused evidence item --------------------------------------
  useEffect(() => {
    if (!map.current || !styleReady || !flyTo) return;
    map.current.flyTo({ center: [flyTo.lon, flyTo.lat], zoom: 10.5, duration: 1200, essential: true });
  }, [flyTo, styleReady]);

  // ---- reveal the source: fit to the trajectory + its evidence -------------
  // Without this the payoff is invisible. Delhi's stubble fires sit 100-260 km
  // upwind, far outside a city-zoom viewport, so a user clicking "open burning"
  // would see an empty map and a list of coordinates. Framing the whole path is
  // the moment the product makes its argument.
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;

    const line = trajectory?.features?.find((f) => f.geometry?.type === "LineString");
    const coords = (line?.geometry?.coordinates as [number, number][] | undefined) ?? [];
    const pts: [number, number][] = [
      ...coords,
      ...(evidence ?? [])
        .filter((e) => e.lat != null && e.lon != null)
        .map((e) => [e.lon!, e.lat!] as [number, number]),
    ];
    if (pts.length < 2) return;

    const b = pts.reduce(
      (acc, p) => acc.extend(p),
      new maplibregl.LngLatBounds(pts[0], pts[0]),
    );
    m.fitBounds(b, { padding: { top: 70, bottom: 60, left: 300, right: 400 }, duration: 1100, maxZoom: 10 });
  }, [trajectory, evidence, styleReady]);

  // ---- layer toggles ------------------------------------------------------
  useEffect(() => {
    const m = map.current;
    if (!m || !styleReady) return;
    const set = (id: string, on: boolean) => {
      if (m.getLayer(id)) m.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    };
    set(LYR_WARD_FILL, layersOn.wardChoropleth);
    set(LYR_WARD_LINE, layersOn.wardChoropleth);
    set(LYR_STATIONS, layersOn.stations);
    set(LYR_TRAJ, layersOn.trajectories);
    set(LYR_CONE, layersOn.trajectories);
    set(LYR_EVIDENCE, layersOn.fires);
  }, [layersOn, styleReady]);

  // ---- city switch: fly, don't rebuild ------------------------------------
  useEffect(() => {
    if (!map.current || !styleReady) return;
    map.current.flyTo({
      center: city.map_center as [number, number],
      zoom: city.map_zoom,
      duration: 900,
      essential: true,
    });
    setHover(null);
  }, [city.id, city.map_center, city.map_zoom, styleReady]);

  return (
    <div className="absolute inset-0">
      <div ref={container} className="h-full w-full" data-testid="map-canvas" />
      {hover && <WardTooltip x={hover.x} y={hover.y} ward={hover.ward} />}
      {mapError && (
        <div className="pointer-events-none absolute bottom-16 left-1/2 z-30 -translate-x-1/2">
          <p className="rounded border border-hazard/40 bg-hazard/10 px-2.5 py-1.5 text-[11px] text-hazard">
            Basemap unavailable — ward data still shown. {mapError}
          </p>
        </div>
      )}
    </div>
  );
}
