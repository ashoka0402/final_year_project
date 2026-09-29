import { create } from "zustand";

import type { AttributionEvidence, SourceCategory } from "@/lib/types";

/**
 * Map/UI state (TRD §9: Zustand for map & UI, TanStack Query for server cache).
 * Layer toggles live here so they persist across a city switch.
 */
export type LayerId = "wardChoropleth" | "stations" | "heatGrid" | "fires" | "trajectories";
export type Basemap = "dark" | "light" | "satellite";

interface CommandState {
  cityId: string;
  selectedWardId: string | null;
  hoveredWardId: string | null;
  layers: Record<LayerId, boolean>;
  basemap: Basemap;
  /** Attribution surface state (Phase 3). */
  trajectoryHours: number;
  selectedCategory: SourceCategory | null;
  hoveredEvidence: AttributionEvidence | null;
  flyTo: { lon: number; lat: number } | null;
  setCity: (id: string) => void;
  selectWard: (id: string | null) => void;
  hoverWard: (id: string | null) => void;
  toggleLayer: (id: LayerId) => void;
  setTrajectoryHours: (h: number) => void;
  selectCategory: (c: SourceCategory | null) => void;
  setHoveredEvidence: (e: AttributionEvidence | null) => void;
  setFlyTo: (p: { lon: number; lat: number } | null) => void;
  setBasemap: (b: Basemap) => void;
}

export const useCommandStore = create<CommandState>((set) => ({
  cityId: "delhi",
  selectedWardId: null,
  hoveredWardId: null,
  layers: {
    wardChoropleth: true,
    stations: true,
    // Off until the agents that produce them land (Phases 2-3). Shown in the
    // layer bar as disabled so the roadmap is legible rather than invisible.
    heatGrid: false,
    // On by default: selecting a ward should show its evidence and trajectory
    // immediately — that IS the product.
    fires: true,
    trajectories: true,
  },
  trajectoryHours: 12,
  selectedCategory: null,
  hoveredEvidence: null,
  flyTo: null,
  basemap: "dark",
  // Switching city must clear ward selection: ward ids are per-city.
  setCity: (id) =>
    set({ cityId: id, selectedWardId: null, hoveredWardId: null, selectedCategory: null, hoveredEvidence: null }),
  // A new ward invalidates the previous ward's category/evidence selection.
  selectWard: (id) => set({ selectedWardId: id, selectedCategory: null, hoveredEvidence: null }),
  hoverWard: (id) => set({ hoveredWardId: id }),
  toggleLayer: (id) =>
    set((s) => ({ layers: { ...s.layers, [id]: !s.layers[id] } })),
  setTrajectoryHours: (h) => set({ trajectoryHours: h }),
  selectCategory: (c) => set({ selectedCategory: c }),
  setHoveredEvidence: (e) => set({ hoveredEvidence: e }),
  setFlyTo: (p) => set({ flyTo: p }),
  setBasemap: (b) => set({ basemap: b }),
}));
