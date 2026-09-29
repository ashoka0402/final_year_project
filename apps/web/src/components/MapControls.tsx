"use client";

import { Flame, Layers, MapPin, Moon, Route, Satellite, Sun, Thermometer } from "lucide-react";

import { AQI_BANDS, readableOn } from "@/lib/aqi";
import { cn } from "@/lib/cn";
import { type Basemap, type LayerId, useCommandStore } from "@/store/useCommandStore";

const BASEMAPS: { id: Basemap; label: string; icon: React.ReactNode }[] = [
  { id: "dark", label: "Dark", icon: <Moon className="h-3 w-3" aria-hidden /> },
  { id: "light", label: "Light", icon: <Sun className="h-3 w-3" aria-hidden /> },
  { id: "satellite", label: "Satellite", icon: <Satellite className="h-3 w-3" aria-hidden /> },
];

/** Basemap style switcher — dark / light / satellite (App Flow: map controls). */
export function BasemapSwitcher() {
  const basemap = useCommandStore((s) => s.basemap);
  const setBasemap = useCommandStore((s) => s.setBasemap);
  return (
    <div className="panel flex items-center gap-1 p-1">
      {BASEMAPS.map((b) => (
        <button
          key={b.id}
          onClick={() => setBasemap(b.id)}
          aria-pressed={basemap === b.id}
          title={`${b.label} basemap`}
          data-testid={`basemap-${b.id}`}
          className={cn(
            "flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium transition-colors duration-150",
            basemap === b.id
              ? "bg-data/15 text-data"
              : "text-slate-400 hover:bg-surface-2 hover:text-slate-200",
          )}
        >
          {b.icon}
          {b.label}
        </button>
      ))}
    </div>
  );
}

const LAYERS: { id: LayerId; label: string; icon: React.ReactNode; phase?: string; hint?: string }[] = [
  { id: "wardChoropleth", label: "Ward AQI", icon: <Layers className="h-3 w-3" aria-hidden /> },
  { id: "stations", label: "Stations", icon: <MapPin className="h-3 w-3" aria-hidden /> },
  {
    id: "fires",
    label: "Evidence",
    icon: <Flame className="h-3 w-3" aria-hidden />,
    hint: "Fire pixels, industry and permits behind the selected ward's attribution",
  },
  {
    id: "trajectories",
    label: "Trajectory",
    icon: <Route className="h-3 w-3" aria-hidden />,
    hint: "Back-trajectory and dispersion cone for the selected ward",
  },
  { id: "heatGrid", label: "Heat grid", icon: <Thermometer className="h-3 w-3" aria-hidden />, phase: "Phase 6" },
];

/** Layer toggle chips, top-left of the map (master prompt §8, Screen 1). */
export function LayerChips() {
  const layers = useCommandStore((s) => s.layers);
  const toggle = useCommandStore((s) => s.toggleLayer);

  return (
    <div className="panel flex items-center gap-1 p-1">
      {LAYERS.map((l) => {
        const locked = Boolean(l.phase);
        const on = layers[l.id];
        return (
          <button
            key={l.id}
            disabled={locked}
            onClick={() => toggle(l.id)}
            aria-pressed={on}
            title={locked ? `${l.label} — ${l.phase}` : l.hint ?? `Toggle ${l.label}`}
            data-testid={`layer-${l.id}`}
            className={cn(
              "flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium transition-colors duration-150",
              locked && "cursor-not-allowed text-slate-600",
              !locked && on && "bg-data/15 text-data",
              !locked && !on && "text-slate-400 hover:bg-surface-2 hover:text-slate-200",
            )}
          >
            {l.icon}
            {l.label}
            {locked && (
              <span className="rounded-sm bg-slate-800 px-1 text-[8px] font-semibold uppercase tracking-wide text-slate-500">
                Soon
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/** CPCB legend — number + label, never colour alone (PRD accessibility). */
export function AqiLegend() {
  return (
    <div className="panel px-2 py-1.5">
      <p className="mb-1 text-[9px] font-medium uppercase tracking-wider text-slate-500">
        CPCB AQI
      </p>
      <div className="flex items-center gap-px">
        {AQI_BANDS.map((b) => (
          <div key={b.label} className="group relative">
            <div
              className="flex h-5 w-[52px] items-center justify-center text-[9px] font-semibold"
              style={{ background: b.color, color: readableOn(b.color) }}
            >
              {b.min}–{b.max}
            </div>
            <span className="mt-0.5 block text-center text-[8px] leading-tight text-slate-500">
              {b.label}
            </span>
            <div className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-1 hidden w-44 -translate-x-1/2 rounded border border-edge bg-surface p-1.5 text-[10px] leading-relaxed text-slate-300 shadow-xl group-hover:block">
              {b.note}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Footer ticker — business impact, stated where the work happens (§12). */
export function ImpactTicker() {
  return (
    <div className="pointer-events-none flex items-center gap-2 text-[10px] text-slate-600">
      <span>1.67M premature deaths/yr (Lancet)</span>
      <span aria-hidden>·</span>
      <span>900+ CAAQMS stations</span>
      <span aria-hidden>·</span>
      <span>69% of monitored cities lack response protocols (CAG 2024)</span>
    </div>
  );
}
