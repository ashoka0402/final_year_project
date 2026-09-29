"use client";

import { bandFor, readableOn } from "@/lib/aqi";
import type { Ward } from "@/lib/types";

/** Hover tooltip: name, AQI, category (App Flow §3.1). */
export function WardTooltip({ x, y, ward }: { x: number; y: number; ward: Ward }) {
  const band = bandFor(ward.aqi);

  return (
    <div
      className="pointer-events-none absolute z-40 min-w-[190px] -translate-x-1/2 -translate-y-[calc(100%+12px)] rounded-lg border border-edge bg-surface/95 p-2.5 shadow-2xl backdrop-blur"
      style={{ left: x, top: y }}
      role="tooltip"
    >
      <p className="truncate text-xs font-semibold text-slate-100">{ward.name}</p>
      <p className="mt-0.5 font-mono text-[10px] text-slate-500">{ward.ward_id}</p>

      {ward.aqi != null && band ? (
        <div className="mt-2 flex items-center gap-2">
          <span
            className="numeral rounded px-1.5 py-0.5 text-base font-bold leading-none"
            style={{ background: band.color, color: readableOn(band.color) }}
          >
            {ward.aqi}
          </span>
          {/* Label always travels with the colour (PRD: never colour alone). */}
          <div className="min-w-0">
            <p className="text-[11px] font-medium text-slate-200">{band.label}</p>
            <p className="numeral text-[10px] text-slate-500">PM2.5 {ward.pm25?.toFixed(0)} µg/m³</p>
          </div>
        </div>
      ) : (
        <p className="mt-2 text-[11px] text-slate-500">No reading — nearest station too far</p>
      )}

      <div className="mt-2 flex items-center justify-between border-t border-edge pt-1.5 text-[10px] text-slate-500">
        <span className="numeral">{ward.population.toLocaleString("en-IN")} people</span>
        {ward.nearest_station_km != null && (
          <span className="numeral">{ward.nearest_station_km} km to station</span>
        )}
      </div>

      {ward.low_confidence && (
        <p className="mt-1.5 rounded bg-warn/10 px-1.5 py-1 text-[10px] leading-tight text-warn">
          Low confidence — interpolated from a distant station
        </p>
      )}
    </div>
  );
}
