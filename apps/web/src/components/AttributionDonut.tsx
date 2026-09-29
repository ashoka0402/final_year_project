"use client";

import { Flame, Factory, HardHat, Car, Wind } from "lucide-react";

import { cn } from "@/lib/cn";
import type { AttributionCategory, SourceCategory } from "@/lib/types";

/**
 * Attribution donut with confidence rings (App Flow §3.2).
 *
 * The confidence ring is the honest part: two wards can both read "42% open
 * burning" while one is backed by twelve fire pixels under a steady wind and the
 * other by one pixel under a wandering breeze. The share alone would present
 * those as identical claims, so confidence is drawn as an outer arc rather than
 * buried in a tooltip.
 *
 * Hand-rolled SVG rather than a chart library: a donut is two arcs of
 * arithmetic, and the ring geometry is not something Recharts exposes.
 */

export const CATEGORY_COLOR: Record<SourceCategory, string> = {
  open_burning: "#EF4444",
  traffic: "#F59E0B",
  construction: "#A78BFA",
  industry: "#22D3EE",
  regional_transport: "#64748B",
};

export const CATEGORY_ICON: Record<SourceCategory, React.ComponentType<{ className?: string }>> = {
  open_burning: Flame,
  traffic: Car,
  construction: HardHat,
  industry: Factory,
  regional_transport: Wind,
};

const SIZE = 168;
const CX = SIZE / 2;
const CY = SIZE / 2;
const R_OUTER = 66;
const R_INNER = 44;
const R_RING = 76; // confidence arc sits outside the donut

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(r0: number, r1: number, start: number, end: number): string {
  // Guard the 360° case: an arc that starts and ends at the same point draws nothing.
  const sweep = Math.min(end - start, 359.99);
  const e = start + sweep;
  const large = sweep > 180 ? 1 : 0;
  const p0 = polar(CX, CY, r1, start);
  const p1 = polar(CX, CY, r1, e);
  const p2 = polar(CX, CY, r0, e);
  const p3 = polar(CX, CY, r0, start);
  return [
    `M ${p0.x} ${p0.y}`,
    `A ${r1} ${r1} 0 ${large} 1 ${p1.x} ${p1.y}`,
    `L ${p2.x} ${p2.y}`,
    `A ${r0} ${r0} 0 ${large} 0 ${p3.x} ${p3.y}`,
    "Z",
  ].join(" ");
}

function ringPath(r: number, start: number, end: number): string {
  const sweep = Math.min(end - start, 359.99);
  const e = start + sweep;
  const large = sweep > 180 ? 1 : 0;
  const p0 = polar(CX, CY, r, start);
  const p1 = polar(CX, CY, r, e);
  return `M ${p0.x} ${p0.y} A ${r} ${r} 0 ${large} 1 ${p1.x} ${p1.y}`;
}

export function AttributionDonut({
  categories,
  selected,
  onSelect,
}: {
  categories: AttributionCategory[];
  selected: SourceCategory | null;
  onSelect: (c: SourceCategory | null) => void;
}) {
  const total = categories.reduce((s, c) => s + c.share_pct, 0) || 1;
  let cursor = 0;
  const slices = categories.map((c) => {
    const start = cursor;
    const sweep = (c.share_pct / total) * 360;
    cursor += sweep;
    return { c, start, end: start + sweep };
  });

  const dominant = categories[0];
  const active = selected ? categories.find((c) => c.category === selected) : null;
  const shown = active ?? dominant;

  return (
    <div className="flex flex-col items-center">
      <svg width={SIZE} height={SIZE} role="img" aria-label="Source attribution">
        {slices.map(({ c, start, end }) => {
          const isSel = selected === c.category;
          const dim = selected !== null && !isSel;
          return (
            <g key={c.category}>
              {/* share slice */}
              <path
                d={arcPath(R_INNER, isSel ? R_OUTER + 4 : R_OUTER, start, end)}
                fill={CATEGORY_COLOR[c.category]}
                opacity={dim ? 0.25 : 0.9}
                className="cursor-pointer transition-opacity duration-150"
                onClick={() => onSelect(isSel ? null : c.category)}
                data-testid={`slice-${c.category}`}
              >
                <title>{`${c.label}: ${c.share_pct}% (confidence ${c.confidence})`}</title>
              </path>
              {/* confidence ring: arc length = confidence x slice width */}
              <path
                d={ringPath(R_RING, start + 1, start + 1 + Math.max((end - start - 2) * c.confidence, 0.5))}
                stroke={CATEGORY_COLOR[c.category]}
                strokeWidth={3}
                strokeLinecap="round"
                fill="none"
                opacity={dim ? 0.2 : 1}
              />
              {/* the ring's unfilled remainder — the doubt, drawn explicitly */}
              <path
                d={ringPath(R_RING, start + 1, start + Math.max(end - start - 1, 0.5))}
                stroke={CATEGORY_COLOR[c.category]}
                strokeWidth={3}
                strokeLinecap="round"
                fill="none"
                opacity={dim ? 0.05 : 0.18}
              />
            </g>
          );
        })}

        {/* centre: dominant or selected source */}
        {shown && (
          <>
            <text
              x={CX}
              y={CY - 6}
              textAnchor="middle"
              className="fill-slate-50 font-mono text-[20px] font-bold"
            >
              {shown.share_pct}%
            </text>
            <text x={CX} y={CY + 9} textAnchor="middle" className="fill-slate-400 text-[8px]">
              {shown.label.toUpperCase()}
            </text>
            <text x={CX} y={CY + 21} textAnchor="middle" className="fill-slate-500 font-mono text-[8px]">
              conf {shown.confidence.toFixed(2)}
            </text>
          </>
        )}
      </svg>

      {/* legend — click to filter evidence */}
      <div className="mt-2 grid w-full grid-cols-2 gap-x-2 gap-y-1">
        {categories.map((c) => {
          const Icon = CATEGORY_ICON[c.category];
          const isSel = selected === c.category;
          return (
            <button
              key={c.category}
              onClick={() => onSelect(isSel ? null : c.category)}
              data-testid={`legend-${c.category}`}
              className={cn(
                "flex items-center gap-1.5 rounded px-1 py-0.5 text-left transition-colors",
                isSel ? "bg-surface-2" : "hover:bg-surface-2/60",
              )}
            >
              <span
                className="h-2 w-2 shrink-0 rounded-sm"
                style={{ background: CATEGORY_COLOR[c.category] }}
                aria-hidden
              />
              <Icon className="h-3 w-3 shrink-0 text-slate-500" />
              <span className="min-w-0 flex-1 truncate text-[10px] text-slate-300">{c.label}</span>
              <span className="numeral shrink-0 text-[10px] font-semibold text-slate-200">
                {c.share_pct}%
              </span>
            </button>
          );
        })}
      </div>

      <p className="mt-1.5 text-center text-[9px] leading-relaxed text-slate-600">
        Outer arc = confidence in that share. A full arc means the evidence is strong and the
        wind field steady; a short one means treat the number with caution.
      </p>
    </div>
  );
}
