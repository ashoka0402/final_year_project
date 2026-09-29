"use client";

import { ExternalLink, MapPin } from "lucide-react";

import { cn } from "@/lib/cn";
import type { AttributionCategory, AttributionEvidence, SourceCategory } from "@/lib/types";
import { CATEGORY_COLOR, CATEGORY_ICON } from "./AttributionDonut";
import { EmptyState } from "./ui/States";

/**
 * The evidence behind a share (PRD B2: "100% of attribution percentages click
 * through to concrete evidence").
 *
 * Hovering an item pulses its marker on the map; clicking flies to it. Anything
 * curated carries its "Sample data" badge here — the badge rides on the
 * evidence item from the API, so this component cannot forget to show it.
 */

function relativeTime(iso: string | null, now: Date): string | null {
  if (!iso) return null;
  const h = (now.getTime() - new Date(iso).getTime()) / 3_600_000;
  if (h < 1) return "just now";
  if (h < 24) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export function EvidenceList({
  categories,
  selected,
  asOf,
  onHover,
  onFocus,
}: {
  categories: AttributionCategory[];
  selected: SourceCategory | null;
  asOf: string;
  onHover: (e: AttributionEvidence | null) => void;
  onFocus: (e: AttributionEvidence) => void;
}) {
  const now = new Date(asOf);
  const shown = selected ? categories.filter((c) => c.category === selected) : categories;
  const items = shown.flatMap((c) => c.evidence.map((e) => ({ e, cat: c.category })));

  if (!items.length) {
    return (
      <EmptyState
        title="No evidence items"
        hint={
          selected
            ? "This source scored from geometry alone — see the formula on Methodology."
            : "Nothing in the trajectory cone for this window."
        }
        className="py-4"
      />
    );
  }

  return (
    <ul className="space-y-1">
      {items.map(({ e, cat }, i) => {
        const Icon = CATEGORY_ICON[cat];
        const isSample = (e.source ?? "").toLowerCase().includes("sample");
        const located = e.lat != null && e.lon != null;
        return (
          <li key={`${cat}-${i}`}>
            <button
              onMouseEnter={() => onHover(e)}
              onMouseLeave={() => onHover(null)}
              onClick={() => located && onFocus(e)}
              disabled={!located}
              data-testid={`evidence-${cat}-${i}`}
              className={cn(
                "w-full rounded border border-edge bg-surface-2/40 p-1.5 text-left transition-colors",
                located ? "cursor-pointer hover:border-slate-600 hover:bg-surface-2" : "cursor-default",
              )}
            >
              <div className="flex items-start gap-1.5">
                <span
                  className="mt-0.5 rounded p-0.5"
                  style={{ background: `${CATEGORY_COLOR[cat]}22` }}
                >
                  <Icon className="h-2.5 w-2.5" />
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[11px] font-medium text-slate-200">{e.label}</p>
                  {e.detail && (
                    <p className="mt-0.5 line-clamp-2 text-[9px] leading-relaxed text-slate-500">
                      {e.detail}
                    </p>
                  )}
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-1.5 text-[9px] text-slate-600">
                    {e.distance_km != null && <span className="numeral">{e.distance_km} km</span>}
                    {relativeTime(e.timestamp, now) && (
                      <>
                        <span aria-hidden>·</span>
                        <span className="numeral">{relativeTime(e.timestamp, now)}</span>
                      </>
                    )}
                    {e.source && !isSample && (
                      <>
                        <span aria-hidden>·</span>
                        <span className="truncate">{e.source}</span>
                      </>
                    )}
                  </div>
                  {/* The badge travels with the evidence from the API. */}
                  {isSample && (
                    <span className="mt-1 inline-block rounded bg-warn/15 px-1 py-0.5 text-[8px] font-semibold uppercase tracking-wide text-warn">
                      Sample data
                    </span>
                  )}
                </div>
                {located && <MapPin className="mt-0.5 h-2.5 w-2.5 shrink-0 text-slate-600" aria-hidden />}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
