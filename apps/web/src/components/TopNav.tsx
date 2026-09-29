"use client";

import { Check, ChevronDown, Wind } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/cn";
import type { City, DataStatus } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";
import { ClockControl } from "./ClockControl";
import { DataPills } from "./DataPills";

/**
 * Top nav (App Flow §1): VAYU ◆ [City switcher] | Command | Interventions |
 * Verify | Methodology + data pills.
 *
 * Routes for phases not yet built are rendered disabled with their phase noted,
 * rather than as dead links — the demo path stays obvious.
 */
const NAV = [
  { label: "Command", href: "/", enabled: true },
  { label: "Interventions", href: "/interventions", enabled: true },
  { label: "Report", href: "/report", enabled: true },
  { label: "Corridors", href: "/corridors", enabled: true },
  { label: "Verify", href: "/verify", enabled: true },
  { label: "Methodology", href: "/methodology", enabled: true },
];

function CitySwitcher({ cities }: { cities: City[] }) {
  const cityId = useCommandStore((s) => s.cityId);
  const setCity = useCommandStore((s) => s.setCity);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onEsc);
    };
  }, []);

  const active = cities.find((c) => c.id === cityId);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        data-testid="city-switcher"
        className="flex items-center gap-1.5 rounded-md border border-edge bg-surface-2 px-2.5 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-data/50"
      >
        {active?.name ?? "Select city"}
        <ChevronDown className={cn("h-3 w-3 text-slate-500 transition-transform", open && "rotate-180")} aria-hidden />
      </button>

      {open && (
        <ul
          role="listbox"
          className="absolute left-0 top-full z-50 mt-1.5 w-56 animate-slide-in-right overflow-hidden rounded-md border border-edge bg-surface shadow-2xl"
        >
          {cities.map((c) => (
            <li key={c.id}>
              <button
                role="option"
                aria-selected={c.id === cityId}
                data-testid={`city-option-${c.id}`}
                onClick={() => {
                  setCity(c.id);
                  setOpen(false);
                }}
                className={cn(
                  "flex w-full items-center justify-between gap-2 px-2.5 py-2 text-left transition-colors hover:bg-surface-2",
                  c.id === cityId && "bg-surface-2",
                )}
              >
                <span>
                  <span className="block text-xs font-medium text-slate-100">{c.name}</span>
                  <span className="numeral block text-[10px] text-slate-500">
                    {c.ward_count} wards · {c.station_count} stations
                  </span>
                </span>
                {c.id === cityId && <Check className="h-3 w-3 text-data" aria-hidden />}
              </button>
            </li>
          ))}
          <li className="border-t border-edge px-2.5 py-1.5">
            {/* The scalability claim, stated where it is demonstrated (PRD G1). */}
            <p className="text-[10px] leading-relaxed text-slate-600">
              A new city is one file in <span className="font-mono">config/cities/</span>.
            </p>
          </li>
        </ul>
      )}
    </div>
  );
}

export function TopNav({
  cities,
  statuses,
  loading,
}: {
  cities: City[];
  statuses?: DataStatus[];
  loading: boolean;
}) {
  return (
    // relative z-40: the header's backdrop-blur creates a stacking context, and
    // <main> is a later sibling — without an explicit z the map and KPI rail
    // paint over the open city-switcher dropdown.
    <header className="relative z-40 flex h-12 shrink-0 items-center justify-between gap-4 border-b border-edge bg-surface/60 px-3 backdrop-blur-md">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <Wind className="h-4 w-4 text-data" aria-hidden />
          <span className="text-sm font-bold tracking-tight text-slate-50">VAYU</span>
          <span className="text-data" aria-hidden>
            ◆
          </span>
        </div>

        {loading ? (
          <div className="h-7 w-24 animate-pulse rounded-md bg-edge/60" />
        ) : (
          <CitySwitcher cities={cities} />
        )}

        <ClockControl />

        <nav className="ml-1 flex items-center gap-0.5" aria-label="Main">
          {NAV.map((item) =>
            item.enabled ? (
              <a
                key={item.label}
                href={item.href}
                aria-current="page"
                className="rounded px-2.5 py-1.5 text-xs font-medium text-slate-100 transition-colors hover:bg-surface-2"
              >
                {item.label}
              </a>
            ) : (
              <span
                key={item.label}
                title={"phase" in item ? `${item.label} — ${item.phase}` : item.label}
                className="cursor-not-allowed rounded px-2.5 py-1.5 text-xs font-medium text-slate-600"
              >
                {item.label}
              </span>
            ),
          )}
          {/* Public surface — a separate URL, linked quietly rather than in the
              commissioner nav proper (App Flow §1). */}
          <a
            href="/citizen"
            className="ml-1 rounded px-2.5 py-1.5 text-xs font-medium text-slate-500 transition-colors hover:bg-surface-2 hover:text-slate-300"
          >
            Citizen ↗
          </a>
        </nav>
      </div>

      <DataPills statuses={statuses} loading={loading} />
    </header>
  );
}
