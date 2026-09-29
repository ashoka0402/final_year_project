"use client";

import { Database } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cn } from "@/lib/cn";
import type { DataStatus, DataStatusValue } from "@/lib/types";
import { Skeleton } from "./ui/States";

/**
 * Per-source freshness (PRD F2 / App Flow §3.1) — the honesty surface.
 *
 * Collapsed into one top-right button so the eight-plus sources never wrap over
 * the map. The button summarises live-vs-sample at a glance; clicking it opens a
 * panel listing every source, its status, age, and what the label means.
 */

const STYLES: Record<DataStatusValue, { dot: string; text: string; label: string }> = {
  live: { dot: "bg-verified", text: "text-verified", label: "live" },
  cached: { dot: "bg-slate-500", text: "text-slate-400", label: "cached" },
  sample: { dot: "bg-warn", text: "text-warn", label: "sample" },
  cams: { dot: "bg-warn", text: "text-warn", label: "reanalysis" },
  "h3-fallback": { dot: "bg-warn", text: "text-warn", label: "zones" },
  unavailable: { dot: "bg-slate-700", text: "text-slate-600", label: "off" },
};

const SOURCE_EXPLAIN: Record<string, string> = {
  permits:
    "No public machine-readable permit feed exists for these cities. Sites are curated on real OSM construction landuse and badged sample; the dust-compliance flag is curated.",
  roads: "OSM major roads, weighted by class — the traffic proxy. There is no free real-time vehicle-count feed for Indian cities.",
  fires: "NASA FIRMS VIIRS active-fire detections.",
  measurements: "Station measurements.",
};

const EXPLAIN: Record<DataStatusValue, string> = {
  live: "Fetched now from the upstream source.",
  cached: "Served from a recent cached response.",
  sample: "Bundled offline copy shipped with the repo.",
  cams: "Modelled ECMWF CAMS reanalysis sampled at real CPCB station coordinates — not a measurement. Add OPENAQ_API_KEY for measured values.",
  "h3-fallback": "Generated hex analysis zones — this city publishes no ward boundaries.",
  unavailable: "No key or no data for this layer; it is hidden rather than faked.",
};

const SOURCE_LABEL: Record<string, string> = {
  measurements: "Air quality",
  stations: "Stations",
  weather: "Weather",
  wards: "Wards",
  fires: "Fires",
  roads: "Roads",
  permits: "Permits",
  osm: "OSM landuse",
  s5p: "S5P NO₂",
};

const ORDER = Object.keys(SOURCE_LABEL);

function relative(iso: string | null): string {
  if (!iso) return "unknown";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.round(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.round(hrs / 24)}d ago`;
}

export function DataPills({ statuses, loading }: { statuses?: DataStatus[]; loading?: boolean }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const ordered = useMemo(
    () =>
      [...(statuses ?? [])].sort(
        (a, b) => (ORDER.indexOf(a.source) + 99) - (ORDER.indexOf(b.source) + 99),
      ),
    [statuses],
  );

  const counts = useMemo(() => {
    let live = 0;
    let other = 0;
    for (const s of ordered) (s.status === "live" ? (live += 1) : (other += 1));
    return { live, other };
  }, [ordered]);

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

  if (loading) return <Skeleton className="h-[30px] w-28 rounded-md" />;
  if (!ordered.length) return null;

  return (
    <div className="relative shrink-0" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        data-testid="data-sources"
        title="Data sources & freshness"
        className="flex items-center gap-1.5 rounded-md border border-edge bg-surface-2 px-2.5 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-data/50"
      >
        <Database className="h-3.5 w-3.5 text-slate-400" aria-hidden />
        <span className="hidden sm:inline">Sources</span>
        {/* At-a-glance dots so the summary reads without opening. */}
        <span className="flex items-center gap-1">
          <span className="flex items-center gap-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-verified" aria-hidden />
            <span className="numeral text-verified">{counts.live}</span>
          </span>
          <span className="flex items-center gap-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-warn" aria-hidden />
            <span className="numeral text-warn">{counts.other}</span>
          </span>
        </span>
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Data sources"
          className="absolute right-0 top-full z-50 mt-1.5 w-80 animate-slide-in-right rounded-md border border-edge bg-surface p-2 shadow-2xl"
        >
          <p className="px-1.5 py-1 text-[10px] font-medium uppercase tracking-wider text-slate-500">
            Data sources & freshness
          </p>
          <ul className="max-h-[70vh] overflow-y-auto">
            {ordered.map((s) => {
              const style = STYLES[s.status] ?? STYLES.unavailable;
              return (
                <li key={s.source} className="rounded px-1.5 py-1.5 hover:bg-surface-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-1.5">
                      <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", style.dot)} aria-hidden />
                      <span className="text-xs font-medium text-slate-200">
                        {SOURCE_LABEL[s.source] ?? s.source}
                      </span>
                    </span>
                    <span className="flex items-center gap-2">
                      <span className={cn("text-[10px] font-medium uppercase tracking-wide", style.text)}>
                        {style.label}
                      </span>
                      <span className="numeral text-[10px] text-slate-600">{relative(s.fetched_ts)}</span>
                    </span>
                  </div>
                  <p className="mt-0.5 pl-3 text-[10px] leading-relaxed text-slate-500">
                    {SOURCE_EXPLAIN[s.source] ?? EXPLAIN[s.status]}
                  </p>
                  {s.detail && (
                    <p className="mt-0.5 pl-3 text-[10px] leading-relaxed text-slate-600">{s.detail}</p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
