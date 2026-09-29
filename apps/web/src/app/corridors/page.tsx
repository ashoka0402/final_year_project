"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { TopNav } from "@/components/TopNav";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { api, ApiError, queryKeys } from "@/lib/api";
import { cn } from "@/lib/cn";

/**
 * Federated corridor bulletins.
 *
 * The point this page has to make visually: pollution does not respect state
 * lines, so the analysis unit is the corridor, and the bulletin is a document
 * several states can act on together. Hence the states are listed prominently,
 * and coverage is shown next to every number — a corridor the satellite could
 * not see must never be mistaken for a clean one.
 */

const DEFAULT_DATE = "2025-11-04"; // peak of the 2025 burning season in the archive

export default function CorridorsPage() {
  const cities = useQuery({ queryKey: queryKeys.cities, queryFn: api.cities });
  const list = useQuery({ queryKey: queryKeys.corridors("india"), queryFn: () => api.corridors("india") });

  const [selected, setSelected] = useState("agra_kanpur_igp");
  const [date, setDate] = useState(DEFAULT_DATE);

  const bulletin = useQuery({
    queryKey: queryKeys.corridorBulletin(selected, date),
    queryFn: () => api.corridorBulletin(selected, date),
    enabled: Boolean(selected && date),
    retry: false,
  });

  return (
    <div className="flex h-dvh flex-col bg-base">
      <TopNav cities={cities.data ?? []} loading={cities.isLoading} />
      <main className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-6 py-6">
        <header className="mb-5">
          <h1 className="text-xl font-semibold text-slate-100">Economic corridors</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-500">
            Pollution follows freight and wind, not municipal boundaries — the
            Amritsar–Kolkata spine alone crosses seven states. Each corridor gets a
            daily bulletin in a versioned, self-describing format that any state
            agency can consume over plain HTTP, without adopting our database, our
            models, or our code.
          </p>
          <p className="mt-2 inline-block rounded border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300/90">
            Historical case study, not live — satellite/fire coverage here is
            the archived Oct–Nov 2025 stubble-burning season. City-level AQI
            elsewhere in VAYU is live; extending this corridor view to a
            live feed needs a standing daily ingestion job, which isn&apos;t
            running yet.
          </p>
        </header>

        {list.isPending && <Skeleton className="h-20 w-full" />}
        {list.error && (
          <ErrorState
            title="Could not load corridors"
            detail={(list.error as ApiError).message}
            onRetry={() => list.refetch()}
          />
        )}

        {/* corridor picker */}
        <div className="mb-4 flex flex-wrap gap-2">
          {list.data?.corridors.map((c) => (
            <button
              key={c.id}
              onClick={() => setSelected(c.id)}
              data-testid={`corridor-${c.id}`}
              className={cn(
                "rounded-md border px-3 py-2 text-left transition-colors",
                selected === c.id
                  ? "border-data/50 bg-data/10"
                  : "border-edge bg-surface-2 hover:border-data/40",
              )}
            >
              <span className="block text-xs font-medium text-slate-100">{c.name}</span>
              <span className="numeral block text-[10px] text-slate-500">
                {c.states.length} states · {c.cells} cells
              </span>
            </button>
          ))}
        </div>

        <label className="mb-4 inline-block">
          <span className="mb-1 block text-[9px] uppercase tracking-wider text-slate-500">
            Bulletin date (archive: Oct–Nov 2025)
          </span>
          <input
            type="date"
            value={date}
            min="2025-10-01"
            max="2025-11-25"
            onChange={(e) => setDate(e.target.value)}
            className="rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs text-slate-100"
          />
        </label>

        {bulletin.isPending && <Skeleton className="h-64 w-full" />}
        {bulletin.error && (
          <ErrorState
            title="No bulletin for that day"
            detail={(bulletin.error as ApiError).message}
            onRetry={() => bulletin.refetch()}
          />
        )}

        {bulletin.data && (
          <>
            <div className="panel mb-4 p-4">
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-sm font-semibold text-slate-100">
                  {bulletin.data.corridor.name}
                </h2>
                <span className="numeral text-[10px] text-slate-600">
                  {bulletin.data.schema} · issued {bulletin.data.issued_utc.slice(0, 16)}Z
                </span>
              </div>
              <div className="mb-3 flex flex-wrap gap-1">
                {bulletin.data.corridor.states.map((s) => (
                  <span key={s} className="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] text-slate-400">
                    {s}
                  </span>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat
                  label="Satellite coverage"
                  value={`${bulletin.data.coverage.coverage_pct}%`}
                  sub={`${bulletin.data.coverage.cells_observed}/${bulletin.data.coverage.cells_total} cells`}
                />
                <Stat
                  label="HCHO hotspots"
                  value={String(bulletin.data.hcho.hotspot_cells)}
                  sub={
                    bulletin.data.hcho.max_anomaly_sigma != null
                      ? `max ${bulletin.data.hcho.max_anomaly_sigma}σ`
                      : "none flagged"
                  }
                  accent={bulletin.data.hcho.hotspot_cells > 0}
                />
                <Stat
                  label="Active fires"
                  value={bulletin.data.fire.count.toLocaleString("en-IN")}
                  sub="VIIRS/MODIS"
                  accent={bulletin.data.fire.count > 0}
                />
                <Stat
                  label="Citizen reports"
                  value={String(bulletin.data.citizen.reports)}
                  sub={`${bulletin.data.citizen.satellite_corroborated} corroborated`}
                />
              </div>

              <p className="mt-3 border-t border-edge pt-2 text-[10px] leading-relaxed text-slate-600">
                {bulletin.data.hcho.source} · {bulletin.data.fire.source} ·{" "}
                {bulletin.data.citizen.note}
              </p>
            </div>

            <h3 className="mb-2 text-sm font-semibold text-slate-200">Strongest anomalies</h3>
            {bulletin.data.top_hotspots.length === 0 ? (
              <EmptyState
                title="No hotspots on this day"
                hint={
                  bulletin.data.coverage.coverage_pct < 50
                    ? "Note the low satellite coverage — this may be cloud, not clean air."
                    : "The corridor looked ordinary against its own baseline."
                }
              />
            ) : (
              <ul className="space-y-2">
                {bulletin.data.top_hotspots.map((h, i) => (
                  <li key={i} className="panel flex items-center justify-between gap-3 p-3">
                    <div>
                      <p className="numeral text-xs text-slate-200">
                        {h.lat.toFixed(3)}, {h.lon.toFixed(3)}
                      </p>
                      <p className="text-[10px] text-slate-500">
                        {h.source_region ? h.source_region.replace(/_/g, " ") : "outside defined source regions"}
                        {h.fire_count > 0 && ` · ${h.fire_count} fires`}
                      </p>
                    </div>
                    <span className="numeral rounded bg-amber-500/15 px-2 py-1 text-xs font-semibold text-amber-300">
                      {h.anomaly_sigma}σ
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div>
      <p className="text-[9px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className={cn("numeral text-lg font-semibold", accent ? "text-amber-300" : "text-slate-100")}>
        {value}
      </p>
      <p className="text-[10px] text-slate-600">{sub}</p>
    </div>
  );
}
