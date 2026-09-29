"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, ClipboardList, Info, Timer } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { AdvisoryCard } from "@/components/interventions/AdvisoryCard";
import { CandidateRow } from "@/components/interventions/CandidateRow";
import { GrapCard } from "@/components/interventions/GrapCard";
import { TopNav } from "@/components/TopNav";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { ApiError, api, queryKeys } from "@/lib/api";
import type { ActionType, Candidate, Order } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";

const ACTION_FILTERS: { id: ActionType | null; label: string }[] = [
  { id: null, label: "All actions" },
  { id: "halt_burning", label: "Burning" },
  { id: "stop_work_construction", label: "Construction" },
  { id: "industrial_curb", label: "Industry" },
  { id: "traffic_restriction", label: "Traffic" },
];

export default function InterventionsPage() {
  return (
    <Suspense fallback={<Shell><Skeleton className="h-64 w-full" /></Shell>}>
      <Interventions />
    </Suspense>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const cities = useQuery({ queryKey: queryKeys.cities, queryFn: api.cities });
  return (
    <div className="flex h-dvh flex-col bg-base">
      <TopNav cities={cities.data ?? []} loading={cities.isLoading} />
      <main className="mx-auto w-full max-w-6xl flex-1 overflow-y-auto px-6 py-6">{children}</main>
    </div>
  );
}

function Interventions() {
  const cityId = useCommandStore((s) => s.cityId);
  const params = useSearchParams();
  const wardParam = params.get("ward");
  const [action, setAction] = useState<ActionType | null>(null);
  const [dispatched, setDispatched] = useState<Record<string, Order>>({});
  const qc = useQueryClient();

  const q = useQuery({
    queryKey: queryKeys.interventions(cityId, wardParam, action),
    queryFn: () => api.interventions(cityId, wardParam, action),
  });

  const dispatch = useMutation({
    // No signal_ts from the client: the server derives it from the forecast run
    // that flagged the ward. Passing the page's compute time made the stopwatch
    // read 0m 0s by construction.
    mutationFn: (c: Candidate) => api.dispatch(c),
    onSuccess: (order) => {
      setDispatched((d) => ({ ...d, [order.id]: order }));
      qc.invalidateQueries({ queryKey: queryKeys.orders() });
    },
  });

  return (
    <Shell>
      <header className="mb-5">
        <div className="flex items-baseline justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-slate-100">Interventions</h1>
            <p className="mt-1 text-sm text-slate-500">
              Ranked by <span className="font-mono text-slate-400">µg/m³ averted × people ÷ teams</span>
              {" — "}the leverage each action buys per unit of effort.
            </p>
          </div>
          <Link
            href="/inspector"
            className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-slate-300 hover:border-white/20"
          >
            <ClipboardList className="h-3.5 w-3.5" />
            Inspector
          </Link>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          {wardParam && (
            <span className="rounded-full border border-sky-500/30 bg-sky-500/10 px-2.5 py-1 text-xs text-sky-300">
              Ward {wardParam}
              <Link href="/interventions" className="ml-1.5 text-sky-400/70 hover:text-sky-300">
                ×
              </Link>
            </span>
          )}
          {ACTION_FILTERS.map((f) => (
            <button
              key={f.label}
              onClick={() => setAction(f.id)}
              className={
                action === f.id
                  ? "rounded-full bg-white/10 px-2.5 py-1 text-xs text-slate-200"
                  : "rounded-full px-2.5 py-1 text-xs text-slate-500 hover:text-slate-300"
              }
            >
              {f.label}
            </button>
          ))}
        </div>
      </header>

      {q.isPending && (
        <div className="space-y-2">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      )}

      {q.isError && (
        <ErrorState
          title="Could not compute interventions"
          detail={q.error instanceof ApiError ? q.error.message : String(q.error)}
          onRetry={() => q.refetch()}
        />
      )}

      {q.data && (
        <>
          {dispatch.isError && (
            <div className="mb-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
              Dispatch failed:{" "}
              {dispatch.error instanceof ApiError ? dispatch.error.message : String(dispatch.error)}
            </div>
          )}

          {Object.values(dispatched).map((o) => (
            <DispatchToast key={o.id} order={o} />
          ))}

          {/* Conditional: renders only when a GRAP stage crossing is drafted. */}
          {!wardParam && <GrapCard />}

          {q.data.candidates.length > 0 ? (
            <div className="space-y-2">
              {q.data.candidates.map((c, i) => (
                <CandidateRow
                  key={c.id}
                  candidate={c}
                  rank={i}
                  onDispatch={(x) => dispatch.mutate(x)}
                  dispatching={dispatch.isPending && dispatch.variables?.id === c.id}
                  dispatched={Boolean(dispatched[c.id])}
                />
              ))}
            </div>
          ) : (
            <EmptyState
              title="No dispatchable action for this air"
              hint={
                q.data.advisories.length
                  ? "The sources driving this air are outside local reach — see below."
                  : "No wards are currently flagged. Run the seeder, or widen the filter."
              }
              icon={<Info className="h-5 w-5" />}
            />
          )}

          {q.data.advisories.length > 0 && (
            <section className="mt-6">
              <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
                Sources you cannot act on
              </h2>
              <div className="space-y-2">
                {q.data.advisories.map((a, i) => (
                  <AdvisoryCard key={`${a.kind}-${a.category}-${i}`} advisory={a} />
                ))}
              </div>
            </section>
          )}

          {/* Coverage, stated. A sweep that silently looked at 12 of 290 wards
              would read as "these are all the options in the city". */}
          <footer className="mt-6 border-t border-white/5 pt-3 text-[11px] text-slate-600">
            {q.data.meta.wards_total
              ? `Evaluated ${q.data.meta.wards_evaluated} of ${q.data.meta.wards_total} wards (${q.data.meta.selection}).`
              : `Evaluated ${q.data.meta.selection}.`}{" "}
            {q.data.meta.city_aqi != null && `City AQI ${q.data.meta.city_aqi}. `}
            Impact is modelled with a Gaussian plume, not measured.
          </footer>
        </>
      )}
    </Shell>
  );
}

/** PRD E2: the signal → dossier stopwatch, shown on dispatch. */
function DispatchToast({ order }: { order: Order }) {
  const s = order.signal_to_dossier_s;
  return (
    <div className="mb-3 flex items-center gap-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3">
      <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
      <p className="flex-1 text-sm text-emerald-200">
        Order <span className="font-mono">{order.id}</span> dispatched to the inspector.
      </p>
      {s != null && (
        <span
          className="flex items-center gap-1 text-xs text-emerald-300/80"
          title="Signal is the forecast run that flagged this ward. In a replayed demo that run and 'now' are the same instant, so the elapsed time is legitimately zero — the pipeline figure is the real wall-clock cost."
        >
          <Timer className="h-3.5 w-3.5" />
          signal → dossier {s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`}
          {order.pipeline_ms != null && (
            <span className="text-emerald-400/50"> · pipeline {order.pipeline_ms}ms</span>
          )}
        </span>
      )}
      <a
        href={api.dossierUrl(order.id)}
        target="_blank"
        rel="noreferrer"
        className="rounded border border-emerald-500/30 px-2 py-1 text-xs text-emerald-300 hover:bg-emerald-500/10"
      >
        Dossier PDF
      </a>
    </div>
  );
}
