"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  CheckCircle2,
  ClipboardCheck,
  Clock,
  Download,
  FileText,
  Inbox,
  MapPin,
  RotateCw,
  Scale,
} from "lucide-react";
import { useState } from "react";

import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { ApiError, api, queryKeys } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { Order, OrderStatus } from "@/lib/types";

/**
 * Inspector (App Flow §3.4) — mobile-first, max-w-md centred on desktop.
 *
 * This is the only VAYU surface used by someone standing in the field rather
 * than sitting at a desk, so it is deliberately the plainest: what the order is,
 * where to go, what to look for, under what authority, and one button to record
 * what happened.
 */

const STATUS_STYLE: Record<OrderStatus, string> = {
  candidate: "border-slate-500/30 bg-slate-500/10 text-slate-400",
  dispatched: "border-sky-500/30 bg-sky-500/10 text-sky-300",
  executed: "border-violet-500/30 bg-violet-500/10 text-violet-300",
  verified: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
};

const ACTION_LABEL: Record<string, string> = {
  halt_burning: "Halt open burning",
  stop_work_construction: "Stop work — construction",
  traffic_restriction: "Restrict traffic corridor",
  industrial_curb: "Curb industrial emissions",
  road_dust_suppression: "Road dust suppression",
};

export default function InspectorPage() {
  const [openId, setOpenId] = useState<string | null>(null);
  return (
    <div className="min-h-dvh bg-base">
      <div className="mx-auto max-w-md px-4 py-5">
        {openId ? (
          <OrderDetail orderId={openId} onBack={() => setOpenId(null)} />
        ) : (
          <OrderList onOpen={setOpenId} />
        )}
      </div>
    </div>
  );
}

function OrderList({ onOpen }: { onOpen: (id: string) => void }) {
  const q = useQuery({ queryKey: queryKeys.orders(), queryFn: () => api.orders() });
  // Age must be measured against the application clock, not the browser's. In
  // demo replay "now" is 3 Nov 2025 while the laptop says 2026, and Date.now()
  // rendered every fresh order as "256d ago".
  const health = useQuery({ queryKey: queryKeys.health, queryFn: api.health });
  const now = health.data?.now;

  return (
    <>
      <header className="mb-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Field orders</h1>
          <p className="text-xs text-slate-500">{q.data?.count ?? 0} assigned</p>
        </div>
        <button
          onClick={() => q.refetch()}
          aria-label="Refresh"
          className="rounded-lg border border-white/10 p-2 text-slate-400 hover:text-slate-200"
        >
          <RotateCw className={cn("h-4 w-4", q.isFetching && "animate-spin")} />
        </button>
      </header>

      {q.isPending && (
        <div className="space-y-2">
          {[0, 1].map((i) => (
            <Skeleton key={i} className="h-24 w-full" />
          ))}
        </div>
      )}

      {q.isError && (
        <ErrorState
          title="Could not load orders"
          detail={q.error instanceof ApiError ? q.error.message : String(q.error)}
          onRetry={() => q.refetch()}
        />
      )}

      {q.data && q.data.orders.length === 0 && (
        <EmptyState
          title="No orders yet"
          hint="Dispatch one from the Interventions leaderboard and it will appear here."
          icon={<Inbox className="h-5 w-5" />}
        />
      )}

      <div className="space-y-2">
        {q.data?.orders.map((o) => (
          <button
            key={o.id}
            onClick={() => onOpen(o.id)}
            className="w-full rounded-xl border border-white/8 bg-white/[0.02] p-3 text-left hover:border-white/15"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="font-mono text-xs text-slate-500">{o.id}</span>
              <StatusChip status={o.status} />
            </div>
            <p className="mt-1.5 text-sm font-medium text-slate-100">
              {ACTION_LABEL[o.action_type] ?? o.action_type}
            </p>
            <p className="mt-0.5 truncate text-xs text-slate-500">{o.title}</p>
            <div className="mt-2 flex items-center gap-3 text-[11px] text-slate-500">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {age(o.dispatched_ts ?? o.created_ts, now)}
              </span>
              <span>{o.effort_units} team{o.effort_units > 1 ? "s" : ""}</span>
              <span>{compact(o.population_protected)} people</span>
            </div>
          </button>
        ))}
      </div>
    </>
  );
}

function OrderDetail({ orderId, onBack }: { orderId: string; onBack: () => void }) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const [checked, setChecked] = useState<Record<number, boolean>>({});

  const q = useQuery({ queryKey: queryKeys.order(orderId), queryFn: () => api.order(orderId) });
  // The leaderboard holds the evidence and citation; the order row holds status.
  const lb = useQuery({
    queryKey: queryKeys.interventions(q.data?.city ?? "", q.data?.ward_id ?? null, null),
    queryFn: () => api.interventions(q.data!.city, q.data!.ward_id, null),
    enabled: Boolean(q.data),
  });
  const detail = lb.data?.candidates.find((c) => c.id === orderId);

  const execute = useMutation({
    mutationFn: () => api.execute(orderId, note),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.order(orderId) });
      qc.invalidateQueries({ queryKey: queryKeys.orders() });
    },
  });

  if (q.isPending) return <Skeleton className="h-96 w-full" />;
  if (q.isError || !q.data)
    return (
      <ErrorState
        title="Could not load order"
        detail={q.error instanceof ApiError ? q.error.message : String(q.error)}
        onRetry={() => q.refetch()}
      />
    );

  const o = q.data;

  if (o.status === "executed" || o.status === "verified") {
    return (
      <>
        <BackButton onBack={onBack} />
        <div className="mt-8 flex flex-col items-center text-center">
          <CheckCircle2 className="h-12 w-12 text-violet-400" />
          <h2 className="mt-3 text-lg font-semibold text-slate-100">Order executed</h2>
          <p className="mt-1 font-mono text-xs text-slate-500">{o.id}</p>
          <div className="mt-6 w-full rounded-xl border border-violet-500/25 bg-violet-500/[0.06] p-4 text-left">
            <p className="text-xs font-medium text-violet-300">Verification pending</p>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">
              VAYU will compare the predicted impact against observed PM2.5 in this ward,
              against weather-matched control wards, once ~48h of readings are in.
            </p>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <BackButton onBack={onBack} />

      <div className="mt-3 flex items-center justify-between gap-2">
        <span className="font-mono text-xs text-slate-500">{o.id}</span>
        <StatusChip status={o.status} />
      </div>
      <h1 className="mt-1.5 text-lg font-semibold text-slate-100">
        {ACTION_LABEL[o.action_type] ?? o.action_type}
      </h1>
      <p className="mt-0.5 text-sm text-slate-500">{o.title}</p>

      <a
        href={`https://www.google.com/maps/search/?api=1&query=${o.source_lat},${o.source_lon}`}
        target="_blank"
        rel="noreferrer"
        className="mt-4 flex items-center gap-2 rounded-xl border border-white/8 bg-white/[0.02] p-3 hover:border-white/15"
      >
        <MapPin className="h-4 w-4 shrink-0 text-rose-400" />
        <div className="min-w-0 flex-1">
          <p className="font-mono text-xs text-slate-200">
            {o.source_lat.toFixed(5)}, {o.source_lon.toFixed(5)}
          </p>
          <p className="text-[11px] text-slate-500">Open directions</p>
        </div>
      </a>

      <a
        href={api.dossierUrl(o.id)}
        target="_blank"
        rel="noreferrer"
        className="mt-2 flex items-center gap-2 rounded-xl border border-white/8 bg-white/[0.02] p-3 hover:border-white/15"
      >
        <FileText className="h-4 w-4 shrink-0 text-sky-400" />
        <div className="min-w-0 flex-1">
          <p className="text-xs text-slate-200">Evidence dossier</p>
          <p className="text-[11px] text-slate-500">PDF — map, evidence, citation, sign-off</p>
        </div>
        <Download className="h-3.5 w-3.5 shrink-0 text-slate-500" />
      </a>

      {detail?.regulation && (
        <div className="mt-2 rounded-xl border border-amber-500/25 bg-amber-500/[0.06] p-3">
          <div className="flex items-center gap-1.5">
            <Scale className="h-3.5 w-3.5 shrink-0 text-amber-400" />
            <p className="text-xs font-medium text-slate-200">{detail.regulation.title}</p>
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-slate-400">
            “{detail.regulation.text}”
          </p>
          <p className="mt-1.5 text-[11px] text-amber-300/80">{detail.regulation.citation}</p>
        </div>
      )}

      {detail && detail.evidence.length > 0 && (
        <section className="mt-4">
          <h2 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
            Evidence checklist
          </h2>
          <ul className="mt-2 space-y-1.5">
            {detail.evidence.map((e, i) => (
              <li key={i}>
                <label className="flex cursor-pointer items-start gap-2 rounded-lg border border-white/8 bg-white/[0.02] p-2.5">
                  <input
                    type="checkbox"
                    checked={Boolean(checked[i])}
                    onChange={() => setChecked((c) => ({ ...c, [i]: !c[i] }))}
                    className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-sky-500"
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs text-slate-200">{e.label}</span>
                    {e.detail && (
                      <span className="block truncate text-[11px] text-slate-500">{e.detail}</span>
                    )}
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="mt-5">
        <label htmlFor="note" className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
          What did you find?
        </label>
        <textarea
          id="note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={3}
          placeholder="burning extinguished 14:20, 3 violations issued"
          className="mt-2 w-full rounded-lg border border-white/10 bg-white/[0.02] p-2.5 text-xs text-slate-200 placeholder:text-slate-600 focus:border-sky-500/40 focus:outline-none"
        />
        {execute.isError && (
          <p className="mt-2 text-xs text-rose-400">
            {execute.error instanceof ApiError ? execute.error.message : String(execute.error)}
          </p>
        )}
        <button
          onClick={() => execute.mutate()}
          disabled={!note.trim() || execute.isPending}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-violet-500/20 py-2.5 text-sm font-medium text-violet-200 hover:bg-violet-500/30 disabled:opacity-40"
        >
          <ClipboardCheck className="h-4 w-4" />
          {execute.isPending ? "Submitting…" : "Mark executed"}
        </button>
        {/* The note is the only record of what actually happened on the ground,
            and verification is measured against it. Requiring it is the point. */}
        <p className="mt-2 text-center text-[11px] text-slate-600">
          A note is required — it is the record verification is measured against.
        </p>
      </section>
    </>
  );
}

function BackButton({ onBack }: { onBack: () => void }) {
  return (
    <button
      onClick={onBack}
      className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200"
    >
      <ArrowLeft className="h-3.5 w-3.5" />
      All orders
    </button>
  );
}

function StatusChip({ status }: { status: OrderStatus }) {
  return (
    <span
      className={cn(
        "rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        STATUS_STYLE[status],
      )}
    >
      {status}
    </span>
  );
}

/**
 * Age against the server's clock.
 *
 * `now` is the application instant from /health — in demo mode the replayed
 * one. Using the browser's Date.now() against a replayed timestamp printed
 * "256d ago" on an order dispatched seconds earlier.
 */
function age(ts: string | null, now?: string): string {
  if (!ts) return "—";
  const ref = now ? new Date(now).getTime() : Date.now();
  const ms = ref - new Date(ts).getTime();
  if (ms < 0) return "just now";
  const h = Math.floor(ms / 3_600_000);
  if (h < 1) {
    const m = Math.floor(ms / 60_000);
    return m < 1 ? "just now" : `${m}m ago`;
  }
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}
