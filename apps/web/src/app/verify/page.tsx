"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Clock, HelpCircle, Minus } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { TopNav } from "@/components/TopNav";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { ApiError, api, queryKeys } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { Verification } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";

/**
 * Verify (App Flow §3.6, PRD E1) — the loop's closing arrow, and the page most
 * likely to embarrass us, which is exactly why it ships.
 *
 * The design rule here is one line: significance leads. A verdict whose
 * confidence interval spans zero cannot be told apart from the weather, and no
 * amount of "150% realized" changes that. So the headline is the CI, not the
 * ratio — a real result and a null result must not look the same.
 */
export default function VerifyPage() {
  const cityId = useCommandStore((s) => s.cityId);
  const cities = useQuery({ queryKey: queryKeys.cities, queryFn: api.cities });
  const q = useQuery({
    queryKey: queryKeys.verifications(cityId),
    queryFn: () => api.verifications(cityId),
  });

  return (
    <div className="flex h-dvh flex-col bg-base">
      <TopNav cities={cities.data ?? []} loading={cities.isLoading} />
      <main className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-6 py-6">
        <header className="mb-5">
          <h1 className="text-xl font-semibold text-slate-100">Verification</h1>
          <p className="mt-1 text-sm text-slate-500">
            Predicted vs. observed, by difference-in-differences against weather-matched control
            wards. A result only counts when it can be told apart from the weather.
          </p>
        </header>

        {q.isPending && (
          <div className="space-y-3">
            {[0, 1].map((i) => (
              <Skeleton key={i} className="h-40 w-full" />
            ))}
          </div>
        )}

        {q.isError && (
          <ErrorState
            title="Could not load verifications"
            detail={q.error instanceof ApiError ? q.error.message : String(q.error)}
            onRetry={() => q.refetch()}
          />
        )}

        {q.data && q.data.verifications.length === 0 && (
          <EmptyState
            title="Nothing to verify yet"
            hint="Execute an order in the Inspector; ~48h later its outcome is measured here."
            icon={<Clock className="h-5 w-5" />}
          />
        )}

        <div className="space-y-4">
          {q.data?.verifications.map((v) => (
            <VerificationCard key={v.intervention_id} v={v} />
          ))}
        </div>
      </main>
    </div>
  );
}

function VerificationCard({ v }: { v: Verification }) {
  if (v.status === "pending") return <PendingCard v={v} />;
  if (v.status === "error" || v.note) return <UnverifiableCard v={v} />;

  const seeded = v.order.seeded;
  return (
    <article className="rounded-xl border border-white/8 bg-white/[0.02] p-5">
      <Header v={v} seeded={seeded} />

      {/* Significance verdict — the headline. */}
      <SignificanceBanner v={v} />

      <div className="mt-4 grid gap-6 md:grid-cols-2">
        <PredictedVsObserved v={v} />
        <DidChart v={v} />
      </div>

      <Method v={v} />
    </article>
  );
}

function Header({ v, seeded }: { v: Verification; seeded: boolean }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-slate-500">{v.intervention_id}</span>
          {seeded && (
            <span
              className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-300"
              title="A demonstration order that was never actually dispatched. The ward, the readings, and the diff-in-diff verdict are all real; only the act of dispatch is fabricated."
            >
              Seeded demo record
            </span>
          )}
        </div>
        <p className="mt-1 text-sm font-medium text-slate-100">
          {v.order.title}
          {v.ward_name && <span className="text-slate-500"> · {v.ward_name}</span>}
        </p>
      </div>
      {v.post_hours != null && (
        <span className="text-[11px] text-slate-500">measured over {v.post_hours}h</span>
      )}
    </div>
  );
}

function SignificanceBanner({ v }: { v: Verification }) {
  const spansZero = v.ci_low <= 0 && v.ci_high >= 0;
  if (v.significant && !spansZero) {
    return (
      <div className="mt-4 flex items-start gap-2 rounded-lg border border-emerald-500/25 bg-emerald-500/[0.06] p-3">
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
        <p className="text-sm text-emerald-200">
          <b>Measurable effect.</b> The order averted{" "}
          <b>{v.observed_reduction.toFixed(1)} µg/m³</b> beyond what the control wards did — a 95%
          interval of [{v.ci_low.toFixed(1)}, {v.ci_high.toFixed(1)}] that excludes zero.{" "}
          {isFinite(v.pct_realized) && `That is ${v.pct_realized.toFixed(0)}% of the predicted ${v.predicted_reduction.toFixed(1)} µg/m³.`}
        </p>
      </div>
    );
  }
  return (
    <div className="mt-4 flex items-start gap-2 rounded-lg border border-slate-500/25 bg-slate-500/[0.06] p-3">
      <HelpCircle className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
      <p className="text-sm text-slate-300">
        <b>Not distinguishable from the weather.</b> The observed change was{" "}
        {v.observed_reduction.toFixed(1)} µg/m³, but the 95% interval [{v.ci_low.toFixed(1)},{" "}
        {v.ci_high.toFixed(1)}] spans zero — this ward and its controls moved together, so the data
        cannot credit the order. We report this rather than claim a win.
      </p>
    </div>
  );
}

function PredictedVsObserved({ v }: { v: Verification }) {
  const max = Math.max(Math.abs(v.predicted_reduction), Math.abs(v.observed_reduction), 1);
  const bar = (val: number, color: string) => (
    <div className="h-5 overflow-hidden rounded bg-white/5">
      <div
        className={cn("h-full rounded", color)}
        style={{ width: `${Math.max((Math.abs(val) / max) * 100, 2)}%` }}
      />
    </div>
  );
  return (
    <div>
      <h4 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        Predicted vs. observed reduction
      </h4>
      <div className="mt-3 space-y-3">
        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="text-slate-400">Predicted</span>
            <span className="font-mono text-slate-300">{v.predicted_reduction.toFixed(1)} µg/m³</span>
          </div>
          {bar(v.predicted_reduction, "bg-sky-400/60")}
        </div>
        <div>
          <div className="mb-1 flex justify-between text-xs">
            <span className="text-slate-400">Observed (DiD)</span>
            <span className="font-mono text-slate-300">
              {v.observed_reduction.toFixed(1)} µg/m³
              <span className="text-slate-600">
                {" "}[{v.ci_low.toFixed(1)}, {v.ci_high.toFixed(1)}]
              </span>
            </span>
          </div>
          {bar(v.observed_reduction, v.significant ? "bg-emerald-400/60" : "bg-slate-400/50")}
        </div>
      </div>
      <p className="mt-3 text-[11px] leading-relaxed text-slate-600">
        Observed = (target after − before) − (controls after − before). Subtracting the controls
        removes the city-wide weather swing that a plain before/after would miscredit to the order.
      </p>
    </div>
  );
}

function DidChart({ v }: { v: Verification }) {
  const rows = (v.series?.days ?? []).map((d, i) => ({
    day: new Date(d).toLocaleDateString("en-IN", { day: "numeric", month: "short" }),
    target: v.series.target[i],
    control: v.series.control[i],
  }));
  // Explicit numeric domain, padded off zero so the band fills the panel.
  // Recharts v3's "dataMin - 15" string form mis-computed the range and hid the
  // lines, so the bounds are derived here from the actual values instead.
  const vals = rows
    .flatMap((r) => [r.target, r.control])
    .filter((x): x is number => typeof x === "number");
  const yMin = vals.length ? Math.floor(Math.min(...vals) - 15) : 0;
  const yMax = vals.length ? Math.ceil(Math.max(...vals) + 15) : 100;
  if (!rows.length) {
    return (
      <div className="flex items-center justify-center text-xs text-slate-600">
        No daily series available.
      </div>
    );
  }
  const execIdx = Math.max(0, Math.floor(rows.length * (7 / 9))); // exec ~day 7 of 9
  return (
    <div>
      <h4 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        Target ward vs. synthetic control
      </h4>
      <div className="mt-2 h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 12, right: 8, bottom: 0, left: -6 }}>
            <CartesianGrid stroke="#ffffff10" vertical={false} />
            <XAxis dataKey="day" tick={{ fontSize: 9, fill: "#64748b" }} tickLine={false} axisLine={false} />
            {/* Domain padded off zero so the 114-185 band fills the panel instead
                of hugging the top; width fits 3-digit µg/m³ labels un-clipped. */}
            <YAxis
              domain={[yMin, yMax]}
              tick={{ fontSize: 9, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              width={46}
            />
            <Tooltip
              contentStyle={{ background: "#0f172a", border: "1px solid #ffffff15", borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: "#94a3b8" }}
            />
            <ReferenceLine x={rows[execIdx]?.day} stroke="#f59e0b" strokeDasharray="3 3"
              label={{ value: "executed", fontSize: 8, fill: "#f59e0b", position: "top" }} />
            <Line type="monotone" dataKey="target" name={v.ward_name ?? "Target"}
              stroke="#22d3ee" strokeWidth={2} dot={false} connectNulls />
            <Line type="monotone" dataKey="control" name="Controls"
              stroke="#64748b" strokeWidth={1.5} strokeDasharray="4 3" dot={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-[11px] text-slate-600">
        If the order worked, the cyan line drops below the grey after the marker. Here they track
        together — the visual form of a null result.
      </p>
    </div>
  );
}

function Method({ v }: { v: Verification }) {
  return (
    <div className="mt-4 border-t border-white/5 pt-3 text-[11px] text-slate-600">
      Controls (chosen on pre-period behaviour only):{" "}
      <span className="text-slate-500">
        {(v.control_ward_names ?? v.control_wards).join(", ") || "—"}
      </span>
      . 95% CI via block bootstrap (n=500) over hourly residuals. Observed PM2.5 is IDW-interpolated
      from CPCB stations onto ward centroids — the same series the map shows.
    </div>
  );
}

function PendingCard({ v }: { v: Verification }) {
  const pct = v.hours_required ? ((v.hours_elapsed ?? 0) / v.hours_required) * 100 : 0;
  return (
    <article className="rounded-xl border border-white/8 bg-white/[0.02] p-5">
      <Header v={v} seeded={v.order.seeded} />
      <div className="mt-4 flex items-center gap-3">
        <Clock className="h-4 w-4 shrink-0 text-sky-400" />
        <div className="flex-1">
          <p className="text-sm text-slate-300">
            Verification completes in ~{v.hours_remaining}h — it needs {v.hours_required}h of
            post-execution readings before the outcome can be measured.
          </p>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5">
            <div className="h-full rounded-full bg-sky-400/60" style={{ width: `${Math.min(pct, 100)}%` }} />
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] text-slate-600">
        A verdict drawn from fewer hours would be noise. We wait rather than guess.
      </p>
    </article>
  );
}

function UnverifiableCard({ v }: { v: Verification }) {
  return (
    <article className="rounded-xl border border-amber-500/20 bg-amber-500/[0.04] p-5">
      <Header v={v} seeded={v.order.seeded} />
      <div className="mt-4 flex items-start gap-2">
        {v.status === "error" ? (
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        ) : (
          <Minus className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
        )}
        <p className="text-sm text-slate-300">{v.note ?? v.detail}</p>
      </div>
    </article>
  );
}
