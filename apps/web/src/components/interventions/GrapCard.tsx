"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Check, Scale, ShieldAlert, TrendingUp, X } from "lucide-react";
import { useState } from "react";

import { api, queryKeys } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { GrapDraft } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";

/**
 * GRAP Autopilot card (App Flow §3.3, PRD C4) — conditional, and the loudest
 * place VAYU says "a human decides".
 *
 * The autopilot drafts the measures an incoming GRAP stage mandates when a
 * crossing is forecast, each with its clause citation, and then waits. Approving
 * it bans construction or restricts vehicles across the NCR — so the card wears
 * a "human-in-the-loop" badge, the primary button is an explicit Approve behind
 * a confirm, and nothing is actioned until a person clicks it.
 */
export function GrapCard() {
  const cityId = useCommandStore((s) => s.cityId);
  const qc = useQueryClient();
  const q = useQuery({ queryKey: queryKeys.grap(cityId), queryFn: () => api.grap(cityId) });
  const [confirming, setConfirming] = useState(false);

  const approve = useMutation({
    mutationFn: (id: string) => api.grapApprove(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.grap(cityId) });
      qc.invalidateQueries({ queryKey: queryKeys.audit });
      setConfirming(false);
    },
  });
  const dismiss = useMutation({
    mutationFn: (id: string) => api.grapDismiss(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.grap(cityId) }),
  });

  const draft = q.data?.draft;
  // No draft, or one already resolved → the card is not shown at all (it is
  // conditional). Showing an empty autopilot would imply inaction where there is
  // simply nothing to escalate.
  if (!draft || draft.status !== "draft") return null;

  const seeded = draft.id.includes("DEMO");

  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-rose-500/25 bg-rose-500/[0.05]">
      <div className="flex items-start gap-3 p-4">
        <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-rose-500/15 text-rose-300">
          <ShieldAlert className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-slate-100">GRAP Autopilot</h3>
            <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-300">
              Human-in-the-loop
            </span>
            {seeded && (
              <span
                className="rounded-full border border-slate-500/30 bg-slate-500/10 px-2 py-0.5 text-[10px] text-slate-400"
                title="On the demo instant no city-wide crossing is forecast; this draft is built from the worst-ward forecast so the approval flow is demonstrable. The measures and citations are real."
              >
                Seeded demo record
              </span>
            )}
          </div>

          <p className="mt-1.5 flex items-center gap-1.5 text-xs text-slate-300">
            <TrendingUp className="h-3.5 w-3.5 text-rose-400" />
            {draft.current_stage_label} → <b className="text-rose-300">{draft.forecast_stage_label}</b>
            {draft.crossing_eta_h != null && (
              <span className="text-slate-500">forecast within {draft.crossing_eta_h}h</span>
            )}
          </p>

          <p className="mt-2 text-xs leading-relaxed text-slate-400">
            {draft.measures.length} measure{draft.measures.length > 1 ? "s" : ""} come into force at{" "}
            {draft.forecast_stage_label}. VAYU has drafted them with citations. They are not active
            until a human approves.
          </p>
        </div>
      </div>

      <ul className="space-y-1.5 px-4 pb-1">
        {draft.measures.map((m) => (
          <MeasureRow key={m.clause_id} measure={m} />
        ))}
      </ul>

      <div className="flex items-center justify-end gap-2 border-t border-white/5 p-3">
        {approve.isError && (
          <span className="mr-auto text-xs text-rose-400">Approval failed — try again.</span>
        )}
        {!confirming ? (
          <>
            <button
              onClick={() => dismiss.mutate(draft.id)}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200"
            >
              <X className="h-3.5 w-3.5" />
              Dismiss
            </button>
            <button
              onClick={() => setConfirming(true)}
              className="flex items-center gap-1.5 rounded-lg bg-rose-500/20 px-3 py-1.5 text-xs font-medium text-rose-200 hover:bg-rose-500/30"
            >
              <Check className="h-3.5 w-3.5" />
              Approve measures
            </button>
          </>
        ) : (
          <div className="flex w-full items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/[0.06] p-2">
            <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400" />
            <p className="flex-1 text-[11px] text-amber-200">
              This activates {draft.measures.length} {draft.forecast_stage_label} measure
              {draft.measures.length > 1 ? "s" : ""} city-wide. Confirm?
            </p>
            <button
              onClick={() => setConfirming(false)}
              className="rounded px-2 py-1 text-[11px] text-slate-400 hover:text-slate-200"
            >
              Cancel
            </button>
            <button
              onClick={() => approve.mutate(draft.id)}
              disabled={approve.isPending}
              className="rounded bg-rose-500/30 px-3 py-1 text-[11px] font-medium text-rose-100 hover:bg-rose-500/40 disabled:opacity-50"
            >
              {approve.isPending ? "Approving…" : "Confirm approval"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function MeasureRow({ measure: m }: { measure: GrapDraft["measures"][number] }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="rounded-lg border border-white/8 bg-white/[0.02]">
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-2 p-2.5 text-left">
        <Scale className="h-3.5 w-3.5 shrink-0 text-amber-400" />
        <span className="flex-1 text-xs text-slate-200">{m.title}</span>
        <span className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-slate-500">{m.clause_id}</span>
      </button>
      <div className={cn("overflow-hidden px-2.5 transition-all", open ? "max-h-40 pb-2.5" : "max-h-0")}>
        <p className="text-[11px] leading-relaxed text-slate-400">&ldquo;{m.text}&rdquo;</p>
        <p className="mt-1 text-[10px] text-amber-300/80">{m.citation}</p>
      </div>
    </li>
  );
}
