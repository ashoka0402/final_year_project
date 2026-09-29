"use client";

import { ArrowUpRight, Info, Wind } from "lucide-react";

import type { Advisory } from "@/lib/types";
import { CATEGORY_COLOR } from "@/components/AttributionDonut";

/**
 * A source that matters but that this city cannot act on.
 *
 * This card carries the finding that most distinguishes VAYU from a dashboard.
 * On a Delhi November morning the stubble driving the smog burns 200-300 km away
 * in Punjab: real, attributed, and beyond both the dispersion model's range and
 * the commissioner's jurisdiction. The leaderboard above is then short or empty,
 * and an empty table alone reads as "nothing to do" — the exact opposite of the
 * truth. Saying "not yours to fix, escalate to CAQM" is the useful answer, and
 * it saves a shift that would otherwise be spent chasing a lever that isn't
 * there.
 */
export function AdvisoryCard({ advisory }: { advisory: Advisory }) {
  const color = CATEGORY_COLOR[advisory.category] ?? "#64748B";
  const Icon = advisory.kind === "out_of_range" ? Wind : Info;

  return (
    <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.06] p-4">
      <div className="flex items-start gap-3">
        <div
          className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg"
          style={{ backgroundColor: `${color}1F`, color }}
        >
          <Icon className="h-4 w-4" />
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium text-slate-100">{advisory.headline}</p>
            <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-300">
              No local lever
            </span>
          </div>

          <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{advisory.detail}</p>

          {advisory.escalate_to && (
            <div className="mt-3 flex items-center gap-1.5 text-xs">
              <ArrowUpRight className="h-3.5 w-3.5 text-amber-400" />
              <span className="text-slate-400">Escalate to</span>
              <span className="font-medium text-amber-300">{advisory.escalate_to}</span>
            </div>
          )}

          {advisory.kind === "out_of_range" && advisory.nearest_km != null && (
            <dl className="mt-3 grid grid-cols-3 gap-3 border-t border-white/5 pt-3">
              <Stat label="Sources" value={String(advisory.source_count)} />
              <Stat
                label="Distance"
                value={`${Math.round(advisory.nearest_km)}–${Math.round(advisory.farthest_km ?? 0)} km`}
              />
              <Stat label="Total FRP" value={`${Math.round(advisory.total_magnitude ?? 0)} MW`} />
            </dl>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-0.5 font-mono text-xs text-slate-200">{value}</dd>
    </div>
  );
}
