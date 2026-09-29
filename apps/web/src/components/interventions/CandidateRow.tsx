"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, MapPin, Scale, Send, Users } from "lucide-react";
import { useState } from "react";

import { CATEGORY_COLOR, CATEGORY_ICON } from "@/components/AttributionDonut";
import { cn } from "@/lib/cn";
import type { Candidate } from "@/lib/types";

const ACTION_LABEL: Record<string, string> = {
  halt_burning: "Halt open burning",
  stop_work_construction: "Stop work",
  traffic_restriction: "Restrict traffic",
  industrial_curb: "Curb industry",
  road_dust_suppression: "Suppress road dust",
};

const MEDAL = ["#FACC15", "#CBD5E1", "#D97706"];

/**
 * One row of the ROI leaderboard (App Flow §3.3).
 *
 * The row shows the three numbers the ROI is made of — averted, people, effort —
 * next to the score, so the arithmetic is checkable on screen rather than taken
 * on trust: score = averted x people / effort / 1000, exactly.
 *
 * Two averted figures are shown deliberately. The headline is the
 * population-weighted mean across every ward the plume reaches (what the ROI
 * multiplies); the ward's own figure sits in the expanded detail. Showing only
 * one would claim that everyone counted got the alerting ward's benefit.
 */
export function CandidateRow({
  candidate: c,
  rank,
  onDispatch,
  dispatching,
  dispatched,
}: {
  candidate: Candidate;
  rank: number;
  onDispatch: (c: Candidate) => void;
  dispatching: boolean;
  dispatched: boolean;
}) {
  const [open, setOpen] = useState(false);
  const Icon = CATEGORY_ICON[c.category] ?? Scale;
  const color = CATEGORY_COLOR[c.category] ?? "#64748B";

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border transition-colors",
        rank === 0
          ? "border-amber-400/30 bg-amber-400/[0.04]"
          : "border-white/8 bg-white/[0.02] hover:border-white/15",
      )}
    >
      <div className="flex items-center gap-3 p-3">
        <div
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold"
          style={{
            backgroundColor: rank < 3 ? `${MEDAL[rank]}1F` : "rgba(255,255,255,0.04)",
            color: rank < 3 ? MEDAL[rank] : "#64748B",
          }}
        >
          {rank + 1}
        </div>

        <div
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
          style={{ backgroundColor: `${color}1F`, color }}
        >
          <Icon className="h-4 w-4" />
        </div>

        <button
          onClick={() => setOpen((v) => !v)}
          className="min-w-0 flex-1 text-left"
          aria-expanded={open}
        >
          {/* The action is identified by its TARGET, not by the ward that
              surfaced it. "Curb industry · Vivek Vihar" reads as "this helps
              Vivek Vihar" — but that ward gains 0.37 µg/m³ while the mean across
              the 229 wards it reaches is 3.84. The ward is where the attribution
              noticed the source, not who benefits. */}
          <p className="truncate text-sm font-medium text-slate-100">
            {ACTION_LABEL[c.action_type] ?? c.action_type}
            <span className="text-slate-500"> · </span>
            <span className="text-slate-400">{targetOf(c)}</span>
          </p>
          <p className="mt-0.5 truncate text-xs text-slate-500">
            Helps {c.wards_protected} ward{c.wards_protected === 1 ? "" : "s"}
            <span className="text-slate-600"> · flagged by {c.ward_name}</span>
          </p>
        </button>

        <dl className="hidden shrink-0 items-center gap-5 md:flex">
          <Metric
            label="µg/m³ avoided"
            value={c.predicted_ugm3_averted.toFixed(2)}
            hint="Population-weighted mean across every ward the plume reaches"
            preserveCase
          />
          <Metric label="people" value={compact(c.population_protected)} />
          <Metric label="teams" value={String(c.effort_units)} />
          <Metric label="conf" value={c.confidence.toFixed(2)} />
        </dl>

        <div className="w-20 shrink-0 text-right">
          <p className="font-mono text-sm font-semibold text-sky-300">
            {compact(Math.round(c.roi_score))}
          </p>
          <p className="text-[10px] uppercase tracking-wide text-slate-500">ROI</p>
        </div>

        <button
          onClick={() => onDispatch(c)}
          disabled={dispatching || dispatched}
          className={cn(
            "flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
            dispatched
              ? "bg-emerald-500/15 text-emerald-300"
              : "bg-sky-500/15 text-sky-300 hover:bg-sky-500/25 disabled:opacity-50",
          )}
        >
          <Send className="h-3.5 w-3.5" />
          {dispatched ? "Dispatched" : dispatching ? "Sending…" : "Dispatch"}
        </button>

        <button
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Collapse detail" : "Expand detail"}
          className="shrink-0 rounded p-1 text-slate-500 hover:text-slate-300"
        >
          <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} />
        </button>
      </div>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            <div className="border-t border-white/5 p-4">
              <p className="text-xs leading-relaxed text-slate-400">{c.rationale}</p>

              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <Counterfactual candidate={c} />

                <div>
                  <h4 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
                    Target &amp; evidence
                  </h4>
                  <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-400">
                    <MapPin className="h-3.5 w-3.5 text-rose-400" />
                    <span className="font-mono">
                      {c.source_lat.toFixed(4)}, {c.source_lon.toFixed(4)}
                    </span>
                    <span className="text-slate-600">·</span>
                    <span>{c.distance_km.toFixed(1)} km from ward</span>
                  </div>
                  <ul className="mt-2 space-y-1">
                    {c.evidence.slice(0, 4).map((e, i) => (
                      <li key={i} className="truncate text-xs text-slate-500">
                        · {e.label}
                        {e.source && <span className="text-slate-600"> — {e.source}</span>}
                      </li>
                    ))}
                    {c.evidence.length > 4 && (
                      <li className="text-xs text-slate-600">
                        + {c.evidence.length - 4} more in the dossier
                      </li>
                    )}
                  </ul>
                </div>
              </div>

              {c.regulation && (
                <div className="mt-4 rounded-lg border border-amber-500/20 bg-amber-500/[0.05] p-3">
                  <div className="flex items-center gap-1.5">
                    <Scale className="h-3.5 w-3.5 text-amber-400" />
                    <p className="text-xs font-medium text-slate-200">{c.regulation.title}</p>
                  </div>
                  <p className="mt-1 text-[11px] text-slate-500">{c.regulation.citation}</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/**
 * With / without the action, at each forecast horizon. The bars are the
 * counterfactual: the model's whole claim in one picture.
 */
function Counterfactual({ candidate: c }: { candidate: Candidate }) {
  const horizons = Object.entries(c.averted_by_horizon).sort(
    (a, b) => Number(a[0]) - Number(b[0]),
  );
  const max = Math.max(...horizons.map(([, v]) => v), c.peak_ugm3_averted, 0.001);

  return (
    <div>
      <h4 className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        Averted in {c.ward_name} if halted
      </h4>
      <div className="mt-2 space-y-1.5">
        {horizons.map(([h, v]) => (
          <div key={h} className="flex items-center gap-2">
            <span className="w-10 shrink-0 font-mono text-[10px] text-slate-500">t+{h}h</span>
            <div className="h-3 flex-1 overflow-hidden rounded-sm bg-white/5">
              <div
                className="h-full rounded-sm bg-sky-400/70"
                style={{ width: `${Math.max((v / max) * 100, 1)}%` }}
              />
            </div>
            <span className="w-14 shrink-0 text-right font-mono text-[10px] text-slate-400">
              {v.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-slate-600">
        {c.ward_name} gains {c.ward_averted_ugm3.toFixed(2)} µg/m³ at t+24h (peak{" "}
        {c.peak_ugm3_averted.toFixed(2)}). The headline {c.predicted_ugm3_averted.toFixed(2)} is the
        mean per person across all {compact(c.population_protected)} people in the{" "}
        {c.wards_protected} wards the plume reaches — often higher than this ward&rsquo;s own
        gain, because the ward that flagged a source is not always the one it hurts most.
      </p>
    </div>
  );
}

function Metric({
  label,
  value,
  hint,
  preserveCase,
}: {
  label: string;
  value: string;
  hint?: string;
  /**
   * Never CSS-uppercase a unit. `text-transform: uppercase` maps µ (U+00B5
   * MICRO SIGN) to Μ (U+039C GREEK CAPITAL MU), which renders as an ordinary M:
   * "µg/m³ avoided" displayed as "MG/M³ AVOIDED". That reads as milligrams — a
   * 1000x overstatement of the number that justifies dispatching a team.
   */
  preserveCase?: boolean;
}) {
  return (
    <div className="text-right" title={hint}>
      <dd className="font-mono text-xs text-slate-200">{value}</dd>
      <dt
        className={cn(
          "text-[10px] tracking-wide text-slate-500",
          !preserveCase && "uppercase",
        )}
      >
        {label}
      </dt>
    </div>
  );
}

/** The site an inspector is sent to — the part of the title after the dash. */
function targetOf(c: Candidate): string {
  const i = c.title.indexOf("—");
  return i >= 0 ? c.title.slice(i + 1).trim() : c.title;
}

function compact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}
