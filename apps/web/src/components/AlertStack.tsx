"use client";

import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, ShieldCheck, X } from "lucide-react";
import { useState } from "react";

import { bandFor, readableOn } from "@/lib/aqi";
import { cn } from "@/lib/cn";
import type { HazardAlert } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";
import { EmptyState, Skeleton } from "./ui/States";

/**
 * Hazard alert cards (PRD A3, App Flow §3.1).
 * Sorted by ETA ascending, max 4 visible + "N more". Click flies to the ward.
 */
const MAX_VISIBLE = 4;

function relativeEta(hours: number): string {
  return hours < 24 ? `${hours}h` : `${Math.round(hours / 24)}d`;
}

export function AlertStack({
  alerts,
  loading,
  onSelect,
}: {
  alerts?: HazardAlert[];
  loading: boolean;
  onSelect: (wardId: string) => void;
}) {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const selected = useCommandStore((s) => s.selectedWardId);

  if (loading) {
    return (
      <div className="space-y-1.5">
        <Skeleton className="h-[68px] w-full" />
        <Skeleton className="h-[68px] w-full" />
      </div>
    );
  }

  const live = (alerts ?? []).filter((a) => !dismissed.has(a.ward_id));

  if (!live.length) {
    return (
      <div className="panel">
        <EmptyState
          icon={<ShieldCheck className="h-4 w-4 text-verified" aria-hidden />}
          title="No hazard crossings predicted"
          hint="Air holding steady across every ward in the forecast window."
        />
      </div>
    );
  }

  const visible = live.slice(0, MAX_VISIBLE);
  const more = live.length - visible.length;

  return (
    <div className="space-y-1.5">
      {/* Needs its own surface: bare text over the map lets basemap place
          labels bleed through the header. */}
      <div className="flex items-center justify-between rounded-md border border-edge bg-surface/85 px-2 py-1 backdrop-blur-md">
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
          Hazard alerts
        </p>
        <span className="numeral rounded-full bg-hazard/15 px-1.5 py-0.5 text-[10px] font-semibold text-hazard">
          {live.length}
        </span>
      </div>

      <AnimatePresence initial={false}>
        {visible.map((a) => {
          const band = bandFor(a.aqi_p50);
          const isSel = selected === a.ward_id;
          return (
            <motion.div
              key={a.ward_id}
              layout
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 12 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <button
                onClick={() => onSelect(a.ward_id)}
                data-testid={`alert-${a.ward_id}`}
                className={cn(
                  "panel w-full p-2.5 text-left transition-colors hover:border-warn/50",
                  isSel && "border-data/60",
                )}
              >
                <div className="flex items-start gap-2">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" aria-hidden />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline justify-between gap-2">
                      <p className="truncate text-xs font-semibold text-slate-100">{a.name}</p>
                      <span
                        className="numeral shrink-0 rounded px-1.5 py-0.5 text-[11px] font-bold"
                        style={{ background: band?.color, color: band ? readableOn(band.color) : undefined }}
                      >
                        {a.aqi_p50}
                      </span>
                    </div>
                    {/* Terse and operational (Copy Tone Guide §8). */}
                    <p className="mt-0.5 text-[11px] leading-snug text-slate-400">
                      AQI {a.aqi_p50} predicted in{" "}
                      <span className="numeral text-slate-200">{relativeEta(a.eta_h)}</span> · confidence{" "}
                      <span className="numeral text-slate-200">{a.confidence.toFixed(2)}</span>
                    </p>
                    <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500">
                      <span className="numeral">{a.population.toLocaleString("en-IN")} people</span>
                      <span aria-hidden>·</span>
                      <span className="numeral">PM2.5 {a.pm25_p50.toFixed(0)} µg/m³</span>
                    </div>
                  </div>
                  <span
                    role="button"
                    tabIndex={0}
                    aria-label={`Dismiss alert for ${a.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      setDismissed((d) => new Set(d).add(a.ward_id));
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.stopPropagation();
                        setDismissed((d) => new Set(d).add(a.ward_id));
                      }
                    }}
                    className="shrink-0 rounded p-0.5 text-slate-600 hover:text-slate-300"
                  >
                    <X className="h-3 w-3" aria-hidden />
                  </span>
                </div>
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>

      {more > 0 && (
        <p className="rounded border border-edge bg-surface/85 px-2 py-1 text-[10px] text-slate-400 backdrop-blur-md">
          {more} more ward{more > 1 ? "s" : ""} crossing — see Interventions
        </p>
      )}
    </div>
  );
}
