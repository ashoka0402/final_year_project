"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Clock as ClockIcon,
  FastForward,
  RotateCcw,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, queryKeys } from "@/lib/api";
import { aqiColor, readableOn } from "@/lib/aqi";
import { cn } from "@/lib/cn";
import { useCommandStore } from "@/store/useCommandStore";

const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;
const IST_OFFSET_MIN = 330; // +5:30

/** ISO instant → IST date ("YYYY-MM-DD") and time ("HH:mm") strings.
 *
 * getTime() is epoch-UTC (browser-tz-independent); adding the fixed +5:30 and
 * then reading UTC components via toISOString() yields IST wall time in ANY
 * browser. (Using getTimezoneOffset() here was a bug: in an IST browser it
 * cancelled the offset, so the picker showed UTC and capped a day early.) */
function istParts(iso: string): { date: string; time: string } {
  const ist = new Date(new Date(iso).getTime() + IST_OFFSET_MIN * 60000);
  const s = ist.toISOString();
  return { date: s.slice(0, 10), time: s.slice(11, 16) };
}

/** IST date + time (wall clock) → UTC ISO the API can pin to. */
function combineIst(date: string, time: string): string {
  const asUtcMs = new Date(`${date}T${time}:00Z`).getTime() - IST_OFFSET_MIN * 60000;
  return new Date(asUtcMs).toISOString();
}

function clampIso(iso: string, min?: string | null, max?: string | null): string {
  let t = new Date(iso).getTime();
  if (min) t = Math.max(t, new Date(min).getTime());
  if (max) t = Math.min(t, new Date(max).getTime());
  return new Date(t).toISOString();
}

/**
 * Archive clock + time-travel control.
 *
 * VAYU reads a fixed historical record (CPCB + satellite through Nov 2025) —
 * there is no live-fetch path, so the picker spans exactly the data on disk.
 * Moving the clock snaps every surface (nowcast, forecast horizons, alerts,
 * GRAP stage, ROI) to the chosen hour; forecasts for a not-yet-scored hour are
 * scored on demand.
 */
export function ClockControl() {
  const qc = useQueryClient();
  const cityId = useCommandStore((s) => s.cityId);
  const clock = useQuery({ queryKey: queryKeys.clock, queryFn: api.clock });
  const notable = useQuery({
    queryKey: queryKeys.notableDates(cityId),
    queryFn: () => api.notableDates(cityId),
    staleTime: Infinity, // curated server-side; static for the life of the app
  });
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const pinned = clock.data?.pinned ?? false;
  const dataMin = clock.data?.data_min ?? null;
  const dataMax = clock.data?.data_max ?? null;
  const maxSelectable = clock.data?.max_selectable ?? dataMax;
  const nowIso = clock.data?.now ?? null;

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

  const setClock = useMutation({
    mutationFn: (asOf: string | null) => api.setClock(asOf),
    onSuccess: (data) => {
      qc.setQueryData(queryKeys.clock, data);
      qc.invalidateQueries(); // "now" drives every read; refresh all of it
    },
  });

  const pinTo = (iso: string) => setClock.mutate(clampIso(iso, dataMin, maxSelectable));
  const shift = (ms: number) => {
    if (!nowIso) return;
    pinTo(new Date(new Date(nowIso).getTime() + ms).toISOString());
  };

  const shown = nowIso ? new Date(nowIso) : null;
  const label = shown
    ? shown.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short", timeZone: "Asia/Kolkata" })
    : "—";
  const parts = nowIso ? istParts(nowIso) : { date: "", time: "" };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        data-testid="clock-control"
        title="Archive clock — click to time-travel"
        className={cn(
          "flex items-center gap-1.5 whitespace-nowrap rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
          pinned
            ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
            : "border-edge bg-surface-2 text-slate-200 hover:border-data/50",
        )}
      >
        <ClockIcon className="h-3 w-3 shrink-0 text-slate-400" aria-hidden />
        <span className="numeral whitespace-nowrap">{label}</span>
        <span
          className={cn(
            "whitespace-nowrap text-[9px] uppercase tracking-wider",
            pinned ? "text-amber-400" : "text-slate-500",
          )}
        >
          IST · {pinned ? "pinned" : "archive"}
        </span>
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1.5 w-[300px] rounded-md border border-edge bg-surface p-3 shadow-2xl">
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-slate-500">
            Time-travel the airshed
          </p>

          {/* Date + time selection (IST), bounded to the archive on disk. */}
          <div className="flex gap-2">
            <label className="flex-1">
              <span className="mb-1 block text-[9px] uppercase tracking-wider text-slate-500">Date</span>
              <input
                type="date"
                data-testid="clock-date"
                value={parts.date}
                min={dataMin ? istParts(dataMin).date : undefined}
                max={maxSelectable ? istParts(maxSelectable).date : undefined}
                onChange={(e) => e.target.value && pinTo(combineIst(e.target.value, parts.time || "06:00"))}
                className="w-full rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs text-slate-100"
              />
            </label>
            <label className="w-24">
              <span className="mb-1 block text-[9px] uppercase tracking-wider text-slate-500">Time</span>
              <input
                type="time"
                data-testid="clock-time"
                value={parts.time}
                onChange={(e) => e.target.value && pinTo(combineIst(parts.date, e.target.value))}
                className="w-full rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs text-slate-100"
              />
            </label>
          </div>

          {/* Steppers — walk hour by hour or day by day. */}
          <div className="mt-2 grid grid-cols-4 gap-1.5">
            <StepBtn onClick={() => shift(-DAY_MS)} label="−1 day">
              <ChevronLeft className="h-3 w-3" aria-hidden />
              1d
            </StepBtn>
            <StepBtn onClick={() => shift(-HOUR_MS)} label="−1 hour">
              <ChevronLeft className="h-3 w-3" aria-hidden />
              1h
            </StepBtn>
            <StepBtn onClick={() => shift(HOUR_MS)} label="+1 hour">
              1h
              <ChevronRight className="h-3 w-3" aria-hidden />
            </StepBtn>
            <StepBtn onClick={() => shift(DAY_MS)} label="+1 day">
              1d
              <ChevronRight className="h-3 w-3" aria-hidden />
            </StepBtn>
          </div>

          {/* Notable episodes — shortcuts, not a restriction. */}
          {(notable.data?.dates?.length ?? 0) > 0 && (
            <>
              <p className="mb-1.5 mt-3 text-[10px] font-medium uppercase tracking-wider text-slate-500">
                Jump to a notable episode
              </p>
              <ul className="space-y-1">
                {(notable.data?.dates ?? []).map((d) => {
                  const selected = nowIso && new Date(nowIso).getTime() === new Date(d.at).getTime();
                  const color = aqiColor(d.aqi);
                  return (
                    <li key={d.at}>
                      <button
                        onClick={() => pinTo(d.at)}
                        disabled={setClock.isPending}
                        data-testid={`notable-date-${d.at}`}
                        className={cn(
                          "flex w-full items-center justify-between gap-2 rounded border px-2 py-1.5 text-left transition-colors disabled:opacity-50",
                          selected ? "border-data/50 bg-data/10" : "border-edge bg-surface-2 hover:border-data/40",
                        )}
                      >
                        <span className="min-w-0">
                          <span className="block truncate text-xs font-medium text-slate-100">{d.label}</span>
                          <span className="numeral block text-[10px] text-slate-500">
                            {new Date(d.at).toLocaleDateString("en-IN", {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                              timeZone: "Asia/Kolkata",
                            })}
                          </span>
                        </span>
                        <span
                          className="numeral shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold"
                          style={{ background: color, color: readableOn(color) }}
                        >
                          {d.aqi}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </>
          )}

          <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
            Snaps the nowcast, forecast, alerts, GRAP stage and ROI to that hour.
            Forecasts are scored on demand.
          </p>

          <div className="mt-2 flex gap-1.5">
            {dataMax && (
              <button
                onClick={() => pinTo(dataMax)}
                disabled={setClock.isPending}
                data-testid="clock-latest"
                className="flex flex-1 items-center justify-center gap-1.5 rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-data/50 disabled:opacity-50"
              >
                <FastForward className="h-3 w-3" aria-hidden />
                Latest
              </button>
            )}
            <button
              onClick={() => setClock.mutate(null)}
              disabled={setClock.isPending}
              data-testid="clock-reset"
              title="Clear the pin and return to the default instant"
              className="flex flex-1 items-center justify-center gap-1.5 rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-data/50 disabled:opacity-50"
            >
              <RotateCcw className="h-3 w-3" aria-hidden />
              {setClock.isPending ? "…" : "Reset"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function StepBtn({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      className="flex items-center justify-center gap-0.5 rounded border border-edge bg-surface-2 px-1 py-1.5 text-[11px] font-medium text-slate-300 transition-colors hover:border-data/50 hover:text-slate-100"
    >
      {children}
    </button>
  );
}
