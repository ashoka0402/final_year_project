"use client";

import { useQueries } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowDownRight, ArrowRight, ArrowUpRight, Building2, Gauge, Radio, Users } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, queryKeys } from "@/lib/api";
import { bandFor, readableOn } from "@/lib/aqi";
import { cn } from "@/lib/cn";
import type { City, Current } from "@/lib/types";
import { Skeleton } from "./ui/States";

const HORIZONS = [24, 48, 72] as const;

/** Count-up animation for KPI numerals (App Flow §3.1: "count-up once per load"). */
function useCountUp(target: number | null, ms = 900) {
  const [value, setValue] = useState(0);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (target == null) return;
    const start = performance.now();
    const from = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      // ease-out cubic — settles rather than slams
      const eased = 1 - (1 - t) ** 3;
      setValue(Math.round(from + (target - from) * eased));
      if (t < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [target, ms]);

  return target == null ? null : value;
}

function Stat({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  sub?: string;
}) {
  return (
    <div className="flex items-start gap-2.5 px-3 py-2.5">
      <div className="mt-0.5 text-slate-500">{icon}</div>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">{label}</p>
        <p className="numeral mt-0.5 text-sm font-semibold text-slate-100">{value}</p>
        {sub && <p className="mt-0.5 truncate text-[10px] text-slate-500">{sub}</p>}
      </div>
    </div>
  );
}

export function KpiRail({
  city,
  current,
  loading,
}: {
  city?: City;
  current?: Current;
  loading: boolean;
}) {
  const aqi = useCountUp(current?.aqi ?? null);
  const band = bandFor(current?.aqi);

  if (loading || !city) {
    return (
      <div className="panel w-[248px] overflow-hidden">
        <Skeleton className="h-[104px] w-full rounded-none" />
        <div className="space-y-2 p-3">
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
        </div>
      </div>
    );
  }

  const scored = current?.wards.filter((w) => w.aqi != null) ?? [];
  const worst = [...scored].sort((a, b) => (b.aqi ?? 0) - (a.aqi ?? 0))[0];
  const exposed = scored
    .filter((w) => (w.aqi ?? 0) > 300)
    .reduce((sum, w) => sum + w.population, 0);

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="panel w-[248px] overflow-hidden"
    >
      {/* City AQI hero */}
      <div
        className="px-3 py-3"
        style={{
          background: band ? `linear-gradient(135deg, ${band.color}22, transparent 70%)` : undefined,
        }}
      >
        <div className="flex items-center justify-between">
          <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
            {city.name} · AQI now
          </p>
          {band && (
            <span
              className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
              style={{ background: band.color, color: readableOn(band.color) }}
            >
              {band.label}
            </span>
          )}
        </div>

        {current?.aqi != null ? (
          <>
            <p className="numeral mt-1 text-[40px] font-bold leading-none text-slate-50">{aqi}</p>
            <p className="mt-1.5 text-[10px] leading-relaxed text-slate-500">
              Population-weighted across {scored.length} wards
            </p>
          </>
        ) : (
          <>
            <p className="numeral mt-1 text-[40px] font-bold leading-none text-slate-600">—</p>
            <p className="mt-1.5 text-[10px] text-slate-500">No readings in window</p>
          </>
        )}
      </div>

      <div className="divide-y divide-edge border-t border-edge">
        <Stat
          icon={<Gauge className="h-3.5 w-3.5" aria-hidden />}
          label="Worst ward"
          value={worst ? `${worst.aqi} · ${worst.name}` : "—"}
          sub={worst ? `${worst.category} · PM2.5 ${worst.pm25?.toFixed(0)} µg/m³` : undefined}
        />
        <Stat
          icon={<Users className="h-3.5 w-3.5" aria-hidden />}
          label="People in AQI > 300"
          value={exposed.toLocaleString("en-IN")}
          sub={exposed === 0 ? "No ward above 300 right now" : "Across flagged wards"}
        />
        <Stat
          icon={<Building2 className="h-3.5 w-3.5" aria-hidden />}
          label="Wards monitored"
          value={city.ward_count.toLocaleString("en-IN")}
          sub={`${city.population.toLocaleString("en-IN")} residents`}
        />
        <Stat
          icon={<Radio className="h-3.5 w-3.5" aria-hidden />}
          label="Stations reporting"
          value={`${scored.length ? current?.stations.filter((s) => s.aqi != null).length ?? 0 : 0} / ${city.station_count}`}
          sub={current?.stations[0]?.provider}
        />
      </div>

      <ForecastChips cityId={city.id} nowAqi={current?.aqi ?? null} />
    </motion.div>
  );
}

/** 24/48/72h population-weighted city AQI with trend arrows (PRD A2). */
function ForecastChips({ cityId, nowAqi }: { cityId: string; nowAqi: number | null }) {
  // useQueries, not useQuery-in-a-loop: hooks must not be called from a map,
  // even when the array happens to be constant.
  const qs = useQueries({
    queries: HORIZONS.map((h) => ({
      queryKey: queryKeys.forecast(cityId, h),
      queryFn: () => api.forecast(cityId, h),
      retry: false,
    })),
  });

  if (qs.every((q) => q.isLoading)) {
    return (
      <div className="flex gap-1.5 border-t border-edge px-3 py-2.5">
        {HORIZONS.map((h) => <Skeleton key={h} className="h-11 flex-1" />)}
      </div>
    );
  }
  if (qs.every((q) => q.error)) {
    return (
      <div className="border-t border-edge px-3 py-2">
        <p className="text-[10px] leading-relaxed text-slate-600">
          No forecast yet — run <span className="font-mono">make seed</span> to train and score.
        </p>
      </div>
    );
  }

  return (
    <div className="border-t border-edge px-3 py-2.5">
      <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-500">
        City forecast
      </p>
      <div className="flex gap-1.5">
        {qs.map((q, i) => {
          const h = HORIZONS[i];
          const wards = q.data?.wards ?? [];
          // Population-weighted, matching how the "now" figure is computed —
          // an unweighted mean would let empty peri-urban wards outvote the core.
          const pop = wards.reduce((s, w) => s + w.population, 0);
          const aqi = pop
            ? Math.round(wards.reduce((s, w) => s + w.aqi_p50 * w.population, 0) / pop)
            : null;
          const band = bandFor(aqi);
          const delta = aqi != null && nowAqi != null ? aqi - nowAqi : null;
          const Arrow =
            delta == null || Math.abs(delta) < 10 ? ArrowRight : delta > 0 ? ArrowUpRight : ArrowDownRight;

          return (
            <div key={h} className="flex-1 rounded border border-edge bg-surface-2/60 px-1.5 py-1">
              <p className="text-[9px] font-medium text-slate-500">+{h}h</p>
              {aqi == null ? (
                <p className="numeral text-sm font-bold text-slate-600">—</p>
              ) : (
                <>
                  <div className="flex items-center gap-0.5">
                    <span
                      className="numeral rounded px-1 text-xs font-bold"
                      style={{ background: band?.color, color: band ? readableOn(band.color) : undefined }}
                    >
                      {aqi}
                    </span>
                    <Arrow
                      className={cn(
                        "h-3 w-3",
                        delta == null || Math.abs(delta) < 10
                          ? "text-slate-500"
                          : delta > 0
                            ? "text-hazard"
                            : "text-verified",
                      )}
                      aria-hidden
                    />
                  </div>
                  <p className="numeral mt-0.5 text-[9px] text-slate-500">
                    {delta == null ? "" : `${delta > 0 ? "+" : ""}${delta}`}
                  </p>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
