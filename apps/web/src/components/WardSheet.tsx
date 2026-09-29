"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowRight, Info, X } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { AttributionDonut } from "./AttributionDonut";
import { EvidenceList } from "./EvidenceList";
import { api, queryKeys } from "@/lib/api";
import { bandFor, readableOn } from "@/lib/aqi";
import { cn } from "@/lib/cn";
import type { Current, Ward } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";
import { ErrorState, Skeleton } from "./ui/States";

const HORIZONS = [24, 48, 72] as const;

/** Ward Detail (App Flow §3.2): AQI dial, forecast band, SHAP, attribution CTA. */
export function WardSheet({ current }: { current?: Current }) {
  const cityId = useCommandStore((s) => s.cityId);
  const wardId = useCommandStore((s) => s.selectedWardId);
  const close = useCommandStore((s) => s.selectWard);
  const [horizon, setHorizon] = useState<number>(48);

  const ward: Ward | undefined = useMemo(
    () => current?.wards.find((w) => w.ward_id === wardId),
    [current, wardId],
  );

  const forecasts = useQuery({
    queryKey: ["forecast-all", cityId],
    queryFn: async () => Promise.all(HORIZONS.map((h) => api.forecast(cityId, h))),
    enabled: Boolean(wardId),
  });

  const explain = useQuery({
    queryKey: queryKeys.explain(cityId, wardId ?? "", horizon),
    queryFn: () => api.explain(cityId, wardId!, horizon),
    enabled: Boolean(wardId),
    retry: false,
  });

  if (!wardId || !ward) return null;

  // Observed "now" + the three forecast horizons, as a single series.
  const series = (forecasts.data ?? [])
    .map((f, i) => {
      const w = f.wards.find((x) => x.ward_id === wardId);
      return w ? { t: `+${HORIZONS[i]}h`, p10: w.p10, p50: w.p50, p90: w.p90, aqi: w.aqi_p50 } : null;
    })
    .filter(Boolean) as { t: string; p10: number; p50: number; p90: number; aqi: number }[];

  const chartData = ward.pm25 != null
    ? [{ t: "now", p10: ward.pm25, p50: ward.pm25, p90: ward.pm25, aqi: ward.aqi ?? 0 }, ...series]
    : series;

  const band = bandFor(ward.aqi);
  const crossing = series.find((s) => s.aqi >= 300);

  return (
    <motion.aside
      initial={{ x: 32, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="panel absolute right-3 top-3 z-30 flex max-h-[calc(100%-24px)] w-[380px] flex-col overflow-hidden"
      aria-label={`Ward detail: ${ward.name}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2 border-b border-edge p-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-50">{ward.name}</p>
          <p className="numeral text-[10px] text-slate-500">
            {ward.ward_id} · {ward.population.toLocaleString("en-IN")} people
            {ward.nearest_station_km != null && ` · ${ward.nearest_station_km} km to station`}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {band && (
            <span
              className="numeral rounded px-2 py-1 text-lg font-bold leading-none"
              style={{ background: band.color, color: readableOn(band.color) }}
            >
              {ward.aqi}
            </span>
          )}
          <button
            onClick={() => close(null)}
            aria-label="Close ward detail"
            className="rounded p-1 text-slate-500 hover:text-slate-200"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </button>
        </div>
      </div>

      {ward.low_confidence && (
        <p className="border-b border-edge bg-warn/10 px-3 py-1.5 text-[10px] text-warn">
          Low confidence — nearest station is {ward.nearest_station_km} km away; values are interpolated.
        </p>
      )}

      <div className="flex-1 overflow-y-auto">
        {/* Forecast chart */}
        <section className="border-b border-edge p-3">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Forecast · PM2.5 µg/m³
            </p>
            <div className="flex gap-0.5">
              {HORIZONS.map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={cn(
                    "rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                    horizon === h ? "bg-data/15 text-data" : "text-slate-500 hover:text-slate-300",
                  )}
                >
                  {h}h
                </button>
              ))}
            </div>
          </div>

          {forecasts.isLoading ? (
            <Skeleton className="h-[150px] w-full" />
          ) : forecasts.error ? (
            <ErrorState
              title="No forecast yet"
              detail="Run `make seed` to train the model and score wards."
              className="py-4"
            />
          ) : chartData.length < 2 ? (
            <p className="py-6 text-center text-[11px] text-slate-500">
              Not enough forecast data for this ward.
            </p>
          ) : (
            <div className="h-[150px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                  <CartesianGrid stroke="#1F2A44" strokeDasharray="2 4" />
                  <XAxis dataKey="t" tick={{ fill: "#64748B", fontSize: 10 }} axisLine={{ stroke: "#1F2A44" }} tickLine={false} />
                  <YAxis tick={{ fill: "#64748B", fontSize: 10 }} axisLine={false} tickLine={false} width={38} />
                  {/* CPCB thresholds — the lines a commissioner acts on. */}
                  <ReferenceLine y={90} stroke="#F29C33" strokeDasharray="3 3" strokeOpacity={0.5}
                    label={{ value: "AQI 200", fill: "#F29C33", fontSize: 8, position: "insideTopRight" }} />
                  <ReferenceLine y={120} stroke="#E93F33" strokeDasharray="3 3" strokeOpacity={0.6}
                    label={{ value: "AQI 300", fill: "#E93F33", fontSize: 8, position: "insideTopRight" }} />
                  <ReferenceLine y={250} stroke="#AF2D24" strokeDasharray="3 3" strokeOpacity={0.6}
                    label={{ value: "AQI 400", fill: "#AF2D24", fontSize: 8, position: "insideTopRight" }} />
                  <Tooltip
                    contentStyle={{ background: "#111827", border: "1px solid #1F2A44", borderRadius: 6, fontSize: 11 }}
                    labelStyle={{ color: "#94A3B8" }}
                    formatter={(v, n) => [`${Number(v).toFixed(0)} µg/m³`, String(n)]}
                  />
                  {/* p10–p90 band: uncertainty is the product, not a caveat. */}
                  <Area type="monotone" dataKey="p90" stroke="none" fill="#22D3EE" fillOpacity={0.14} />
                  <Area type="monotone" dataKey="p10" stroke="none" fill="#0A0E1A" fillOpacity={1} />
                  <Line type="monotone" dataKey="p50" stroke="#22D3EE" strokeWidth={2} dot={{ r: 2.5, fill: "#22D3EE" }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {crossing && (
            <p className="mt-1.5 rounded bg-hazard/10 px-2 py-1 text-[10px] text-hazard">
              Crosses AQI 300 at {crossing.t} (p50 {crossing.p50.toFixed(0)} µg/m³)
            </p>
          )}
          <p className="mt-1.5 text-[9px] leading-relaxed text-slate-600">
            Shaded band = p10–p90. Thresholds are CPCB AQI breakpoints on PM2.5.
          </p>
        </section>

        {/* Why this forecast? — SHAP */}
        <section className="border-b border-edge p-3">
          <div className="mb-2 flex items-center gap-1.5">
            <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Why this forecast?
            </p>
            <Info className="h-3 w-3 text-slate-600" aria-hidden />
          </div>

          {explain.isLoading ? (
            <div className="space-y-1.5">
              {[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-4 w-full" />)}
            </div>
          ) : explain.error || !explain.data?.features.length ? (
            <p className="py-2 text-[11px] text-slate-500">
              Explanation needs a trained model — run `make seed`.
            </p>
          ) : (
            <ShapBars features={explain.data.features} />
          )}

          {explain.data && (
            <p className="mt-2 text-[9px] leading-relaxed text-slate-600">
              Explained via {explain.data.explained_via_station} ({explain.data.station_distance_km} km away) —
              the ward value is interpolated from nearby stations, so the drivers are the nearest one&apos;s.
            </p>
          )}
        </section>

        {/* Attribution — the product's core claim */}
        <AttributionSection wardId={wardId} />
      </div>

      <div className="border-t border-edge p-2.5">
        <InterventionCta wardId={wardId} />
      </div>
    </motion.aside>
  );
}

const TRAJ_HOURS = [6, 12, 24] as const;

/**
 * Footer CTA (App Flow §3.2): disabled below 0.3 attribution confidence.
 *
 * Below that threshold the evidence does not support naming a culprit, and an
 * intervention ranking built on it would dispatch a team on a guess. The button
 * says why rather than going quietly grey.
 */
function InterventionCta({ wardId }: { wardId: string }) {
  const cityId = useCommandStore((s) => s.cityId);
  const hours = useCommandStore((s) => s.trajectoryHours);
  const attribution = useQuery({
    queryKey: queryKeys.attribution(cityId, wardId, hours),
    queryFn: () => api.attribution(cityId, wardId, hours),
    enabled: Boolean(wardId),
  });

  const a = attribution.data;
  const top = a?.categories?.[0]?.confidence ?? 0;
  const stagnant = Boolean(a?.stagnant);
  const blocked = !a || stagnant || top < 0.3;

  const reason = stagnant
    ? "Stagnant conditions — no upwind source can be blamed"
    : !a
      ? "Attribution still loading"
      : `Attribution confidence ${top.toFixed(2)} is below 0.3 — too weak to dispatch on`;

  if (blocked) {
    return (
      <button
        disabled
        title={reason}
        className="w-full cursor-not-allowed rounded-md border border-edge bg-surface-2 px-3 py-2 text-xs font-medium text-slate-600"
      >
        Generate Intervention Options
      </button>
    );
  }

  return (
    <Link
      href={`/interventions?ward=${encodeURIComponent(wardId)}`}
      className="flex w-full items-center justify-center gap-1.5 rounded-md bg-sky-500/20 px-3 py-2 text-xs font-medium text-sky-200 transition-colors hover:bg-sky-500/30"
    >
      Generate Intervention Options
      <ArrowRight className="h-3.5 w-3.5" />
    </Link>
  );
}

function AttributionSection({ wardId }: { wardId: string }) {
  const cityId = useCommandStore((s) => s.cityId);
  const hours = useCommandStore((s) => s.trajectoryHours);
  const setHours = useCommandStore((s) => s.setTrajectoryHours);
  const selectedCat = useCommandStore((s) => s.selectedCategory);
  const selectCat = useCommandStore((s) => s.selectCategory);
  const setHoverEvidence = useCommandStore((s) => s.setHoveredEvidence);
  const setFlyTo = useCommandStore((s) => s.setFlyTo);

  const attribution = useQuery({
    queryKey: queryKeys.attribution(cityId, wardId, hours),
    queryFn: () => api.attribution(cityId, wardId, hours),
    retry: false,
  });

  return (
    <section className="p-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[10px] font-medium uppercase tracking-wider text-slate-500">
          Where did this air come from?
        </p>
        <div className="flex gap-0.5">
          {TRAJ_HOURS.map((h) => (
            <button
              key={h}
              onClick={() => setHours(h)}
              className={cn(
                "rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                hours === h ? "bg-data/15 text-data" : "text-slate-500 hover:text-slate-300",
              )}
            >
              {h}h
            </button>
          ))}
        </div>
      </div>

      {attribution.isLoading ? (
        <div className="space-y-2">
          <Skeleton className="mx-auto h-[168px] w-[168px] rounded-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      ) : attribution.error ? (
        <ErrorState
          title="Attribution unavailable"
          detail={(attribution.error as Error).message}
          onRetry={() => attribution.refetch()}
          className="py-4"
        />
      ) : attribution.data?.stagnant || !attribution.data?.categories.length ? (
        // Honest refusal, not a blank panel (App Flow §3.2).
        <div className="rounded border border-warn/30 bg-warn/10 p-2.5">
          <p className="text-[11px] font-medium text-warn">Attribution unavailable</p>
          <p className="mt-1 text-[10px] leading-relaxed text-slate-400">
            {attribution.data?.note ??
              "No attributable evidence in the trajectory cone for this window."}
          </p>
        </div>
      ) : (
        <>
          <AttributionDonut
            categories={attribution.data.categories}
            selected={selectedCat}
            onSelect={selectCat}
          />

          <div className="mt-2 flex items-center justify-between rounded border border-edge bg-surface-2/40 px-2 py-1 text-[9px] text-slate-500">
            <span className="numeral">
              air travelled {attribution.data.trajectory.length_km} km @{" "}
              {attribution.data.trajectory.mean_speed_kmh} km/h
            </span>
            <span className="numeral">
              station agreement {attribution.data.station_agreement?.toFixed(2) ?? "—"}
            </span>
          </div>

          <div className="mt-2">
            <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-slate-500">
              Evidence {selectedCat ? `· ${selectedCat.replace("_", " ")}` : "· all sources"}
            </p>
            <div className="max-h-[220px] overflow-y-auto pr-0.5">
              <EvidenceList
                categories={attribution.data.categories}
                selected={selectedCat}
                asOf={attribution.data.computed_ts}
                onHover={setHoverEvidence}
                onFocus={(e) => e.lat != null && e.lon != null && setFlyTo({ lon: e.lon, lat: e.lat })}
              />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function ShapBars({ features }: { features: { label: string; contribution: number; direction: string }[] }) {
  const bars = features.filter((f) => f.direction !== "base");
  const max = Math.max(...bars.map((b) => Math.abs(b.contribution)), 1);

  return (
    <div className="space-y-1.5">
      {bars.map((f, i) => {
        const pct = (Math.abs(f.contribution) / max) * 100;
        const up = f.contribution > 0;
        return (
          <div key={`${f.label}-${i}`} className="flex items-center gap-2">
            <span className="w-[104px] shrink-0 truncate text-[10px] text-slate-400" title={f.label}>
              {f.label}
            </span>
            <div className="relative h-3 flex-1 rounded-sm bg-surface-2">
              <div
                className={cn("absolute inset-y-0 rounded-sm", up ? "left-1/2 bg-hazard/70" : "right-1/2 bg-verified/70")}
                style={{ width: `${pct / 2}%` }}
              />
              <div className="absolute inset-y-0 left-1/2 w-px bg-edge" />
            </div>
            <span className={cn("numeral w-9 shrink-0 text-right text-[10px]", up ? "text-hazard" : "text-verified")}>
              {up ? "+" : ""}{f.contribution.toFixed(0)}
            </span>
          </div>
        );
      })}
      <p className="pt-0.5 text-[9px] text-slate-600">
        µg/m³ contribution to this prediction (SHAP). Red pushes the forecast up, green down.
      </p>
    </div>
  );
}
