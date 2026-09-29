"use client";

import { useQuery } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { useMemo } from "react";

import { AlertStack } from "@/components/AlertStack";
import { AqiLegend, BasemapSwitcher, ImpactTicker, LayerChips } from "@/components/MapControls";
import { KpiRail } from "@/components/KpiRail";
import { TopNav } from "@/components/TopNav";
import { WardSheet } from "@/components/WardSheet";
import { ErrorState, Skeleton } from "@/components/ui/States";
import { api, ApiError, queryKeys } from "@/lib/api";
import { useCommandStore } from "@/store/useCommandStore";

// deck.gl + maplibre are heavy and browser-only: keep them out of the server
// bundle and off the critical path (TRD §10: initial JS < 450KB gz).
const MapCanvas = dynamic(() => import("@/components/map/MapCanvas").then((m) => m.MapCanvas), {
  ssr: false,
  loading: () => <MapSkeleton />,
});

function MapSkeleton() {
  return (
    <div className="absolute inset-0 bg-base">
      <Skeleton className="h-full w-full rounded-none" />
      <p className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-xs text-slate-500">
        Loading airshed…
      </p>
    </div>
  );
}

export default function CommandCenter() {
  const cityId = useCommandStore((s) => s.cityId);
  const selectWard = useCommandStore((s) => s.selectWard);

  const cities = useQuery({ queryKey: queryKeys.cities, queryFn: api.cities });
  // Alerts are optional: before the Forecaster has run there simply aren't any,
  // which is a designed empty state ("air holding steady"), not an error.
  const alerts = useQuery({
    queryKey: queryKeys.alerts(cityId),
    queryFn: () => api.alerts(cityId),
    enabled: Boolean(cityId),
    retry: false,
  });
  const current = useQuery({
    queryKey: queryKeys.current(cityId),
    queryFn: () => api.current(cityId),
    enabled: Boolean(cityId),
  });
  // Geometry is immutable per city and cached hard by the browser; keeping it in
  // a separate query means a city switch re-renders values instantly while
  // polygons come from cache.
  const wards = useQuery({
    queryKey: queryKeys.wards(cityId),
    queryFn: () => api.wards(cityId),
    enabled: Boolean(cityId),
    staleTime: Infinity,
    gcTime: Infinity,
  });

  const city = useMemo(() => cities.data?.find((c) => c.id === cityId), [cities.data, cityId]);
  const selectedWardId = useCommandStore((s) => s.selectedWardId);
  const trajectoryHours = useCommandStore((s) => s.trajectoryHours);
  const selectedCategory = useCommandStore((s) => s.selectedCategory);
  const hoveredEvidence = useCommandStore((s) => s.hoveredEvidence);
  const flyTo = useCommandStore((s) => s.flyTo);

  // Trajectory + attribution for the selected ward drive the map's evidence
  // layers. They live here rather than in the sheet so the map keeps rendering
  // them while the sheet scrolls or re-renders.
  const trajectory = useQuery({
    queryKey: queryKeys.trajectory(cityId, selectedWardId ?? "", trajectoryHours),
    queryFn: () => api.trajectory(cityId, selectedWardId!, trajectoryHours),
    enabled: Boolean(selectedWardId),
    retry: false,
  });
  const attribution = useQuery({
    queryKey: queryKeys.attribution(cityId, selectedWardId ?? "", trajectoryHours),
    queryFn: () => api.attribution(cityId, selectedWardId!, trajectoryHours),
    enabled: Boolean(selectedWardId),
    retry: false,
  });

  // Only the selected category's evidence goes on the map when a slice is
  // active — clicking "open burning" should highlight fires, not everything.
  const evidence = useMemo(() => {
    const cats = attribution.data?.categories ?? [];
    const use = selectedCategory ? cats.filter((c) => c.category === selectedCategory) : cats;
    return use.flatMap((c) => c.evidence).filter((e) => e.lat != null && e.lon != null);
  }, [attribution.data, selectedCategory]);

  const fatal = cities.error as ApiError | null;
  const dataError = (current.error ?? wards.error) as ApiError | null;
  const loading = cities.isLoading || current.isLoading || wards.isLoading;

  // The API being unreachable is the one case where there is nothing to draw.
  if (fatal) {
    return (
      <div className="flex h-screen items-center justify-center bg-base">
        <div className="panel max-w-md p-6">
          <ErrorState
            title="Cannot reach the VAYU API"
            detail={
              fatal.status === 0
                ? "The API is not responding on :8000. Start it with `make dev`, or seed first with `make seed`."
                : fatal.message
            }
            onRetry={() => cities.refetch()}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-base">
      <TopNav
        cities={cities.data ?? []}
        statuses={current.data?.data_status}
        loading={cities.isLoading}
      />

      <main id="main" className="relative flex-1">
        {city && !dataError && (
          <MapCanvas
            city={city}
            current={current.data}
            wards={wards.data}
            trajectory={trajectory.data}
            evidence={evidence}
            hoveredEvidence={hoveredEvidence}
            flyTo={flyTo}
          />
        )}
        {loading && !city && <MapSkeleton />}

        {dataError && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="panel max-w-md p-6">
              <ErrorState
                title={`Could not load ${cityId}`}
                detail={dataError.message}
                onRetry={() => {
                  current.refetch();
                  wards.refetch();
                }}
              />
            </div>
          </div>
        )}

        {/* Left rail: city KPIs */}
        <div className="pointer-events-none absolute left-3 top-3 z-20">
          <div className="pointer-events-auto">
            <KpiRail city={city} current={current.data} loading={loading} />
          </div>
        </div>

        {/* Top-left of map: layer chips + basemap switcher (offset past the rail) */}
        <div className="pointer-events-none absolute left-[268px] top-3 z-20 flex items-center gap-2">
          <div className="pointer-events-auto">
            <LayerChips />
          </div>
          <div className="pointer-events-auto">
            <BasemapSwitcher />
          </div>
        </div>

        {/* Bottom-left: CPCB legend (kept on the map — it reads against the data) */}
        <div className="pointer-events-none absolute bottom-3 left-3 z-20">
          <div className="pointer-events-auto">
            <AqiLegend />
          </div>
        </div>

        {/* Top-right: hazard alert stack (PRD A3). Hidden while the ward sheet
            is open — they occupy the same rail. */}
        {!selectedWardId && (
          <div className="absolute right-3 top-3 z-20 w-[300px]">
            <AlertStack
              alerts={alerts.data}
              loading={alerts.isLoading}
              onSelect={(id) => selectWard(id)}
            />
          </div>
        )}

        {/* Ward Detail sheet */}
        <WardSheet current={current.data} />

      </main>

      {/* Footer strip: impact ticker + demo-mode disclosure (§12, App Flow §7).
          A real footer rather than a floating overlay, so neither can collide
          with the map controls at any viewport width. */}
      <footer className="flex h-7 shrink-0 items-center justify-between gap-4 border-t border-edge bg-surface/60 px-3 backdrop-blur-md">
        <ImpactTicker />
        {current.data?.demo_mode ? (
          <p className="shrink-0 whitespace-nowrap text-[10px] text-slate-500">
            Running on bundled sample data · clock pinned to{" "}
            <span className="numeral text-slate-400">
              {new Date(current.data.as_of).toLocaleString("en-IN", {
                dateStyle: "medium",
                timeStyle: "short",
                timeZone: "Asia/Kolkata",
              })}{" "}
              IST
            </span>
          </p>
        ) : (
          <span className="shrink-0 text-[10px] text-verified">Live feeds</span>
        )}
      </footer>
    </div>
  );
}

