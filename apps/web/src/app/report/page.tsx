"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import { TopNav } from "@/components/TopNav";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { api, ApiError, queryKeys, type CitizenVerdict } from "@/lib/api";
import { cn } from "@/lib/cn";

/**
 * Citizen reporting — a photo, read by Gemini, then judged against the satellite.
 *
 * The design decision that matters here is showing the verdict honestly. It
 * would be friendlier to thank every reporter and move on; instead the page
 * tells people plainly when the satellite does not back their report, and why.
 * A crowd-sourced layer that flatters its contributors is worth nothing to the
 * official who has to act on it.
 */

const VERDICT_STYLE: Record<string, { label: string; cls: string; blurb: string }> = {
  corroborated: {
    label: "Corroborated by satellite",
    cls: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
    blurb: "Independent satellite measurements agree. This report will inform hotspot detection.",
  },
  unsupported: {
    label: "Not yet supported",
    cls: "border-amber-500/40 bg-amber-500/10 text-amber-300",
    blurb: "The satellite neither confirms nor contradicts this. Kept on record, but it will not move the analysis.",
  },
  contradicted: {
    label: "Contradicted by satellite",
    cls: "border-rose-500/40 bg-rose-500/10 text-rose-300",
    blurb: "The satellite saw a normal atmosphere and no fire here. Could be very small, very local, or after the overpass.",
  },
  no_satellite_data: {
    label: "No satellite coverage",
    cls: "border-slate-500/40 bg-slate-500/10 text-slate-300",
    blurb: "Cloud or a gap in the overpass — we could not check this one. That is our limitation, not your error.",
  },
  unusable: {
    label: "Could not read this photo",
    cls: "border-slate-600/40 bg-slate-700/20 text-slate-400",
    blurb: "It does not appear to show outdoor air, or was too ambiguous to judge.",
  },
};

// Real cell from the 2025 archive with a strong anomaly and active fires —
// gives a first-time visitor a working example instead of an empty form.
const EXAMPLE = { lat: "25.875", lon: "78.375", when: "2025-11-21T07:30" };

export default function ReportPage() {
  const qc = useQueryClient();
  const cities = useQuery({ queryKey: queryKeys.cities, queryFn: api.cities });
  const reports = useQuery({
    queryKey: queryKeys.citizenReports("india", "all"),
    queryFn: () => api.citizenReports("india", "all"),
  });

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [lat, setLat] = useState(EXAMPLE.lat);
  const [lon, setLon] = useState(EXAMPLE.lon);
  const [when, setWhen] = useState(EXAMPLE.when);
  const [note, setNote] = useState("");
  const [result, setResult] = useState<CitizenVerdict | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const submit = useMutation({
    mutationFn: async () => {
      if (!file) throw new ApiError("Choose a photo first", 400);
      const fd = new FormData();
      fd.append("photo", file);
      fd.append("lat", lat);
      fd.append("lon", lon);
      fd.append("region_id", "india");
      if (when) fd.append("when", new Date(when).toISOString());
      if (note) fd.append("note", note);
      return api.submitPhoto(fd);
    },
    onSuccess: (r) => {
      setResult(r);
      qc.invalidateQueries({ queryKey: ["citizenReports"] });
    },
  });

  const onPick = (f: File | null) => {
    setFile(f);
    setResult(null);
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const aiOff = reports.data && !reports.data.google_ai_enabled;

  return (
    <div className="flex h-dvh flex-col bg-base">
      <TopNav cities={cities.data ?? []} loading={cities.isLoading} />
      <main className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-6 py-6">
        <header className="mb-5">
          <h1 className="text-xl font-semibold text-slate-100">Report what you can see</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-500">
            India has roughly 900 air-quality monitors for 1.4 billion people, and a
            satellite passes overhead once a day. Neither sees the field burning at the
            edge of your village at 7&nbsp;a.m. — a phone does. Every photo is read by
            Google Gemini and then checked against{" "}
            <span className="text-slate-300">independent satellite measurements</span>{" "}
            for the same square. We tell you either way.
          </p>
        </header>

        {aiOff && (
          <div className="panel mb-4 border-amber-500/40 p-3 text-xs text-amber-300">
            Photo analysis is offline — <code>GOOGLE_API_KEY</code> is not configured on
            the server. Sensor reports still work.
          </div>
        )}

        <div className="grid gap-5 md:grid-cols-2">
          {/* ---- submission ---- */}
          <section className="panel p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">1 · Your photo</h2>

            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="flex h-44 w-full items-center justify-center overflow-hidden rounded-md border border-dashed border-edge bg-surface-2 transition-colors hover:border-data/50"
            >
              {preview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={preview} alt="Selected" className="h-full w-full object-cover" />
              ) : (
                <span className="text-xs text-slate-500">Tap to choose a photo of the sky or the source</span>
              )}
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
              data-testid="photo-input"
            />

            <h2 className="mb-2 mt-4 text-sm font-semibold text-slate-200">2 · Where and when</h2>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Latitude" value={lat} onChange={setLat} />
              <Field label="Longitude" value={lon} onChange={setLon} />
            </div>
            <label className="mt-2 block">
              <span className="mb-1 block text-[9px] uppercase tracking-wider text-slate-500">
                When (archive covers Oct–Nov 2025)
              </span>
              <input
                type="datetime-local"
                value={when}
                onChange={(e) => setWhen(e.target.value)}
                className="w-full rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs text-slate-100"
              />
            </label>
            <label className="mt-2 block">
              <span className="mb-1 block text-[9px] uppercase tracking-wider text-slate-500">Note (optional)</span>
              <input
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="Thick smoke from the fields since dawn"
                className="w-full rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs text-slate-100"
              />
            </label>

            <button
              onClick={() => submit.mutate()}
              disabled={!file || submit.isPending}
              data-testid="submit-report"
              className="mt-3 w-full rounded-md bg-data/20 px-3 py-2 text-sm font-medium text-data transition-colors hover:bg-data/30 disabled:opacity-40"
            >
              {submit.isPending ? "Gemini is reading your photo…" : "Submit for analysis"}
            </button>

            {submit.error && (
              <p className="mt-2 text-xs text-rose-400">{(submit.error as ApiError).message}</p>
            )}
          </section>

          {/* ---- verdict ---- */}
          <section className="panel p-4">
            <h2 className="mb-3 text-sm font-semibold text-slate-200">3 · What we found</h2>
            {!result && !submit.isPending && (
              <EmptyState
                title="No analysis yet"
                hint="Submit a photo and Gemini will describe what it sees, then we check it against the satellite record for that square."
              />
            )}
            {submit.isPending && <Skeleton className="h-40 w-full" />}
            {result && <Verdict r={result} />}
          </section>
        </div>

        {/* ---- review queue ---- */}
        <section className="mt-6">
          <div className="mb-2 flex items-baseline justify-between">
            <h2 className="text-sm font-semibold text-slate-200">Community reports</h2>
            {reports.data && (
              <p className="text-[10px] text-slate-500">
                {Object.entries(reports.data.by_verdict)
                  .map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`)
                  .join(" · ")}
              </p>
            )}
          </div>
          <p className="mb-3 text-[11px] leading-relaxed text-slate-600">
            Rejected reports stay visible on purpose. A filter you cannot inspect is
            indistinguishable from no filter at all.
          </p>

          {reports.isPending && <Skeleton className="h-24 w-full" />}
          {reports.error && (
            <ErrorState
              title="Could not load reports"
              detail={(reports.error as ApiError).message}
              onRetry={() => reports.refetch()}
            />
          )}
          {reports.data?.items.length === 0 && (
            <EmptyState title="No reports yet" hint="Yours would be the first." />
          )}

          <ul className="space-y-2">
            {reports.data?.items.slice(0, 12).map((r) => {
              const st = VERDICT_STYLE[r.verdict] ?? VERDICT_STYLE.unusable;
              return (
                <li key={r.id} className="panel flex items-start justify-between gap-3 p-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={cn("rounded border px-1.5 py-0.5 text-[10px] font-medium", st.cls)}>
                        {st.label}
                      </span>
                      <span className="text-[10px] uppercase tracking-wide text-slate-500">{r.kind}</span>
                      {r.haze_severity && (
                        <span className="numeral text-[10px] text-slate-400">haze: {r.haze_severity}</span>
                      )}
                      {r.pm25 != null && (
                        <span className="numeral text-[10px] text-slate-400">PM2.5 {r.pm25}</span>
                      )}
                    </div>
                    <p className="mt-1 text-xs leading-relaxed text-slate-400">{r.verdict_detail}</p>
                    {r.ai_reasoning && (
                      <p className="mt-0.5 text-[11px] italic leading-relaxed text-slate-600">
                        Gemini: “{r.ai_reasoning}”
                      </p>
                    )}
                  </div>
                  <span className="numeral shrink-0 text-[10px] text-slate-600">
                    {r.grid_lat.toFixed(2)}, {r.grid_lon.toFixed(2)}
                  </span>
                </li>
              );
            })}
          </ul>
        </section>
      </main>
    </div>
  );
}

function Field({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[9px] uppercase tracking-wider text-slate-500">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="numeral w-full rounded border border-edge bg-surface-2 px-2 py-1.5 text-xs text-slate-100"
      />
    </label>
  );
}

function Verdict({ r }: { r: CitizenVerdict }) {
  const st = VERDICT_STYLE[r.verdict] ?? VERDICT_STYLE.unusable;
  return (
    <div data-testid="verdict">
      <div className={cn("rounded-md border p-3", st.cls)}>
        <p className="text-sm font-semibold">{st.label}</p>
        <p className="mt-1 text-xs leading-relaxed opacity-90">{st.blurb}</p>
      </div>
      <dl className="mt-3 space-y-2 text-xs">
        {r.haze_severity && (
          <Row k="Gemini read the air as" v={<span className="capitalize">{r.haze_severity}</span>} />
        )}
        {r.source_type && r.source_type !== "none_visible" && (
          <Row k="Visible source" v={r.source_type.replace(/_/g, " ")} />
        )}
        <Row k="Evidence" v={<span className="text-slate-400">{r.detail}</span>} />
        <Row
          k="Influences analysis"
          v={
            <span className={r.may_influence ? "text-emerald-400" : "text-slate-500"}>
              {r.may_influence ? "Yes — satellite-backed" : "No — not independently confirmed"}
            </span>
          }
        />
      </dl>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <dt className="w-40 shrink-0 text-slate-500">{k}</dt>
      <dd className="min-w-0 text-slate-200">{v}</dd>
    </div>
  );
}
