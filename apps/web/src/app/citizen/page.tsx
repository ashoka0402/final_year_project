"use client";

import { useQuery } from "@tanstack/react-query";
import { MapPin, Search, Wind } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { EmptyState, ErrorState, Skeleton } from "@/components/ui/States";
import { ApiError, api, queryKeys } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { CitizenBrief, WardCollection } from "@/lib/types";
import { useCommandStore } from "@/store/useCommandStore";

/**
 * Citizen (App Flow §3.5) — VAYU's only surface aimed at the public, and a
 * separate URL from the commissioner tools. Deliberately plain and reassuring:
 * one big number, when it's safest to be outside, and what to do — in the
 * reader's own language.
 */

const LANG_STORE_KEY = "vayu-lang";

function faceFor(aqi: number | null): string {
  if (aqi == null) return "❓";
  if (aqi <= 100) return "🙂";
  if (aqi <= 200) return "😐";
  if (aqi <= 300) return "😷";
  if (aqi <= 400) return "🤢";
  return "☠️";
}

const UI = {
  en: { title: "Air today", pick: "Choose your area", locate: "Use my location",
        clean: "Clean hours (next 48h)", best: "Best time outside", none: "No clearly cleaner window",
        advice: "What to do", search: "Search area…", est: "Estimated — nearest monitor is far" },
  hi: { title: "आज की हवा", pick: "अपना क्षेत्र चुनें", locate: "मेरा स्थान इस्तेमाल करें",
        clean: "साफ़ घंटे (अगले 48 घंटे)", best: "बाहर जाने का सबसे अच्छा समय", none: "कोई साफ़ समय नहीं",
        advice: "क्या करें", search: "क्षेत्र खोजें…", est: "अनुमानित — निकटतम मॉनिटर दूर है" },
  pa: { title: "ਅੱਜ ਦੀ ਹਵਾ", pick: "ਆਪਣਾ ਇਲਾਕਾ ਚੁਣੋ", locate: "ਮੇਰਾ ਟਿਕਾਣਾ ਵਰਤੋ",
        clean: "ਸਾਫ਼ ਘੰਟੇ (ਅਗਲੇ 48 ਘੰਟੇ)", best: "ਬਾਹਰ ਜਾਣ ਦਾ ਵਧੀਆ ਸਮਾਂ", none: "ਕੋਈ ਸਾਫ਼ ਸਮਾਂ ਨਹੀਂ",
        advice: "ਕੀ ਕਰਨਾ ਹੈ", search: "ਇਲਾਕਾ ਲੱਭੋ…", est: "ਅੰਦਾਜ਼ਨ — ਨੇੜਲਾ ਮਾਨੀਟਰ ਦੂਰ ਹੈ" },
} as const;

export default function CitizenPage() {
  const cityId = useCommandStore((s) => s.cityId);
  const [lang, setLang] = useState<keyof typeof UI>("en");
  const [wardId, setWardId] = useState<string | null>(null);
  const [audience, setAudience] = useState("general");
  const [search, setSearch] = useState("");

  // Language persists across visits (cookie is fine in the real app; here a
  // simple document.cookie keeps it out of React re-renders).
  useEffect(() => {
    const m = document.cookie.match(/vayu-lang=(\w+)/);
    if (m && m[1] in UI) setLang(m[1] as keyof typeof UI);
  }, []);
  const pickLang = (l: keyof typeof UI) => {
    setLang(l);
    document.cookie = `${LANG_STORE_KEY}=${l};path=/;max-age=31536000`;
  };

  const t = UI[lang];
  const wards = useQuery({ queryKey: queryKeys.wards(cityId), queryFn: () => api.wards(cityId) });
  const brief = useQuery({
    queryKey: queryKeys.citizen(cityId, wardId, lang),
    queryFn: () => api.citizen(cityId, wardId, lang),
  });

  const wardList = useMemo(() => {
    const feats = (wards.data as WardCollection | undefined)?.features ?? [];
    return feats
      .map((f) => ({ id: f.properties.ward_id, name: f.properties.name }))
      .filter((w) => w.name.toLowerCase().includes(search.toLowerCase()))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [wards.data, search]);

  const useLocation = () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const feats = (wards.data as WardCollection | undefined)?.features ?? [];
        let best: { id: string; d: number } | null = null;
        for (const f of feats) {
          const p = f.properties;
          // centroid is [lon, lat] (GeoJSON order).
          const [lon, lat] = p.centroid;
          const d = (lat - pos.coords.latitude) ** 2 + (lon - pos.coords.longitude) ** 2;
          if (!best || d < best.d) best = { id: p.ward_id, d };
        }
        if (best) setWardId(best.id);
      },
      () => {
        /* denial: the searchable selector below is the fallback */
      },
    );
  };

  return (
    <div className="min-h-dvh bg-gradient-to-b from-slate-950 to-slate-900">
      <div className="mx-auto max-w-lg px-4 py-6">
        {/* header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wind className="h-5 w-5 text-sky-400" />
            <span className="text-sm font-semibold tracking-wide text-slate-200">VAYU</span>
          </div>
          <div className="flex gap-1 rounded-full border border-white/10 p-0.5">
            {(Object.keys(UI) as (keyof typeof UI)[]).map((l) => (
              <button
                key={l}
                onClick={() => pickLang(l)}
                className={cn(
                  "rounded-full px-2.5 py-1 text-xs transition-colors",
                  lang === l ? "bg-white/15 text-slate-100" : "text-slate-400 hover:text-slate-200",
                )}
              >
                {l === "en" ? "EN" : l === "hi" ? "हिं" : "ਪੰ"}
              </button>
            ))}
          </div>
        </div>

        {/* area picker */}
        <div className="mt-5">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-slate-500" />
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t.search}
                className="w-full rounded-lg border border-white/10 bg-white/[0.03] py-2 pl-8 pr-3 text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500/40 focus:outline-none"
              />
            </div>
            <button
              onClick={useLocation}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-2 text-xs text-slate-300 hover:border-white/20"
            >
              <MapPin className="h-3.5 w-3.5" />
              {t.locate}
            </button>
          </div>
          {search && (
            <div className="mt-1 max-h-40 overflow-y-auto rounded-lg border border-white/10 bg-slate-900">
              {wardList.slice(0, 12).map((w) => (
                <button
                  key={w.id}
                  onClick={() => {
                    setWardId(w.id);
                    setSearch("");
                  }}
                  className="block w-full px-3 py-2 text-left text-sm text-slate-300 hover:bg-white/5"
                >
                  {w.name}
                </button>
              ))}
              {wardList.length === 0 && <p className="px-3 py-2 text-xs text-slate-500">—</p>}
            </div>
          )}
        </div>

        {brief.isPending && <Skeleton className="mt-6 h-96 w-full rounded-2xl" />}
        {brief.isError && (
          <ErrorState
            className="mt-6"
            title="Could not load air quality"
            detail={brief.error instanceof ApiError ? brief.error.message : String(brief.error)}
            onRetry={() => brief.refetch()}
          />
        )}

        {brief.data && (
          <Brief brief={brief.data} t={t} audience={audience} setAudience={setAudience} />
        )}
      </div>
    </div>
  );
}

function Brief({
  brief,
  t,
  audience,
  setAudience,
}: {
  brief: CitizenBrief;
  t: (typeof UI)[keyof typeof UI];
  audience: string;
  setAudience: (a: string) => void;
}) {
  const aqi = brief.now_aqi;
  const color = brief.now_color ?? "#64748b";
  const shown = useCountUp(aqi ?? 0);
  const adv = brief.advisories.find((a) => a.audience === audience) ?? brief.advisories[0];

  return (
    <>
      {/* AQI hero */}
      <div
        className="mt-6 rounded-2xl border p-6 text-center"
        style={{ borderColor: `${color}40`, backgroundColor: `${color}12` }}
      >
        <p className="text-sm text-slate-400">{brief.ward_name}</p>
        <div className="mt-2 text-7xl leading-none">{faceFor(aqi)}</div>
        <div className="mt-3 font-mono text-6xl font-bold" style={{ color }}>
          {aqi == null ? "—" : shown}
        </div>
        <p className="mt-1 text-lg font-medium" style={{ color }}>
          {brief.now_category ?? "No reading"}
        </p>
        {brief.low_confidence && <p className="mt-2 text-[11px] text-slate-500">{t.est}</p>}
      </div>

      {/* clean hours strip */}
      <section className="mt-6">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-medium text-slate-300">{t.clean}</h2>
          {brief.clean_hours.best_window ? (
            <span className="text-xs text-emerald-400">
              {t.best}: {brief.clean_hours.best_window}
            </span>
          ) : (
            <span className="text-xs text-slate-500">{t.none}</span>
          )}
        </div>
        <div className="mt-2 flex gap-0.5 overflow-hidden rounded-lg">
          {brief.clean_hours.blocks.map((b, i) => (
            <div
              key={i}
              className="group relative h-8 flex-1"
              style={{ backgroundColor: b.color, opacity: b.clean ? 1 : 0.5 }}
              title={`${new Date(b.ts).toLocaleString("en-IN", { hour: "numeric", day: "numeric", month: "short" })} · AQI ${b.aqi}`}
            />
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-slate-600">
          <span>now</span>
          <span>+24h</span>
          <span>+48h</span>
        </div>
      </section>

      {/* advisories */}
      <section className="mt-6">
        <h2 className="text-sm font-medium text-slate-300">{t.advice}</h2>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {brief.advisories.map((a) => (
            <button
              key={a.audience}
              onClick={() => setAudience(a.audience)}
              className={cn(
                "rounded-full px-3 py-1 text-xs transition-colors",
                audience === a.audience
                  ? "bg-sky-500/20 text-sky-200"
                  : "border border-white/10 text-slate-400 hover:text-slate-200",
              )}
            >
              {a.audience_label}
            </button>
          ))}
        </div>
        {adv && (
          <div className="mt-3 rounded-xl border border-white/8 bg-white/[0.02] p-4">
            <p className="text-sm leading-relaxed text-slate-200">{adv.text}</p>
            <p className="mt-3 text-[11px] text-slate-600">{adv.source}</p>
          </div>
        )}
      </section>

      <p className="mt-6 text-center text-[11px] text-slate-600">
        Prototype · not an official government advisory
      </p>
    </>
  );
}

/** Count-up on the AQI numeral (App Flow §3.5). */
function useCountUp(target: number, ms = 700): number {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!target) {
      setN(0);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / ms, 1);
      setN(Math.round(target * (1 - Math.pow(1 - p, 3))));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return n;
}
