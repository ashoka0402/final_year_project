"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Activity, Brain, ClipboardCheck, Factory, Megaphone, Radio, Send, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, queryKeys } from "@/lib/api";
import { cn } from "@/lib/cn";
import type { AuditEntry } from "@/lib/types";
import { useQuery } from "@tanstack/react-query";

/**
 * Agent Activity drawer (PRD F1) — the trust surface, reachable from every
 * screen via a right-edge tab.
 *
 * VAYU's claim is that a human can trust an automated enforcement
 * recommendation. This drawer is where that claim is kept honest: every
 * automated step, its reasoning, and its confidence, in one stream. It seeds
 * from GET /audit and then live-updates over SSE, so a dispatch made during a
 * demo appears here within a couple of seconds.
 */

const AGENT_META: Record<string, { icon: typeof Brain; color: string; label: string }> = {
  forecaster: { icon: Activity, color: "#38bdf8", label: "Forecaster" },
  attributor: { icon: Brain, color: "#a78bfa", label: "Attributor" },
  enforcer: { icon: Factory, color: "#f59e0b", label: "Enforcer" },
  herald: { icon: Megaphone, color: "#34d399", label: "Herald" },
  inspector: { icon: ClipboardCheck, color: "#c084fc", label: "Inspector" },
  verifier: { icon: Radio, color: "#22d3ee", label: "Verifier" },
};

function metaFor(agent: string) {
  return AGENT_META[agent] ?? { icon: Send, color: "#64748b", label: agent };
}

export function AgentDrawer() {
  const [open, setOpen] = useState(false);
  const [live, setLive] = useState<AuditEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const seenIds = useRef<Set<number>>(new Set());

  // Initial fill from the REST endpoint (works even if SSE is blocked).
  const seed = useQuery({ queryKey: queryKeys.audit, queryFn: () => api.audit(100) });

  useEffect(() => {
    if (seed.data) {
      for (const e of seed.data.entries) seenIds.current.add(e.id);
      setLive(seed.data.entries);
    }
  }, [seed.data]);

  // Live stream. Only opened while the drawer is open, to avoid a hanging
  // connection on every page for a log that rarely changes.
  useEffect(() => {
    if (!open) return;
    const es = new EventSource(api.auditStreamUrl());
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (ev) => {
      try {
        const e: AuditEntry = JSON.parse(ev.data);
        if (seenIds.current.has(e.id)) return;
        seenIds.current.add(e.id);
        setLive((prev) => [e, ...prev].slice(0, 200));
      } catch {
        /* comment/heartbeat line */
      }
    };
    return () => es.close();
  }, [open]);

  return (
    <>
      {/* right-edge tab, present on every screen */}
      <button
        onClick={() => setOpen(true)}
        className="fixed right-0 top-1/2 z-40 flex -translate-y-1/2 items-center gap-1.5 rounded-l-lg border border-r-0 border-white/10 bg-slate-900/90 py-3 pl-2 pr-1.5 text-slate-300 backdrop-blur hover:bg-slate-800/90"
        style={{ writingMode: "vertical-rl" }}
        aria-label="Open Agent Activity"
      >
        <Activity className="h-3.5 w-3.5 rotate-90" />
        <span className="text-[11px] font-medium tracking-wide">Agent Activity</span>
      </button>

      <AnimatePresence>
        {open && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setOpen(false)}
              className="fixed inset-0 z-40 bg-black/40"
            />
            <motion.aside
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "tween", duration: 0.22, ease: "easeOut" }}
              className="fixed right-0 top-0 z-50 flex h-dvh w-full max-w-md flex-col border-l border-white/10 bg-slate-950"
            >
              <header className="flex items-center justify-between border-b border-white/8 px-4 py-3">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-sky-400" />
                  <h2 className="text-sm font-semibold text-slate-100">Agent Activity</h2>
                  <span
                    className={cn(
                      "flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px]",
                      connected ? "bg-emerald-500/15 text-emerald-300" : "bg-slate-500/15 text-slate-400",
                    )}
                  >
                    <span className={cn("h-1.5 w-1.5 rounded-full", connected ? "bg-emerald-400" : "bg-slate-500")} />
                    {connected ? "live" : "idle"}
                  </span>
                </div>
                <button onClick={() => setOpen(false)} aria-label="Close" className="rounded p-1 text-slate-500 hover:text-slate-300">
                  <X className="h-4 w-4" />
                </button>
              </header>

              <p className="border-b border-white/5 px-4 py-2 text-[11px] text-slate-500">
                Every automated decision, its reasoning, and its confidence. Human-in-the-loop:
                nothing here was actioned without a person.
              </p>

              <div className="flex-1 overflow-y-auto p-3">
                {live.length === 0 && (
                  <p className="mt-8 text-center text-xs text-slate-600">No activity recorded yet.</p>
                )}
                <ol className="space-y-2">
                  {live.map((e) => (
                    <AuditRow key={e.id} e={e} />
                  ))}
                </ol>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

function AuditRow({ e }: { e: AuditEntry }) {
  const m = metaFor(e.agent);
  const Icon = m.icon;
  return (
    <li className="rounded-lg border border-white/8 bg-white/[0.02] p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span
            className="flex h-5 w-5 items-center justify-center rounded"
            style={{ backgroundColor: `${m.color}22`, color: m.color }}
          >
            <Icon className="h-3 w-3" />
          </span>
          <span className="text-xs font-medium text-slate-200">{m.label}</span>
        </div>
        <div className="flex items-center gap-2">
          {e.confidence != null && (
            <span className="font-mono text-[10px] text-slate-500">conf {e.confidence.toFixed(2)}</span>
          )}
          <time className="text-[10px] text-slate-600">{fmt(e.ts)}</time>
        </div>
      </div>
      <p className="mt-1.5 text-xs text-slate-200">{e.decision}</p>
      {e.reasoning && <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{e.reasoning}</p>}
      {e.duration_ms != null && (
        <p className="mt-1 text-[10px] text-slate-600">{e.duration_ms} ms</p>
      )}
    </li>
  );
}

function fmt(ts: string | null): string {
  if (!ts) return "";
  return new Date(ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}
