"use client";

import { AlertTriangle, Inbox, RotateCw } from "lucide-react";
import { cn } from "@/lib/cn";

/**
 * The four states every async region must implement (App Flow §3):
 * loading (skeleton) / ready / empty (designed) / error (message + retry).
 * "Never a blank div, never a spinner-only page."
 */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded bg-edge/60", className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/[0.07] to-transparent" />
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  icon,
  className,
}: {
  title: string;
  hint?: string;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 px-4 py-8 text-center", className)}>
      <div className="rounded-full border border-edge bg-surface-2 p-2.5 text-slate-500">
        {icon ?? <Inbox className="h-4 w-4" aria-hidden />}
      </div>
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {hint && <p className="max-w-[38ch] text-xs leading-relaxed text-slate-500">{hint}</p>}
    </div>
  );
}

export function ErrorState({
  title = "Something went wrong",
  detail,
  onRetry,
  className,
}: {
  title?: string;
  detail?: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn("flex flex-col items-center justify-center gap-3 px-4 py-8 text-center", className)}
    >
      <div className="rounded-full border border-hazard/30 bg-hazard/10 p-2.5 text-hazard">
        <AlertTriangle className="h-4 w-4" aria-hidden />
      </div>
      <div>
        <p className="text-sm font-medium text-slate-200">{title}</p>
        {detail && <p className="mt-1 max-w-[42ch] text-xs leading-relaxed text-slate-500">{detail}</p>}
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 rounded-md border border-edge bg-surface-2 px-3 py-1.5 text-xs font-medium text-slate-200 transition-colors hover:border-data/50 hover:text-data"
        >
          <RotateCw className="h-3 w-3" aria-hidden />
          Retry
        </button>
      )}
    </div>
  );
}
