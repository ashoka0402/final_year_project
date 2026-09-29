"use client";

import { usePathname } from "next/navigation";

import { AgentDrawer } from "./AgentDrawer";

/**
 * Mounts the Agent Activity drawer on the commissioner surfaces only.
 *
 * The drawer exposes VAYU's internal reasoning — the right audience is the
 * operator, not the public. So it is hidden on /citizen (a separate public URL)
 * and /inspector (the field officer's phone view), and present everywhere else.
 */
const HIDDEN_PREFIXES = ["/citizen", "/inspector"];

export function AgentDrawerMount() {
  const pathname = usePathname();
  if (HIDDEN_PREFIXES.some((p) => pathname.startsWith(p))) return null;
  return <AgentDrawer />;
}
