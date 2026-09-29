"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // In DEMO_MODE the clock is pinned, so aggressive refetching buys
            // nothing and risks a flicker mid-demo.
            staleTime: 60_000,
            refetchOnWindowFocus: false,
            retry: (failureCount, error) => {
              // Don't retry a 404 (unknown city) — only transient failures.
              const status = (error as { status?: number })?.status;
              if (status && status >= 400 && status < 500) return false;
              return failureCount < 2;
            },
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
