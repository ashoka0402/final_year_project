import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import "maplibre-gl/dist/maplibre-gl.css";
import { AgentDrawerMount } from "@/components/AgentDrawerMount";
import { Providers } from "./providers";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "VAYU — Verifiable Airshed Intelligence & Enforcement",
  description:
    "Dashboards measure pollution. VAYU prosecutes it. Evidence-backed intervention orders for Indian cities.",
};

export const viewport: Viewport = {
  themeColor: "#0A0E1A",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
      <body className="min-h-screen bg-base">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-data focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-base"
        >
          Skip to content
        </a>
        <Providers>
          {children}
          <AgentDrawerMount />
        </Providers>
      </body>
    </html>
  );
}
