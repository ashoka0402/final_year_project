import type { Config } from "tailwindcss";

/**
 * VAYU design system (master prompt §8).
 * Command Center is dark and operational; the Citizen surface (Phase 5) is light.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Command Center
        base: "#0A0E1A",
        surface: "#111827",
        "surface-2": "#161F35",
        edge: "#1F2A44",
        // Accents
        data: "#22D3EE", // electric cyan — data
        warn: "#F59E0B", // amber — warnings
        hazard: "#EF4444", // red — hazard
        verified: "#10B981", // green — verified
        // CPCB AQI buckets (authoritative; mirrored from vayu_core/aqi.py)
        aqi: {
          good: "#009865",
          satisfactory: "#A3C853",
          moderate: "#FFF833",
          poor: "#F29C33",
          "very-poor": "#E93F33",
          severe: "#AF2D24",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
        // Transform-only on purpose. An entrance that starts at opacity:0 leaves
        // the element invisible in any environment where the animation clock
        // doesn't advance (screenshot harnesses, some a11y tooling) — visibility
        // must never depend on an animation frame. The slide still reads as motion.
        "slide-in-right": {
          "0%": { transform: "translateX(10px)" },
          "100%": { transform: "translateX(0)" },
        },
        "pulse-ring": {
          "0%": { boxShadow: "0 0 0 0 rgba(34,211,238,0.45)" },
          "70%": { boxShadow: "0 0 0 10px rgba(34,211,238,0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(34,211,238,0)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
        "slide-in-right": "slide-in-right 220ms ease-out",
        "pulse-ring": "pulse-ring 2s infinite",
      },
    },
  },
  plugins: [],
};
export default config;
