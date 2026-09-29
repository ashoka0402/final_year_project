import type { AqiCategory } from "./types";

/**
 * CPCB buckets, mirroring vayu_core/aqi.py.
 *
 * The API already sends a category and colour per ward, so this exists for the
 * cases where the client has only a number (legend, KPI tiles, tooltips) — and
 * every consumer must render the label alongside the colour: PRD accessibility
 * requires AQI never be conveyed by colour alone.
 */
export const AQI_BANDS: {
  min: number;
  max: number;
  label: AqiCategory;
  color: string;
  /** Advice tone for the commissioner surface — terse, operational. */
  note: string;
}[] = [
  { min: 0, max: 50, label: "Good", color: "#009865", note: "Minimal impact" },
  { min: 51, max: 100, label: "Satisfactory", color: "#A3C853", note: "Minor breathing discomfort to sensitive people" },
  { min: 101, max: 200, label: "Moderate", color: "#FFF833", note: "Breathing discomfort to people with lung disease" },
  { min: 201, max: 300, label: "Poor", color: "#F29C33", note: "Breathing discomfort on prolonged exposure" },
  { min: 301, max: 400, label: "Very Poor", color: "#E93F33", note: "Respiratory illness on prolonged exposure" },
  { min: 401, max: 500, label: "Severe", color: "#AF2D24", note: "Affects healthy people; serious impact on those with illness" },
];

export function bandFor(aqi: number | null | undefined) {
  if (aqi == null || Number.isNaN(aqi)) return null;
  return AQI_BANDS.find((b) => aqi >= b.min && aqi <= b.max) ?? AQI_BANDS[AQI_BANDS.length - 1];
}

/** Colour for a ward polygon; grey when there is no reading (never green). */
export function aqiColor(aqi: number | null | undefined): string {
  return bandFor(aqi)?.color ?? "#334155";
}

/** "#RRGGBB" -> deck.gl [r,g,b,a] */
export function hexToRgba(hex: string, alpha = 255): [number, number, number, number] {
  const h = hex.replace("#", "");
  const n = parseInt(
    h.length === 3
      ? h
          .split("")
          .map((c) => c + c)
          .join("")
      : h,
    16,
  );
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255, alpha];
}

/** Text colour with adequate contrast against an AQI swatch (WCAG AA). */
export function readableOn(hex: string): string {
  const [r, g, b] = hexToRgba(hex);
  // Relative luminance, sRGB.
  const lum = [r, g, b]
    .map((c) => c / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
    .reduce((acc, c, i) => acc + c * [0.2126, 0.7152, 0.0722][i], 0);
  return lum > 0.45 ? "#0A0E1A" : "#F8FAFC";
}
