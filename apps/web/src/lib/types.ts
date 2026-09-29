/** Mirrors services/api/schemas.py. Phase 6 will generate these from openapi.json. */

import type { MultiPolygon, Polygon } from "geojson";

export type DataStatusValue =
  | "live"
  | "cached"
  | "sample"
  | "cams"
  | "h3-fallback"
  | "unavailable";

export type AqiCategory =
  | "Good"
  | "Satisfactory"
  | "Moderate"
  | "Poor"
  | "Very Poor"
  | "Severe";

export interface City {
  id: string;
  name: string;
  timezone: string;
  bbox: [number, number, number, number];
  map_center: [number, number];
  map_zoom: number;
  languages: string[];
  grap_applicable: boolean;
  ward_count: number;
  station_count: number;
  population: number;
  population_source: string;
}

export interface DataStatus {
  source: string;
  status: DataStatusValue;
  detail: string;
  rows_loaded: number;
  fetched_ts: string | null;
}

export interface Station {
  station_id: string;
  name: string;
  lat: number;
  lon: number;
  provider: string;
  ts: string | null;
  pm25: number | null;
  aqi: number | null;
  category: AqiCategory | null;
  color: string | null;
  source: string | null;
}

export interface Ward {
  ward_id: string;
  name: string;
  pm25: number | null;
  aqi: number | null;
  category: AqiCategory | null;
  color: string | null;
  population: number;
  nearest_station_km: number | null;
  low_confidence: boolean;
}

export interface Current {
  city: string;
  as_of: string;
  demo_mode: boolean;
  aqi: number | null;
  category: AqiCategory | null;
  color: string | null;
  dominant_param: string;
  sources: string[];
  wards: Ward[];
  stations: Station[];
  data_status: DataStatus[];
}

/** Ward polygon properties. deck.gl's GeoJsonLayer generic is the *properties*
 *  type, so this is what parameterises the layer. */
export interface WardProps {
  ward_id: string;
  name: string;
  population: number;
  area_km2: number;
  centroid: [number, number];
}

export interface WardFeature {
  type: "Feature";
  id: string;
  properties: WardProps;
  geometry: Polygon | MultiPolygon;
}

export interface WardCollection {
  type: "FeatureCollection";
  attribution?: string;
  features: WardFeature[];
}

export interface ForecastWard {
  ward_id: string;
  name: string;
  population: number;
  p10: number;
  p50: number;
  p90: number;
  aqi_p50: number;
  category: AqiCategory;
  color: string;
}

export interface Forecast {
  city: string;
  run_ts: string;
  horizon_h: number;
  model_ver: string;
  target_ts: string | null;
  wards: ForecastWard[];
}

export interface HazardAlert {
  ward_id: string;
  name: string;
  population: number;
  eta_h: number;
  target_ts: string;
  aqi_p50: number;
  pm25_p50: number;
  /** Share of the p10–p90 band above the threshold. */
  confidence: number;
}

export interface ExplainFeature {
  feature: string;
  label: string;
  contribution: number;
  direction: "increases" | "decreases" | "base";
  value: number | null;
}

export interface Explain {
  city: string;
  ward_id: string;
  horizon_h: number;
  explained_via_station: string;
  station_distance_km: number;
  features: ExplainFeature[];
}

export interface Health {
  status: "ok" | "degraded";
  demo_mode: boolean;
  now: string;
  cities: string[];
  seeded: boolean;
  detail: string;
}

export type SourceCategory =
  | "open_burning"
  | "traffic"
  | "construction"
  | "industry"
  | "regional_transport";

export interface AttributionEvidence {
  type: "fire" | "industry" | "permit" | "traffic" | "regional";
  label: string;
  lat: number | null;
  lon: number | null;
  distance_km: number | null;
  timestamp: string | null;
  detail: string | null;
  source: string | null;
  weight: number;
}

export interface AttributionCategory {
  category: SourceCategory;
  label: string;
  share_pct: number;
  confidence: number;
  raw_score: number;
  evidence: AttributionEvidence[];
}

export interface Attribution {
  city: string;
  ward_id: string;
  ward_name?: string;
  computed_ts: string;
  window_h: number;
  stagnant: boolean;
  note: string | null;
  trajectory_ref: string;
  station_agreement?: number;
  categories: AttributionCategory[];
  trajectory: {
    hours: number;
    length_km: number;
    mean_speed_kmh: number;
    stagnant: boolean;
  };
}

export interface TrajectoryCollection {
  type: "FeatureCollection";
  properties: {
    ward_id: string;
    ward_name: string;
    hours: number;
    length_km: number;
    mean_speed_kmh: number;
    stagnant: boolean;
    run_ts: string;
  };
  features: GeoJSONFeature[];
}

// Loose feature shape — the trajectory payload carries a LineString and a Polygon.
export interface GeoJSONFeature {
  type: "Feature";
  properties: Record<string, unknown>;
  geometry: { type: string; coordinates: unknown } | null;
}

// ---- interventions ---------------------------------------------------------

export type ActionType =
  | "halt_burning"
  | "stop_work_construction"
  | "traffic_restriction"
  | "industrial_curb"
  | "road_dust_suppression";

export type OrderStatus = "candidate" | "dispatched" | "executed" | "verified";

export interface Regulation {
  id: string;
  instrument: string;
  stage: number | null;
  clause: string;
  title: string;
  text: string;
  citation: string;
  penalty_reference?: string;
  action_supported?: string;
}

export interface Candidate {
  id: string;
  city: string;
  ward_id: string;
  ward_name: string;
  action_type: ActionType;
  title: string;
  category: SourceCategory;
  source_lat: number;
  source_lon: number;
  distance_km: number;
  /** Population-weighted mean µg/m³ averted across every ward helped — the ROI
   *  numerator. Not this ward's own figure; that is `ward_averted_ugm3`. */
  predicted_ugm3_averted: number;
  ward_averted_ugm3: number;
  wards_protected: number;
  peak_ugm3_averted: number;
  averted_by_horizon: Record<string, number>;
  population_protected: number;
  effort_units: number;
  confidence: number;
  roi_score: number;
  rationale: string;
  evidence: AttributionEvidence[];
  regulation: Regulation | null;
}

/** A source that matters but that this city cannot act on. Rendered alongside
 *  the leaderboard: an empty table alone reads as "nothing to do", when the
 *  truth is "not yours to fix — escalate". */
export interface Advisory {
  kind: "out_of_range" | "no_local_lever";
  category: SourceCategory;
  headline: string;
  detail: string;
  share_pct: number;
  escalate_to: string | null;
  source_count: number;
  nearest_km: number | null;
  farthest_km: number | null;
  total_magnitude: number | null;
}

export interface Leaderboard {
  city: string;
  ward_id: string | null;
  computed_ts: string;
  candidates: Candidate[];
  advisories: Advisory[];
  meta: {
    wards_evaluated: number;
    wards_total: number | null;
    selection: string;
    city_aqi: number | null;
  };
}

export interface Order {
  id: string;
  city: string;
  ward_id: string;
  created_ts: string | null;
  action_type: ActionType;
  title: string;
  source_lat: number;
  source_lon: number;
  predicted_ugm3_averted: number;
  population_protected: number;
  effort_units: number;
  confidence: number;
  roi_score: number;
  status: OrderStatus;
  signal_ts: string | null;
  dispatched_ts: string | null;
  executed_ts: string | null;
  seeded: boolean;
  has_dossier: boolean;
  signal_to_dossier_s?: number;
  /** Real wall-clock ms to turn the signal into a rendered dossier. */
  pipeline_ms?: number;
}

export interface OrderList {
  orders: Order[];
  count: number;
}

// ---- verification ----------------------------------------------------------

export interface DidSeries {
  days: string[];
  target: (number | null)[];
  control: (number | null)[];
}

export interface Verification {
  intervention_id: string;
  status?: "verified" | "pending" | "error";
  method: string;
  control_wards: string[];
  control_ward_names?: string[];
  ward_name?: string;
  predicted_reduction: number;
  observed_reduction: number;
  ci_low: number;
  ci_high: number;
  /** observed/predicted, clamped [0,150]. Meaningless when `significant` is
   *  false — the CI spans zero and the effect cannot be told from weather. */
  pct_realized: number;
  computed_ts: string;
  target_pre: number;
  target_post: number;
  control_pre: number;
  control_post: number;
  post_hours: number;
  significant: boolean;
  note: string | null;
  series: DidSeries;
  order: Order;
  hours_elapsed?: number;
  hours_required?: number;
  hours_remaining?: number;
  detail?: string;
}

export interface VerificationList {
  verifications: Verification[];
  count: number;
}

// ---- citizen (Herald) ------------------------------------------------------

export interface HourBlock {
  ts: string;
  aqi: number;
  category: string;
  color: string;
  clean: boolean;
}

export interface CitizenAdvisory {
  audience: string;
  audience_label: string;
  text: string;
  source: string;
}

export interface CitizenBrief {
  ward_id: string;
  ward_name: string;
  language: string;
  now_aqi: number | null;
  now_category: string | null;
  now_color: string | null;
  low_confidence: boolean;
  clean_hours: {
    blocks: HourBlock[];
    best_window: string | null;
    best_window_start: string | null;
    best_aqi: number | null;
  };
  advisories: CitizenAdvisory[];
  languages: { code: string; label: string }[];
}

// ---- audit (Agent Activity) ------------------------------------------------

export interface AuditEntry {
  id: number;
  ts: string | null;
  agent: string;
  trigger: string;
  decision: string;
  reasoning: string;
  confidence: number | null;
  duration_ms: number | null;
}

export interface AuditList {
  entries: AuditEntry[];
  count: number;
}

// ---- evaluation (Methodology) ----------------------------------------------

export interface EvalMetric {
  model: "VAYU" | "Persistence" | "Climatology";
  horizon_h: number;
  n: number;
  rmse: number;
  mae: number;
  bucket_accuracy: number;
  crossing_precision: number;
  crossing_recall: number;
  crossing_events: number;
}

export interface Evaluation {
  generated_ts: string;
  protocol: {
    method: string;
    holdout_days: number;
    holdout_from: string;
    holdout_to: string;
    crossing_threshold_aqi: number;
    cities: string[];
  };
  metrics: EvalMetric[];
  calibration_p10_p90: Record<string, number>;
  charts: string[];
}

// ---- GRAP autopilot --------------------------------------------------------

export interface GrapMeasure {
  clause_id: string;
  title: string;
  text: string;
  citation: string;
  action_supported: string | null;
}

export interface GrapDraft {
  id: string;
  city: string;
  current_stage: number;
  current_stage_label: string;
  forecast_stage: number;
  forecast_stage_label: string;
  forecast_aqi: number;
  trigger_forecast_ts: string;
  crossing_eta_h: number | null;
  status: "draft" | "approved" | "dismissed";
  measures: GrapMeasure[];
}

export interface GrapState {
  city: string;
  current_stage: number;
  current_stage_label: string;
  observed_city_aqi: number | null;
  crossing_forecast: boolean;
  forecast_series: { horizon_h: number; city_aqi: number }[];
  draft: GrapDraft | null;
}
