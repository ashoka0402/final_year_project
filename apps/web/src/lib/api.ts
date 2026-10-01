import type {
  Attribution,
  AuditList,
  Evaluation,
  Candidate,
  City,
  Current,
  Explain,
  Forecast,
  GrapState,
  HazardAlert,
  Health,
  Leaderboard,
  Order,
  OrderList,
  CitizenBrief,
  OrderStatus,
  TrajectoryCollection,
  VerificationList,
  WardCollection,
} from "./types";

/**
 * Requests go to a same-origin /api/v1 path, which next.config.mjs rewrites to
 * the API. One origin => no CORS preflight on the hot path.
 */
const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers: { Accept: "application/json" } });
  } catch (e) {
    // Network/offline: distinguish from an API that answered with an error, so
    // the ErrorState can say something true about what went wrong.
    throw new ApiError("Cannot reach the VAYU API", 0, e instanceof Error ? e.message : undefined);
  }

  if (!res.ok) {
    // The API speaks RFC7807; surface its `detail` rather than a bare status.
    let detail: string | undefined;
    try {
      const problem = await res.json();
      detail = problem?.detail ?? problem?.title;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail ?? `Request failed (${res.status})`, res.status, detail);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new ApiError("Cannot reach the VAYU API", 0, e instanceof Error ? e.message : undefined);
  }
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const problem = await res.json();
      detail = problem?.detail ?? problem?.title;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail ?? `Request failed (${res.status})`, res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // --- Citizen + corridors ---------------------------------------------------
  submitPhoto: async (form: FormData): Promise<CitizenVerdict> => {
    const res = await fetch(`${BASE}/citizen/report/photo`, { method: "POST", body: form });
    if (!res.ok) {
      let detail = `Upload failed (${res.status})`;
      try {
        const body = await res.json();
        detail = body.detail ?? detail;
      } catch { /* problem+json not returned */ }
      throw new ApiError(detail, res.status, detail);
    }
    return res.json() as Promise<CitizenVerdict>;
  },
  citizenReports: (regionId = "india", verdict = "all") =>
    get<CitizenReportList>(`/citizen/reports?region_id=${regionId}&verdict=${verdict}`),
  corridors: (regionId = "india") =>
    get<{ corridors: CorridorSummary[] }>(`/corridors?region_id=${regionId}`),
  corridorBulletin: (id: string, date: string, regionId = "india") =>
    get<CorridorBulletin>(`/corridors/${id}/bulletin?date=${date}&region_id=${regionId}`),

  health: () => get<Health>("/health"),
  cities: () => get<City[]>("/cities"),
  current: (cityId: string) => get<Current>(`/cities/${cityId}/current`),
  wards: (cityId: string) => get<WardCollection>(`/cities/${cityId}/wards.geojson`),
  forecast: (cityId: string, h: number) => get<Forecast>(`/cities/${cityId}/forecast?h=${h}`),
  alerts: (cityId: string) => get<HazardAlert[]>(`/cities/${cityId}/alerts`),
  explain: (cityId: string, wardId: string, h: number) =>
    get<Explain>(`/cities/${cityId}/forecast/explain/${encodeURIComponent(wardId)}?h=${h}`),
  attribution: (cityId: string, wardId: string, hours: number) =>
    get<Attribution>(`/cities/${cityId}/attribution/${encodeURIComponent(wardId)}?hours=${hours}`),
  trajectory: (cityId: string, wardId: string, hours: number) =>
    get<TrajectoryCollection>(`/cities/${cityId}/trajectory/${encodeURIComponent(wardId)}?hours=${hours}`),

  interventions: (cityId: string, wardId?: string | null, actionType?: string | null) => {
    const q = new URLSearchParams();
    if (wardId) q.set("ward_id", wardId);
    if (actionType) q.set("action_type", actionType);
    const qs = q.toString();
    return get<Leaderboard>(`/cities/${cityId}/interventions${qs ? `?${qs}` : ""}`);
  },

  /** Renders the dossier and creates the order. The only write on this path —
   *  everything before it is a recommendation, this is a record. */
  dispatch: (candidate: Candidate, signalTs?: string) =>
    post<Order>("/interventions/dispatch", { candidate, signal_ts: signalTs ?? null }),

  orders: (cityId?: string, status?: OrderStatus) => {
    const q = new URLSearchParams();
    if (cityId) q.set("city_id", cityId);
    if (status) q.set("status", status);
    const qs = q.toString();
    return get<OrderList>(`/interventions${qs ? `?${qs}` : ""}`);
  },

  order: (orderId: string) => get<Order>(`/interventions/${encodeURIComponent(orderId)}`),

  execute: (orderId: string, note: string) =>
    post<Order>(`/interventions/${encodeURIComponent(orderId)}/execute`, { note }),

  dossierUrl: (orderId: string) => `${BASE}/interventions/${encodeURIComponent(orderId)}/dossier`,

  verifications: (cityId?: string) =>
    get<VerificationList>(`/verifications${cityId ? `?city_id=${cityId}` : ""}`),

  audit: (limit = 100) => get<AuditList>(`/audit?limit=${limit}`),
  evaluation: () => get<Evaluation>(`/meta/evaluation`),

  grap: (cityId: string) => get<GrapState>(`/cities/${cityId}/grap`),
  grapApprove: (draftId: string) => post<{ id: string; status: string }>(`/grap/${encodeURIComponent(draftId)}/approve`, {}),
  grapDismiss: (draftId: string) => post<{ id: string; status: string }>(`/grap/${encodeURIComponent(draftId)}/dismiss`, {}),
  auditStreamUrl: () => `${BASE}/audit/stream`,

  citizen: (cityId: string, wardId: string | null, lang: string) =>
    get<CitizenBrief>(
      wardId
        ? `/cities/${cityId}/citizen/${encodeURIComponent(wardId)}?lang=${lang}`
        : `/cities/${cityId}/citizen?lang=${lang}`,
    ),

  clock: () => get<Clock>("/clock"),
  notableDates: (cityId: string) => get<{ city: string; dates: NotableDate[] }>(`/notable-dates?city_id=${cityId}`),
  /** Pin the app clock to an instant (ISO), or null to clear the pin. */
  setClock: (asOf: string | null) => post<Clock>("/clock", { as_of: asOf }),
};

export interface Clock {
  now: string;
  source: "override" | "archive";
  pinned: boolean;
  data_min: string | null;
  data_max: string | null;
  max_selectable: string | null;
}

/** A quick-jump episode for the date picker — see meta.py:NOTABLE_DATES. */
export interface NotableDate {
  at: string;
  label: string;
  aqi: number;
  category: string;
}

export const queryKeys = {
  health: ["health"] as const,
  clock: ["clock"] as const,
  notableDates: (cityId: string) => ["notableDates", cityId] as const,
  cities: ["cities"] as const,
  current: (cityId: string) => ["current", cityId] as const,
  wards: (cityId: string) => ["wards", cityId] as const,
  forecast: (cityId: string, h: number) => ["forecast", cityId, h] as const,
  alerts: (cityId: string) => ["alerts", cityId] as const,
  explain: (cityId: string, wardId: string, h: number) => ["explain", cityId, wardId, h] as const,
  attribution: (cityId: string, wardId: string, hours: number) =>
    ["attribution", cityId, wardId, hours] as const,
  trajectory: (cityId: string, wardId: string, hours: number) =>
    ["trajectory", cityId, wardId, hours] as const,
  interventions: (cityId: string, wardId?: string | null, actionType?: string | null) =>
    ["interventions", cityId, wardId ?? null, actionType ?? null] as const,
  orders: (cityId?: string, status?: string) => ["orders", cityId ?? null, status ?? null] as const,
  order: (orderId: string) => ["order", orderId] as const,
  verifications: (cityId?: string) => ["verifications", cityId ?? null] as const,
  citizen: (cityId: string, wardId: string | null, lang: string) =>
    ["citizen", cityId, wardId ?? null, lang] as const,
  audit: ["audit"] as const,
  evaluation: ["evaluation"] as const,
  grap: (cityId: string) => ["grap", cityId] as const,
  citizenReports: (regionId: string, verdict: string) =>
    ["citizenReports", regionId, verdict] as const,
  corridors: (regionId: string) => ["corridors", regionId] as const,
  corridorBulletin: (id: string, date: string) =>
    ["corridorBulletin", id, date] as const,
};

// --- Citizen reports + federated corridors (Code for Communities track) ------
export interface CitizenVerdict {
  id: string;
  verdict: "corroborated" | "unsupported" | "contradicted" | "no_satellite_data" | "unusable";
  may_influence: boolean;
  usable: boolean;
  detail: string;
  haze_severity: string | null;
  source_type: string | null;
}

export interface CitizenReportRow extends CitizenVerdict {
  lat: number; lon: number; grid_lat: number; grid_lon: number;
  date: string; reported_ts: string; kind: "photo" | "sensor";
  visible_smoke: boolean | null; ai_confidence: number | null;
  ai_reasoning: string | null; ai_model: string | null;
  pm25: number | null; hcho_z: number | null; fire_count: number | null;
  verdict_detail: string; note: string | null;
}

export interface CitizenReportList {
  region: string; count: number;
  by_verdict: Record<string, number>;
  google_ai_enabled: boolean;
  items: CitizenReportRow[];
}

export interface CorridorSummary {
  id: string; name: string; states: string[];
  cells: number; buffer_deg: number; waypoints: number[][];
}

export interface CorridorBulletin {
  schema: string; issued_utc: string; date: string;
  corridor: { id: string; name: string; states: string[] };
  coverage: { cells_total: number; cells_observed: number; coverage_pct: number };
  hcho: { mean: number | null; unit: string; max_anomaly_sigma: number | null; hotspot_cells: number; source: string };
  fire: { count: number; source: string };
  citizen: { reports: number; satellite_corroborated: number; note: string };
  top_hotspots: { lat: number; lon: number; anomaly_sigma: number; fire_count: number; source_region: string | null }[];
}
