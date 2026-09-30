/**
 * Typed API client.
 *
 * Two behaviours matter more than the plumbing:
 *
 * 1. **A failure is never an empty success.** Every error path produces an
 *    `ApiError`, so a panel renders an explicit error state. A dashboard that
 *    renders empty on failure is indistinguishable from one reporting genuinely
 *    empty roads, and in a traffic system that confusion is the dangerous one.
 *
 * 2. **501 is modelled, not treated as a fault.** During phased development the
 *    backend legitimately reports "this capability does not exist yet, it is
 *    phase N". The client surfaces that as a distinct, structured outcome so the
 *    UI can label the gap precisely instead of showing a red failure.
 */

import type { ApiErrorResponse, UploadLocation } from "./types";

const BASE_URL: string = import.meta.env["VITE_API_BASE_URL"] ?? "";
const TIMEOUT_MS: number = Number(import.meta.env["VITE_API_TIMEOUT_MS"] ?? 10_000);

/** An API call that did not succeed. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(
    message: string,
    status: number,
    code: string,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  /** True when the capability belongs to a later development phase. */
  get isNotAvailableYet(): boolean {
    return this.status === 501 || this.code === "not_available_yet";
  }

  /** Owning phase, when the backend reported one. */
  get phase(): number | null {
    const value = this.details["phase"];
    return typeof value === "number" ? value : null;
  }

  /** True when the backend could not be reached at all. */
  get isUnreachable(): boolean {
    return this.status === 0;
  }
}

function isErrorEnvelope(value: unknown): value is ApiErrorResponse {
  if (typeof value !== "object" || value === null) return false;
  const candidate = (value as { error?: unknown }).error;
  return typeof candidate === "object" && candidate !== null && "code" in candidate;
}

/**
 * Perform a typed GET.
 *
 * @throws {ApiError} on any non-2xx response, timeout, or network failure.
 */
export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  // Caller-initiated aborts (component unmount) are chained onto our own
  // timeout controller so both can cancel the same request.
  const onAbort = (): void => controller.abort();
  signal?.addEventListener("abort", onAbort);

  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });

    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      if (isErrorEnvelope(payload)) {
        throw new ApiError(
          payload.error.message,
          response.status,
          payload.error.code,
          payload.error.details ?? {},
        );
      }
      throw new ApiError(
        `Request to ${path} failed with status ${response.status}.`,
        response.status,
        "http_error",
      );
    }

    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        `Request to ${path} timed out after ${TIMEOUT_MS} ms.`,
        0,
        "timeout",
      );
    }
    throw new ApiError(
      `Cannot reach the backend at ${BASE_URL || "the same origin"}${path}. ` +
        `Is the API running?`,
      0,
      "unreachable",
    );
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
  }
}

/** Shared non-GET request helper. Same error discipline as {@link apiGet}. */
async function apiSend<T>(
  path: string,
  init: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);
  const onAbort = (): void => controller.abort();
  signal?.addEventListener("abort", onAbort);

  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (response.status === 204) {
      return undefined as T;
    }

    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      if (isErrorEnvelope(payload)) {
        throw new ApiError(
          payload.error.message,
          response.status,
          payload.error.code,
          payload.error.details ?? {},
        );
      }
      throw new ApiError(
        `Request to ${path} failed with status ${response.status}.`,
        response.status,
        "http_error",
      );
    }

    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(`Request to ${path} timed out.`, 0, "timeout");
    }
    throw new ApiError(
      `Cannot reach the backend at ${BASE_URL || "the same origin"}${path}.`,
      0,
      "unreachable",
    );
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", onAbort);
  }
}

/** Uploads get a longer budget than a metadata read: a video is not a JSON blob. */
const UPLOAD_TIMEOUT_MS = 120_000;

/** POST a JSON body and parse the typed response. */
export async function apiPost<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return apiSend<T>(
    path,
    {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    },
    signal,
  );
}

/**
 * Upload a file as multipart/form-data, optionally with where it was filmed.
 *
 * The `Content-Type` header is deliberately not set: the browser must generate
 * it so it can include the multipart boundary. Setting it by hand produces a
 * body the server cannot parse.
 *
 * Passing a `location` also records an `ObservationSite` for the junction, and
 * changes nothing about the asset that comes back: the response is the same
 * `MediaAsset` either way, because the file and the place are separate things.
 * The site is then readable at `endpoints.site(observationAreaIdForMedia(id))`.
 * Omitting `location` reproduces the previous request byte for byte.
 */
export async function apiUpload<T>(
  path: string,
  file: File,
  location?: UploadLocation,
  signal?: AbortSignal,
): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  if (location !== undefined) {
    // Sent as strings because that is what multipart carries; the backend
    // parses and bounds-checks them before a single byte is stored.
    form.append("latitude", String(location.latitude));
    form.append("longitude", String(location.longitude));
    if (location.bearing_degrees !== undefined) {
      form.append("bearing_degrees", String(location.bearing_degrees));
    }
    if (location.accuracy_m !== undefined) {
      form.append("accuracy_m", String(location.accuracy_m));
    }
    if (location.site_label !== undefined) {
      form.append("site_label", location.site_label);
    }
  }
  return apiSend<T>(
    path,
    { method: "POST", headers: { Accept: "application/json" }, body: form },
    signal,
  );
}

/**
 * Observation area id for footage ingested under `mediaId`.
 *
 * Mirrors `observation_area_id_for_media` on the backend. The convention is
 * stated in both places because the join between a congestion verdict, a
 * routing binding and a site location is this exact string.
 */
export function observationAreaIdForMedia(mediaId: string): string {
  return `observation-area:${mediaId}`;
}

/** DELETE a resource. Resolves on 204. */
export async function apiDelete(path: string, signal?: AbortSignal): Promise<void> {
  await apiSend<void>(path, { method: "DELETE", headers: { Accept: "application/json" } }, signal);
}

/** Absolute URL for media content, for use in `<img>` and `<video>` elements. */
export function mediaContentUrl(mediaId: string): string {
  return `${BASE_URL}/api/media/${encodeURIComponent(mediaId)}/content`;
}

/**
 * URL of the detection overlay this application rendered.
 *
 * A cache-busting token is appended because the overlay is regenerated on each
 * analysis at the same URL; without it the browser would keep showing the
 * previous run's boxes.
 */
export function overlayUrl(mediaId: string, token: string): string {
  return `${BASE_URL}/api/vision/images/${encodeURIComponent(mediaId)}/overlay?v=${encodeURIComponent(token)}`;
}

/** Endpoint paths, centralised so a route rename is a single edit. */
export const endpoints = {
  mediaFormats: "/api/media/formats",
  mediaList: "/api/media",
  mediaLibrary: "/api/media/library",
  uploadImage: "/api/media/images",
  uploadVideo: "/api/media/videos",
  mediaSelect: "/api/media/select",
  liveConnect: "/api/media/live/connect",

  // Observation sites: where the junction behind an observation area is.
  // Persistent and independent of the media store.
  sites: "/api/sites",
  site: (observationAreaId: string): string =>
    `/api/sites/${encodeURIComponent(observationAreaId)}`,

  // Nearest-junction proposal (GET, changes nothing) and its explicit
  // confirmation (POST, the only thing that creates a binding).
  siteCandidate: (observationAreaId: string): string =>
    `/api/sites/${encodeURIComponent(observationAreaId)}/candidate`,
  siteBinding: (observationAreaId: string): string =>
    `/api/sites/${encodeURIComponent(observationAreaId)}/binding`,

  // Directed segments at the confirmed junction, and the separate confirmation
  // that refines a junction binding to one of them.
  siteSegments: (observationAreaId: string): string =>
    `/api/sites/${encodeURIComponent(observationAreaId)}/segments`,
  siteSegmentBinding: (observationAreaId: string): string =>
    `/api/sites/${encodeURIComponent(observationAreaId)}/binding/segment`,
  confirmedBindings: "/api/sites/confirmed",
  // The mapping in force, composed from the file and the confirmations.
  // Read-only and derived; asking for it changes nothing, including route costs.
  effectiveMapping: "/api/sites/effective-mapping",

  // AI Traffic Advisor: an explanation layer over facts the pipeline already
  // computed. Never a second source of traffic figures.
  advisorStatus: "/api/advisor/status",
  advisorAnalyze: "/api/advisor/analyze",
  media: (id: string): string => `/api/media/${encodeURIComponent(id)}`,
  playback: (id: string): string =>
    `/api/media/videos/${encodeURIComponent(id)}/playback`,
  playbackAction: (id: string, action: string): string =>
    `/api/media/videos/${encodeURIComponent(id)}/playback/${action}`,
  analyticsConfig: "/api/analytics/config",
  analyseImageTraffic: (id: string): string =>
    `/api/analytics/images/${encodeURIComponent(id)}`,
  analyseJobTraffic: (jobId: string): string =>
    `/api/analytics/jobs/${encodeURIComponent(jobId)}`,
  visionStatus: "/api/vision/status",
  analyzeImage: (id: string): string =>
    `/api/vision/images/${encodeURIComponent(id)}/analyze`,
  imageAnalysis: (id: string): string =>
    `/api/vision/images/${encodeURIComponent(id)}/analysis`,
  analyzeVideo: (id: string): string =>
    `/api/vision/videos/${encodeURIComponent(id)}/analyze`,
  job: (id: string): string => `/api/vision/jobs/${encodeURIComponent(id)}`,
  jobFrames: (id: string): string =>
    `/api/vision/jobs/${encodeURIComponent(id)}/frames`,
  jobTracks: (id: string): string =>
    `/api/vision/jobs/${encodeURIComponent(id)}/tracks`,
  health: "/api/health",
  versions: "/api/meta/versions",
  phases: "/api/meta/phases",
  sourceCatalog: "/api/sources/catalog",
  configuredSources: "/api/sources",
  safetyCatalog: "/api/safety/catalog",
  network: "/api/network",
  networkState: "/api/network/state",
  emergencyScenarios: "/api/emergency/scenarios",
  emergencyRecommendation: "/api/emergency/recommendation",
  routeConfig: "/api/routes/config",
  routeMapping: "/api/routes/mapping",
  loadMapping: "/api/routes/mapping/load",
  optimizeRoute: "/api/routes/optimize",

  // Traffic simulation (phase 8).
  simulationConfig: "/api/simulation/config",
  simulationRun: "/api/simulation/run",
  simulationCompare: "/api/simulation/compare",
  simulationRuns: "/api/simulation/runs",
  simulationRunById: (id: string): string =>
    `/api/simulation/runs/${encodeURIComponent(id)}`,

  // Traffic state (phase 5) — interpretation of the phase 4 measurements.
  stateConfig: "/api/state/config",
  interpretImageState: (id: string): string =>
    `/api/state/images/${encodeURIComponent(id)}`,
  interpretJobState: (jobId: string): string =>
    `/api/state/jobs/${encodeURIComponent(jobId)}`,
  networkTrafficState: "/api/state/network",

  // Road network (phase 6).
  networkStatus: "/api/network/status",
  loadNetwork: "/api/network/load",
  networkFacilities: "/api/network/facilities",
  networkBlocked: "/api/network/blocked",

  // Incidents (phase 6).
  incidents: "/api/incidents",
  incident: (id: string): string => `/api/incidents/${encodeURIComponent(id)}`,
  incidentStatus: (id: string): string =>
    `/api/incidents/${encodeURIComponent(id)}/status`,

  // Emergency declaration and lifecycle (phase 6).
  declareEmergency: "/api/emergency/declare",
  emergency: (id: string): string => `/api/emergency/${encodeURIComponent(id)}`,
  emergencyStatus: (id: string): string =>
    `/api/emergency/${encodeURIComponent(id)}/status`,
  emergencyPosition: (id: string): string =>
    `/api/emergency/${encodeURIComponent(id)}/position`,
} as const;
