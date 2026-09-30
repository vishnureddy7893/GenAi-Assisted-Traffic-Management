/**
 * Wire types mirroring the backend API schemas.
 *
 * These describe the *shape* of backend responses. They contain no domain
 * values: no junction ids, no road ids, no capability lists, no phase numbers.
 * Everything the dashboard renders comes from a response typed by this file, so
 * the backend stays the single source of truth for what the network contains.
 *
 * Hand-maintained in Phase 1 and deliberately narrow — only the endpoints this
 * dashboard actually calls. Generating them from the published OpenAPI document
 * is a Phase 13 hardening task; until then `npm run typecheck` catches drift
 * against the components that consume them.
 */

// ---------------------------------------------------------------------------
// Provenance primitives
// ---------------------------------------------------------------------------

/**
 * How a value came to exist. Closed set, mirroring the backend enum — note the
 * deliberate absence of any "mock" or "default" member.
 */
export type ValueSource =
  | "observed"
  | "derived"
  | "predicted"
  | "simulated"
  | "static"
  | "assumed";

export interface Provenance {
  source: ValueSource;
  producer: string;
  producer_version: string;
  observed_at: string;
  computed_at: string;
  confidence: number | null;
  input_refs: string[];
  notes: string | null;
}

/**
 * A quantity with its unit and origin. Every fabricable number on the wire
 * arrives in this envelope, so the UI can always show where a value came from.
 */
export interface Measurement<T = number> {
  value: T;
  unit: string;
  provenance: Provenance;
}

export interface GeoPoint {
  latitude: number;
  longitude: number;
}

// ---------------------------------------------------------------------------
// Health and metadata
// ---------------------------------------------------------------------------

export type ComponentStatus =
  | "ok"
  | "not_implemented"
  | "not_configured"
  | "degraded"
  | "failed";

export interface ComponentHealth {
  name: string;
  status: ComponentStatus;
  detail: string | null;
  /** Phase that will implement this subsystem, when not yet implemented. */
  phase: number | null;
}

export interface HealthResponse {
  /** Liveness: is the backend serving requests correctly? */
  status: ComponentStatus;
  /**
   * Whether an end-to-end decision can be produced. Separate from `status` on
   * purpose: the backend can be healthy AND unable to produce a decision, and
   * one flag cannot honestly express both.
   */
  pipeline_ready: boolean;
  pipeline_blockers: string[];
  app_version: string;
  schema_version: string;
  implemented_phase: number;
  implemented_phase_name: string;
  environment: string;
  server_time: string;
  uptime_s: number;
  components: ComponentHealth[];
}

export interface VersionsResponse {
  app_version: string;
  schema_version: string;
  safety_rules_version: string;
  route_weights_version: string;
  implemented_phase: number;
  implemented_phase_name: string;
  model_versions: Record<string, string>;
}

export interface PhaseInfo {
  phase: number;
  name: string;
  implemented: boolean;
  summary: string;
}

export interface PhasesResponse {
  current_phase: number;
  phases: PhaseInfo[];
}

// ---------------------------------------------------------------------------
// Road network topology (/api/network)
// ---------------------------------------------------------------------------

export interface Approach {
  approach_id: string;
  incoming_segment_id: string;
  bearing_degrees: number | null;
  has_pedestrian_crossing: boolean;
  crossing_is_signalized: boolean | null;
}

export interface TurnMovement {
  movement_id: string;
  junction_id: string;
  from_segment_id: string;
  to_segment_id: string;
  movement_type: string;
  is_permitted: boolean;
  signal_group_id: string | null;
}

export interface Junction {
  junction_id: string;
  name: string | null;
  kind: string;
  /** Optional: the layout falls back to a deterministic arrangement without it. */
  position: GeoPoint | null;
  approaches: Approach[];
  movements: TurnMovement[];
}

export interface PhysicalRestrictions {
  max_height_m: Measurement | null;
  max_weight_t: Measurement | null;
  max_width_m: Measurement | null;
  excluded_vehicle_types: string[];
  notes: string | null;
}

/** A directed road segment. A two-way road is two segments. */
export interface RoadSegment {
  segment_id: string;
  from_junction_id: string;
  to_junction_id: string;
  name: string | null;
  road_class: string;
  length_m: Measurement;
  lane_count: number;
  free_flow_speed_kph: Measurement | null;
  speed_limit_kph: Measurement | null;
  capacity_veh_per_hour: Measurement | null;
  physical_restrictions: PhysicalRestrictions;
  geometry: GeoPoint[];
}

export interface Facility {
  facility_id: string;
  name: string;
  facility_type: string;
  access_junction_id: string;
  position: GeoPoint | null;
  accepts_emergency_types: string[];
  notes: string | null;
}

export interface RoadNetwork {
  network_id: string;
  name: string | null;
  version: string;
  driving_side: string;
  provenance: Provenance;
  junctions: Junction[];
  segments: RoadSegment[];
  facilities: Facility[];
}

// ---------------------------------------------------------------------------
// Data sources (/api/sources/catalog)
// ---------------------------------------------------------------------------

export type PayloadKind =
  | "video_frame"
  | "detection_record"
  | "tabular_record"
  | "observation";

export interface SourceKindProfile {
  kind: string;
  /** "A" = recorded real-world data (primary), "B" = live traffic. */
  mode: string;
  display_name: string;
  description: string;
  payload_kind: PayloadKind;
  seekable: boolean;
  bounded: boolean;
  replayable: boolean;
  realtime: boolean;
  requires_backpressure: boolean;
  /** Whether a working implementation exists right now. */
  implemented: boolean;
  available_from_phase: number;
  required_dependencies: string[];
}

export interface SourceModeGroup {
  mode: string;
  title: string;
  description: string;
  is_primary: boolean;
  profiles: SourceKindProfile[];
}

export interface SourceCatalogResponse {
  modes: SourceModeGroup[];
  implemented_kinds: string[];
  /**
   * Stated explicitly rather than inferred from an empty `implemented_kinds`.
   * A client reading only the positive list cannot tell whether nothing is
   * implemented or whether the catalogue itself failed to load.
   */
  unavailable_kinds: string[];
  any_source_available: boolean;
  note: string;
}

// ---------------------------------------------------------------------------
// Emergency scenarios (/api/emergency/scenarios)
// ---------------------------------------------------------------------------

export interface EmergencyVehicle {
  vehicle_id: string;
  vehicle_type: string;
  callsign: string | null;
  height_m: Measurement | null;
  weight_t: Measurement | null;
  requires_wide_turns: boolean | null;
}

export interface NetworkPosition {
  segment_id: string | null;
  offset_m: Measurement | null;
  junction_id: string | null;
  coordinate: GeoPoint | null;
  heading_degrees: number | null;
  observed_at: string | null;
}

export interface EmergencyDestination {
  facility_id: string | null;
  incident_id: string | null;
  position: NetworkPosition | null;
  facility_type: string | null;
  label: string | null;
}

/**
 * A declared emergency response.
 *
 * Declaration landed in Phase 6, so this collection can now be non-empty. It
 * is still empty whenever no road network is loaded: a response is anchored to
 * junctions and facilities that have to exist on a map first.
 */
export interface EmergencyScenario {
  emergency_id: string;
  vehicle: EmergencyVehicle;
  priority: string;
  status: string;
  declared_at: string;
  updated_at: string;
  completed_at?: string | null;
  current_position: NetworkPosition;
  destination: EmergencyDestination;
  triggering_incident_id?: string | null;
  assigned_route_id: string | null;
  notes: string | null;
}

/**
 * Returns 200 with an empty collection rather than 501: a scenario exists only
 * because someone declared it, so "none declared" is a complete and true answer.
 * Contrast `/api/network/state`, which must 501 — traffic exists whether or not
 * we observe it, so an empty state there would assert "no congestion".
 */
export interface EmergencyScenariosResponse {
  scenarios: EmergencyScenario[];
  count: number;
  active_count: number;
  /** Whether an emergency can currently be declared at all. */
  declaration_supported: boolean;
  declaration_phase: number | null;
  /** Why the collection is empty, so it cannot read as "the emergency cleared". */
  note: string;
}

// ---------------------------------------------------------------------------
// Media ingestion (/api/media/*)
// ---------------------------------------------------------------------------

export type MediaKind = "image" | "video";

/**
 * Measured properties of a still image.
 *
 * Everything here was read from the file. Nothing describes its *content* —
 * there is no vehicle count, no congestion level and no classification, because
 * Phase 2 ingests raw input and does not interpret it.
 */
export interface ImageMetadata {
  width: number;
  height: number;
  format: string;
  /** Pillow colour mode (RGB, RGBA, L). `null` when not reported. */
  mode: string | null;
  channels: number | null;
  is_animated: boolean | null;
  /** Frames in an animated image. `null` for a still — not 1. */
  frame_count: number | null;
}

/**
 * Measured properties of a video container.
 *
 * `fps`, `frame_count` and `duration_s` are `null` when the container does not
 * declare them. They are never zero-filled and duration is never estimated.
 */
export interface VideoMetadata {
  width: number;
  height: number;
  fps: number | null;
  frame_count: number | null;
  duration_s: number | null;
  codec_fourcc: string | null;
}

export interface MediaAsset {
  media_id: string;
  kind: MediaKind;
  format: string;
  media_type: string;
  original_filename: string;
  size_bytes: number;
  ingested_at: string;
  image: ImageMetadata | null;
  video: VideoMetadata | null;
}

export interface MediaListResponse {
  assets: MediaAsset[];
  count: number;
}

export interface SupportedFormat {
  name: string;
  kind: MediaKind;
  extensions: string[];
  media_type: string;
}

/** Drives the file picker's `accept` attribute, so it cannot drift from the
 *  validation the backend actually performs. */
export interface FormatsResponse {
  formats: SupportedFormat[];
  max_image_upload_bytes: number;
  max_video_upload_bytes: number;
}

/** Replay position of an opened video source. Position only — no frame pixels. */
export interface PlaybackState {
  media_id: string;
  status: string;
  is_open: boolean;
  current_frame_index: number;
  position_s: number;
  frames_read: number;
  exhausted: boolean;
  fps: number | null;
  frame_count: number | null;
  duration_s: number | null;
  last_read_succeeded: boolean | null;
}

// ---------------------------------------------------------------------------
// Observation sites (/api/sites)
// ---------------------------------------------------------------------------

/**
 * Where the junction behind an observation area physically is.
 *
 * Deliberately separate from `MediaAsset`: an asset describes a file, a site
 * describes a place, and one junction outlives many uploads of it. A site is
 * persistent server-side — it survives a restart and outlives the footage that
 * prompted it — so the dashboard must never treat it as upload-scoped state.
 */
export interface ObservationSite {
  /** The area this location belongs to, e.g. `observation-area:<media_id>`. */
  observation_area_id: string;
  location: GeoPoint;
  /** Camera bearing: 0 = true north, clockwise. Null means unsurveyed, not north. */
  bearing_degrees: number | null;
  /** Radius within which the true position lies. Null means unreported, not good. */
  accuracy_m: number | null;
  label: string | null;
  provenance: Provenance;
  recorded_at: string;
  updated_at: string;
}

/** A surveyed location being recorded. Latitude and longitude are required. */
export interface SiteRequest {
  observation_area_id: string;
  latitude: number;
  longitude: number;
  bearing_degrees?: number;
  accuracy_m?: number;
  label?: string;
  recorded_by?: string;
  source?: ValueSource;
  source_notes?: string;
}

export interface SiteResponse {
  site: ObservationSite;
  /** True when this recording corrected a location already on file (HTTP 200
   *  rather than 201), so a caller who meant to create one learns it moved one. */
  replaced_existing: boolean;
}

export interface SiteListResponse {
  sites: ObservationSite[];
  count: number;
  /** Why the collection is what it is. An empty list means no camera has been
   *  located yet — a true statement about the records, not a failure. */
  note: string;
}

/** Optional location attached to an upload. Both coordinates or neither. */
export interface UploadLocation {
  latitude: number;
  longitude: number;
  bearing_degrees?: number;
  accuracy_m?: number;
  site_label?: string;
}

// ---------------------------------------------------------------------------
// Site-to-junction candidate bindings (/api/sites/{id}/candidate, .../binding)
// ---------------------------------------------------------------------------

/**
 * Outcome of measuring a site against the loaded network.
 *
 * Every value is a complete answer. "No network loaded" and "no such site" are
 * not among them: those are refusals, carrying their own error codes, because
 * neither is a statement about where the junction is.
 */
export type CandidateStatus =
  | "candidate_found"
  | "out_of_range"
  | "no_positioned_junction";

export interface JunctionCandidate {
  junction_id: string;
  junction_name: string | null;
  distance_m: number;
  /** True when the distance is inside the site's own accuracy radius, so the
   *  coordinate cannot tell this junction from anything else in that circle. */
  within_site_accuracy: boolean;
}

/**
 * A proposed binding. Never a binding.
 *
 * `requires_human_confirmation` is always true and the backend refuses to
 * construct one where it is not. The UI must render this as a question put to
 * the operator, never as a fact about which road the camera watches.
 */
export interface SiteBindingCandidate {
  observation_area_id: string;
  network_id: string;
  network_version: string;
  status: CandidateStatus;
  nearest: JunctionCandidate | null;
  runners_up: JunctionCandidate[];
  max_distance_m: number;
  site_accuracy_m: number | null;
  positioned_junction_count: number;
  /** Junctions with no surveyed position. Not measured, so not candidates —
   *  named rather than dropped, because the nearest of those that could be
   *  tested is not necessarily the nearest junction. */
  unpositioned_junction_ids: string[];
  /** False when the ranking is decided by the position's own noise rather than
   *  by geography. */
  is_discriminating: boolean;
  requires_human_confirmation: boolean;
  provenance: Provenance;
  warnings: string[];
  note: string;
}

export type BindingConfirmationMethod =
  | "accepted_nearest"
  /** The confirmed segment lies along the camera's surveyed bearing. Records
   *  that the geometry agreed, never that it decided: the opposite direction of
   *  the same road shares that line. */
  | "aligned_with_bearing"
  | "chose_different"
  | "surveyed";

/** Which way a directed segment runs relative to the junction it touches. */
export type SegmentDirection = "incoming" | "outgoing";

export interface BindingConfirmation {
  confirmed_by: string;
  confirmed_at: string;
  method: BindingConfirmationMethod;
  /** The distance the operator was shown. Retained so the decision can be
   *  reviewed against its evidence even after the site is re-surveyed. */
  candidate_distance_m: number | null;
  site_accuracy_m: number | null;
  was_nearest_candidate: boolean | null;
  notes: string | null;
}

export interface ConfirmedSiteBinding {
  observation_area_id: string;
  junction_id: string;
  network_id: string;
  network_version: string | null;
  /** Where the site was when this was confirmed. */
  site_location: GeoPoint;
  confirmation: BindingConfirmation;
  provenance: Provenance;

  /** The directed segment an operator later confirmed, if one has been.
   *  Null means the segment has not been decided yet — never that the camera
   *  watches no road. */
  segment_id: string | null;
  segment_direction: SegmentDirection | null;
  /** Only an incoming segment has an approach, and only when the operator
   *  named one. Never filled in automatically. */
  approach_id: string | null;
  /** The second human act, kept apart from the junction's so both survive. */
  segment_confirmation: BindingConfirmation | null;
}

/** One directed segment at a junction, offered for selection. */
export interface SegmentCandidate {
  segment_id: string;
  segment_name: string | null;
  direction: SegmentDirection;
  from_junction_id: string;
  to_junction_id: string;
  /** The junction at the far end — what an operator actually recognises. */
  other_junction_id: string;
  length_m: number | null;
  approach_id: string | null;
  /** The bearing the survey recorded for that approach, distinct from the one
   *  computed from geometry below so the two can visibly disagree. */
  approach_bearing_degrees: number | null;
  /** Compass bearing from the junction toward the segment's far end. */
  bearing_from_junction_degrees: number | null;
  /** Angle between the camera's facing and this segment's line. A small value
   *  means the camera points along this road. It does NOT mean the camera
   *  watches this direction of travel: the opposite segment shares the line. */
  camera_bearing_offset_degrees: number | null;
}

/**
 * The directed segments an operator may choose between at one junction.
 *
 * Ordered by segment id, never by bearing: ordering by bearing would put one
 * segment at the top of the list, which is selecting by another name.
 */
export interface SegmentCandidateSet {
  observation_area_id: string;
  junction_id: string;
  junction_name: string | null;
  network_id: string;
  network_version: string;
  candidates: SegmentCandidate[];
  site_bearing_degrees: number | null;
  /** False when the site has no surveyed bearing or the junction no position,
   *  in which case the operator is choosing on local knowledge alone. */
  bearing_evidence_available: boolean;
  /** Structurally always null. This type cannot carry a selection. */
  selected_segment_id: null;
  requires_human_confirmation: boolean;
  provenance: Provenance;
  warnings: string[];
  note: string;
}

export interface SegmentCandidatesResponse {
  site: ObservationSite;
  /** The junction binding being refined. */
  confirmed: ConfirmedSiteBinding;
  segments: SegmentCandidateSet;
  note: string;
}

export interface ConfirmSegmentRequest {
  junction_id: string;
  segment_id: string;
  approach_id?: string;
  /** Must be true. Bearing narrows the choice and never makes it. */
  confirm: boolean;
  confirmed_by?: string;
  /** Required when the choice runs against the geometry. */
  notes?: string;
}

export interface CandidateResponse {
  site: ObservationSite;
  candidate: SiteBindingCandidate;
  /** The binding already in force, when there is one. Confirming again would
   *  replace a decision rather than make a first one. */
  confirmed: ConfirmedSiteBinding | null;
  note: string;
}

export interface ConfirmBindingRequest {
  junction_id: string;
  /** Must be true. The backend refuses anything else — a binding is never
   *  created by accident, and never by proximity alone. */
  confirm: boolean;
  confirmed_by?: string;
  notes?: string;
}

/**
 * The routing layer's own binding type.
 *
 * A confirmed site binding names `junction_id` and leaves `segment_id` null,
 * which by existing mapping semantics prices and prunes nothing: knowing which
 * intersection a camera watches is not knowing which directed segment its queue
 * belongs to.
 */
export interface ObservationBinding {
  observation_area_id: string;
  segment_id: string | null;
  junction_id: string | null;
  approach_id: string | null;
  coverage_note: string | null;
  /** Present when an operator confirmed a proposal; null when the binding came
   *  from a hand-authored survey file. */
  confirmation: BindingConfirmation | null;
}

export interface ConfirmedBindingResponse {
  confirmed: ConfirmedSiteBinding;
  /** The same decision in the routing layer's own type: a junction binding,
   *  which prices and prunes nothing. */
  binding: ObservationBinding;
  replaced_existing: boolean;
  note: string;
}

export interface ConfirmedBindingListResponse {
  bindings: ConfirmedSiteBinding[];
  count: number;
  network_id: string | null;
  stale_version_area_ids: string[];
  /** How many bindings have been refined to a directed segment. */
  segment_bound_count: number;
  segment_bound_observation_area_ids: string[];
  /** Always false. Stated by the backend rather than inferred from an
   *  unchanged route, so the UI can say so plainly. */
  affects_route_cost: boolean;
  route_cost_note: string;
  note: string;
}

// ---------------------------------------------------------------------------
// Evidence attribution (/api/state/network)
// ---------------------------------------------------------------------------

/**
 * Whether one observation area's evidence reached a road, and if not why.
 *
 * Four values rather than mapped/unmapped, because they call for four different
 * fixes. Only `attributed` carries a `segment_id`; the rest leave it null and
 * explain themselves.
 */
export type AttributionStatus =
  | "attributed"
  | "junction_only"
  | "excluded"
  | "unmapped";

export interface EvidenceAttribution {
  /** Also the site key: camera, area and site are one identifier. */
  observation_area_id: string;
  status: AttributionStatus;
  /** Null for everything except `attributed`. Never filled from proximity or
   *  bearing — only an explicit mapping attributes evidence to a road. */
  segment_id: string | null;
  junction_id: string | null;
  approach_id: string | null;
  segment_direction: string | null;
  site_location: GeoPoint | null;
  network_id: string;
  network_version: string;
  source: BindingSource | null;
  resolution: BindingResolution | null;
  exclusion_reason: ExclusionReason | null;
  provenance: Provenance | null;
  congestion_level: CongestionLevel | null;
  freshness: string | null;
  vehicle_count: number | null;
  /** Why this area is where it is. Always present, including on success. */
  reason: string;
}

export interface EvidenceAttributionReport {
  network_id: string;
  network_version: string;
  effective_mapping_hash: string;
  effective_mapping_version: string;
  attributions: EvidenceAttribution[];
  attributed_area_ids: string[];
  /** Areas whose evidence reached no segment — includes junction-only and
   *  excluded, all three being evidence that reaches no directed road. */
  unmapped_observation_area_ids: string[];
  excluded_area_ids: string[];
  junction_only_area_ids: string[];
  attributed_segment_ids: string[];
  /** Always false: attribution is reported, the optimizer is unchanged. */
  used_by_route_costing: boolean;
  note: string;
}

/**
 * The fused traffic-state snapshot, plus where its evidence lands.
 *
 * Every pre-existing field stays at the top level, so a client written before
 * attribution reads exactly what it always read.
 */
export interface NetworkTrafficStateResponse {
  state_id: string;
  as_of: string;
  assembled_at: string;
  policy: Record<string, unknown>;
  elements: IntersectionStateElement[];
  schema_version: string;
  attribution: EvidenceAttributionReport | null;
  /** Why attribution is absent, when it is. */
  attribution_unavailable_reason: string | null;
}

// ---------------------------------------------------------------------------
// Effective observation mapping (/api/sites/effective-mapping)
// ---------------------------------------------------------------------------

/** Where a loaded mapping file came from, and what exactly was read. */
export interface MappingSource {
  relative_path: string;
  sha256: string;
  size_bytes: number;
  loaded_at: string;
  mapping_id: string;
  version: string;
  binding_count: number;
}

/** A versioned set of observation-to-network bindings. */
export interface ObservationMapping {
  mapping_id: string;
  version: string;
  network_id: string;
  network_version: string | null;
  provenance: Provenance;
  bindings: ObservationBinding[];
}

/** Which survey a binding in the effective mapping came from. */
export type BindingSource = "mapping_file" | "confirmed_binding";

/** Why a binding is the effective one for its observation area. */
export type BindingResolution =
  | "file_only"
  | "confirmed_only"
  | "sources_agree"
  | "confirmed_binding_precedence";

/** Why a confirmed binding is not in force. Never silent. */
export type ExclusionReason =
  | "different_network"
  | "stale_network_version"
  | "unknown_network_element";

export interface EffectiveBinding {
  observation_area_id: string;
  binding: ObservationBinding;
  source: BindingSource;
  resolution: BindingResolution;
  /** The whole confirmation record, when one is behind this binding. */
  confirmed: ConfirmedSiteBinding | null;
  superseded_segment_id: string | null;
  superseded_junction_id: string | null;
}

export interface MappingOverride {
  observation_area_id: string;
  file_binding: string | null;
  confirmed_binding: string | null;
  effective_binding: string | null;
  resolution: BindingResolution;
  note: string;
}

export interface ExcludedConfirmedBinding {
  observation_area_id: string;
  junction_id: string | null;
  segment_id: string | null;
  network_id: string | null;
  network_version: string | null;
  reason: ExclusionReason;
  detail: string;
}

/**
 * The mapping in force, composed from the file and the confirmations.
 *
 * Derived on request and never stored, so it cannot drift from its inputs.
 * `used_by_route_costing` is always false in this step: the optimizer still
 * reads the mapping file alone, and the UI must not imply otherwise.
 */
export interface EffectiveObservationMapping {
  mapping_id: string | null;
  mapping_version: string | null;
  mapping_sha256: string | null;
  mapping_source: MappingSource | null;
  confirmed_bindings_hash: string;
  confirmed_binding_count: number;
  network_id: string;
  network_version: string;
  mapping: ObservationMapping;
  bindings: EffectiveBinding[];
  overrides: MappingOverride[];
  excluded: ExcludedConfirmedBinding[];
  effective_mapping_hash: string;
  effective_mapping_version: string;
  provenance: Provenance;
  used_by_route_costing: boolean;
  warnings: string[];
  note: string;
}

export interface EffectiveMappingResponse {
  effective: EffectiveObservationMapping;
  effective_mapping_in_use: boolean;
  /** What the optimizer actually pins into a route result today. */
  optimizer_mapping_id: string | null;
  optimizer_mapping_version: string | null;
  note: string;
}

// ---------------------------------------------------------------------------
// AI Traffic Advisor (/api/advisor/*)
// ---------------------------------------------------------------------------

/**
 * How the advisory text was produced.
 *
 * The UI must render these differently. Showing deterministic template text
 * under an "AI generated" label would misattribute it, which is the same class
 * of dishonesty the fact table exists to prevent, one level up.
 */
export type AdvisoryMode = "llm" | "deterministic_fallback";

/** Result of scanning generated prose against the fact table. */
export interface NarrativeValidation {
  unsupported_numbers: number[];
  unknown_fact_ids: string[];
  passed: boolean;
}

/** One citable quantity the advisory was allowed to mention. */
export interface AdvisorFact {
  fact_id: string;
  label: string;
  value: number | string | boolean;
  unit: string | null;
  provenance: Provenance;
  display_variants: string[];
}

export interface AdvisorFactTable {
  facts: AdvisorFact[];
  entity_tokens: string[];
}

/** A value the advisor wanted and did not have. Never rendered as zero. */
export interface UnavailableFact {
  fact_id: string;
  label: string;
  reason: string;
}

/** The facts the advisory was built from, returned so the panel can show them. */
export interface TrafficAdvisorFacts {
  observation_area_id: string;
  site_label: string | null;
  latitude: number | null;
  longitude: number | null;
  confirmed_junction_id: string | null;
  confirmed_segment_id: string | null;
  confirmed_segment_direction: string | null;
  traffic_state: CongestionLevel | null;
  congestion_index: number | null;
  vehicle_count: number | null;
  stationary_vehicle_count: number | null;
  vehicle_class_counts: Record<string, number>;
  pedestrian_count: number | null;
  cyclist_count: number | null;
  image_occupancy_ratio: number | null;
  spatial_summary: string | null;
  quality_grade: string | null;
  state_as_of: string | null;
  incident_summaries: string[];
  emergency_summaries: string[];
  has_route_comparison: boolean;
  route_baseline_s: number | null;
  route_alternative_s: number | null;
  route_improvement_pct: number | null;
  route_selected_id: string | null;
  has_simulation: boolean;
  simulation_run_id: string | null;
  simulation_travel_time_delta_s: number | null;
  simulation_improvement_pct: number | null;
  fact_table: AdvisorFactTable;
  unavailable: UnavailableFact[];
}

export interface TrafficAdvisory {
  mode: AdvisoryMode;
  observation_area_id: string;
  summary: string;
  traffic_assessment: string;
  suggestions: string[];
  safety_considerations: string[];
  route_advice: string;
  limitations: string[];
  evidence_ids: string[];
  /** Null in fallback mode: naming a model that was never called would
   *  misattribute the text. */
  model_id: string | null;
  generated_at: string;
  validation: NarrativeValidation | null;
  /** Always true, enforced by the backend schema. */
  advisory_only: boolean;
}

export interface AdvisorResponse {
  advisory: TrafficAdvisory;
  facts: TrafficAdvisorFacts;
}

export interface AdvisorStatusResponse {
  provider_configured: boolean;
  mode: AdvisoryMode;
  /** The configured model name. Never the key. */
  model_id: string | null;
  advisory_only: boolean;
  note: string;
}

export interface AdvisorRequest {
  media_id?: string;
  job_id?: string;
  /** Both, to include route evidence from the existing optimizer. */
  origin_junction_id?: string;
  destination_junction_id?: string;
  simulation_run_id?: string;
}

// ---------------------------------------------------------------------------
// Safety catalog (/api/safety/catalog)
// ---------------------------------------------------------------------------

export type ConstraintTier = "hard" | "soft";

export interface SafetyRule {
  rule_id: string;
  name: string;
  tier: ConstraintTier;
  scope: string;
  description: string;
  rationale: string;
  /** Null until Phase 11, when penalty weights are calibrated. */
  default_penalty_equivalent_s: number | null;
}

export interface SafetyCatalogResponse {
  version: string;
  rules: SafetyRule[];
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export interface ApiErrorBody {
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface ApiErrorResponse {
  error: ApiErrorBody;
}

// ---------------------------------------------------------------------------
// Computer vision (/api/vision/*)
// ---------------------------------------------------------------------------

/** Road-user classes the detector reports. Closed set, mirroring the backend. */
export type RoadUserClass =
  | "car"
  | "motorcycle"
  | "bus"
  | "truck"
  | "bicycle"
  | "person"
  | "light_commercial"
  | "auto_rickshaw"
  | "ambulance"
  | "fire_engine"
  | "police"
  | "other"
  | "unknown";

export interface ModelIdentity {
  name: string;
  backend: string;
  file_name: string | null;
  /** Checksum of the weights actually loaded — what makes a result reproducible. */
  sha256: string | null;
  size_bytes: number | null;
  task: string | null;
  class_count: number | null;
  source: string | null;
}

export interface InferenceMetadata {
  backend: string;
  model: ModelIdentity;
  device: string;
  device_kind: string;
  /** Why this device was chosen, so a surprising CPU fallback is explained. */
  device_reason: string;
  image_size: number;
  confidence_threshold: number;
  iou_threshold: number;
  tracker: string | null;
}

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

/**
 * Whether a detection may be counted as a road user.
 *
 * `provisional` and `rejected` boxes are still reported and still drawn — a
 * filtered-away detection is invisible to an operator trying to work out why a
 * count is wrong. What changes is that the measurement layer does not count
 * them.
 */
export type DetectionQuality = "confirmed" | "provisional" | "rejected";

export type QualityFlag =
  | "below_class_confidence"
  | "duplicate"
  | "implausible_area"
  | "implausible_aspect"
  | "edge_truncated_low_confidence";

export interface DetectionOut {
  road_user_class: RoadUserClass;
  confidence: number;
  bbox: DetectionBox;
  /** Present only for video, where tracking establishes identity across frames. */
  track_id: number | null;
  frame_index: number | null;
  timestamp: string | null;
  source_id: string | null;
  quality: DetectionQuality;
  quality_flags: QualityFlag[];
  /** Why it is not confirmed. Present whenever the verdict is not `confirmed`. */
  quality_reason: string | null;
  area_fraction: number | null;
  aspect_ratio: number | null;
  touches_frame_edge: boolean;
  duplicate_of_index: number | null;
}

/** The rules deciding which detections may be counted. */
export interface QualityPolicy {
  enabled: boolean;
  confirm_confidence: Partial<Record<RoadUserClass, number>>;
  default_confidence: number;
  max_area_fraction: number;
  min_aspect_ratio: number;
  max_aspect_ratio: number;
  duplicate_iou: number;
  duplicate_containment: number;
  edge_margin_px: number;
  edge_confidence_margin: number;
}

/**
 * Per-class counts.
 *
 * A tally of boxes or tracks — explicitly not a traffic measurement. Turning
 * these into flow needs counting lines and a time window, which is a later
 * phase's job.
 */
export interface ClassTally {
  by_class: Partial<Record<RoadUserClass, number>>;
  total: number;
  /** Subtotal over classes that are actually vehicles. */
  vehicles: number;
}

export interface ImageAnalysis {
  analysis_id: string;
  media_id: string;
  source_id: string;
  analysed_at: string;
  inference: InferenceMetadata;
  detections: DetectionOut[];
  /** Every box reported, whatever its quality. Not a traffic figure. */
  detection_tally: ClassTally;
  /** The subset a measurement may count. This is what analytics consumes. */
  confirmed_tally: ClassTally;
  quality_policy: QualityPolicy | null;
  frame_width: number | null;
  frame_height: number | null;
  overlay_available: boolean;
}

export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface VideoAnalysisSummary {
  job_id: string;
  media_id: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  frames_analysed: number;
  frames_total_estimate: number | null;
  frame_stride: number;
  progress: number | null;
  detection_tally: ClassTally;
  /** The subset that passed the quality checks, across every frame. */
  confirmed_detection_tally: ClassTally;
  track_tally: ClassTally;
  unique_track_count: number;
  suspected_id_switches: number;
  inference: InferenceMetadata | null;
  /** The real failure message. Never replaced by empty results. */
  error: string | null;
}

export interface FrameResult {
  frame_index: number;
  media_time_s: number | null;
  timestamp: string;
  inference_ms: number | null;
  detections: DetectionOut[];
  active_track_ids: number[];
}

export interface FrameListResponse {
  job_id: string;
  frames: FrameResult[];
  offset: number;
  limit: number;
  total: number;
}

export interface TrackSummary {
  track_id: number;
  road_user_class: RoadUserClass;
  first_frame_index: number;
  last_frame_index: number;
  first_seen_at: string;
  last_seen_at: string;
  frames_tracked: number;
  is_confirmed: boolean;
  id_switch_suspected: boolean;
  mean_confidence: number | null;
}

export interface TrackListResponse {
  job_id: string;
  tracks: TrackSummary[];
  count: number;
}

export interface BackendStatus {
  name: string;
  description: string;
  available: boolean;
  missing_modules: string[];
  default_model: string;
  is_selected: boolean;
}

export interface DetectorStatus {
  detection_available: boolean;
  selected_backend: string | null;
  configured_model: string;
  configured_device: string;
  /** True only once weights are genuinely in memory. */
  model_loaded: boolean;
  load_error: string | null;
  backends: BackendStatus[];
  confidence_threshold: number;
  iou_threshold: number;
  image_size: number;
  /** Ceiling on detections per frame. A binding ceiling truncates a busy frame. */
  max_detections: number;
  quality_policy: QualityPolicy;
  frame_stride: number;
  max_frames: number;
  tracker: string;
}

// ---------------------------------------------------------------------------
// Traffic analytics (/api/analytics/*)
// ---------------------------------------------------------------------------

/**
 * Why a metric is or is not present.
 *
 * Granular on purpose: "needs a calibrated camera" and "clip too short" call for
 * different operator responses, and a bare `null` loses that.
 */
export type MetricAvailability =
  | "available"
  | "unavailable_no_calibration"
  | "unavailable_no_geometry"
  | "unavailable_insufficient_data"
  | "unavailable_not_applicable";

/**
 * A derived quantity, or an explicit statement that it could not be computed.
 *
 * `value` is null whenever `availability` is not `"available"`, and `reason`
 * then explains why. A metric is never rendered as `0` to mean "unknown".
 */
export interface MetricValue<T = number> {
  availability: MetricAvailability;
  value: T | null;
  unit: string | null;
  reason: string | null;
  method: string | null;
  confidence: number | null;
  /** True for pixel/frame-relative values, which must never be shown as physical. */
  is_image_space: boolean;
}

export type DirectionLabel =
  | "right"
  | "down_right"
  | "down"
  | "down_left"
  | "left"
  | "up_left"
  | "up"
  | "up_right";

export interface AnalyticsProvenance {
  media_id: string;
  source_id: string;
  frame_index: number | null;
  frame_range: [number, number] | null;
  timestamp: string | null;
  time_range: [string, string] | null;
  model: ModelIdentity;
  tracker: string | null;
  method: string;
  method_version: string;
  analytics_schema_version: string;
  computed_at: string;
}

export type QualityGrade = "good" | "degraded" | "poor";

export interface MeasurementQuality {
  grade: QualityGrade;
  frames_analysed: number;
  mean_detection_confidence: number | null;
  low_confidence_fraction: number | null;
  tracks_total: number;
  tracks_confirmed: number;
  suspected_id_switches: number;
  calibration_available: boolean;
  road_geometry_available: boolean;
  sampling_interval_s: number | null;
  warnings: string[];
}

export interface ImagePoint2D {
  x: number;
  y: number;
  x_norm: number;
  y_norm: number;
}

export interface ObjectPosition {
  road_user_class: string;
  confidence: number;
  track_id: number | null;
  centre: ImagePoint2D;
  /** Bottom-centre of the box: where the object meets the road. */
  ground_contact: ImagePoint2D;
  box_width_px: number;
  box_height_px: number;
  box_area_fraction: number;
}

export interface TrajectoryMetrics {
  track_id: number;
  road_user_class: string;
  frames_tracked: number;
  first_frame_index: number;
  last_frame_index: number;
  duration_s: MetricValue;
  total_displacement_px: MetricValue;
  net_displacement_px: MetricValue;
  mean_step_displacement_px: MetricValue;
  image_speed_px_per_s: MetricValue;
  image_speed_diagonals_per_s: MetricValue;
  direction_deg: MetricValue;
  direction_label: MetricValue<DirectionLabel>;
  is_stationary: boolean;
  stationary_frames: number;
  id_switch_suspected: boolean;
  physical_speed_mps: MetricValue;
  physical_speed_kph: MetricValue;
}

export interface QueueGroup {
  group_id: number;
  member_track_ids: number[];
  member_count: number;
  road_user_classes: Record<string, number>;
  extent_px: MetricValue;
  longest_stationary_duration_s: MetricValue;
  length_m: MetricValue;
  mean_confidence: number | null;
}

export interface FlowBucket {
  window_start_s: number;
  window_end_s: number;
  arrivals: number;
  active: number;
}

export interface SpatialDistribution {
  centroid: ImagePoint2D | null;
  spread_x_norm: MetricValue;
  spread_y_norm: MetricValue;
  grid_rows: number;
  grid_cols: number;
  grid_counts: number[][];
  /** Fraction of frame area covered. Image-space — not traffic density. */
  image_occupancy_ratio: MetricValue;
}

export interface AnalyticsClassTally {
  by_class: Record<string, number>;
  total: number;
  vehicles: number;
  pedestrians: number;
  cyclists: number;
}

export type ObservationMode = "snapshot" | "sequence";

export interface TrafficAnalytics {
  analytics_id: string;
  mode: ObservationMode;
  /** The camera's field of view — not a surveyed junction. */
  observation_area_id: string;
  provenance: AnalyticsProvenance;
  quality: MeasurementQuality;
  frame_width: number;
  frame_height: number;
  /** Detection boxes. In a sequence, one object counts once per frame. */
  detection_tally: AnalyticsClassTally;
  /** Distinct tracked objects. */
  tracked_object_tally: AnalyticsClassTally;
  active_tracked_objects: MetricValue<number>;
  object_positions: ObjectPosition[];
  spatial_distribution: SpatialDistribution;
  trajectories: TrajectoryMetrics[];
  mean_image_speed_diagonals_per_s: MetricValue;
  direction_histogram: Partial<Record<DirectionLabel, number>>;
  mean_trajectory_duration_s: MetricValue;
  flow_series: FlowBucket[];
  arrival_rate_per_minute: MetricValue;
  stationary_vehicle_count: MetricValue<number>;
  queue_groups: QueueGroup[];
  total_queue_length_m: MetricValue;
  traffic_density_veh_per_km_per_lane: MetricValue;
  mean_speed_kph: MetricValue;
}

export interface AnalyticsConfigResponse {
  method: string;
  method_version: string;
  schema_version: string;
  stationary_speed_frac: number;
  min_stationary_frames: number;
  queue_proximity_factor: number;
  flow_window_s: number;
  min_track_frames: number;
  calibration_configured: boolean;
  road_geometry_configured: boolean;
}

// ---------------------------------------------------------------------------
// Route optimization (/api/routes/*)
// ---------------------------------------------------------------------------

/**
 * How a leg's traversal time was arrived at.
 *
 * Rendered rather than collapsed: a time derived from a surveyed free-flow
 * speed describes an empty road, and presenting it identically to a measured
 * one would hide the difference from the person deciding on it.
 */
export type TimeBasis =
  | "measured_emergency"
  | "measured_nominal"
  | "derived_from_free_flow"
  | "unavailable";

/** How far a segment's traffic evidence could be trusted, if any existed. */
export type EvidenceStatus =
  | "mapped_classified"
  | "mapped_stale"
  | "mapped_unclassified"
  | "unmapped";

export interface RouteCostBreakdown {
  base_travel_time_s: number;
  congestion_penalty_s: number;
  incident_penalty_s: number;
  safety_penalty_s: number;
  accessibility_penalty_s: number;
  turn_penalty_s: number;
  uncertainty_penalty_s: number;
  distance_penalty_s: number;
  weights_version: string;
}

export interface RouteLeg {
  segment_id: string;
  from_junction_id: string;
  to_junction_id: string;
  entry_movement_id: string | null;
  time_basis: TimeBasis;
}

export interface RouteCandidate {
  route_id: string;
  legs: RouteLeg[];
  cost: RouteCostBreakdown;
  warnings: string[];
  emergency_assumptions: string[];
}

export interface SegmentEvidence {
  segment_id: string;
  status: EvidenceStatus;
  congestion_level: string;
  observation_area_id: string | null;
  media_id: string | null;
  congestion_penalty_s: number;
  uncertainty_penalty_s: number;
  incident_penalty_s: number;
  reason: string | null;
}

export interface RankedRoute {
  rank: number;
  candidate: RouteCandidate;
  /** Null whenever any leg lacked a time basis. Never a partial sum. */
  estimated_travel_time_s: number | null;
  travel_time_available: boolean;
  unpriced_segment_ids: string[];
  total_distance_m: number;
  junction_sequence: string[];
  segment_sequence: string[];
  evidence: SegmentEvidence[];
  mapped_segment_count: number;
}

export interface RouteReason {
  code: string;
  detail: string;
  value: number | null;
  comparison_value: number | null;
}

export interface RouteRecommendation {
  selected_route_id: string;
  reasons: RouteReason[];
  alternatives_considered: number;
  cost_margin_s: number | null;
}

export interface RouteRejection {
  reason: string;
  message: string;
  segment_ids: string[];
  junction_ids: string[];
  movement_ids: string[];
  incident_ids: string[];
}

export interface RouteOptimizationResult {
  request: {
    request_id: string;
    mode: string;
    origin_junction_id: string;
    destination_junction_id: string | null;
    destination_facility_id: string | null;
    emergency_id: string | null;
    vehicle_type: string | null;
  };
  status: "routes_found" | "no_admissible_route";
  routes: RankedRoute[];
  recommendation: RouteRecommendation | null;
  rejections: RouteRejection[];
  pruned_segment_ids: string[];
  unmapped_observation_area_ids: string[];
  warnings: string[];
  algorithm: string;
  algorithm_version: string;
  network_id: string;
  network_version: string;
  mapping_id: string | null;
}

// ---------------------------------------------------------------------------
// Traffic simulation (/api/simulation/*) — phase 8
// ---------------------------------------------------------------------------

/**
 * Where one simulation input came from.
 *
 * Rendered beside every figure it produced. A queue built on a surveyed
 * capacity and one built on a number an operator typed are not the same claim,
 * and the panel must not let them look alike.
 */
export type InputBasis = "surveyed" | "observed" | "stated" | "unavailable";

export type SimulationStatus = "completed" | "unavailable";
export type ArmStatus = "simulated" | "unavailable";
export type LinkStatus = "modelled" | "not_modelled";
export type ArmRole = "baseline" | "recommended" | "evaluated";
export type BetterArm = "baseline" | "alternative" | "no_difference" | "unknown";
export type BaselineStrategy = "shortest_distance" | "runner_up" | "explicit";

export interface SimulationRejection {
  reason: string;
  message: string;
  segment_ids: string[];
  junction_ids: string[];
  route_ids: string[];
  incident_ids: string[];
}

export interface SimulationParameters {
  horizon_s: number;
  step_s: number;
  step_count: number;
  departure_offset_s: number;
  emergency_queue_bypass_fraction: number;
  use_observed_demand: boolean;
  baseline_strategy: BaselineStrategy;
  parameters_version: string;
  model_name: string;
  model_version: string;
  is_stochastic: boolean;
}

export interface SimulationLinkInputs {
  segment_id: string;
  free_flow_traversal_s: number | null;
  capacity_veh_per_hour: number | null;
  capacity_basis: InputBasis;
  demand_veh_per_hour: number | null;
  demand_basis: InputBasis;
  initial_queue_veh: number | null;
  initial_queue_basis: InputBasis;
  observation_area_id: string | null;
  is_blocked: boolean;
  demand_reason: string | null;
  capacity_reason: string | null;
}

export interface SimulationScenario {
  scenario_id: string;
  network_id: string;
  network_version: string;
  parameters: SimulationParameters;
  links: SimulationLinkInputs[];
  traffic_state_id: string | null;
  mapping_id: string | null;
  active_incident_ids: string[];
  input_fingerprint: string;
}

export interface SimulationLeg {
  segment_id: string;
  from_junction_id: string;
  to_junction_id: string;
  entry_time_s: number;
  exit_time_s: number;
  free_flow_traversal_s: number;
  time_basis: TimeBasis;
  /** Null where the link was not modelled. Never zero for "unknown". */
  queue_delay_s: number | null;
  queue_modelled: boolean;
  queue_at_arrival_veh: number | null;
  reason: string | null;
}

export interface RouteSimulationResult {
  role: ArmRole;
  route_id: string;
  label: string;
  source: string;
  mode: string;
  status: ArmStatus;
  segment_ids: string[];
  junction_sequence: string[];
  legs: SimulationLeg[];
  travel_time_s: Measurement | null;
  free_flow_travel_time_s: Measurement | null;
  delay_s: Measurement | null;
  completion_time_s: Measurement | null;
  mean_speed_kph: Measurement | null;
  total_distance_m: Measurement | null;
  completed_within_horizon: boolean;
  unmodelled_segment_ids: string[];
  modelled_leg_count: number;
  rejections: SimulationRejection[];
  warnings: string[];
}

export interface SimulationLinkResult {
  segment_id: string;
  status: LinkStatus;
  reason: string | null;
  demand_veh_per_hour: number | null;
  demand_basis: InputBasis;
  capacity_veh_per_hour: number | null;
  capacity_basis: InputBasis;
  saturation_ratio: Measurement | null;
  peak_queue_veh: Measurement | null;
  final_queue_veh: Measurement | null;
  mean_queue_veh: Measurement | null;
  throughput_veh: Measurement | null;
  throughput_veh_per_hour: Measurement | null;
  total_delay_veh_s: Measurement | null;
  mean_delay_per_vehicle_s: Measurement | null;
  queue_series_veh: number[];
}

export interface NetworkSimulationSummary {
  modelled_link_count: number;
  unmodelled_link_count: number;
  modelled_throughput_veh: Measurement | null;
  modelled_peak_queue_veh: Measurement | null;
  modelled_final_queue_veh: Measurement | null;
  modelled_total_delay_veh_s: Measurement | null;
  oversaturated_segment_ids: string[];
  note: string;
}

export interface SimulationComparison {
  baseline_route_id: string;
  alternative_route_id: string;
  comparable: boolean;
  reason: string | null;
  travel_time_delta_s: Measurement | null;
  delay_delta_s: Measurement | null;
  completion_time_delta_s: Measurement | null;
  distance_delta_m: Measurement | null;
  improvement_pct: number | null;
  better: BetterArm;
  rejections: SimulationRejection[];
}

export interface SimulationRun {
  run_id: string;
  status: SimulationStatus;
  scenario: SimulationScenario;
  routes: RouteSimulationResult[];
  links: SimulationLinkResult[];
  network: NetworkSimulationSummary | null;
  comparison: SimulationComparison | null;
  rejections: SimulationRejection[];
  warnings: string[];
  model_name: string;
  model_version: string;
  network_id: string;
  network_version: string;
  computed_at: string;
}

export interface SimulationConfigResponse {
  model_name: string;
  model_version: string;
  parameters_version: string;
  horizon_s: number;
  step_s: number;
  step_count: number;
  departure_offset_s: number;
  emergency_queue_bypass_fraction: number;
  use_observed_demand: boolean;
  default_baseline: BaselineStrategy;
  max_steps: number;
  max_stored_runs: number;
  is_stochastic: boolean;
  stored_runs: number;
  model_description: string;
  limitations: string[];
}

export interface MappingStatusResponse {
  loaded: boolean;
  source?: MappingSource | null;
  mapping?: ObservationMapping | null;
  bound_segment_ids: string[];
  bound_observation_area_ids: string[];
  interpreted_area_ids: string[];
  unmapped_observation_area_ids: string[];
  /** Human-confirmed site-to-junction bindings, reported beside the loaded
   *  mapping file rather than merged into it — a surveyed binding and an
   *  operator accepting a proximity proposal are not the same evidence. */
  confirmed_binding_count: number;
  confirmed_observation_area_ids: string[];
  confirmed_stale_version_area_ids: string[];
  note: string;
}

// ---------------------------------------------------------------------------
// Traffic state (/api/state/*) — phase 5 interpretation
// ---------------------------------------------------------------------------

export type CongestionLevel =
  | "free_flow"
  | "light"
  | "moderate"
  | "heavy"
  | "severe"
  | "gridlock"
  | "unknown";

export type FreshnessStatus = "fresh" | "stale" | "expired" | "unknown";
export type EvidenceBasis = "physical" | "image_space" | "insufficient";

/**
 * One signal behind a congestion verdict.
 *
 * Unavailable signals are returned too, carrying zero weight and a reason. The
 * panel renders them, because an assessment resting on two signals out of three
 * is a weaker claim than one resting on all three, and hiding the missing one
 * would erase that difference.
 */
export interface CongestionSignal {
  name: string;
  availability: string;
  raw_value: number | null;
  unit: string | null;
  score: number | null;
  weight: number;
  reason: string | null;
  is_image_space: boolean;
}

export interface CongestionAssessment {
  level: CongestionLevel;
  /** Null exactly when the level is `unknown`. Never a scored refusal. */
  index: number | null;
  basis: EvidenceBasis;
  signals: CongestionSignal[];
  confidence: number | null;
  /** Required when the level is `unknown`; explains the refusal. */
  reason: string | null;
  method: string;
  method_version: string;
}

export interface IntersectionTrafficState {
  state_id: string;
  observation_area_id: string;
  mode: string;
  as_of: string;
  assessed_at: string;
  congestion: CongestionAssessment;
  vehicle_count: number;
  stationary_vehicle_count: number | null;
  quality_grade: QualityGrade;
  schema_version: string;
}

export interface IntersectionStateElement {
  state: IntersectionTrafficState;
  freshness: FreshnessStatus;
  evidence_age_s: number;
}

export interface NetworkTrafficState {
  state_id: string;
  as_of: string;
  assembled_at: string;
  policy: {
    max_staleness_s: number;
    drop_after_s: number;
    fusion_strategy: string;
  };
  elements: IntersectionStateElement[];
  schema_version: string;
}

// ---------------------------------------------------------------------------
// Road network management (/api/network/*) — phase 6
// ---------------------------------------------------------------------------

export interface NetworkSource {
  relative_path: string;
  sha256: string;
  size_bytes: number;
  loaded_at: string;
  network_id: string;
  version: string;
  junction_count: number;
  segment_count: number;
  facility_count: number;
}

export interface NetworkStatusResponse {
  loaded: boolean;
  source: NetworkSource | null;
  /** Files that exist and end in `.json`, not files known to parse. */
  available_files: string[];
  networks_dir_exists: boolean;
  note: string;
}

export interface LoadNetworkResponse {
  source: NetworkSource;
  junction_count: number;
  segment_count: number;
  facility_count: number;
  cleared_incident_count: number;
  cleared_emergency_count: number;
  note: string;
}

export interface BlockedSegmentsResponse {
  network_id: string;
  network_version: string;
  blocked_segment_ids: string[];
  /** Passable but degraded: priced by a soft penalty, never pruned. */
  degraded_segment_ids: string[];
  active_incident_count: number;
  as_of: string;
}

// ---------------------------------------------------------------------------
// Incidents (/api/incidents) — phase 6
// ---------------------------------------------------------------------------

export type IncidentType =
  | "accident"
  | "fire"
  | "road_blockage"
  | "stalled_vehicle"
  | "flooding"
  | "construction"
  | "debris"
  | "protest_or_event"
  | "signal_failure"
  | "structural_failure"
  | "other";

export type IncidentSeverity = "minor" | "moderate" | "major" | "critical";

export type IncidentStatus =
  | "reported"
  | "confirmed"
  | "responding"
  | "contained"
  | "cleared"
  | "false_alarm";

export interface Incident {
  incident_id: string;
  incident_type: IncidentType;
  severity: IncidentSeverity;
  status: IncidentStatus;
  location: {
    segment_id: string | null;
    junction_id: string | null;
    description: string | null;
  };
  reported_at: string;
  cleared_at: string | null;
  blocks_segment_ids: string[];
  affected_segment_ids: string[];
  exclusion_zone: { excluded_segment_ids: string[]; reason: string } | null;
  description: string | null;
}

export interface IncidentListResponse {
  incidents: Incident[];
  count: number;
  active_count: number;
  network_id: string | null;
  note: string;
}

export interface IncidentResponse {
  incident: Incident;
  zone_resolution: { warnings: string[] } | null;
  /** Empty for a terminal incident, which is how the UI knows to stop offering moves. */
  permitted_next_statuses: IncidentStatus[];
}

// ---------------------------------------------------------------------------
// Emergency declaration and lifecycle (/api/emergency/*) — phase 6
// ---------------------------------------------------------------------------

export type EmergencyVehicleType =
  | "ambulance"
  | "fire_engine"
  | "police"
  | "rescue"
  | "other";

export type EmergencyPriority = "P1_immediate" | "P2_urgent" | "P3_routine";

export type EmergencyStatus =
  | "declared"
  | "en_route"
  | "arrived"
  | "returning"
  | "completed"
  | "cancelled";

export interface EmergencyResponseBody {
  emergency: EmergencyScenario;
  permitted_next_statuses: EmergencyStatus[];
  /** Only P1 warrants corridor pre-emption; the backend decides, not the panel. */
  warrants_corridor_preemption: boolean;
  elapsed_response_time_s: number | null;
}
