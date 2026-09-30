/**
 * Test fixtures.
 *
 * Builders that produce API-shaped responses. They live in the test tree, never
 * in `src/components` or `src/api`, so no fabricated domain value can reach the
 * running application.
 *
 * Identifiers used by callers are deliberately unusual ("XQ-77", "ZZ9-NORTH")
 * so a hard-coded topology could not accidentally satisfy an assertion.
 */

import type {
  EmergencyScenario,
  EmergencyScenariosResponse,
  Facility,
  GeoPoint,
  HealthResponse,
  Junction,
  Measurement,
  Provenance,
  RoadNetwork,
  RoadSegment,
  SourceCatalogResponse,
  SourceKindProfile,
} from "../api/types";

const T0 = "2024-03-14T08:30:00Z";

export const PROVENANCE: Provenance = {
  source: "static",
  producer: "test.fixture",
  producer_version: "1.0",
  observed_at: T0,
  computed_at: T0,
  confidence: null,
  input_refs: [],
  notes: null,
};

export function measure(value: number, unit = "m"): Measurement {
  return { value, unit, provenance: PROVENANCE };
}

export function makeJunction(
  id: string,
  overrides: Partial<Junction> = {},
): Junction {
  return {
    junction_id: id,
    name: null,
    kind: "unsignalized",
    position: null,
    approaches: [],
    movements: [],
    ...overrides,
  };
}

export function makeSegment(
  id: string,
  from: string,
  to: string,
  overrides: Partial<RoadSegment> = {},
): RoadSegment {
  return {
    segment_id: id,
    from_junction_id: from,
    to_junction_id: to,
    name: null,
    road_class: "unknown",
    length_m: measure(400),
    lane_count: 2,
    free_flow_speed_kph: null,
    speed_limit_kph: null,
    capacity_veh_per_hour: null,
    physical_restrictions: {
      max_height_m: null,
      max_weight_t: null,
      max_width_m: null,
      excluded_vehicle_types: [],
      notes: null,
    },
    geometry: [],
    ...overrides,
  };
}

export function makeFacility(
  id: string,
  name: string,
  accessJunctionId: string,
  overrides: Partial<Facility> = {},
): Facility {
  return {
    facility_id: id,
    name,
    facility_type: "hospital",
    access_junction_id: accessJunctionId,
    position: null,
    accepts_emergency_types: [],
    notes: null,
    ...overrides,
  };
}

export function makeNetwork(overrides: Partial<RoadNetwork> = {}): RoadNetwork {
  return {
    network_id: "net-under-test",
    name: null,
    version: "9.9",
    driving_side: "left",
    provenance: PROVENANCE,
    junctions: [],
    segments: [],
    facilities: [],
    ...overrides,
  };
}

export function makeGeoPoint(latitude: number, longitude: number): GeoPoint {
  return { latitude, longitude };
}

export function makeScenario(
  id: string,
  overrides: Partial<EmergencyScenario> = {},
): EmergencyScenario {
  return {
    emergency_id: id,
    vehicle: {
      vehicle_id: `VEH-${id}`,
      vehicle_type: "ambulance",
      callsign: null,
      height_m: null,
      weight_t: null,
      requires_wide_turns: null,
    },
    priority: "P3_routine",
    status: "declared",
    declared_at: T0,
    updated_at: T0,
    current_position: {
      segment_id: null,
      offset_m: null,
      junction_id: null,
      coordinate: null,
      heading_degrees: null,
      observed_at: null,
    },
    destination: {
      facility_id: null,
      incident_id: null,
      position: null,
      facility_type: null,
      label: null,
    },
    assigned_route_id: null,
    notes: null,
    ...overrides,
  };
}

export function makeScenariosResponse(
  scenarios: EmergencyScenario[] = [],
  overrides: Partial<EmergencyScenariosResponse> = {},
): EmergencyScenariosResponse {
  return {
    scenarios,
    count: scenarios.length,
    active_count: 0,
    declaration_supported: false,
    declaration_phase: 6,
    note: "No emergency response has been declared. It is empty because nothing was declared.",
    ...overrides,
  };
}

export function makeProfile(
  kind: string,
  overrides: Partial<SourceKindProfile> = {},
): SourceKindProfile {
  return {
    kind,
    mode: "A",
    display_name: kind,
    description: `Description for ${kind}`,
    payload_kind: "tabular_record",
    seekable: true,
    bounded: true,
    replayable: true,
    realtime: false,
    requires_backpressure: false,
    implemented: false,
    available_from_phase: 2,
    required_dependencies: [],
    ...overrides,
  };
}

export function makeCatalog(
  overrides: Partial<SourceCatalogResponse> = {},
): SourceCatalogResponse {
  return {
    modes: [],
    implemented_kinds: [],
    unavailable_kinds: [],
    any_source_available: false,
    note: "",
    ...overrides,
  };
}

export function makeHealth(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return {
    status: "ok",
    pipeline_ready: false,
    pipeline_blockers: [],
    app_version: "0.0.0-test",
    schema_version: "0.0.0-test",
    implemented_phase: 4,
    implemented_phase_name: "Phase under test",
    environment: "test-env",
    server_time: T0,
    uptime_s: 1.5,
    components: [],
    ...overrides,
  };
}
