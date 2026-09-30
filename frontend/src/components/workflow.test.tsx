/**
 * Tests for the workflow surfaces connected to Phase 5–6 backend capabilities.
 *
 * The properties under test are the same ones the backend enforces, checked at
 * the edge where an operator actually reads them: a network that is not loaded
 * says so rather than showing an example, a blocked segment is visibly distinct
 * from a degraded one, an unclassified traffic state shows its reason instead of
 * a band, and every option offered comes from the API rather than from a list
 * written in the component.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { IncidentsPanel } from "./IncidentsPanel";
import { NetworkPanel } from "./NetworkPanel";
import { TrafficStateSection } from "./TrafficState";
import { WorkflowStage } from "./WorkflowStage";
import type {
  Incident,
  IntersectionTrafficState,
  RoadNetwork,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import {
  makeFacility,
  makeJunction,
  makeNetwork,
  makeSegment,
} from "../test/fixtures";

afterEach(() => {
  vi.restoreAllMocks();
});

const noop = (): void => undefined;

function success<T>(data: T): Resource<T> {
  return { state: "success", data };
}

const NETWORK: RoadNetwork = makeNetwork({
  network_id: "net-1",
  version: "1.0",
  junctions: [makeJunction("J1", { name: "North" }), makeJunction("J2")],
  segments: [
    makeSegment("SEG-12", "J1", "J2"),
    makeSegment("SEG-21", "J2", "J1"),
  ],
  facilities: [makeFacility("FAC-1", "St Jude Clinic", "J2")],
});

const LOADING: Resource<RoadNetwork> = { state: "loading" };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Routes fetches by URL, so a panel making several requests is served properly. */
function mockRoutes(table: Array<[string, unknown]>): void {
  vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = typeof input === "string" ? input : String(input);
    for (const [fragment, body] of table) {
      if (url.includes(fragment)) return Promise.resolve(json(body));
    }
    return Promise.resolve(json({ detail: "not mocked" }, 404));
  });
}

const NETWORK_STATUS = {
  loaded: true,
  source: null,
  available_files: ["example-grid.json", "other.json"],
  networks_dir_exists: true,
  note: "Network 'net-1' version 1.0 loaded.",
};

const BLOCKED = {
  network_id: "net-1",
  network_version: "1.0",
  blocked_segment_ids: ["SEG-12"],
  degraded_segment_ids: ["SEG-21"],
  active_incident_count: 1,
  as_of: "2026-08-28T00:00:00Z",
};

function makeIncident(overrides: Partial<Incident> = {}): Incident {
  return {
    incident_id: "INC-1",
    incident_type: "road_blockage",
    severity: "critical",
    status: "reported",
    location: { segment_id: "SEG-12", junction_id: null, description: null },
    reported_at: "2026-08-28T00:00:00Z",
    cleared_at: null,
    blocks_segment_ids: ["SEG-12"],
    affected_segment_ids: [],
    exclusion_zone: null,
    description: null,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
describe("WorkflowStage", () => {
  it("marks a future stage apart and never renders its controls", () => {
    render(
      <WorkflowStage
        step={5}
        title="Simulation"
        summary="Baseline versus recommended."
        future
        futureNote="Not implemented. Traffic simulation is phase 8."
      >
        <button type="button">Run simulation</button>
      </WorkflowStage>,
    );

    expect(screen.getByText(/not yet implemented/i)).toBeInTheDocument();
    expect(screen.getByText(/phase 8/i)).toBeInTheDocument();
    // The child is deliberately not rendered: a control for a capability that
    // does not exist is worse than no control.
    expect(
      screen.queryByRole("button", { name: /run simulation/i }),
    ).not.toBeInTheDocument();
  });

  it("renders an implemented stage's children", () => {
    render(
      <WorkflowStage step={1} title="Input" summary="Ingest media.">
        <p>panel body</p>
      </WorkflowStage>,
    );
    expect(screen.getByText("panel body")).toBeInTheDocument();
    expect(screen.queryByText(/not yet implemented/i)).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe("NetworkPanel", () => {
  it("says no network is loaded rather than showing an example", async () => {
    mockRoutes([["/api/network/status", { ...NETWORK_STATUS, loaded: false }]]);
    render(<NetworkPanel network={LOADING} onNetworkChanged={noop} />);

    expect(
      await screen.findByText(/No road network is loaded/i),
    ).toBeInTheDocument();
    expect(screen.queryByText("SEG-12")).not.toBeInTheDocument();
  });

  it("offers only the files the backend listed", async () => {
    mockRoutes([
      ["/api/network/status", NETWORK_STATUS],
      ["/api/network/blocked", BLOCKED],
    ]);
    render(<NetworkPanel network={success(NETWORK)} onNetworkChanged={noop} />);

    expect(
      await screen.findByRole("option", { name: "example-grid.json" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "other.json" })).toBeInTheDocument();
  });

  it("renders junctions, segments and facilities from the API", async () => {
    mockRoutes([
      ["/api/network/status", NETWORK_STATUS],
      ["/api/network/blocked", BLOCKED],
    ]);
    render(<NetworkPanel network={success(NETWORK)} onNetworkChanged={noop} />);

    expect(await screen.findByText("J1")).toBeInTheDocument();
    expect(screen.getByText("North")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /segments/i }));
    expect(screen.getByText("SEG-12")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /facilities/i }));
    expect(screen.getByText("St Jude Clinic")).toBeInTheDocument();
  });

  it("marks blocked and degraded segments distinctly", async () => {
    mockRoutes([
      ["/api/network/status", NETWORK_STATUS],
      ["/api/network/blocked", BLOCKED],
    ]);
    render(<NetworkPanel network={success(NETWORK)} onNetworkChanged={noop} />);

    await userEvent.click(await screen.findByRole("tab", { name: /segments/i }));
    // Blocked is pruned before search; degraded is priced and still usable.
    expect(screen.getByText("blocked")).toBeInTheDocument();
    expect(screen.getByText("degraded")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe("IncidentsPanel", () => {
  const emptyList = {
    incidents: [],
    count: 0,
    active_count: 0,
    network_id: "net-1",
    note: "Incidents are held in memory for this process.",
  };

  it("refuses to offer declaration without a network", async () => {
    mockRoutes([["/api/incidents", emptyList]]);
    render(<IncidentsPanel network={LOADING} onIncidentsChanged={noop} />);

    expect(
      await screen.findByText(/No road network is loaded/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /declare incident/i }),
    ).not.toBeInTheDocument();
  });

  it("explains an empty list with the backend's own note", async () => {
    mockRoutes([["/api/incidents", emptyList]]);
    render(<IncidentsPanel network={success(NETWORK)} onIncidentsChanged={noop} />);

    expect(
      await screen.findByText(/No incident has been declared/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/held in memory for this process/i)).toBeInTheDocument();
  });

  it("offers only segments that exist on the loaded network", async () => {
    mockRoutes([["/api/incidents", emptyList]]);
    render(<IncidentsPanel network={success(NETWORK)} onIncidentsChanged={noop} />);

    await userEvent.click(
      await screen.findByRole("button", { name: /declare incident/i }),
    );
    expect(
      screen.getByRole("option", { name: /SEG-12 \(J1 → J2\)/ }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: /SEG-99/ }),
    ).not.toBeInTheDocument();
  });

  it("keeps blocked and degraded visibly distinct on a declared incident", async () => {
    mockRoutes([
      [
        "/api/incidents/INC-1",
        { incident: makeIncident(), zone_resolution: null, permitted_next_statuses: ["confirmed"] },
      ],
      [
        "/api/incidents",
        {
          ...emptyList,
          incidents: [
            makeIncident({
              blocks_segment_ids: ["SEG-12"],
              affected_segment_ids: ["SEG-21"],
            }),
          ],
          count: 1,
          active_count: 1,
        },
      ],
    ]);
    render(<IncidentsPanel network={success(NETWORK)} onIncidentsChanged={noop} />);

    expect(await screen.findByTestId("incident-INC-1")).toBeInTheDocument();
    expect(screen.getByText("blocked")).toBeInTheDocument();
    expect(screen.getByText("degraded")).toBeInTheDocument();
  });

  it("offers only the lifecycle moves the backend permits", async () => {
    mockRoutes([
      [
        "/api/incidents/INC-1",
        {
          incident: makeIncident(),
          zone_resolution: null,
          permitted_next_statuses: ["confirmed", "false_alarm"],
        },
      ],
      [
        "/api/incidents",
        { ...emptyList, incidents: [makeIncident()], count: 1, active_count: 1 },
      ],
    ]);
    render(<IncidentsPanel network={success(NETWORK)} onIncidentsChanged={noop} />);

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /mark confirmed/i }),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByRole("button", { name: /mark false alarm/i }),
    ).toBeInTheDocument();
    // Not permitted by the backend, so not offered here.
    expect(
      screen.queryByRole("button", { name: /mark cleared/i }),
    ).not.toBeInTheDocument();
  });

  it("says a terminal incident is terminal instead of offering moves", async () => {
    mockRoutes([
      [
        "/api/incidents/INC-1",
        {
          incident: makeIncident({ status: "cleared" }),
          zone_resolution: null,
          permitted_next_statuses: [],
        },
      ],
      [
        "/api/incidents",
        {
          ...emptyList,
          incidents: [makeIncident({ status: "cleared" })],
          count: 1,
          active_count: 0,
        },
      ],
    ]);
    render(<IncidentsPanel network={success(NETWORK)} onIncidentsChanged={noop} />);

    expect(await screen.findByText(/is terminal/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
describe("TrafficStateSection", () => {
  function makeState(
    overrides: Partial<IntersectionTrafficState> = {},
  ): IntersectionTrafficState {
    return {
      state_id: "state-1",
      observation_area_id: "observation-area:media-1",
      mode: "sequence",
      as_of: "2026-08-28T00:00:00Z",
      assessed_at: "2026-08-28T00:00:00Z",
      congestion: {
        level: "moderate",
        index: 0.4512,
        basis: "image_space",
        signals: [
          {
            name: "image_occupancy",
            availability: "available",
            raw_value: 0.18,
            unit: "fraction",
            score: 0.51,
            weight: 0.4,
            reason: null,
            is_image_space: true,
          },
          {
            name: "image_speed",
            availability: "unavailable_insufficient_data",
            raw_value: null,
            unit: null,
            score: null,
            weight: 0,
            reason: "No trajectory had enough points to measure speed.",
            is_image_space: true,
          },
        ],
        confidence: 0.42,
        reason: null,
        method: "weighted-threshold-congestion",
        method_version: "1.0",
      },
      vehicle_count: 12,
      stationary_vehicle_count: null,
      quality_grade: "good",
      schema_version: "1.0",
      ...overrides,
    };
  }

  it("prompts for analysis before anything is interpreted", () => {
    render(<TrafficStateSection mediaId="media-1" />);
    expect(
      screen.getByText(/No traffic state has been interpreted/i),
    ).toBeInTheDocument();
  });

  it("renders the band, index and the signals behind it", async () => {
    mockRoutes([["/api/state/images/media-1", makeState()]]);
    render(<TrafficStateSection mediaId="media-1" />);

    await userEvent.click(
      screen.getByRole("button", { name: /interpret traffic state/i }),
    );

    expect(await screen.findByText("Moderate")).toBeInTheDocument();
    expect(screen.getByText(/index 0\.4512/)).toBeInTheDocument();
    expect(screen.getByText("Signals used")).toBeInTheDocument();
  });

  it("lists unavailable signals with their reason rather than hiding them", async () => {
    mockRoutes([["/api/state/images/media-1", makeState()]]);
    render(<TrafficStateSection mediaId="media-1" />);

    await userEvent.click(
      screen.getByRole("button", { name: /interpret traffic state/i }),
    );

    expect(
      await screen.findByText(/Signals not available \(1\)/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/No trajectory had enough points/i),
    ).toBeInTheDocument();
  });

  it("shows an unclassified verdict as a refusal with its reason, not a band", async () => {
    mockRoutes([
      [
        "/api/state/images/media-1",
        makeState({
          congestion: {
            ...makeState().congestion,
            level: "unknown",
            index: null,
            basis: "insufficient",
            confidence: null,
            reason: "Measurement quality is poor, so no congestion level is inferred.",
          },
        }),
      ],
    ]);
    render(<TrafficStateSection mediaId="media-1" />);

    await userEvent.click(
      screen.getByRole("button", { name: /interpret traffic state/i }),
    );

    expect(await screen.findByText("Unclassified")).toBeInTheDocument();
    expect(screen.getByText("no index")).toBeInTheDocument();
    expect(
      screen.getByText(/Measurement quality is poor/i),
    ).toBeInTheDocument();
  });

  it("renders an unmeasured stationary count as unavailable, not zero", async () => {
    mockRoutes([["/api/state/images/media-1", makeState()]]);
    render(<TrafficStateSection mediaId="media-1" />);

    await userEvent.click(
      screen.getByRole("button", { name: /interpret traffic state/i }),
    );

    const absent = await screen.findByText("unavailable");
    expect(absent).toHaveClass("tstate__absent");
    expect(absent.previousElementSibling).toHaveTextContent("Stationary");
  });
});
