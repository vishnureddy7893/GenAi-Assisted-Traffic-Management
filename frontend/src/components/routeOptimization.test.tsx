/**
 * Route Optimization panel tests.
 *
 * The panel must render exactly what the backend sent and nothing more. The
 * assertions here are mostly about absence: an unavailable travel time must
 * read as unavailable rather than as a zero, a refusal must not produce a
 * route, and there must be no control that approves or executes anything —
 * that belongs to the Decision Engine, after the step where a person can say
 * no.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RouteOptimizationPanel } from "./RouteOptimizationPanel";
import type {
  RankedRoute,
  RoadNetwork,
  RouteOptimizationResult,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { makeNetwork, makeJunction, makeSegment, makeFacility } from "../test/fixtures";

afterEach(() => {
  vi.restoreAllMocks();
});

function success<T>(data: T): Resource<T> {
  return { state: "success", data };
}

const NETWORK: RoadNetwork = makeNetwork({
  network_id: "net-1",
  junctions: [makeJunction("J1"), makeJunction("J2"), makeJunction("J3")],
  segments: [makeSegment("SEG-12", "J1", "J2"), makeSegment("SEG-23", "J2", "J3")],
  facilities: [makeFacility("FAC-1", "St Jude Clinic", "J3")],
});

function makeRoute(overrides: Partial<RankedRoute> = {}): RankedRoute {
  return {
    rank: 1,
    candidate: {
      route_id: "route-1",
      legs: [],
      cost: {
        base_travel_time_s: 88,
        congestion_penalty_s: 40,
        incident_penalty_s: 0,
        safety_penalty_s: 0,
        accessibility_penalty_s: 0,
        turn_penalty_s: 0,
        uncertainty_penalty_s: 30,
        distance_penalty_s: 0,
        weights_version: "0.1.0",
      },
      warnings: [],
      emergency_assumptions: [],
    },
    estimated_travel_time_s: 88,
    travel_time_available: true,
    unpriced_segment_ids: [],
    total_distance_m: 1100,
    junction_sequence: ["J1", "J2", "J3"],
    segment_sequence: ["SEG-12", "SEG-23"],
    evidence: [],
    mapped_segment_count: 1,
    ...overrides,
  };
}

function makeResult(
  overrides: Partial<RouteOptimizationResult> = {},
): RouteOptimizationResult {
  return {
    request: {
      request_id: "req-1",
      mode: "normal",
      origin_junction_id: "J1",
      destination_junction_id: "J3",
      destination_facility_id: null,
      emergency_id: null,
      vehicle_type: null,
    },
    status: "routes_found",
    routes: [makeRoute()],
    recommendation: {
      selected_route_id: "route-1",
      reasons: [
        {
          code: "lower_total_cost",
          detail: "Lowest risk-adjusted arrival time.",
          value: 158,
          comparison_value: 240,
        },
      ],
      alternatives_considered: 1,
      cost_margin_s: 82,
    },
    rejections: [],
    pruned_segment_ids: [],
    unmapped_observation_area_ids: [],
    warnings: [],
    algorithm: "dijkstra+yen-k-shortest",
    algorithm_version: "1.0",
    network_id: "net-1",
    network_version: "1.0",
    mapping_id: null,
    ...overrides,
  };
}

/** Empty mapping: no camera is bound, which is the default operator state. */
const NO_MAPPING = {
  loaded: false,
  source: null,
  mapping: null,
  bound_segment_ids: [],
  bound_observation_area_ids: [],
  interpreted_area_ids: [],
  unmapped_observation_area_ids: [],
  note: "No observation mapping is loaded.",
};

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

async function optimizeWith(result: RouteOptimizationResult): Promise<void> {
  // Routed per URL and constructed per call: a Response body can only be read
  // once, and the panel now asks for mapping coverage as well as a route.
  vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = typeof input === "string" ? input : String(input);
    return Promise.resolve(
      url.includes("/api/routes/mapping") ? json(NO_MAPPING) : json(result),
    );
  });
  render(<RouteOptimizationPanel network={success(NETWORK)} />);

  await userEvent.selectOptions(screen.getByLabelText(/origin/i), "J1");
  await userEvent.selectOptions(screen.getByLabelText(/destination/i), "J3");
  await userEvent.click(screen.getByRole("button", { name: /optimize route/i }));
}

describe("RouteOptimizationPanel", () => {
  it("says so plainly when no network is loaded", () => {
    render(
      <RouteOptimizationPanel network={{ state: "loading" } as Resource<RoadNetwork>} />,
    );
    expect(screen.getByText(/No road network is loaded/i)).toBeInTheDocument();
  });

  it("offers junctions and facilities from the API, never a hard-coded list", () => {
    render(<RouteOptimizationPanel network={success(NETWORK)} />);
    expect(screen.getAllByRole("option", { name: /J1/ }).length).toBeGreaterThan(0);
    expect(screen.getByRole("option", { name: /St Jude Clinic/ })).toBeInTheDocument();
  });

  it("renders the recommended route and its reasons", async () => {
    await optimizeWith(makeResult());

    expect(await screen.findByText("Recommended route")).toBeInTheDocument();
    expect(screen.getByText("J1 → J2 → J3")).toBeInTheDocument();
    expect(screen.getByText(/Lower total cost/)).toBeInTheDocument();
    expect(screen.getByText(/158\.0 vs 240\.0/)).toBeInTheDocument();
  });

  it("renders an unavailable travel time as unavailable, not as zero", async () => {
    await optimizeWith(
      makeResult({
        routes: [
          makeRoute({
            estimated_travel_time_s: null,
            travel_time_available: false,
            unpriced_segment_ids: ["SEG-23"],
          }),
        ],
      }),
    );

    // The travel-time field itself must read as absent. A genuinely-zero cost
    // elsewhere on the card (an incident penalty of 0 s) is a real measurement
    // and is left alone -- the distinction under test is missing versus zero.
    const absent = await screen.findByText("unavailable");
    expect(absent).toBeInTheDocument();
    expect(absent).toHaveClass("route__absent");
    expect(absent.previousElementSibling).toHaveTextContent(
      "Estimated travel time",
    );
  });

  it("shows a refusal with its causes and no route", async () => {
    await optimizeWith(
      makeResult({
        status: "no_admissible_route",
        routes: [],
        recommendation: null,
        rejections: [
          {
            reason: "disconnected_by_pruning",
            message: "Every route crosses a segment removed by a hard constraint.",
            segment_ids: ["SEG-12"],
            junction_ids: [],
            movement_ids: [],
            incident_ids: ["INC-1"],
          },
        ],
      }),
    );

    expect(
      await screen.findByText(/Nothing is recommended in its place/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/disconnected by pruning/i)).toBeInTheDocument();
    expect(screen.getByText(/Segments: SEG-12/)).toBeInTheDocument();
    expect(screen.queryByText("Recommended route")).not.toBeInTheDocument();
  });

  it("names the segments hard constraints removed before search", async () => {
    await optimizeWith(makeResult({ pruned_segment_ids: ["SEG-99"] }));
    expect(
      await screen.findByText(/Excluded by hard constraints before search: SEG-99/),
    ).toBeInTheDocument();
  });

  it("surfaces the emergency fallback assumption verbatim", async () => {
    await optimizeWith(
      makeResult({
        routes: [
          makeRoute({
            candidate: {
              ...makeRoute().candidate,
              emergency_assumptions: [
                "No emergency-specific travel time exists, so no advantage is modelled.",
              ],
            },
          }),
        ],
      }),
    );
    expect(
      await screen.findByText(/no advantage is modelled/i),
    ).toBeInTheDocument();
  });

  it("offers no approve or execute control", async () => {
    await optimizeWith(makeResult());
    await screen.findByText("Recommended route");

    for (const label of [/approve/i, /execute/i, /dispatch/i, /apply/i]) {
      expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
    }
  });

  it("shows the algorithm and network version behind the result", async () => {
    await optimizeWith(makeResult());
    expect(
      await screen.findByText(/dijkstra\+yen-k-shortest v1\.0 · network net-1 v1\.0/),
    ).toBeInTheDocument();
  });
});
