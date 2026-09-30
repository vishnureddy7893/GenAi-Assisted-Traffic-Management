/**
 * Traffic simulation panel tests.
 *
 * The assertions are mostly about honesty rather than layout:
 *
 * - a modelled figure is visibly tagged as simulated, so it cannot be read as a
 *   measurement of a real road;
 * - an unavailable value reads as "unavailable", never as a dash or a zero;
 * - an incomparable pair shows no improvement percentage at all;
 * - a blocked route shows the refusal and its cause instead of a travel time;
 * - there is no approve, dispatch or apply control anywhere on the panel.
 *
 * Every number rendered comes from the fixture response. The panel does no
 * arithmetic, so these tests are checking presentation of backend facts rather
 * than a second implementation of the model.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SimulationPanel, type SimulationContext } from "./SimulationPanel";
import type {
  Measurement,
  RouteSimulationResult,
  SimulationComparison,
  SimulationLinkResult,
  SimulationRun,
  RoadNetwork,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { makeNetwork, makeJunction, makeSegment } from "../test/fixtures";

afterEach(() => {
  vi.restoreAllMocks();
});

function success<T>(data: T): Resource<T> {
  return { state: "success", data };
}

const NETWORK: RoadNetwork = makeNetwork({
  network_id: "net-1",
  junctions: [makeJunction("XQ-77"), makeJunction("XQ-78"), makeJunction("XQ-79")],
  segments: [
    makeSegment("SEG-A", "XQ-77", "XQ-78"),
    makeSegment("SEG-B", "XQ-78", "XQ-79"),
    makeSegment("SEG-C", "XQ-77", "XQ-79"),
  ],
});

const CONTEXT: SimulationContext = {
  originJunctionId: "XQ-77",
  destinationJunctionId: "XQ-79",
  destinationFacilityId: null,
  emergencyId: null,
  mode: "normal",
  hasRoutes: true,
};

function simulated(value: number, unit = "s"): Measurement {
  return {
    value,
    unit,
    provenance: {
      source: "simulated",
      producer: "simulation.point_queue",
      producer_version: "1.0",
      observed_at: "2024-03-14T08:30:00Z",
      computed_at: "2024-03-14T08:30:00Z",
      confidence: null,
      input_refs: [],
      notes: null,
    },
  };
}

function derived(value: number, unit = "s"): Measurement {
  return {
    ...simulated(value, unit),
    provenance: { ...simulated(value, unit).provenance, source: "derived" },
  };
}

function makeArm(
  overrides: Partial<RouteSimulationResult> = {},
): RouteSimulationResult {
  return {
    role: "baseline",
    route_id: "route-baseline",
    label: "Shortest route by distance",
    source: "shortest_distance_search",
    mode: "normal",
    status: "simulated",
    segment_ids: ["SEG-C"],
    junction_sequence: ["XQ-77", "XQ-79"],
    legs: [
      {
        segment_id: "SEG-C",
        from_junction_id: "XQ-77",
        to_junction_id: "XQ-79",
        entry_time_s: 0,
        exit_time_s: 120,
        free_flow_traversal_s: 90,
        time_basis: "derived_from_free_flow",
        queue_delay_s: 30,
        queue_modelled: true,
        queue_at_arrival_veh: 7.5,
        reason: null,
      },
    ],
    travel_time_s: simulated(120),
    free_flow_travel_time_s: derived(90),
    delay_s: simulated(30),
    completion_time_s: simulated(120),
    mean_speed_kph: simulated(24, "km/h"),
    total_distance_m: derived(800, "m"),
    completed_within_horizon: true,
    unmodelled_segment_ids: [],
    modelled_leg_count: 1,
    rejections: [],
    warnings: [],
    ...overrides,
  };
}

function makeLink(
  overrides: Partial<SimulationLinkResult> = {},
): SimulationLinkResult {
  return {
    segment_id: "SEG-C",
    status: "modelled",
    reason: null,
    demand_veh_per_hour: 1400,
    demand_basis: "stated",
    capacity_veh_per_hour: 900,
    capacity_basis: "stated",
    saturation_ratio: derived(1.5556, "ratio"),
    peak_queue_veh: simulated(42.5, "veh"),
    final_queue_veh: simulated(42.5, "veh"),
    mean_queue_veh: simulated(20, "veh"),
    throughput_veh: simulated(210, "veh"),
    throughput_veh_per_hour: simulated(840, "veh/h"),
    total_delay_veh_s: simulated(9000),
    mean_delay_per_vehicle_s: simulated(42.8),
    queue_series_veh: [0, 1, 2],
    ...overrides,
  };
}

function makeComparison(
  overrides: Partial<SimulationComparison> = {},
): SimulationComparison {
  return {
    baseline_route_id: "route-baseline",
    alternative_route_id: "route-recommended",
    comparable: true,
    reason: null,
    travel_time_delta_s: simulated(28),
    delay_delta_s: simulated(28),
    completion_time_delta_s: simulated(28),
    distance_delta_m: derived(-40, "m"),
    improvement_pct: 23.3,
    better: "alternative",
    rejections: [],
    ...overrides,
  };
}

function makeRun(overrides: Partial<SimulationRun> = {}): SimulationRun {
  return {
    run_id: "sim-abc123def456",
    status: "completed",
    scenario: {
      scenario_id: "scn-abc123def456",
      network_id: "net-1",
      network_version: "1.0",
      parameters: {
        horizon_s: 900,
        step_s: 5,
        step_count: 180,
        departure_offset_s: 0,
        emergency_queue_bypass_fraction: 0,
        use_observed_demand: true,
        baseline_strategy: "shortest_distance",
        parameters_version: "0.1.0",
        model_name: "deterministic-point-queue",
        model_version: "1.0",
        is_stochastic: false,
      },
      links: [],
      traffic_state_id: null,
      mapping_id: null,
      active_incident_ids: [],
      input_fingerprint: "abc123def456789",
    },
    routes: [
      makeArm(),
      makeArm({
        role: "recommended",
        route_id: "route-recommended",
        label: "Recommended route",
        source: "optimizer_selected",
        segment_ids: ["SEG-A", "SEG-B"],
        junction_sequence: ["XQ-77", "XQ-78", "XQ-79"],
        travel_time_s: simulated(92),
        delay_s: simulated(0),
      }),
    ],
    links: [makeLink()],
    network: {
      modelled_link_count: 1,
      unmodelled_link_count: 2,
      modelled_throughput_veh: simulated(210, "veh"),
      modelled_peak_queue_veh: simulated(42.5, "veh"),
      modelled_final_queue_veh: simulated(42.5, "veh"),
      modelled_total_delay_veh_s: simulated(9000),
      oversaturated_segment_ids: ["SEG-C"],
      note: "Totals cover the 1 link(s) that had every input.",
    },
    comparison: makeComparison(),
    rejections: [],
    warnings: ["2 of 3 segment(s) could not be modelled."],
    model_name: "deterministic-point-queue",
    model_version: "1.0",
    network_id: "net-1",
    network_version: "1.0",
    computed_at: "2024-03-14T08:30:00Z",
    ...overrides,
  };
}

const CONFIG = {
  model_name: "deterministic-point-queue",
  model_version: "1.0",
  parameters_version: "0.1.0",
  horizon_s: 900,
  step_s: 5,
  step_count: 180,
  departure_offset_s: 0,
  emergency_queue_bypass_fraction: 0,
  use_observed_demand: true,
  default_baseline: "shortest_distance",
  max_steps: 20000,
  max_stored_runs: 50,
  is_stochastic: false,
  stored_runs: 0,
  model_description: "Deterministic point-queue link model.",
  limitations: ["Queues are vertical."],
};

/** Stubs `fetch`: GET returns the config, POST returns the given run. */
function stubApi(run: SimulationRun | null, status = 200): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_url: string, init?: RequestInit) => {
      const isPost = init?.method === "POST";
      return {
        ok: status < 400,
        status,
        json: async () => (isPost ? run : CONFIG),
      } as Response;
    }),
  );
}

async function runSimulation(run: SimulationRun | null): Promise<void> {
  stubApi(run);
  render(<SimulationPanel network={success(NETWORK)} context={CONTEXT} />);
  await userEvent.click(
    screen.getByRole("button", { name: /simulate baseline vs recommended/i }),
  );
}

describe("SimulationPanel", () => {
  it("asks for a route before offering to simulate one", () => {
    render(<SimulationPanel network={success(NETWORK)} context={null} />);
    expect(screen.getByText(/no route has been optimized yet/i)).toBeVisible();
    expect(
      screen.queryByRole("button", { name: /simulate/i }),
    ).not.toBeInTheDocument();
  });

  it("says so when no network is loaded", () => {
    render(
      <SimulationPanel
        network={{ state: "loading" }}
        context={CONTEXT}
      />,
    );
    expect(screen.getByText(/no road network is loaded/i)).toBeVisible();
  });

  it("does not offer to simulate when the optimizer found no route", () => {
    render(
      <SimulationPanel
        network={success(NETWORK)}
        context={{ ...CONTEXT, hasRoutes: false }}
      />,
    );
    expect(screen.getByText(/no route has been optimized yet/i)).toBeVisible();
  });

  it("states that every figure is a modelled outcome", async () => {
    await runSimulation(makeRun());
    expect(
      screen.getByText(/modelled outcome of a hypothetical/i),
    ).toBeVisible();
  });

  it("renders both arms with their travel times", async () => {
    await runSimulation(makeRun());
    expect(await screen.findByText("Shortest route by distance")).toBeVisible();
    expect(screen.getByText("Recommended route")).toBeVisible();
    // Both arms report a travel time and a completion time, so each figure
    // legitimately appears more than once.
    expect(screen.getAllByText(/120\.0 s/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/92\.0 s/).length).toBeGreaterThan(0);
  });

  it("tags a simulated figure as simulated and the reference as derived", async () => {
    await runSimulation(makeRun());
    await screen.findByText("Shortest route by distance");
    expect(screen.getAllByText("simulated").length).toBeGreaterThan(0);
    expect(screen.getAllByText("derived").length).toBeGreaterThan(0);
  });

  it("shows the improvement the backend computed", async () => {
    await runSimulation(makeRun());
    expect(
      await screen.findByText(/the recommended route arrives sooner/i),
    ).toBeVisible();
    expect(screen.getByText(/23\.3 %/)).toBeVisible();
    expect(screen.getAllByText(/28\.0 s/).length).toBeGreaterThan(0);
  });

  it("shows no improvement figure when the pair is not comparable", async () => {
    await runSimulation(
      makeRun({
        comparison: makeComparison({
          comparable: false,
          reason: "The baseline produced no travel time.",
          travel_time_delta_s: null,
          delay_delta_s: null,
          completion_time_delta_s: null,
          distance_delta_m: null,
          improvement_pct: null,
          better: "unknown",
        }),
      }),
    );
    expect(
      await screen.findByText(/no improvement figure exists/i),
    ).toBeVisible();
    expect(screen.getByText(/produced no travel time/i)).toBeVisible();
    expect(screen.queryByText(/23\.3 %/)).not.toBeInTheDocument();
    expect(screen.queryByText(/%$/)).not.toBeInTheDocument();
  });

  it("renders a blocked route as a refusal with its cause, not a time", async () => {
    await runSimulation(
      makeRun({
        routes: [
          makeArm({
            status: "unavailable",
            travel_time_s: null,
            free_flow_travel_time_s: null,
            delay_s: null,
            completion_time_s: null,
            mean_speed_kph: null,
            rejections: [
              {
                reason: "segment_blocked",
                message: "Route crosses a segment an active incident blocks.",
                segment_ids: ["SEG-C"],
                junction_ids: [],
                route_ids: ["route-baseline"],
                incident_ids: ["INC-1"],
              },
            ],
          }),
        ],
        comparison: null,
      }),
    );
    expect(await screen.findByText(/no travel time for this route/i)).toBeVisible();
    expect(screen.getByText("segment blocked")).toBeVisible();
    expect(screen.getByText(/active incident blocks/i)).toBeVisible();
    // The figures block is not rendered at all: there is no travel time, no
    // delay and no mean speed to show, so none of those labels appear.
    expect(screen.queryByText("Travel time")).not.toBeInTheDocument();
    expect(screen.queryByText("Delay")).not.toBeInTheDocument();
    expect(screen.queryByText("Mean speed")).not.toBeInTheDocument();
  });

  it("renders an unmodelled link as unavailable rather than as zero", async () => {
    await runSimulation(
      makeRun({
        links: [
          makeLink({
            segment_id: "SEG-A",
            status: "not_modelled",
            reason: "This segment has no surveyed capacity.",
            demand_veh_per_hour: null,
            demand_basis: "unavailable",
            capacity_veh_per_hour: null,
            capacity_basis: "unavailable",
            saturation_ratio: null,
            peak_queue_veh: null,
            final_queue_veh: null,
            mean_queue_veh: null,
            throughput_veh: null,
            throughput_veh_per_hour: null,
            total_delay_veh_s: null,
            mean_delay_per_vehicle_s: null,
            queue_series_veh: [],
          }),
        ],
      }),
    );
    const tables = await screen.findAllByRole("table");
    const linkTable = tables[tables.length - 1] as HTMLElement;
    const row = within(linkTable).getByText("SEG-A").closest("tr") as HTMLElement;
    expect(within(row).getAllByText(/unavailable/i)).toHaveLength(2);
    expect(within(row).getAllByText(/not modelled/i)).toHaveLength(2);
    expect(within(row).queryByText("0")).not.toBeInTheDocument();
  });

  it("shows the queue and throughput a modelled link produced", async () => {
    await runSimulation(makeRun());
    await screen.findByText("Shortest route by distance");
    // The peak queue appears both in the network summary and in the link row.
    expect(screen.getAllByText(/42\.5 veh/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/210 veh/).length).toBeGreaterThan(0);
    expect(screen.getByText(/1 modelled · 2 not modelled/)).toBeVisible();
  });

  it("reports which segments were left out of the totals", async () => {
    await runSimulation(makeRun());
    expect(
      await screen.findByText(/could not be modelled/i),
    ).toBeVisible();
  });

  it("marks a leg whose queue was not modelled", async () => {
    await runSimulation(
      makeRun({
        routes: [
          makeArm({
            legs: [
              {
                segment_id: "SEG-C",
                from_junction_id: "XQ-77",
                to_junction_id: "XQ-79",
                entry_time_s: 0,
                exit_time_s: 90,
                free_flow_traversal_s: 90,
                time_basis: "derived_from_free_flow",
                queue_delay_s: null,
                queue_modelled: false,
                queue_at_arrival_veh: null,
                reason: "No capacity for this segment.",
              },
            ],
          }),
        ],
        links: [],
        comparison: null,
      }),
    );
    const legTable = (await screen.findAllByRole("table"))[0] as HTMLElement;
    const row = within(legTable).getByText("SEG-C").closest("tr") as HTMLElement;
    expect(within(row).getByText("not modelled")).toBeVisible();
  });

  it("surfaces an unavailable run with its causes and no outcome", async () => {
    await runSimulation(
      makeRun({
        status: "unavailable",
        routes: [],
        links: [],
        network: null,
        comparison: null,
        rejections: [
          {
            reason: "no_admissible_route",
            message: "The optimizer found no admissible route.",
            segment_ids: [],
            junction_ids: [],
            route_ids: [],
            incident_ids: [],
          },
        ],
      }),
    );
    expect(await screen.findByText(/nothing could be simulated/i)).toBeVisible();
    expect(screen.getByText("no admissible route")).toBeVisible();
    expect(screen.queryByText(/travel time/i)).not.toBeInTheDocument();
  });

  it("records the input fingerprint so a run can be reproduced", async () => {
    await runSimulation(makeRun());
    // Once as the run id in the header pill, once as the input fingerprint.
    expect((await screen.findAllByText(/abc123def456/)).length).toBe(2);
    expect(screen.getByText(/identical inputs reproduce this run/i)).toBeVisible();
  });

  it("requires an assumption to justify itself before it can be added", async () => {
    stubApi(makeRun());
    render(<SimulationPanel network={success(NETWORK)} context={CONTEXT} />);
    const add = screen.getByRole("button", { name: /add assumption/i });
    expect(add).toBeDisabled();

    await userEvent.selectOptions(screen.getByLabelText(/segment/i), "SEG-A");
    expect(add).toBeDisabled();

    await userEvent.type(screen.getByLabelText(/why/i), "Peak-hour estimate.");
    expect(add).toBeEnabled();

    await userEvent.click(add);
    expect(screen.getByText("Peak-hour estimate.")).toBeVisible();
  });

  it("sends stated assumptions with the comparison request", async () => {
    stubApi(makeRun());
    render(<SimulationPanel network={success(NETWORK)} context={CONTEXT} />);

    await userEvent.selectOptions(screen.getByLabelText(/segment/i), "SEG-A");
    await userEvent.type(screen.getByLabelText(/demand/i), "1400");
    await userEvent.type(screen.getByLabelText(/capacity/i), "900");
    await userEvent.type(screen.getByLabelText(/why/i), "Roadworks assumption.");
    await userEvent.click(screen.getByRole("button", { name: /add assumption/i }));
    await userEvent.click(
      screen.getByRole("button", { name: /simulate baseline vs recommended/i }),
    );

    const calls = (globalThis.fetch as unknown as { mock: { calls: unknown[][] } })
      .mock.calls;
    const post = calls.find(
      (call) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(post).toBeDefined();
    const body = JSON.parse((post?.[1] as RequestInit).body as string);
    expect(body.segment_inputs).toEqual([
      {
        segment_id: "SEG-A",
        note: "Roadworks assumption.",
        demand_veh_per_hour: 1400,
        capacity_veh_per_hour: 900,
      },
    ]);
    expect(body.origin_junction_id).toBe("XQ-77");
    expect(body.destination_junction_id).toBe("XQ-79");
  });

  it("has no approve, dispatch or apply control", async () => {
    await runSimulation(makeRun());
    await screen.findByText("Shortest route by distance");
    for (const label of [/approve/i, /dispatch/i, /apply/i, /execute/i]) {
      expect(screen.queryByRole("button", { name: label })).not.toBeInTheDocument();
    }
  });

  it("surfaces a failed request rather than rendering an empty panel", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => {
        if (init?.method === "POST") {
          return {
            ok: false,
            status: 422,
            json: async () => ({
              error: {
                code: "validation_failed",
                message: "Stated input names an unknown segment.",
                details: {},
              },
            }),
          } as Response;
        }
        return { ok: true, status: 200, json: async () => CONFIG } as Response;
      }),
    );
    render(<SimulationPanel network={success(NETWORK)} context={CONTEXT} />);
    await userEvent.click(
      screen.getByRole("button", { name: /simulate baseline vs recommended/i }),
    );
    expect(
      await screen.findByText(/stated input names an unknown segment/i),
    ).toBeVisible();
  });
});
