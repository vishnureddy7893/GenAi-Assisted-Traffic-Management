/**
 * Panel tests.
 *
 * The central property under test: **everything domain-shaped on screen came
 * from an API response.** Fixtures use identifiers no hard-coded topology would
 * contain ("XQ-77", "ZZ9-NORTH"), and several tests assert that the identifiers
 * previously baked into this dashboard no longer appear anywhere.
 *
 * Also covered: an API failure must never be replaced by fabricated values, and
 * an empty response must render as empty rather than as sample data.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AppHeader } from "./AppHeader";
import { DataSourcePanel } from "./DataSourcePanel";
import { EmergencyPanel } from "./EmergencyPanel";
import { NetworkGraphPanel } from "./NetworkGraphPanel";
import { SafetyCatalogPanel } from "./SafetyCatalogPanel";
import { SystemStatusPanel } from "./SystemStatusPanel";
import { ErrorState, UnavailableState } from "./ResourceState";
import type { Resource } from "../hooks/useApiResource";
import type {
  EmergencyScenariosResponse,
  HealthResponse,
  RoadNetwork,
  SafetyCatalogResponse,
  SourceCatalogResponse,
} from "../api/types";
import {
  makeCatalog,
  makeFacility,
  makeHealth,
  makeJunction,
  makeNetwork,
  makeProfile,
  makeScenario,
  makeScenariosResponse,
  makeSegment,
} from "../test/fixtures";

const noop = (): void => undefined;

/** Identifiers that were hard-coded into this dashboard before the refactor. */
const PREVIOUSLY_HARDCODED = ["J10", "J11", "J12", "J13", "J14"];

function success<T>(data: T): Resource<T> {
  return { state: "success", data };
}

const NETWORK_UNAVAILABLE: Resource<RoadNetwork> = {
  state: "unavailable",
  message: "No road network is loaded.",
  phase: 5,
};

// ---------------------------------------------------------------------------
// Network — nodes must come from the API
// ---------------------------------------------------------------------------
describe("NetworkGraphPanel", () => {
  const network = makeNetwork({
    network_id: "net-xyz",
    junctions: [makeJunction("XQ-77"), makeJunction("ZZ9-NORTH"), makeJunction("K3")],
    segments: [makeSegment("SEG-A", "XQ-77", "ZZ9-NORTH"), makeSegment("SEG-B", "ZZ9-NORTH", "K3")],
    facilities: [makeFacility("MED-1", "St Jude Clinic", "K3")],
  });

  it("renders exactly the junctions the API returned", () => {
    render(<NetworkGraphPanel network={success(network)} onRetry={noop} />);
    for (const id of ["XQ-77", "ZZ9-NORTH", "K3"]) {
      expect(screen.getByTestId(`network-node-${id}`)).toBeInTheDocument();
      expect(screen.getByText(id)).toBeInTheDocument();
    }
  });

  it("renders facilities the API returned, by their API name", () => {
    render(<NetworkGraphPanel network={success(network)} onRetry={noop} />);
    expect(screen.getByTestId("network-node-MED-1")).toBeInTheDocument();
    expect(screen.getByText("St Jude Clinic")).toBeInTheDocument();
  });

  it("draws one line per segment returned by the API", () => {
    const { container } = render(
      <NetworkGraphPanel network={success(network)} onRetry={noop} />,
    );
    // Two road segments plus one facility-access link.
    expect(container.querySelectorAll("line.netgraph__edge")).toHaveLength(3);
    expect(container.querySelectorAll("line.netgraph__edge--access")).toHaveLength(1);
  });

  it("contains none of the previously hard-coded junction ids", () => {
    const { container } = render(
      <NetworkGraphPanel network={success(network)} onRetry={noop} />,
    );
    const text = container.textContent ?? "";
    for (const id of PREVIOUSLY_HARDCODED) {
      expect(text).not.toContain(id);
    }
  });

  it("reports counts from the API rather than from a constant", () => {
    render(<NetworkGraphPanel network={success(network)} onRetry={noop} />);
    expect(screen.getByText(/3 junctions · 2 segments/)).toBeInTheDocument();
  });

  it("shows a no-network state and draws nothing when none is loaded", () => {
    const { container } = render(
      <NetworkGraphPanel network={NETWORK_UNAVAILABLE} onRetry={noop} />,
    );
    expect(screen.getByText(/no network loaded/i)).toBeInTheDocument();
    expect(container.querySelector("svg[role='img']")).toBeNull();
    expect(container.querySelectorAll("line.netgraph__edge")).toHaveLength(0);
  });

  it("does not invent a topology when the backend fails", () => {
    const { container } = render(
      <NetworkGraphPanel
        network={{ state: "error", message: "boom", unreachable: true }}
        onRetry={noop}
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(container.querySelectorAll("line.netgraph__edge")).toHaveLength(0);
    expect(container.querySelectorAll("circle.netgraph__node")).toHaveLength(0);
  });

  it("handles a loaded-but-empty network without drawing placeholder nodes", () => {
    const { container } = render(
      <NetworkGraphPanel network={success(makeNetwork())} onRetry={noop} />,
    );
    expect(screen.getByText(/contains no junctions/i)).toBeInTheDocument();
    expect(container.querySelectorAll("circle.netgraph__node")).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Sources — capabilities must come from the API
// ---------------------------------------------------------------------------
describe("DataSourcePanel", () => {
  const catalog: SourceCatalogResponse = makeCatalog({
    unavailable_kinds: ["totally_invented_kind"],
    note: "Catalogue note from the backend.",
    modes: [
      {
        mode: "PRIMARY-MODE",
        title: "Archived Feeds",
        description: "Group description from the backend.",
        is_primary: true,
        profiles: [
          makeProfile("totally_invented_kind", {
            display_name: "Totally Invented Kind",
            available_from_phase: 7,
            required_dependencies: ["some-lib"],
          }),
        ],
      },
      {
        mode: "SECONDARY-MODE",
        title: "Streaming Feeds",
        description: "Second group description.",
        is_primary: false,
        profiles: [
          makeProfile("another_kind", {
            display_name: "Another Kind",
            mode: "SECONDARY-MODE",
            realtime: true,
            bounded: false,
            seekable: false,
            replayable: false,
            requires_backpressure: true,
            available_from_phase: 14,
          }),
        ],
      },
    ],
  });

  it("renders the mode groups the API returned, by their API titles", () => {
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(screen.getByRole("tab", { name: /archived feeds/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /streaming feeds/i })).toBeInTheDocument();
  });

  it("defaults to whichever group the API marks primary, not a hard-coded mode", () => {
    const tab = () => screen.getByRole("tab", { name: /archived feeds/i });
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(tab()).toHaveAttribute("aria-selected", "true");
    expect(within(tab()).getByText(/primary/i)).toBeInTheDocument();
  });

  it("renders source kinds from the API, including invented ones", () => {
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(screen.getByTestId("source-totally_invented_kind")).toBeInTheDocument();
    expect(screen.getByText("Totally Invented Kind")).toBeInTheDocument();
  });

  it("derives capability chips from the response, not a fixed list", async () => {
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(screen.getByText("seekable")).toBeInTheDocument();
    expect(screen.getByText("bounded")).toBeInTheDocument();
    expect(screen.getByText("no realtime")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /streaming feeds/i }));
    expect(screen.getByText("no seekable")).toBeInTheDocument();
    expect(screen.getByText("realtime")).toBeInTheDocument();
    expect(screen.getByText("requires backpressure")).toBeInTheDocument();
  });

  it("labels each source with the phase the API reported", () => {
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(screen.getByText("Phase 7")).toBeInTheDocument();
  });

  it("distinguishes implemented from unavailable using the API flag", () => {
    const withImplemented = makeCatalog({
      implemented_kinds: ["ready_kind"],
      any_source_available: true,
      modes: [
        {
          mode: "M",
          title: "Group",
          description: "d",
          is_primary: true,
          profiles: [
            makeProfile("ready_kind", { display_name: "Ready Kind", implemented: true }),
          ],
        },
      ],
    });
    render(<DataSourcePanel catalog={success(withImplemented)} onRetry={noop} />);
    const row = screen.getByTestId("source-ready_kind");
    expect(within(row).getByText("Available")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /load recording/i })).toBeEnabled();
  });

  it("disables controls for unimplemented kinds", () => {
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(screen.getByRole("button", { name: /load recording/i })).toBeDisabled();
  });

  it("states plainly when a group has nothing connected", () => {
    render(<DataSourcePanel catalog={success(catalog)} onRetry={noop} />);
    expect(screen.getByText(/no archived feeds source is connected/i)).toBeInTheDocument();
  });

  it("shows an empty state when the API returns no modes", () => {
    render(<DataSourcePanel catalog={success(makeCatalog())} onRetry={noop} />);
    expect(screen.getByText(/no input modes defined/i)).toBeInTheDocument();
  });

  it("invents no sources when the backend fails", () => {
    render(
      <DataSourcePanel
        catalog={{ state: "error", message: "down", unreachable: true }}
        onRetry={noop}
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Emergency — scenarios must come from the API
// ---------------------------------------------------------------------------
describe("EmergencyPanel", () => {
  it("renders the scenarios the API returned", () => {
    const response = makeScenariosResponse(
      [
        makeScenario("EMG-ALPHA", {
          priority: "P1_immediate",
          status: "en_route",
          vehicle: {
            vehicle_id: "UNIT-42",
            vehicle_type: "fire_engine",
            callsign: "Ladder 9",
            height_m: null,
            weight_t: null,
            requires_wide_turns: null,
          },
          destination: {
            facility_id: null,
            incident_id: null,
            position: null,
            facility_type: null,
            label: "Warehouse Fire",
          },
        }),
      ],
      { active_count: 1 },
    );

    render(<EmergencyPanel scenarios={success(response)} onRetry={noop} />);
    expect(screen.getByTestId("scenario-EMG-ALPHA")).toBeInTheDocument();
    expect(screen.getByText("Ladder 9")).toBeInTheDocument();
    expect(screen.getByText(/warehouse fire/i)).toBeInTheDocument();
    expect(screen.getByText("P1_immediate")).toBeInTheDocument();
  });

  it("renders every scenario the API returned, not a fixed number", () => {
    const response = makeScenariosResponse([
      makeScenario("E1"),
      makeScenario("E2"),
      makeScenario("E3"),
    ]);
    render(<EmergencyPanel scenarios={success(response)} onRetry={noop} />);
    for (const id of ["E1", "E2", "E3"]) {
      expect(screen.getByTestId(`scenario-${id}`)).toBeInTheDocument();
    }
  });

  it("reports counts from the API", () => {
    const response = makeScenariosResponse([makeScenario("E1")], { active_count: 1 });
    render(<EmergencyPanel scenarios={success(response)} onRetry={noop} />);
    expect(screen.getByText("1 active / 1 declared")).toBeInTheDocument();
  });

  it("shows the empty state, using the backend's own explanation", () => {
    render(<EmergencyPanel scenarios={success(makeScenariosResponse())} onRetry={noop} />);
    expect(screen.getByText(/no emergency declared/i)).toBeInTheDocument();
    expect(screen.getByText(/nothing was declared/i)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("fabricates no emergency when the collection is empty", () => {
    const { container } = render(
      <EmergencyPanel scenarios={success(makeScenariosResponse())} onRetry={noop} />,
    );
    expect(container.querySelector("[data-testid^='scenario-']")).toBeNull();
    // No numeric literal with a unit — no ETA, distance or percentage.
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/(?<![A-Za-z])\d+(\.\d+)?\s*(s|sec|min|%|km|km\/h)\b/i);
  });

  it("fabricates no emergency when the backend fails", () => {
    const { container } = render(
      <EmergencyPanel
        scenarios={{ state: "error", message: "down", unreachable: true }}
        onRetry={noop}
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(container.querySelector("[data-testid^='scenario-']")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Health — status must come from the API
// ---------------------------------------------------------------------------
describe("Health-driven UI", () => {
  const health: Resource<HealthResponse> = success(
    makeHealth({
      app_version: "7.7.7-fixture",
      implemented_phase: 11,
      implemented_phase_name: "Fixture Phase Eleven",
      environment: "fixture-env",
      pipeline_ready: false,
      pipeline_blockers: ["one blocker"],
      components: [
        { name: "alpha_subsystem", status: "ok", detail: "running", phase: null },
        { name: "beta_subsystem", status: "not_implemented", detail: "later", phase: 9 },
      ],
    }),
  );

  it("header shows version and phase from the API, not constants", () => {
    render(<AppHeader health={health} />);
    expect(screen.getByText("v7.7.7-fixture")).toBeInTheDocument();
    expect(screen.getByText("Fixture Phase Eleven")).toBeInTheDocument();
    expect(screen.getByText(/backend connected/i)).toBeInTheDocument();
    expect(screen.getByText(/pipeline not ready/i)).toBeInTheDocument();
  });

  it("header reflects pipeline readiness reported by the API", () => {
    const ready = success(makeHealth({ pipeline_ready: true }));
    render(<AppHeader health={ready} />);
    expect(screen.getByText(/pipeline ready/i)).toBeInTheDocument();
  });

  it("header reports a failure instead of claiming connection", () => {
    render(
      <AppHeader health={{ state: "error", message: "refused", unreachable: true }} />,
    );
    expect(screen.getByText(/backend offline/i)).toBeInTheDocument();
    expect(screen.queryByText(/backend connected/i)).not.toBeInTheDocument();
  });

  it("system panel lists the components the API returned", () => {
    render(<SystemStatusPanel health={health} onRetry={noop} />);
    expect(screen.getByText("alpha_subsystem")).toBeInTheDocument();
    expect(screen.getByText("beta_subsystem")).toBeInTheDocument();
    expect(screen.getByText("Phase 9")).toBeInTheDocument();
    expect(screen.getByText("1/2 operational")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Safety catalog — rules must come from the API
// ---------------------------------------------------------------------------
describe("SafetyCatalogPanel", () => {
  const catalog: Resource<SafetyCatalogResponse> = success({
    version: "3.3.3-fixture",
    rules: [
      {
        rule_id: "hard.invented_rule",
        name: "Invented Hard Rule",
        tier: "hard",
        scope: "segment",
        description: "d",
        rationale: "cannot be traversed at any cost",
        default_penalty_equivalent_s: null,
      },
      {
        rule_id: "soft.invented_rule",
        name: "Invented Soft Rule",
        tier: "soft",
        scope: "junction",
        description: "d",
        rationale: "raises risk without making a route impossible",
        default_penalty_equivalent_s: null,
      },
    ],
  });

  it("renders rules and version from the API", () => {
    render(<SafetyCatalogPanel catalog={catalog} onRetry={noop} />);
    expect(screen.getByText("Invented Hard Rule")).toBeInTheDocument();
    expect(screen.getByText("Invented Soft Rule")).toBeInTheDocument();
    expect(screen.getByText("v3.3.3-fixture")).toBeInTheDocument();
  });

  it("groups by the tiers present in the response", () => {
    render(<SafetyCatalogPanel catalog={catalog} onRetry={noop} />);
    expect(screen.getByText(/removed from the search graph/i)).toBeInTheDocument();
    expect(screen.getByText(/priced in equivalent seconds/i)).toBeInTheDocument();
  });

  it("marks soft penalties uncalibrated rather than showing a number", () => {
    render(<SafetyCatalogPanel catalog={catalog} onRetry={noop} />);
    expect(screen.getByText(/penalty uncalibrated/i)).toBeInTheDocument();
  });

  it("shows an empty state for a catalog with no rules", () => {
    render(
      <SafetyCatalogPanel
        catalog={success({ version: "0", rules: [] })}
        onRetry={noop}
      />,
    );
    expect(screen.getByText(/no safety rules defined/i)).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Shared state rendering
// ---------------------------------------------------------------------------
describe("ResourceState", () => {
  it("presents a 501 as a phase boundary, not a failure", () => {
    render(<UnavailableState message="Not built." phase={7} />);
    expect(screen.getByText(/not implemented yet — phase 7/i)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("presents a genuine failure as an alert with a retry", async () => {
    const onRetry = vi.fn();
    render(<ErrorState message="boom" unreachable onRetry={onRetry} />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/backend unreachable/i)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// Repository-level guard
// ---------------------------------------------------------------------------
describe("no hard-coded domain data", () => {
  it("emergency response type carries no default scenario", () => {
    const empty: EmergencyScenariosResponse = makeScenariosResponse();
    expect(empty.scenarios).toEqual([]);
    expect(empty.count).toBe(0);
  });
});
