/**
 * AI Traffic Advisor panel tests.
 *
 * The panel's job is to render an advisory without adding anything to it, and
 * without letting the reader mistake one mode for the other. So the assertions
 * are about labelling and about absence: a deterministic advisory must say so
 * and must not name a model, limitations and unavailable facts must be on
 * screen rather than tucked away, and the advisory-only disclaimer must survive
 * every state.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AdvisorPanel } from "./AdvisorPanel";
import type {
  AdvisorResponse,
  AdvisorStatusResponse,
  Provenance,
  TrafficAdvisory,
  TrafficAdvisorFacts,
} from "../api/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const MEDIA_ID = "media-1";
const AREA = `observation-area:${MEDIA_ID}`;

const PROVENANCE: Provenance = {
  source: "derived",
  producer: "advisor.fact_builder",
  producer_version: "0.1.0",
  observed_at: "2026-09-22T09:00:00Z",
  computed_at: "2026-09-22T09:00:00Z",
  confidence: null,
  input_refs: [`analytics:${MEDIA_ID}`],
  notes: null,
};

function makeFacts(overrides: Partial<TrafficAdvisorFacts> = {}): TrafficAdvisorFacts {
  return {
    observation_area_id: AREA,
    site_label: "Benz Circle NW mast",
    latitude: 16.5051,
    longitude: 80.6001,
    confirmed_junction_id: "J1",
    confirmed_segment_id: null,
    confirmed_segment_direction: null,
    traffic_state: "heavy",
    congestion_index: 0.74,
    vehicle_count: 46,
    stationary_vehicle_count: 12,
    vehicle_class_counts: { car: 38, bus: 1 },
    pedestrian_count: 3,
    cyclist_count: 0,
    image_occupancy_ratio: 0.31,
    spatial_summary: null,
    quality_grade: "good",
    state_as_of: "2026-09-22T09:00:00Z",
    incident_summaries: [],
    emergency_summaries: [],
    has_route_comparison: false,
    route_baseline_s: null,
    route_alternative_s: null,
    route_improvement_pct: null,
    route_selected_id: null,
    has_simulation: false,
    simulation_run_id: null,
    simulation_travel_time_delta_s: null,
    simulation_improvement_pct: null,
    fact_table: {
      facts: [
        {
          fact_id: "state.level",
          label: "Traffic state band",
          value: "heavy",
          unit: null,
          provenance: PROVENANCE,
          display_variants: [],
        },
        {
          fact_id: "state.vehicle_count",
          label: "Vehicles detected in the observation area",
          value: 46,
          unit: "vehicles",
          provenance: PROVENANCE,
          display_variants: ["46 vehicles"],
        },
      ],
      entity_tokens: [AREA],
    },
    unavailable: [
      {
        fact_id: "route.comparison",
        label: "Route comparison",
        reason: "no route optimization with at least two ranked candidates is available",
      },
    ],
    ...overrides,
  };
}

function makeAdvisory(overrides: Partial<TrafficAdvisory> = {}): TrafficAdvisory {
  return {
    mode: "deterministic_fallback",
    observation_area_id: AREA,
    summary: "The traffic-state system classifies junction J1 as heavy.",
    traffic_assessment: "Heavy traffic is indicated on the monitored area.",
    suggestions: ["Evaluate alternate routes using current routing information."],
    safety_considerations: [
      "This advisory is decision support for a human operator.",
    ],
    route_advice:
      "No route comparison is available, so no route can be described as faster.",
    limitations: ["Route comparison is unavailable: no route optimization is available."],
    evidence_ids: ["state.level", "state.vehicle_count"],
    model_id: null,
    generated_at: "2026-09-22T09:00:00Z",
    validation: null,
    advisory_only: true,
    ...overrides,
  };
}

function makeStatus(
  overrides: Partial<AdvisorStatusResponse> = {},
): AdvisorStatusResponse {
  return {
    provider_configured: false,
    mode: "deterministic_fallback",
    model_id: null,
    advisory_only: true,
    note: "No language model is configured. Advisories are composed from the recorded facts by fixed rules and are labelled as such.",
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubApi(
  options: {
    status?: AdvisorStatusResponse;
    result?: AdvisorResponse;
    analyzeError?: { code: string; message: string; status: number };
    /** Held open until resolved, so a loading assertion cannot lose a race
     *  against a timer under a loaded suite. */
    analyzeGate?: Promise<void>;
  } = {},
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.includes("/advisor/analyze") && init?.method === "POST") {
      if (options.analyzeError !== undefined) {
        return jsonResponse(
          {
            error: {
              code: options.analyzeError.code,
              message: options.analyzeError.message,
              details: {},
            },
          },
          options.analyzeError.status,
        );
      }
      if (options.analyzeGate !== undefined) {
        await options.analyzeGate;
      }
      return jsonResponse(
        options.result ?? { advisory: makeAdvisory(), facts: makeFacts() },
      );
    }
    return jsonResponse(options.status ?? makeStatus());
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function generate(): Promise<void> {
  await userEvent.click(screen.getByRole("button", { name: /generate advisory/i }));
  await waitFor(() =>
    expect(screen.getByTestId("advisor-result")).toBeInTheDocument(),
  );
}

describe("AdvisorPanel", () => {
  it("renders and asks for media rather than an empty advisory", () => {
    stubApi();
    render(<AdvisorPanel mediaId={null} />);
    expect(screen.getByText("AI Traffic Advisor")).toBeInTheDocument();
    expect(screen.getByText(/No media selected/i)).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /generate advisory/i }),
    ).not.toBeInTheDocument();
  });

  it("says it is in deterministic mode before anything is generated", async () => {
    stubApi();
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await waitFor(() =>
      expect(screen.getByText(/Deterministic advisory mode/i)).toBeInTheDocument(),
    );
  });

  it("says a model is configured when one is", async () => {
    stubApi({
      status: makeStatus({
        provider_configured: true,
        mode: "llm",
        model_id: "anthropic/claude-3.5-haiku",
        note: "A language model is configured.",
      }),
    });
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await waitFor(() =>
      expect(screen.getByText(/LLM configured/i)).toBeInTheDocument(),
    );
  });

  it("shows a loading state while generating", async () => {
    let release = (): void => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    stubApi({ analyzeGate: gate });
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await userEvent.click(screen.getByRole("button", { name: /generate advisory/i }));

    // The request cannot complete until the gate is released, so this is not a
    // race against a timer.
    await waitFor(() =>
      expect(screen.getByTestId("advisor-loading")).toBeInTheDocument(),
    );

    release();
    await waitFor(() =>
      expect(screen.queryByTestId("advisor-loading")).not.toBeInTheDocument(),
    );
  });

  it("renders every section of a deterministic advisory", async () => {
    stubApi();
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    expect(screen.getByText(/classifies junction J1 as heavy/)).toBeInTheDocument();
    expect(screen.getByTestId("advisor-suggestions")).toHaveTextContent(
      "Evaluate alternate routes",
    );
    expect(screen.getByTestId("advisor-safety")).toHaveTextContent(
      "decision support for a human operator",
    );
    expect(screen.getByTestId("advisor-route")).toHaveTextContent(
      "no route can be described as faster",
    );
    expect(screen.getByTestId("advisor-limitations")).toBeInTheDocument();
    expect(screen.getByTestId("advisor-evidence")).toBeInTheDocument();
  });

  it("labels a deterministic advisory as deterministic and names no model", async () => {
    stubApi();
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    expect(screen.getByText(/Deterministic advisory mode/i)).toBeInTheDocument();
    expect(screen.getByText(/No language model was involved/i)).toBeInTheDocument();
    expect(screen.queryByText(/LLM generated/i)).not.toBeInTheDocument();
  });

  it("labels an LLM advisory as generated and names the model", async () => {
    stubApi({
      result: {
        advisory: makeAdvisory({
          mode: "llm",
          model_id: "anthropic/claude-3.5-haiku",
          validation: { unsupported_numbers: [], unknown_fact_ids: [], passed: true },
        }),
        facts: makeFacts(),
      },
    });
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    expect(screen.getByText(/LLM generated/i)).toBeInTheDocument();
    expect(screen.getByText(/anthropic\/claude-3.5-haiku/)).toBeInTheDocument();
    expect(
      screen.queryByText(/No language model was involved/i),
    ).not.toBeInTheDocument();
  });

  it("shows the advisory-only disclaimer", async () => {
    stubApi();
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    expect(screen.getByText("Advisory only")).toBeInTheDocument();
    expect(
      screen.getByText(/no control path to any traffic signal/i),
    ).toBeInTheDocument();
  });

  it("shows the facts the advisory used, with their provenance", async () => {
    stubApi();
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    const evidence = screen.getByTestId("advisor-evidence");
    expect(evidence).toHaveTextContent("Vehicles detected in the observation area");
    expect(evidence).toHaveTextContent("46");
    expect(evidence).toHaveTextContent("derived");
  });

  it("shows missing data as unavailable with a reason, never as zero", async () => {
    stubApi({
      result: {
        advisory: makeAdvisory(),
        facts: makeFacts({
          vehicle_count: null,
          traffic_state: null,
          unavailable: [
            {
              fact_id: "state.vehicle_count",
              label: "Vehicle count",
              reason: "no traffic state has been interpreted, so no count exists",
            },
          ],
        }),
      },
    });
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    const unavailable = screen.getByTestId("advisor-unavailable");
    expect(unavailable).toHaveTextContent("Vehicle count");
    expect(unavailable).toHaveTextContent("unavailable");
    expect(unavailable).toHaveTextContent("no traffic state has been interpreted");
    expect(unavailable).not.toHaveTextContent(/:\s*0\b/);
  });

  it("surfaces a failed advisory rather than rendering an empty one", async () => {
    stubApi({
      analyzeError: {
        code: "not_found",
        message: "Nothing has been analysed under 'media-1'.",
        status: 404,
      },
    });
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await userEvent.click(screen.getByRole("button", { name: /generate advisory/i }));

    await waitFor(() =>
      expect(screen.getByTestId("advisor-error")).toHaveTextContent(
        /Nothing has been analysed/,
      ),
    );
    expect(screen.queryByTestId("advisor-result")).not.toBeInTheDocument();
  });

  it("sends the media id it was given", async () => {
    const fetchMock = stubApi();
    render(<AdvisorPanel mediaId={MEDIA_ID} />);
    await generate();

    const posted = fetchMock.mock.calls.find(
      (call) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(posted).toBeDefined();
    const body = JSON.parse(String((posted?.[1] as RequestInit).body)) as {
      media_id: string;
    };
    expect(body.media_id).toBe(MEDIA_ID);
  });
});
