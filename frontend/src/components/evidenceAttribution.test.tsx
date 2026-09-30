/**
 * Evidence attribution panel tests.
 *
 * The panel renders a chain — camera → road → observed state — that looks very
 * much like something already steering traffic. So the assertions are about
 * what it must keep visible: that an unmapped camera reads as *unmapped* rather
 * than as a blank, that the four non-attributed outcomes stay distinct, that
 * the backend's own reason for each row is shown, and that nothing here offers
 * to create a mapping.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EvidenceAttributionPanel } from "./EvidenceAttributionPanel";
import type {
  EvidenceAttribution,
  EvidenceAttributionReport,
  NetworkTrafficStateResponse,
  Provenance,
} from "../api/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const AREA_A = "observation-area:camera-A";
const AREA_B = "observation-area:camera-B";
const HASH = "a".repeat(64);

const PROVENANCE: Provenance = {
  source: "observed",
  producer: "operator",
  producer_version: "0.1.0",
  observed_at: "2026-09-22T09:00:00Z",
  computed_at: "2026-09-22T09:00:00Z",
  confidence: null,
  input_refs: [],
  notes: null,
};

function attributed(
  overrides: Partial<EvidenceAttribution> = {},
): EvidenceAttribution {
  return {
    observation_area_id: AREA_A,
    status: "attributed",
    segment_id: "SEG-14",
    junction_id: "J2",
    approach_id: "J2-A-SEG-14",
    segment_direction: "incoming",
    site_location: { latitude: 16.505, longitude: 80.6 },
    network_id: "example-grid",
    network_version: "1.0.0",
    source: "confirmed_binding",
    resolution: "confirmed_only",
    exclusion_reason: null,
    provenance: PROVENANCE,
    congestion_level: "heavy",
    freshness: "fresh",
    vehicle_count: 46,
    reason: "Evidence is attributed to segment SEG-14 by the effective mapping.",
    ...overrides,
  };
}

function unmapped(overrides: Partial<EvidenceAttribution> = {}): EvidenceAttribution {
  return {
    ...attributed(),
    observation_area_id: AREA_B,
    status: "unmapped",
    segment_id: null,
    junction_id: null,
    approach_id: null,
    segment_direction: null,
    site_location: null,
    source: null,
    resolution: null,
    provenance: null,
    congestion_level: "light",
    vehicle_count: 4,
    reason:
      "No effective binding exists for this observation area, so its evidence is real and unattributable.",
    ...overrides,
  };
}

function report(
  overrides: Partial<EvidenceAttributionReport> = {},
): EvidenceAttributionReport {
  return {
    network_id: "example-grid",
    network_version: "1.0.0",
    effective_mapping_hash: HASH,
    effective_mapping_version: HASH.slice(0, 16),
    attributions: [attributed(), unmapped()],
    attributed_area_ids: [AREA_A],
    unmapped_observation_area_ids: [AREA_B],
    excluded_area_ids: [],
    junction_only_area_ids: [],
    attributed_segment_ids: ["SEG-14"],
    used_by_route_costing: false,
    note: "1 of 2 interpreted observation area(s) reach a directed segment. Attribution is reported only: route costs are unchanged.",
    ...overrides,
  };
}

function response(
  overrides: Partial<NetworkTrafficStateResponse> = {},
): NetworkTrafficStateResponse {
  return {
    state_id: "snapshot-1",
    as_of: "2026-09-22T09:00:00Z",
    assembled_at: "2026-09-22T09:00:00Z",
    policy: {},
    elements: [],
    schema_version: "1.0",
    attribution: report(),
    attribution_unavailable_reason: null,
    ...overrides,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubApi(body: unknown, status = 200): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async () => jsonResponse(body, status));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function rendered(): Promise<void> {
  await waitFor(() =>
    expect(screen.getByTestId("attribution-list")).toBeInTheDocument(),
  );
}

describe("EvidenceAttributionPanel", () => {
  it("renders the camera to road to state chain", async () => {
    stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    const row = screen.getByTestId(`attribution-${AREA_A}`);
    expect(row).toHaveTextContent(AREA_A);
    expect(row).toHaveTextContent("SEG-14");
    expect(row).toHaveTextContent("heavy");
  });

  it("shows an unmapped camera as unmapped rather than as a blank", async () => {
    stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    const row = screen.getByTestId(`attribution-${AREA_B}`);
    expect(row).toHaveTextContent("Segment: Unmapped");
    expect(row).toHaveTextContent("unmapped");
    expect(row).toHaveTextContent(/real and unattributable/);
    expect(row).toHaveTextContent(/No effective binding exists/);
    // The observed state is still shown: the evidence is real, only its road
    // is unknown.
    expect(row).toHaveTextContent("light");
  });

  it("names the source and how the binding was resolved", async () => {
    stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    const row = screen.getByTestId(`attribution-${AREA_A}`);
    expect(row).toHaveTextContent("Human-confirmed binding");
    expect(row).toHaveTextContent("confirmed binding only");
    expect(row).toHaveTextContent("junction J2");
    expect(row).toHaveTextContent("approach J2-A-SEG-14");
  });

  it("distinguishes a file-only mapping from a confirmation", async () => {
    stubApi(
      response({
        attribution: report({
          attributions: [
            attributed({ source: "mapping_file", resolution: "file_only" }),
          ],
        }),
      }),
    );
    render(<EvidenceAttributionPanel />);
    await rendered();

    expect(screen.getByTestId(`attribution-${AREA_A}`)).toHaveTextContent(
      "Mapping file",
    );
  });

  it("shows a junction-only binding as reaching no segment", async () => {
    stubApi(
      response({
        attribution: report({
          attributions: [
            unmapped({
              observation_area_id: AREA_A,
              status: "junction_only",
              junction_id: "J1",
              source: "confirmed_binding",
              resolution: "confirmed_only",
              reason:
                "Bound to junction J1 but to no directed segment, so the evidence belongs to an intersection and to no direction of travel.",
            }),
          ],
          attributed_area_ids: [],
          unmapped_observation_area_ids: [AREA_A],
          junction_only_area_ids: [AREA_A],
          attributed_segment_ids: [],
        }),
      }),
    );
    render(<EvidenceAttributionPanel />);
    await rendered();

    const row = screen.getByTestId(`attribution-${AREA_A}`);
    expect(row).toHaveTextContent("junction only");
    expect(row).toHaveTextContent("Segment: Unmapped");
    expect(row).toHaveTextContent(/no direction of travel/);
  });

  it("shows an excluded stale binding with its reason", async () => {
    stubApi(
      response({
        attribution: report({
          attributions: [
            unmapped({
              observation_area_id: AREA_A,
              status: "excluded",
              exclusion_reason: "stale_network_version",
              reason:
                "A confirmed binding exists but is not in force: confirmed against network version '0.9'. Re-confirm it.",
            }),
          ],
          attributed_area_ids: [],
          unmapped_observation_area_ids: [AREA_A],
          excluded_area_ids: [AREA_A],
          attributed_segment_ids: [],
        }),
      }),
    );
    render(<EvidenceAttributionPanel />);
    await rendered();

    const row = screen.getByTestId(`attribution-${AREA_A}`);
    expect(row).toHaveTextContent("excluded");
    expect(row).toHaveTextContent(/Re-confirm it/);
    expect(row).toHaveTextContent("Segment: Unmapped");
  });

  it("shows the counts and the effective mapping version", async () => {
    stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    expect(screen.getByTestId("attribution-attributed-count")).toHaveTextContent("1");
    expect(screen.getByTestId("attribution-unmapped-count")).toHaveTextContent("1");
    expect(screen.getByTestId("attribution-segment-count")).toHaveTextContent("1");
    const version = screen.getByTestId("attribution-mapping-version");
    expect(version).toHaveTextContent(HASH.slice(0, 16));
    expect(within(version).getByTitle(HASH)).toBeInTheDocument();
  });

  it("states that attribution does not drive route costing", async () => {
    stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    expect(screen.getByText(/not used for route costing/i)).toBeInTheDocument();
    expect(screen.getByText(/route costs are unchanged/i)).toBeInTheDocument();
  });

  it("offers no control that creates a mapping", async () => {
    stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByRole("radio")).not.toBeInTheDocument();
  });

  it("says why attribution is unavailable rather than showing nothing", async () => {
    stubApi(
      response({
        attribution: null,
        attribution_unavailable_reason:
          "No road network is loaded, so no observation area can be resolved to a road.",
      }),
    );
    render(<EvidenceAttributionPanel />);
    await waitFor(() =>
      expect(
        screen.getByText(/No road network is loaded/i),
      ).toBeInTheDocument(),
    );
    expect(screen.queryByTestId("attribution-list")).not.toBeInTheDocument();
  });

  it("says when nothing has been interpreted yet", async () => {
    stubApi(
      response({
        attribution: report({
          attributions: [],
          attributed_area_ids: [],
          unmapped_observation_area_ids: [],
          attributed_segment_ids: [],
        }),
      }),
    );
    render(<EvidenceAttributionPanel />);
    await waitFor(() =>
      expect(
        screen.getByText(/No observation area has been interpreted/i),
      ).toBeInTheDocument(),
    );
  });

  it("surfaces a failed read instead of rendering an empty panel", async () => {
    stubApi(
      {
        error: {
          code: "internal_error",
          message: "Traffic state could not be assembled.",
          details: {},
        },
      },
      500,
    );
    render(<EvidenceAttributionPanel />);
    await waitFor(() =>
      expect(screen.getByTestId("attribution-error")).toHaveTextContent(
        /could not be assembled/,
      ),
    );
  });

  it("reads once and writes nothing", async () => {
    const fetchMock = stubApi(response());
    render(<EvidenceAttributionPanel />);
    await rendered();

    const writes = fetchMock.mock.calls.filter(
      (call) =>
        (call[1] as RequestInit | undefined)?.method !== undefined &&
        (call[1] as RequestInit).method !== "GET",
    );
    expect(writes).toHaveLength(0);
  });
});
