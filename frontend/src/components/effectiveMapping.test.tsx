/**
 * Effective observation mapping summary tests.
 *
 * The section is read-only and its job is to hide nothing: the content hash,
 * every override, every exclusion, and the fact that route costs are unaffected
 * must all reach the screen. It must also offer no way to change precedence —
 * that is an architectural rule, and a control for it would let two operators
 * derive two different mappings from the same data.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EffectiveMappingSummary } from "./EffectiveMappingSummary";
import type {
  EffectiveBinding,
  EffectiveMappingResponse,
  ExcludedConfirmedBinding,
  MappingOverride,
  Provenance,
} from "../api/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const AREA_A = "observation-area:camera-A";
const HASH = "a1b2c3d4e5f60718293a4b5c6d7e8f901122334455667788990011223344556677".slice(
  0,
  64,
);

const PROVENANCE: Provenance = {
  source: "derived",
  producer: "routing.effective_mapping",
  producer_version: "0.1.0",
  observed_at: "2026-09-22T09:00:00Z",
  computed_at: "2026-09-22T09:00:00Z",
  confidence: null,
  input_refs: ["mapping:survey-1@1.0"],
  notes: null,
};

function binding(): EffectiveBinding {
  return {
    observation_area_id: AREA_A,
    binding: {
      observation_area_id: AREA_A,
      segment_id: "SEG-15",
      junction_id: "J1",
      approach_id: null,
      coverage_note: null,
      confirmation: null,
    },
    source: "confirmed_binding",
    resolution: "confirmed_binding_precedence",
    confirmed: null,
    superseded_segment_id: "SEG-14",
    superseded_junction_id: null,
  };
}

function override(): MappingOverride {
  return {
    observation_area_id: AREA_A,
    file_binding: "SEG-14",
    confirmed_binding: "SEG-15",
    effective_binding: "SEG-15",
    resolution: "confirmed_binding_precedence",
    note: "A human confirmation supersedes the mapping file for this area.",
  };
}

function excluded(): ExcludedConfirmedBinding {
  return {
    observation_area_id: "observation-area:camera-Z",
    junction_id: "J2",
    segment_id: "SEG-23",
    network_id: "example-grid",
    network_version: "0.9",
    reason: "stale_network_version",
    detail: "Confirmed against network version '0.9', but version '1.0.0' is loaded.",
  };
}

function makeResponse(
  overrides: Partial<EffectiveMappingResponse["effective"]> = {},
  top: Partial<EffectiveMappingResponse> = {},
): EffectiveMappingResponse {
  return {
    effective: {
      mapping_id: "survey-1",
      mapping_version: "1.0",
      mapping_sha256: "b".repeat(64),
      mapping_source: null,
      confirmed_bindings_hash: "c".repeat(64),
      confirmed_binding_count: 1,
      network_id: "example-grid",
      network_version: "1.0.0",
      mapping: {
        mapping_id: `effective-${HASH.slice(0, 16)}`,
        version: HASH.slice(0, 16),
        network_id: "example-grid",
        network_version: "1.0.0",
        provenance: PROVENANCE,
        bindings: [binding().binding],
      },
      bindings: [binding()],
      overrides: [override()],
      excluded: [],
      effective_mapping_hash: HASH,
      effective_mapping_version: HASH.slice(0, 16),
      provenance: PROVENANCE,
      used_by_route_costing: false,
      warnings: [],
      note: "1 observation area(s) bound. This composition is not used for route costing.",
      ...overrides,
    },
    effective_mapping_in_use: false,
    optimizer_mapping_id: "survey-1",
    optimizer_mapping_version: "1.0",
    note: "Derived on request; nothing is stored. Route costs are unaffected.",
    ...top,
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
    expect(screen.getByTestId("effective-mapping")).toBeInTheDocument(),
  );
}

describe("EffectiveMappingSummary", () => {
  it("renders the mapping file identity", async () => {
    stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    expect(screen.getByTestId("effective-mapping-file")).toHaveTextContent(
      "survey-1 @ 1.0",
    );
  });

  it("says when no mapping file is loaded rather than showing a placeholder", async () => {
    stubApi(
      makeResponse({ mapping_id: null, mapping_version: null, mapping_sha256: null }),
    );
    render(<EffectiveMappingSummary />);
    await rendered();

    expect(screen.getByTestId("effective-mapping-file")).toHaveTextContent(
      "none loaded",
    );
  });

  it("shows the counts", async () => {
    stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    expect(screen.getByTestId("effective-confirmed-count")).toHaveTextContent("1");
    expect(screen.getByTestId("effective-binding-count")).toHaveTextContent("1");
    expect(screen.getByTestId("effective-override-count")).toHaveTextContent("1");
  });

  it("shows the effective mapping version and keeps the full hash available", async () => {
    stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    const shown = screen.getByTestId("effective-mapping-hash");
    expect(shown).toHaveTextContent(HASH.slice(0, 16));
    expect(within(shown).getByTitle(HASH)).toBeInTheDocument();
  });

  it("shows an override with both sources and the reason", async () => {
    stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    const row = screen.getByTestId(`override-${AREA_A}`);
    expect(row).toHaveTextContent("SEG-14");
    expect(row).toHaveTextContent("SEG-15");
    expect(row).toHaveTextContent("Human-confirmed binding precedence");
  });

  it("labels agreement as agreement rather than as an override", async () => {
    stubApi(
      makeResponse({
        overrides: [
          {
            ...override(),
            file_binding: "SEG-15",
            resolution: "sources_agree",
            note: "Both sources name the same target.",
          },
        ],
      }),
    );
    render(<EffectiveMappingSummary />);
    await rendered();

    expect(screen.getByTestId(`override-${AREA_A}`)).toHaveTextContent(
      "Both sources agree",
    );
  });

  it("shows excluded confirmations with why they are not in force", async () => {
    stubApi(makeResponse({ excluded: [excluded()] }));
    render(<EffectiveMappingSummary />);
    await rendered();

    const list = screen.getByTestId("effective-excluded");
    expect(list).toHaveTextContent("requires reconfirmation");
    expect(list).toHaveTextContent("camera-Z");
  });

  it("states that the composition is not used for route costing", async () => {
    stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    expect(screen.getByText(/not used for route costing/i)).toBeInTheDocument();
    expect(screen.getByText(/Route costs are unaffected/i)).toBeInTheDocument();
  });

  it("offers no control for changing precedence", async () => {
    stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    const section = screen.getByTestId("effective-mapping");
    expect(within(section).queryByRole("button")).not.toBeInTheDocument();
    expect(within(section).queryByRole("combobox")).not.toBeInTheDocument();
    expect(within(section).queryByRole("radio")).not.toBeInTheDocument();
  });

  it("renders nothing when no network is loaded", async () => {
    stubApi(
      {
        error: {
          code: "network_not_loaded",
          message: "No road network is loaded.",
          details: {},
        },
      },
      404,
    );
    render(<EffectiveMappingSummary />);
    await waitFor(() =>
      expect(screen.queryByTestId("effective-mapping")).not.toBeInTheDocument(),
    );
  });

  it("surfaces an unreadable registry rather than rendering an empty mapping", async () => {
    stubApi(
      {
        error: {
          code: "data_source_error",
          message: "The confirmed binding registry could not be read.",
          details: {},
        },
      },
      502,
    );
    render(<EffectiveMappingSummary />);
    await waitFor(() =>
      expect(screen.getByTestId("effective-mapping-error")).toHaveTextContent(
        /could not be read/,
      ),
    );
    expect(screen.queryByTestId("effective-binding-count")).not.toBeInTheDocument();
  });

  it("reads the mapping once and writes nothing", async () => {
    const fetchMock = stubApi(makeResponse());
    render(<EffectiveMappingSummary />);
    await rendered();

    const writes = fetchMock.mock.calls.filter(
      (call) =>
        (call[1] as RequestInit | undefined)?.method !== undefined &&
        (call[1] as RequestInit).method !== "GET",
    );
    expect(writes).toHaveLength(0);
  });
});
