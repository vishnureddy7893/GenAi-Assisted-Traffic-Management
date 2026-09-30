/**
 * Segment stage of the site binding panel.
 *
 * The assertions are about what the panel refuses to do. It must show the
 * bearing offset without ordering by it, must pre-select nothing, must not send
 * anything until the operator both picks a segment and presses Confirm, and must
 * state plainly that a confirmed segment changes no route cost. A panel that
 * pre-ticked the best-aligned segment would be making the one decision the whole
 * step exists to put to a person.
 */

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SiteBindingPanel } from "./SiteBindingPanel";
import type {
  CandidateResponse,
  ConfirmedSiteBinding,
  ObservationSite,
  Provenance,
  RoadNetwork,
  SegmentCandidate,
  SegmentCandidatesResponse,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { makeNetwork, makeJunction, makeSegment } from "../test/fixtures";

afterEach(() => {
  vi.restoreAllMocks();
});

const AREA = "observation-area:cam-1";

const NETWORK: RoadNetwork = makeNetwork({
  network_id: "example-grid",
  junctions: [makeJunction("J1"), makeJunction("J2"), makeJunction("J4")],
  segments: [makeSegment("SEG-12", "J1", "J2"), makeSegment("SEG-21", "J2", "J1")],
});

const PROVENANCE: Provenance = {
  source: "derived",
  producer: "sites.incident_segments",
  producer_version: "0.1.0",
  observed_at: "2026-09-17T08:00:00Z",
  computed_at: "2026-09-17T08:00:00Z",
  confidence: null,
  input_refs: [`observation_site:${AREA}`, "junction:J1"],
  notes: null,
};

function makeSite(): ObservationSite {
  return {
    observation_area_id: AREA,
    location: { latitude: 16.5051, longitude: 80.6001 },
    bearing_degrees: 90,
    accuracy_m: null,
    label: "North West mast",
    provenance: { ...PROVENANCE, source: "observed", producer: "operator" },
    recorded_at: "2026-09-17T08:00:00Z",
    updated_at: "2026-09-17T08:00:00Z",
  };
}

function makeConfirmed(
  overrides: Partial<ConfirmedSiteBinding> = {},
): ConfirmedSiteBinding {
  return {
    observation_area_id: AREA,
    junction_id: "J1",
    network_id: "example-grid",
    network_version: "1.0.0",
    site_location: { latitude: 16.5051, longitude: 80.6001 },
    confirmation: {
      confirmed_by: "surveyor.junction",
      confirmed_at: "2026-09-17T08:05:00Z",
      method: "accepted_nearest",
      candidate_distance_m: 15.4,
      site_accuracy_m: null,
      was_nearest_candidate: true,
      notes: null,
    },
    provenance: { ...PROVENANCE, source: "observed", producer: "surveyor.junction" },
    segment_id: null,
    segment_direction: null,
    approach_id: null,
    segment_confirmation: null,
    ...overrides,
  };
}

function segment(
  id: string,
  direction: "incoming" | "outgoing",
  other: string,
  offset: number | null,
  approach: string | null = null,
): SegmentCandidate {
  return {
    segment_id: id,
    segment_name: `${other} arm`,
    direction,
    from_junction_id: direction === "incoming" ? other : "J1",
    to_junction_id: direction === "incoming" ? "J1" : other,
    other_junction_id: other,
    length_m: 550,
    approach_id: approach,
    approach_bearing_degrees: null,
    bearing_from_junction_degrees: offset === null ? null : 90,
    camera_bearing_offset_degrees: offset,
  };
}

/**
 * Four arms at J1, ordered by segment id. SEG-12 and SEG-21 are the two
 * directions of the east road and share a bearing; SEG-14 and SEG-41 run south.
 */
function makeSegments(
  overrides: Partial<SegmentCandidatesResponse["segments"]> = {},
  confirmedOverrides: Partial<ConfirmedSiteBinding> = {},
): SegmentCandidatesResponse {
  return {
    site: makeSite(),
    confirmed: makeConfirmed(confirmedOverrides),
    segments: {
      observation_area_id: AREA,
      junction_id: "J1",
      junction_name: "North West junction",
      network_id: "example-grid",
      network_version: "1.0.0",
      candidates: [
        segment("SEG-12", "outgoing", "J2", 0),
        segment("SEG-14", "outgoing", "J4", 90),
        segment("SEG-21", "incoming", "J2", 0, "J1-A-SEG-21"),
        segment("SEG-41", "incoming", "J4", 90, "J1-A-SEG-41"),
      ],
      site_bearing_degrees: 90,
      bearing_evidence_available: true,
      selected_segment_id: null,
      requires_human_confirmation: true,
      provenance: PROVENANCE,
      warnings: [
        "Bearing cannot separate SEG-12 and SEG-21. These run along the same line from the junction.",
      ],
      note: "4 directed segment(s) meet junction J1.",
      ...overrides,
    },
    note: "Junction J1 is confirmed; no segment has been chosen yet.",
  };
}

function makeCandidateResponse(
  confirmedOverrides: Partial<ConfirmedSiteBinding> = {},
): CandidateResponse {
  return {
    site: makeSite(),
    candidate: {
      observation_area_id: AREA,
      network_id: "example-grid",
      network_version: "1.0.0",
      status: "candidate_found",
      nearest: {
        junction_id: "J1",
        junction_name: "North West junction",
        distance_m: 15.4,
        within_site_accuracy: false,
      },
      runners_up: [],
      max_distance_m: 250,
      site_accuracy_m: null,
      positioned_junction_count: 3,
      unpositioned_junction_ids: [],
      is_discriminating: true,
      requires_human_confirmation: true,
      provenance: PROVENANCE,
      warnings: [],
      note: "Nearest junction J1 is 15.4 m from the site.",
    },
    confirmed: makeConfirmed(confirmedOverrides),
    note: "This area is already bound to junction J1.",
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function success<T>(data: T): Resource<T> {
  return { state: "success", data };
}

function stubApi(
  options: {
    segments?: SegmentCandidatesResponse;
    confirmedOverrides?: Partial<ConfirmedSiteBinding>;
    segmentConfirmError?: { code: string; message: string; status: number };
  } = {},
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.includes("/binding/segment") && init?.method === "POST") {
      if (options.segmentConfirmError !== undefined) {
        return jsonResponse(
          {
            error: {
              code: options.segmentConfirmError.code,
              message: options.segmentConfirmError.message,
              details: {},
            },
          },
          options.segmentConfirmError.status,
        );
      }
      return jsonResponse(
        {
          confirmed: makeConfirmed({
            segment_id: "SEG-21",
            segment_direction: "incoming",
          }),
          binding: {
            observation_area_id: AREA,
            segment_id: "SEG-21",
            junction_id: "J1",
            approach_id: null,
            coverage_note: "x",
            confirmation: null,
          },
          replaced_existing: false,
          note: "Observation area is bound to segment SEG-21 (incoming at J1). A confirmed binding does not change any route cost.",
        },
        201,
      );
    }
    if (url.includes("/effective-mapping")) {
      // The panel now embeds the effective-mapping summary, which reads this
      // on mount. Served here so these tests stay about the binding flow.
      return jsonResponse(
        {
          error: {
            code: "network_not_loaded",
            message: "No road network is loaded.",
            details: {},
          },
        },
        404,
      );
    }
    if (url.includes("/segments")) {
      return jsonResponse(
        options.segments ?? makeSegments({}, options.confirmedOverrides ?? {}),
      );
    }
    if (url.includes("/candidate")) {
      return jsonResponse(makeCandidateResponse(options.confirmedOverrides ?? {}));
    }
    return jsonResponse({ sites: [makeSite()], count: 1, note: "" });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function openSegmentStage(): Promise<void> {
  await waitFor(() =>
    expect(screen.getByLabelText("Observation site")).toBeInTheDocument(),
  );
  await userEvent.selectOptions(screen.getByLabelText("Observation site"), AREA);
  await waitFor(() =>
    expect(screen.getByTestId("segment-stage")).toBeInTheDocument(),
  );
}

describe("SiteBindingPanel — segment stage", () => {
  it("shows the confirmed junction the segments belong to", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    const heading = screen.getByTestId("segment-junction");
    expect(heading).toHaveTextContent("J1");
    expect(heading).toHaveTextContent("North West junction");
    expect(heading).toHaveTextContent("surveyor.junction");
  });

  it("lists every directed segment with its direction and far end", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    const outgoing = screen.getByTestId("segment-SEG-12");
    expect(outgoing).toHaveTextContent("outgoing");
    expect(outgoing).toHaveTextContent("toward J2");

    const incoming = screen.getByTestId("segment-SEG-21");
    expect(incoming).toHaveTextContent("incoming");
    expect(incoming).toHaveTextContent("from J2");
    expect(incoming).toHaveTextContent("J1-A-SEG-21");
  });

  it("shows the bearing offset as evidence beside each segment", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    expect(screen.getByTestId("segment-SEG-12")).toHaveTextContent("0° off camera");
    expect(screen.getByTestId("segment-SEG-14")).toHaveTextContent("90° off camera");
  });

  it("does not order the list by bearing", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    const rendered = screen
      .getAllByTestId(/^segment-SEG-/)
      .map((node) => node.getAttribute("data-testid"));
    expect(rendered).toEqual([
      "segment-SEG-12",
      "segment-SEG-14",
      "segment-SEG-21",
      "segment-SEG-41",
    ]);
  });

  it("pre-selects no segment, not even the best-aligned one", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    for (const id of ["SEG-12", "SEG-14", "SEG-21", "SEG-41"]) {
      const row = screen.getByTestId(`segment-${id}`);
      expect(within(row).getByRole("radio")).not.toBeChecked();
    }
    expect(screen.getByRole("button", { name: /confirm segment/i })).toBeDisabled();
  });

  it("surfaces that bearing cannot separate the two directions of one road", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    expect(
      screen.getByText(/Bearing cannot separate SEG-12 and SEG-21/),
    ).toBeInTheDocument();
  });

  it("says when there is no bearing evidence at all", async () => {
    stubApi({
      segments: makeSegments({
        site_bearing_degrees: null,
        bearing_evidence_available: false,
        candidates: [segment("SEG-12", "outgoing", "J2", null)],
        warnings: [],
      }),
    });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    expect(screen.getByTestId("no-bearing-evidence")).toBeInTheDocument();
    expect(screen.getByTestId("segment-SEG-12")).toHaveTextContent(
      "no bearing evidence",
    );
  });

  it("states that bearing never decides the direction of travel", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    expect(
      screen.getByText(/never which direction of travel it watches/i),
    ).toBeInTheDocument();
  });

  it("states that a confirmed segment changes no route cost", async () => {
    stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    expect(
      screen.getByText(/does not change any route cost/i),
    ).toBeInTheDocument();
  });

  it("creates nothing while the operator is only looking", async () => {
    const fetchMock = stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();
    await userEvent.click(
      within(screen.getByTestId("segment-SEG-21")).getByRole("radio"),
    );

    const posted = fetchMock.mock.calls.filter(
      (call) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(posted).toHaveLength(0);
  });

  it("sends an explicit confirmation naming the junction and the segment", async () => {
    const fetchMock = stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    await userEvent.click(
      within(screen.getByTestId("segment-SEG-21")).getByRole("radio"),
    );
    await userEvent.click(screen.getByRole("button", { name: /confirm segment/i }));

    await waitFor(() =>
      expect(screen.getByTestId("confirm-outcome")).toBeInTheDocument(),
    );

    const posted = fetchMock.mock.calls.find(
      (call) =>
        (call[1] as RequestInit | undefined)?.method === "POST" &&
        String(call[0]).includes("/binding/segment"),
    );
    expect(posted).toBeDefined();
    const body = JSON.parse(String((posted?.[1] as RequestInit).body)) as {
      junction_id: string;
      segment_id: string;
      confirm: boolean;
    };
    expect(body.confirm).toBe(true);
    expect(body.junction_id).toBe("J1");
    expect(body.segment_id).toBe("SEG-21");
  });

  it("sends the written reason when one is given", async () => {
    const fetchMock = stubApi();
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    await userEvent.click(
      within(screen.getByTestId("segment-SEG-14")).getByRole("radio"),
    );
    await userEvent.type(
      screen.getByLabelText(/Why this segment/i),
      "the lens is angled down the south arm",
    );
    await userEvent.click(screen.getByRole("button", { name: /confirm segment/i }));

    await waitFor(() => {
      const posted = fetchMock.mock.calls.find(
        (call) =>
          (call[1] as RequestInit | undefined)?.method === "POST" &&
          String(call[0]).includes("/binding/segment"),
      );
      const body = JSON.parse(String((posted?.[1] as RequestInit).body)) as {
        notes?: string;
      };
      expect(body.notes).toBe("the lens is angled down the south arm");
    });
  });

  it("shows the segment already on record and that confirming replaces it", async () => {
    stubApi({
      confirmedOverrides: { segment_id: "SEG-21", segment_direction: "incoming" },
    });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    const existing = screen.getByTestId("existing-segment");
    expect(existing).toHaveTextContent("SEG-21");
    expect(existing).toHaveTextContent("incoming");
    expect(existing).toHaveTextContent(/replaces that/i);
    expect(
      within(screen.getByTestId("segment-SEG-21")).getByRole("radio"),
    ).toBeChecked();
  });

  it("surfaces a refused segment confirmation instead of reporting success", async () => {
    stubApi({
      segmentConfirmError: {
        code: "validation_failed",
        message:
          "The camera's bearing points 90 degrees away from segment 'SEG-14'.",
        status: 422,
      },
    });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await openSegmentStage();

    await userEvent.click(
      within(screen.getByTestId("segment-SEG-14")).getByRole("radio"),
    );
    await userEvent.click(screen.getByRole("button", { name: /confirm segment/i }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/90 degrees away/),
    );
    expect(screen.queryByTestId("confirm-outcome")).not.toBeInTheDocument();
  });

  it("does not show the segment stage before a junction is confirmed", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (url.includes("/candidate")) {
        const body = makeCandidateResponse();
        return jsonResponse({ ...body, confirmed: null });
      }
      return jsonResponse({ sites: [makeSite()], count: 1, note: "" });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<SiteBindingPanel network={success(NETWORK)} />);
    await waitFor(() =>
      expect(screen.getByLabelText("Observation site")).toBeInTheDocument(),
    );
    await userEvent.selectOptions(screen.getByLabelText("Observation site"), AREA);
    await waitFor(() =>
      expect(screen.getByTestId("candidate-result")).toBeInTheDocument(),
    );

    expect(screen.queryByTestId("segment-stage")).not.toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter((call) => String(call[0]).includes("/segments")),
    ).toHaveLength(0);
  });
});
