/**
 * Site-to-junction binding panel tests.
 *
 * The assertions are mostly about restraint. A nearest-junction result must
 * render as a proposal and never as a fact; selecting a site must not create
 * anything; the only request that can create a binding must carry an explicit
 * confirmation; and an override of the geometry must not be submittable without
 * a reason. A panel that quietly bound a camera to whatever sorted first would
 * defeat the entire point of computing the candidate.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SiteBindingPanel } from "./SiteBindingPanel";
import type {
  CandidateResponse,
  ConfirmedSiteBinding,
  JunctionCandidate,
  ObservationSite,
  Provenance,
  RoadNetwork,
  SegmentCandidatesResponse,
  SiteBindingCandidate,
} from "../api/types";
import type { Resource } from "../hooks/useApiResource";
import { makeNetwork, makeJunction, makeSegment } from "../test/fixtures";

afterEach(() => {
  vi.restoreAllMocks();
});

const AREA = "observation-area:cam-1";

const NETWORK: RoadNetwork = makeNetwork({
  network_id: "example-grid",
  junctions: [makeJunction("J1"), makeJunction("J2"), makeJunction("J3")],
  segments: [makeSegment("SEG-12", "J1", "J2")],
});

const PROVENANCE: Provenance = {
  source: "derived",
  producer: "sites.nearest_junction",
  producer_version: "0.1.0",
  observed_at: "2026-09-17T08:00:00Z",
  computed_at: "2026-09-17T08:00:00Z",
  confidence: null,
  input_refs: [`observation_site:${AREA}`, "network:example-grid@1.0.0"],
  notes: null,
};

function makeSite(overrides: Partial<ObservationSite> = {}): ObservationSite {
  return {
    observation_area_id: AREA,
    location: { latitude: 16.5051, longitude: 80.6001 },
    bearing_degrees: null,
    accuracy_m: null,
    label: "North West mast",
    provenance: { ...PROVENANCE, source: "observed", producer: "operator" },
    recorded_at: "2026-09-17T08:00:00Z",
    updated_at: "2026-09-17T08:00:00Z",
    ...overrides,
  };
}

function junction(
  id: string,
  distance: number,
  withinAccuracy = false,
): JunctionCandidate {
  return {
    junction_id: id,
    junction_name: `Junction ${id}`,
    distance_m: distance,
    within_site_accuracy: withinAccuracy,
  };
}

function makeCandidate(
  overrides: Partial<SiteBindingCandidate> = {},
): SiteBindingCandidate {
  return {
    observation_area_id: AREA,
    network_id: "example-grid",
    network_version: "1.0.0",
    status: "candidate_found",
    nearest: junction("J1", 15.4),
    runners_up: [junction("J2", 522.5)],
    max_distance_m: 250,
    site_accuracy_m: null,
    positioned_junction_count: 3,
    unpositioned_junction_ids: [],
    is_discriminating: true,
    requires_human_confirmation: true,
    provenance: PROVENANCE,
    warnings: [],
    note: "Nearest junction J1 is 15.4 m from the site. This is proximity evidence only: nothing is bound until an operator confirms it.",
    ...overrides,
  };
}

function makeResponse(
  overrides: Partial<CandidateResponse> = {},
): CandidateResponse {
  return {
    site: makeSite(),
    candidate: makeCandidate(),
    confirmed: null,
    note: "Nothing is bound yet. This is a proposal; it becomes a binding only when an operator confirms it by naming the junction.",
    ...overrides,
  };
}

function makeConfirmed(): ConfirmedSiteBinding {
  return {
    observation_area_id: AREA,
    junction_id: "J1",
    network_id: "example-grid",
    network_version: "1.0.0",
    site_location: { latitude: 16.5051, longitude: 80.6001 },
    confirmation: {
      confirmed_by: "surveyor.a",
      confirmed_at: "2026-09-17T08:05:00Z",
      method: "accepted_nearest",
      candidate_distance_m: 15.4,
      site_accuracy_m: null,
      was_nearest_candidate: true,
      notes: null,
    },
    provenance: { ...PROVENANCE, source: "observed", producer: "surveyor.a" },
    segment_id: null,
    segment_direction: null,
    approach_id: null,
    segment_confirmation: null,
  };
}

/**
 * A segment stage with nothing selected.
 *
 * Present in the stub because a candidate that reports an existing junction
 * binding makes the panel fetch the segment stage next. Serving it here keeps
 * these tests about the junction step.
 */
function makeSegments(): SegmentCandidatesResponse {
  return {
    site: makeSite(),
    confirmed: makeConfirmed(),
    segments: {
      observation_area_id: AREA,
      junction_id: "J1",
      junction_name: "Junction J1",
      network_id: "example-grid",
      network_version: "1.0.0",
      candidates: [],
      site_bearing_degrees: null,
      bearing_evidence_available: false,
      selected_segment_id: null,
      requires_human_confirmation: true,
      provenance: PROVENANCE,
      warnings: [],
      note: "no segments",
    },
    note: "Junction J1 is confirmed; no segment has been chosen yet.",
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

/**
 * Route stubbed fetches by path, so a test asserts on the request the panel
 * actually made rather than on call ordering.
 */
function stubApi(
  handlers: { sites?: ObservationSite[]; candidate?: CandidateResponse; confirm?: unknown },
  confirmStatus = 201,
): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.includes("/candidate")) {
      return jsonResponse(handlers.candidate ?? makeResponse());
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
      return jsonResponse(makeSegments());
    }
    if (url.includes("/binding") && init?.method === "POST") {
      return jsonResponse(
        handlers.confirm ?? {
          confirmed: makeConfirmed(),
          binding: {
            observation_area_id: AREA,
            segment_id: null,
            junction_id: "J1",
            approach_id: null,
            coverage_note: "Confirmed by operator.dashboard",
            confirmation: makeConfirmed().confirmation,
          },
          replaced_existing: false,
          note: "Observation area is bound to junction J1 on network 'example-grid'.",
        },
        confirmStatus,
      );
    }
    return jsonResponse({
      sites: handlers.sites ?? [makeSite()],
      count: (handlers.sites ?? [makeSite()]).length,
      note: "",
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function selectSite(): Promise<void> {
  await waitFor(() =>
    expect(screen.getByLabelText("Observation site")).toBeInTheDocument(),
  );
  await userEvent.selectOptions(screen.getByLabelText("Observation site"), AREA);
  await waitFor(() =>
    expect(screen.getByTestId("candidate-result")).toBeInTheDocument(),
  );
}

describe("SiteBindingPanel", () => {
  it("says a network is needed rather than rendering an empty binding form", () => {
    stubApi({});
    render(<SiteBindingPanel network={{ state: "loading" }} />);
    expect(screen.getByText(/No road network is loaded/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /confirm/i })).not.toBeInTheDocument();
  });

  it("says no site has been recorded rather than showing an empty picker", async () => {
    stubApi({ sites: [] });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await waitFor(() =>
      expect(
        screen.getByText(/No observation site has been recorded/i),
      ).toBeInTheDocument(),
    );
  });

  it("renders the selected site, the nearest junction and the distance", async () => {
    stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    expect(screen.getByTestId("candidate-result")).toHaveTextContent(
      "North West mast",
    );
    expect(screen.getByTestId("candidate-result")).toHaveTextContent("16.50510");
    const nearest = screen.getByTestId("nearest-junction");
    expect(nearest).toHaveTextContent("J1");
    expect(nearest).toHaveTextContent("15.4 m");
  });

  it("presents the candidate as a proposal, not as a binding", async () => {
    stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    expect(screen.getByText(/Nothing is bound until you confirm/i)).toBeInTheDocument();
    expect(
      screen.getByText(/proximity evidence, not a statement/i),
    ).toBeInTheDocument();
  });

  it("does not create a binding merely by evaluating a candidate", async () => {
    const fetchMock = stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    const posted = fetchMock.mock.calls.filter(
      (call) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(posted).toHaveLength(0);
  });

  it("shows the runner-up junctions so the operator can see how close it was", async () => {
    stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    expect(screen.getByTestId("candidate-J1")).toHaveTextContent("15.4 m");
    expect(screen.getByTestId("candidate-J2")).toHaveTextContent("523 m");
  });

  it("says when the ranking is decided by noise rather than geography", async () => {
    stubApi({
      candidate: makeResponse({
        site: makeSite({ accuracy_m: 100 }),
        candidate: makeCandidate({
          site_accuracy_m: 100,
          is_discriminating: false,
          nearest: junction("J1", 15.4, true),
          warnings: ["The site is 15.4 m from J1, inside its own 100.0 m position accuracy."],
        }),
      }),
    });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    expect(screen.getByTestId("not-discriminating")).toBeInTheDocument();
    expect(screen.getByText(/inside its own 100.0 m position accuracy/)).toBeInTheDocument();
    expect(screen.getByTestId("candidate-J1")).toHaveTextContent(
      /within position accuracy/i,
    );
  });

  it("reports an out-of-range candidate with its distance and offers no assurance", async () => {
    stubApi({
      candidate: makeResponse({
        candidate: makeCandidate({
          status: "out_of_range",
          nearest: junction("J1", 4210),
          note: "Nearest junction J1 is 4210.0 m from the site, outside the 250 m evidentiary range.",
          warnings: ["The nearest junction is 4210.0 m away, beyond the 250 m range."],
        }),
        note: "Nearest junction J1 is 4210.0 m from the site, outside the 250 m evidentiary range.",
      }),
    });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    expect(screen.getByText(/out of range/i)).toBeInTheDocument();
    expect(screen.getByTestId("nearest-junction")).toHaveTextContent("4210 m");
    expect(screen.getByText(/outside the 250 m evidentiary range/)).toBeInTheDocument();
  });

  it("says an unsurveyed map cannot answer the question", async () => {
    stubApi({
      candidate: makeResponse({
        candidate: makeCandidate({
          status: "no_positioned_junction",
          nearest: null,
          runners_up: [],
          positioned_junction_count: 0,
          unpositioned_junction_ids: ["J1", "J2", "J3"],
        }),
      }),
    });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    expect(
      screen.getByText(/No junction on this map carries a surveyed position/i),
    ).toBeInTheDocument();
  });

  it("sends an explicit confirmation naming the junction", async () => {
    const fetchMock = stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    await userEvent.click(screen.getByRole("button", { name: /confirm binding/i }));

    await waitFor(() => expect(screen.getByTestId("confirm-outcome")).toBeInTheDocument());

    const posted = fetchMock.mock.calls.find(
      (call) => (call[1] as RequestInit | undefined)?.method === "POST",
    );
    expect(posted).toBeDefined();
    expect(String(posted?.[0])).toContain(encodeURIComponent(AREA));
    expect(String(posted?.[0])).toContain("/binding");
    const body = JSON.parse(String((posted?.[1] as RequestInit).body)) as {
      junction_id: string;
      confirm: boolean;
    };
    expect(body.confirm).toBe(true);
    expect(body.junction_id).toBe("J1");
  });

  it("will not submit an override of the nearest junction without a reason", async () => {
    stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    await userEvent.selectOptions(screen.getByLabelText("Junction to confirm"), "J3");

    const button = screen.getByRole("button", { name: /confirm binding/i });
    expect(button).toBeDisabled();

    await userEvent.type(
      screen.getByLabelText(/Why this junction rather than the nearest/i),
      "the camera faces south across the flyover",
    );
    expect(button).toBeEnabled();
  });

  it("sends the reason with an override", async () => {
    const fetchMock = stubApi({});
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    await userEvent.selectOptions(screen.getByLabelText("Junction to confirm"), "J3");
    await userEvent.type(
      screen.getByLabelText(/Why this junction rather than the nearest/i),
      "faces south",
    );
    await userEvent.click(screen.getByRole("button", { name: /confirm binding/i }));

    await waitFor(() => {
      const posted = fetchMock.mock.calls.find(
        (call) => (call[1] as RequestInit | undefined)?.method === "POST",
      );
      expect(posted).toBeDefined();
      const body = JSON.parse(String((posted?.[1] as RequestInit).body)) as {
        junction_id: string;
        notes?: string;
      };
      expect(body.junction_id).toBe("J3");
      expect(body.notes).toBe("faces south");
    });
  });

  it("says when confirming again would replace an existing decision", async () => {
    stubApi({ candidate: makeResponse({ confirmed: makeConfirmed() }) });
    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();

    const existing = screen.getByTestId("existing-binding");
    expect(existing).toHaveTextContent("J1");
    expect(existing).toHaveTextContent("surveyor.a");
    expect(existing).toHaveTextContent(/replaces that decision/i);
  });

  it("surfaces a refused confirmation instead of reporting success", async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (url.includes("/candidate")) return jsonResponse(makeResponse());
      if (url.includes("/binding") && init?.method === "POST") {
        return jsonResponse(
          {
            error: {
              code: "validation_failed",
              message: "Junction 'J3' is not the nearest to this site.",
              details: {},
            },
          },
          422,
        );
      }
      return jsonResponse({ sites: [makeSite()], count: 1, note: "" });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<SiteBindingPanel network={success(NETWORK)} />);
    await selectSite();
    await userEvent.click(screen.getByRole("button", { name: /confirm binding/i }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/not the nearest/i),
    );
    expect(screen.queryByTestId("confirm-outcome")).not.toBeInTheDocument();
  });
});
