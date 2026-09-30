/**
 * Traffic Analysis section tests.
 *
 * Fixtures use distinctive values (0.1229 diagonals/s, 272.7 arrivals/min,
 * track #77) that no default could produce, so a hard-coded number could not
 * satisfy these assertions.
 *
 * The properties under test are the honesty ones: unavailable measurements are
 * shown with their reason rather than as zero, image-space values are labelled,
 * and no traffic-management conclusion is ever rendered.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { TrafficAnalysisSection } from "./TrafficAnalysis";
import type { MetricValue, TrafficAnalytics } from "../api/types";

const MODEL = {
  name: "yolov8n.pt",
  backend: "ultralytics",
  file_name: "yolov8n.pt",
  sha256: "f59b3d833e2f" + "0".repeat(52),
  size_bytes: 6_549_796,
  task: null,
  class_count: null,
  source: "ultralytics release asset",
};

function metric(value: number, unit: string, imageSpace = false): MetricValue {
  return {
    availability: "available",
    value,
    unit,
    reason: null,
    method: "test",
    confidence: null,
    is_image_space: imageSpace,
  };
}

function unavailable(
  availability: MetricValue["availability"],
  reason: string,
): MetricValue {
  return {
    availability,
    value: null,
    unit: null,
    reason,
    method: null,
    confidence: null,
    is_image_space: false,
  };
}

const NO_CALIBRATION = "Physical speed unavailable — camera calibration not configured.";

const SEQUENCE: TrafficAnalytics = {
  analytics_id: "an-1",
  mode: "sequence",
  observation_area_id: "observation-area:vid-1",
  provenance: {
    media_id: "vid-1",
    source_id: "video:vid-1",
    frame_index: null,
    frame_range: [0, 11],
    timestamp: null,
    time_range: ["2024-03-14T08:30:00Z", "2024-03-14T08:30:01Z"],
    model: MODEL,
    tracker: "bytetrack-lite",
    method: "image-space-traffic-analytics",
    method_version: "1.0",
    analytics_schema_version: "0.1.0",
    computed_at: "2024-03-14T08:30:02Z",
  },
  quality: {
    grade: "degraded",
    frames_analysed: 12,
    mean_detection_confidence: 0.7412,
    low_confidence_fraction: 0.1,
    tracks_total: 5,
    tracks_confirmed: 5,
    suspected_id_switches: 0,
    calibration_available: false,
    road_geometry_available: false,
    sampling_interval_s: 0.1,
    warnings: ["No camera calibration: all measurements are image-space only."],
  },
  frame_width: 405,
  frame_height: 540,
  detection_tally: {
    by_class: { person: 43, bus: 15 },
    total: 58,
    vehicles: 15,
    pedestrians: 43,
    cyclists: 0,
  },
  tracked_object_tally: {
    by_class: { person: 3, bus: 2 },
    total: 5,
    vehicles: 2,
    pedestrians: 3,
    cyclists: 0,
  },
  active_tracked_objects: { ...metric(4, "objects"), is_image_space: false },
  object_positions: [],
  spatial_distribution: {
    centroid: { x: 200, y: 500, x_norm: 0.49, y_norm: 0.93 },
    spread_x_norm: metric(0.21, "fraction_of_width", true),
    spread_y_norm: metric(0.04, "fraction_of_height", true),
    grid_rows: 3,
    grid_cols: 3,
    grid_counts: [
      [0, 0, 0],
      [0, 0, 0],
      [2, 2, 1],
    ],
    image_occupancy_ratio: metric(0.553, "fraction_of_frame_area", true),
  },
  trajectories: [
    {
      track_id: 77,
      road_user_class: "bus",
      frames_tracked: 12,
      first_frame_index: 0,
      last_frame_index: 11,
      duration_s: metric(1.1, "s"),
      total_displacement_px: metric(81.8, "px", true),
      net_displacement_px: metric(81.8, "px", true),
      mean_step_displacement_px: metric(7.4, "px", true),
      image_speed_px_per_s: metric(74.4, "px/s", true),
      image_speed_diagonals_per_s: metric(0.1103, "frame_diagonals/s", true),
      direction_deg: metric(0, "degrees", true),
      direction_label: {
        availability: "available",
        value: "right",
        unit: null,
        reason: null,
        method: "eight_sector_image_space",
        confidence: null,
        is_image_space: true,
      },
      is_stationary: false,
      stationary_frames: 0,
      id_switch_suspected: false,
      physical_speed_mps: unavailable("unavailable_no_calibration", NO_CALIBRATION),
      physical_speed_kph: unavailable("unavailable_no_calibration", NO_CALIBRATION),
    },
  ],
  mean_image_speed_diagonals_per_s: metric(0.1229, "frame_diagonals/s", true),
  direction_histogram: { right: 4, down: 1 },
  mean_trajectory_duration_s: metric(0.82, "s"),
  flow_series: [{ window_start_s: 0, window_end_s: 5, arrivals: 5, active: 5 }],
  arrival_rate_per_minute: metric(272.7, "objects/min"),
  stationary_vehicle_count: { ...metric(0, "vehicles"), is_image_space: false },
  queue_groups: [],
  total_queue_length_m: unavailable(
    "unavailable_no_calibration",
    "Total queue length in metres requires camera calibration.",
  ),
  traffic_density_veh_per_km_per_lane: unavailable(
    "unavailable_no_geometry",
    "Traffic density requires surveyed road geometry (length of road in view and lane count).",
  ),
  mean_speed_kph: unavailable("unavailable_no_calibration", NO_CALIBRATION),
};

const SNAPSHOT: TrafficAnalytics = {
  ...SEQUENCE,
  analytics_id: "an-2",
  mode: "snapshot",
  observation_area_id: "observation-area:img-1",
  provenance: {
    ...SEQUENCE.provenance,
    media_id: "img-1",
    source_id: "image:img-1",
    frame_index: 0,
    frame_range: null,
    tracker: null,
  },
  detection_tally: { by_class: { person: 4, bus: 1 }, total: 5, vehicles: 1, pedestrians: 4, cyclists: 0 },
  tracked_object_tally: { by_class: {}, total: 0, vehicles: 0, pedestrians: 0, cyclists: 0 },
  active_tracked_objects: unavailable(
    "unavailable_not_applicable",
    "Tracking requires a sequence of frames; a single image has no temporal information.",
  ),
  trajectories: [],
  mean_image_speed_diagonals_per_s: unavailable(
    "unavailable_insufficient_data",
    "No trajectory had enough points to measure speed.",
  ),
  direction_histogram: {},
  mean_trajectory_duration_s: unavailable(
    "unavailable_insufficient_data",
    "No trajectory spanned a measurable interval.",
  ),
  flow_series: [],
  arrival_rate_per_minute: unavailable(
    "unavailable_not_applicable",
    "Flow is a rate over time; a single image spans no interval.",
  ),
  stationary_vehicle_count: unavailable(
    "unavailable_not_applicable",
    "Queue estimation needs persistence across frames: a single image cannot distinguish a stopped vehicle from a moving one.",
  ),
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubFetch(response: Response): void {
  vi.stubGlobal("fetch", vi.fn(async () => response.clone()));
}

async function derive(): Promise<void> {
  await userEvent.click(screen.getByRole("button", { name: /derive measurements/i }));
}

/** Traffic-management conclusions that must never appear at this phase. */
const FORBIDDEN = /congestion level|\bETA\b|recommended route|travel time|should be|advis/i;

// ---------------------------------------------------------------------------
describe("TrafficAnalysisSection labelling", () => {
  it("is labelled Traffic Analysis", () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    expect(screen.getByText("Traffic Analysis")).toBeInTheDocument();
  });

  it("separates detection, measurement and management", () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    const section = screen.getByTestId("traffic-analysis");
    expect(section).toHaveTextContent(/what was detected/i);
    expect(section).toHaveTextContent(/what measurements were derived/i);
    expect(section).toHaveTextContent(/Traffic Management, a later phase/i);
  });

  it("shows an empty state before measurements are derived", () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    expect(screen.getByTestId("analytics-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("analytics-results")).not.toBeInTheDocument();
  });

  it("fabricates no measurement while empty", () => {
    stubFetch(jsonResponse(SNAPSHOT));
    const { container } = render(<TrafficAnalysisSection mediaId="img-1" />);
    expect(container.textContent ?? "").not.toMatch(/\d+\s*(px\/s|objects\/min|km\/h)/);
  });

  it("disables derivation when there is nothing to measure from", () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(
      <TrafficAnalysisSection disabled disabledReason="Run detection and tracking first." />,
    );
    expect(screen.getByRole("button", { name: /derive measurements/i })).toBeDisabled();
    expect(screen.getByText(/run detection and tracking first/i)).toBeInTheDocument();
  });
});

describe("image (snapshot) measurements", () => {
  it("renders counts returned by the API", async () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();

    const tally = await screen.findByTestId("tally-detected-objects");
    expect(within(tally).getByText("5")).toBeInTheDocument();
    expect(within(tally).getByText(/person 4/)).toBeInTheDocument();
    expect(within(tally).getByText(/bus 1/)).toBeInTheDocument();
    expect(within(tally).getByText(/vehicles 1/)).toBeInTheDocument();
    expect(within(tally).getByText(/pedestrians 4/)).toBeInTheDocument();
  });

  it("renders the spatial distribution grid", async () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();
    expect(await screen.findByTestId("spatial-distribution")).toBeInTheDocument();
  });

  it("shows temporal measurements as unavailable, not as zero", async () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();

    const active = await screen.findByTestId("metric-active-tracked-objects");
    expect(active).toHaveTextContent(/not applicable/i);
    expect(active).toHaveTextContent(/single image has no temporal information/i);
    expect(within(active).queryByText("0")).not.toBeInTheDocument();
  });

  it("omits sequence-only sections for a still image", async () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();
    await screen.findByTestId("analytics-results");
    expect(screen.queryByTestId("direction-histogram")).not.toBeInTheDocument();
    expect(screen.queryByTestId("queue-section")).not.toBeInTheDocument();
  });

  it("reports the tracker as not applicable for a still", async () => {
    stubFetch(jsonResponse(SNAPSHOT));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();
    const provenance = await screen.findByTestId("analytics-provenance");
    expect(provenance).toHaveTextContent(/not applicable \(single frame\)/i);
  });
});

describe("video (sequence) measurements", () => {
  it("distinguishes detections from tracked objects", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const detections = await screen.findByTestId("tally-detected-objects");
    const tracked = screen.getByTestId("tally-tracked-objects");
    expect(within(detections).getByText("58")).toBeInTheDocument();
    expect(within(tracked).getByText("5")).toBeInTheDocument();
    expect(detections).toHaveTextContent(/once per frame it is visible in/i);
  });

  it("renders movement directions as image-space, not compass", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const histogram = await screen.findByTestId("direction-histogram");
    expect(histogram).toHaveTextContent(/right 4/);
    expect(histogram).toHaveTextContent(/image space/i);
    expect(histogram).toHaveTextContent(/not compass bearings/i);
  });

  it("renders trajectory statistics from the API", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const row = await screen.findByTestId("trajectory-77");
    expect(row).toHaveTextContent(/#77 bus/);
    expect(row).toHaveTextContent(/frames 0–11/);
    expect(row).toHaveTextContent("74.4 px/s");
  });

  it("labels relative speed as image-space", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const speed = await screen.findByTestId("metric-mean-movement-speed");
    expect(speed).toHaveTextContent("0.123 frame_diagonals/s");
    expect(within(speed).getByText(/image-space/i)).toBeInTheDocument();
  });

  it("renders the arrival rate from the API", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();
    expect(await screen.findByTestId("metric-arrival-rate")).toHaveTextContent(
      "273 objects/min",
    );
  });

  it("shows queue evidence and explains what a queue is", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const queue = await screen.findByTestId("queue-section");
    expect(queue).toHaveTextContent(/movement, persistence and proximity/i);
    expect(queue).toHaveTextContent(/never from a headcount/i);
    expect(queue).toHaveTextContent(/requires camera calibration/i);
  });
});

describe("unavailable measurements", () => {
  it("shows the calibration message verbatim for physical speed", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const speed = await screen.findByTestId("metric-physical-speed");
    expect(speed).toHaveTextContent(NO_CALIBRATION);
    expect(speed).toHaveTextContent(/needs calibration/i);
  });

  it("never renders km/h when calibration is absent", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    const { container } = render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();
    await screen.findByTestId("analytics-results");
    expect(container.textContent ?? "").not.toMatch(/\d+(\.\d+)?\s*km\/h/);
  });

  it("distinguishes a missing-geometry reason from a missing-calibration one", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const density = await screen.findByTestId("metric-traffic-density");
    expect(density).toHaveTextContent(/needs road geometry/i);
    expect(density).toHaveTextContent(/surveyed road geometry/i);
  });

  it("labels image occupancy as not being traffic density", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();
    expect(await screen.findByTestId("metric-image-occupancy")).toHaveTextContent(
      /not traffic density/i,
    );
  });
});

describe("quality and provenance", () => {
  it("renders provenance from the API", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const provenance = await screen.findByTestId("analytics-provenance");
    expect(provenance).toHaveTextContent("video:vid-1");
    expect(provenance).toHaveTextContent("bytetrack-lite");
    expect(provenance).toHaveTextContent("f59b3d833e2f");
    expect(provenance).toHaveTextContent("image-space-traffic-analytics v1.0");
    expect(provenance).toHaveTextContent(/0–11/);
    expect(provenance).toHaveTextContent("74.1%");
  });

  it("renders the quality grade and warnings", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    expect(await screen.findByText("degraded")).toBeInTheDocument();
    expect(screen.getByTestId("quality-warnings")).toHaveTextContent(
      /image-space only/i,
    );
  });
});

describe("error handling", () => {
  it("shows the real error and no measurements", async () => {
    stubFetch(
      jsonResponse(
        {
          error: {
            code: "validation_failed",
            message: "Job 'job-1' is running; analytics require a finished analysis.",
            details: {},
          },
        },
        422,
      ),
    );
    render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();

    const error = await screen.findByTestId("analytics-error");
    expect(error).toHaveTextContent(/require a finished analysis/i);
    expect(screen.queryByTestId("analytics-results")).not.toBeInTheDocument();
  });

  it("fabricates nothing when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();

    expect(await screen.findByTestId("analytics-error")).toBeInTheDocument();
    expect(screen.queryByTestId("analytics-results")).not.toBeInTheDocument();
  });
});

describe("phase boundary", () => {
  it("renders no traffic-management conclusion for a sequence", async () => {
    stubFetch(jsonResponse(SEQUENCE));
    const { container } = render(<TrafficAnalysisSection jobId="job-1" />);
    await derive();
    await screen.findByTestId("analytics-results");
    expect(container.textContent ?? "").not.toMatch(FORBIDDEN);
  });

  it("renders no traffic-management conclusion for a snapshot", async () => {
    stubFetch(jsonResponse(SNAPSHOT));
    const { container } = render(<TrafficAnalysisSection mediaId="img-1" />);
    await derive();
    await screen.findByTestId("analytics-results");
    expect(container.textContent ?? "").not.toMatch(FORBIDDEN);
  });
});
