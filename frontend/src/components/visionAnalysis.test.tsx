/**
 * Computer Vision Analysis panel tests.
 *
 * Fixtures use distinctive values (confidence 0.873, track #77, sha 3f9c…) that
 * no default or placeholder would produce, so a hard-coded value could not
 * satisfy these assertions.
 *
 * The properties under test are the honesty ones: results come from the API, an
 * error is shown as an error rather than as "nothing found", and no traffic
 * conclusion is ever rendered.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";

import { ImageVisionAnalysis, VideoVisionAnalysis } from "./VisionAnalysis";
import type {
  DetectorStatus,
  ImageAnalysis,
  InferenceMetadata,
  TrackListResponse,
  VideoAnalysisSummary,
} from "../api/types";

const INFERENCE: InferenceMetadata = {
  backend: "ultralytics",
  model: {
    name: "yolov8n.pt",
    backend: "ultralytics",
    file_name: "yolov8n.pt",
    sha256: "3f9c" + "0".repeat(60),
    size_bytes: 6_549_796,
    task: null,
    class_count: null,
    source: "ultralytics release asset",
  },
  device: "cpu",
  device_kind: "cpu",
  device_reason: "No CUDA device detected: torch reports no available CUDA device",
  image_size: 640,
  confidence_threshold: 0.25,
  iou_threshold: 0.45,
  tracker: "bytetrack-lite",
};

const IMAGE_ANALYSIS: ImageAnalysis = {
  analysis_id: "analysis-fixture-1",
  media_id: "img-1",
  source_id: "image:img-1",
  analysed_at: "2024-03-14T08:30:00Z",
  inference: INFERENCE,
  detections: [
    {
      road_user_class: "bus",
      confidence: 0.873,
      bbox: { x1: 22, y1: 229, x2: 805, y2: 750, width: 783, height: 521 },
      track_id: null,
      frame_index: 0,
      timestamp: "2024-03-14T08:30:00Z",
      source_id: "image:img-1",
      quality: "confirmed",
      quality_flags: [],
      quality_reason: null,
      area_fraction: 0.466,
      aspect_ratio: 1.503,
      touches_frame_edge: false,
      duplicate_of_index: null,
    },
    {
      road_user_class: "person",
      confidence: 0.611,
      bbox: { x1: 49, y1: 399, x2: 244, y2: 903, width: 195, height: 504 },
      track_id: null,
      frame_index: 0,
      timestamp: "2024-03-14T08:30:00Z",
      source_id: "image:img-1",
      quality: "confirmed",
      quality_flags: [],
      quality_reason: null,
      area_fraction: 0.112,
      aspect_ratio: 0.387,
      touches_frame_edge: false,
      duplicate_of_index: null,
    },
  ],
  detection_tally: { by_class: { bus: 1, person: 1 }, total: 2, vehicles: 1 },
  confirmed_tally: { by_class: { bus: 1, person: 1 }, total: 2, vehicles: 1 },
  quality_policy: {
    enabled: true,
    confirm_confidence: { car: 0.4, bus: 0.4, truck: 0.35 },
    default_confidence: 0.2,
    max_area_fraction: 0.05,
    min_aspect_ratio: 0.15,
    max_aspect_ratio: 6.0,
    duplicate_iou: 0.55,
    duplicate_containment: 0.7,
    edge_margin_px: 1,
    edge_confidence_margin: 0.1,
  },
  frame_width: 810,
  frame_height: 1080,
  overlay_available: true,
};

const STATUS: DetectorStatus = {
  detection_available: true,
  selected_backend: "ultralytics",
  configured_model: "yolov8s.pt",
  configured_device: "auto",
  model_loaded: true,
  load_error: null,
  backends: [],
  confidence_threshold: 0.2,
  iou_threshold: 0.45,
  image_size: 1280,
  max_detections: 300,
  quality_policy: {
    enabled: true,
    confirm_confidence: { car: 0.4, bus: 0.4, truck: 0.35 },
    default_confidence: 0.2,
    max_area_fraction: 0.05,
    min_aspect_ratio: 0.15,
    max_aspect_ratio: 6.0,
    duplicate_iou: 0.55,
    duplicate_containment: 0.7,
    edge_margin_px: 1,
    edge_confidence_margin: 0.1,
  },
  frame_stride: 1,
  max_frames: 900,
  tracker: "bytetrack-lite",
};

const COMPLETED_JOB: VideoAnalysisSummary = {
  job_id: "job-fixture-1",
  media_id: "vid-1",
  status: "completed",
  created_at: "2024-03-14T08:30:00Z",
  started_at: "2024-03-14T08:30:01Z",
  finished_at: "2024-03-14T08:30:09Z",
  frames_analysed: 143,
  frames_total_estimate: 143,
  frame_stride: 1,
  progress: 1,
  detection_tally: { by_class: { car: 412, person: 55 }, total: 467, vehicles: 412 },
  confirmed_detection_tally: {
    by_class: { car: 380, person: 44 },
    total: 424,
    vehicles: 380,
  },
  track_tally: { by_class: { car: 9, person: 3 }, total: 12, vehicles: 9 },
  unique_track_count: 12,
  suspected_id_switches: 2,
  inference: INFERENCE,
  error: null,
};

const TRACKS: TrackListResponse = {
  job_id: "job-fixture-1",
  count: 1,
  tracks: [
    {
      track_id: 77,
      road_user_class: "car",
      first_frame_index: 4,
      last_frame_index: 96,
      first_seen_at: "2024-03-14T08:30:01Z",
      last_seen_at: "2024-03-14T08:30:08Z",
      frames_tracked: 88,
      is_confirmed: true,
      id_switch_suspected: false,
      mean_confidence: 0.79,
    },
  ],
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubFetch(overrides: Record<string, Response> = {}): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === "string" ? input : input.toString();
      for (const [fragment, response] of Object.entries(overrides)) {
        if (url.includes(fragment)) return response.clone();
      }
      if (url.includes("/analyze") && url.includes("/images/")) {
        return jsonResponse(IMAGE_ANALYSIS);
      }
      if (url.includes("/analyze") && url.includes("/videos/")) {
        return jsonResponse(COMPLETED_JOB, 202);
      }
      if (url.includes("/tracks")) return jsonResponse(TRACKS);
      if (url.includes("/frames")) {
        return jsonResponse({
          job_id: "job-fixture-1",
          frames: [],
          offset: 0,
          limit: 500,
          total: 0,
        });
      }
      if (url.includes("/api/vision/jobs/")) return jsonResponse(COMPLETED_JOB);
      return jsonResponse({ error: { code: "http_error", message: "x", details: {} } }, 404);
    }),
  );
}

/** Traffic conclusions that must never appear at this phase. */
const FORBIDDEN = /congestion|queue length|travel time|recommended route|emergency|\bETA\b/i;

// ---------------------------------------------------------------------------
// Image
// ---------------------------------------------------------------------------
describe("ImageVisionAnalysis", () => {
  it("is labelled Computer Vision Analysis", () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    expect(screen.getByText("Computer Vision Analysis")).toBeInTheDocument();
  });

  it("shows an empty state before analysis is run", () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    expect(screen.getByTestId("analysis-empty")).toBeInTheDocument();
    expect(screen.getByText(/no analysis run yet/i)).toBeInTheDocument();
    expect(screen.queryByTestId("detection-overlay")).not.toBeInTheDocument();
  });

  it("renders detections returned by the API", async () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    const list = await screen.findByTestId("detection-list");
    expect(within(list).getByText(/bus/)).toBeInTheDocument();
    expect(within(list).getByText("87.3%")).toBeInTheDocument();
    expect(within(list).getByText("61.1%")).toBeInTheDocument();
  });

  it("renders the overlay this application generated", async () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    const overlay = await screen.findByTestId("detection-overlay");
    expect(overlay).toHaveAttribute("src", expect.stringContaining("/overlay"));
    // Cache-busted per analysis, or the previous run's boxes would persist.
    expect(overlay).toHaveAttribute("src", expect.stringContaining("analysis-fixture-1"));
  });

  it("shows detection counts from the API tally", async () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    const tally = await screen.findByTestId("detection-tally");
    expect(within(tally).getByText("2")).toBeInTheDocument();
    expect(within(tally).getByText(/bus 1/)).toBeInTheDocument();
    expect(within(tally).getByText(/person 1/)).toBeInTheDocument();
  });

  it("labels counts as detections rather than traffic volume", async () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    const tally = await screen.findByTestId("detection-tally");
    expect(tally).toHaveTextContent(/not a traffic volume/i);
  });

  it("shows the model and device that produced the results", async () => {
    stubFetch();
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    const meta = await screen.findByTestId("inference-metadata");
    expect(within(meta).getByText("yolov8n.pt")).toBeInTheDocument();
    expect(within(meta).getByText("ultralytics")).toBeInTheDocument();
    expect(within(meta).getByText("CPU")).toBeInTheDocument();
    expect(within(meta).getByText("3f9c00000000")).toBeInTheDocument();
    expect(meta).toHaveTextContent(/no cuda device detected/i);
  });

  it("shows the real error and no results when analysis fails", async () => {
    stubFetch({
      "/analyze": jsonResponse(
        {
          error: {
            code: "data_source_error",
            message: "Failed to load model yolov8n.pt on cpu: OSError",
            details: {},
          },
        },
        502,
      ),
    });
    render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    const error = await screen.findByTestId("analysis-error");
    expect(error).toHaveTextContent(/failed to load model/i);
    // A failure must never look like "no vehicles found".
    expect(screen.queryByTestId("detection-list")).not.toBeInTheDocument();
    expect(screen.queryByTestId("detection-tally")).not.toBeInTheDocument();
    expect(screen.queryByTestId("detection-overlay")).not.toBeInTheDocument();
  });

  it("disables analysis when no detector is available", () => {
    stubFetch();
    render(
      <ImageVisionAnalysis
        mediaId="img-1"
        status={{ ...STATUS, detection_available: false, selected_backend: null }}
      />,
    );
    expect(screen.getByRole("button", { name: /run detection/i })).toBeDisabled();
    expect(screen.getByText(/no detector backend is available/i)).toBeInTheDocument();
  });

  it("renders no traffic conclusion", async () => {
    stubFetch();
    const { container } = render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await screen.findByTestId("detection-list");
    expect(container.textContent ?? "").not.toMatch(FORBIDDEN);
  });
});

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------
describe("VideoVisionAnalysis", () => {
  function renderVideo(status: DetectorStatus = STATUS): HTMLElement {
    const ref = createRef<HTMLVideoElement>();
    const { container } = render(
      <VideoVisionAnalysis
        mediaId="vid-1"
        videoRef={ref}
        frameWidth={1280}
        frameHeight={720}
        status={status}
      />,
    );
    return container;
  }

  it("is labelled Computer Vision Analysis", () => {
    stubFetch();
    renderVideo();
    expect(screen.getByText("Computer Vision Analysis")).toBeInTheDocument();
  });

  it("shows an empty state before analysis", () => {
    stubFetch();
    renderVideo();
    expect(screen.getByTestId("analysis-empty")).toBeInTheDocument();
    expect(screen.queryByTestId("track-list")).not.toBeInTheDocument();
  });

  it("reports processed frame information from the API", async () => {
    stubFetch();
    renderVideo();
    await userEvent.click(screen.getByRole("button", { name: /run detection & tracking/i }));

    await waitFor(
      () => expect(screen.getByTestId("processing-info")).toBeInTheDocument(),
      { timeout: 4000 },
    );
    const info = screen.getByTestId("processing-info");
    expect(within(info).getByText("143")).toBeInTheDocument();
    expect(within(info).getByText("2")).toBeInTheDocument();
  });

  it("shows tracked-object counts distinctly from detection counts", async () => {
    stubFetch();
    renderVideo();
    await userEvent.click(screen.getByRole("button", { name: /run detection & tracking/i }));

    await waitFor(() => expect(screen.getByTestId("track-tally")).toBeInTheDocument(), {
      timeout: 4000,
    });
    const trackTally = screen.getByTestId("track-tally");
    const detectionTally = screen.getByTestId("detection-tally");
    expect(within(trackTally).getByText("12")).toBeInTheDocument();
    expect(within(detectionTally).getByText("467")).toBeInTheDocument();
    // The distinction must be stated, not left for the reader to infer.
    expect(detectionTally).toHaveTextContent(/once per frame it is visible in/i);
  });

  it("renders tracks returned by the API with their ids", async () => {
    stubFetch();
    renderVideo();
    await userEvent.click(screen.getByRole("button", { name: /run detection & tracking/i }));

    await waitFor(() => expect(screen.getByTestId("track-list")).toBeInTheDocument(), {
      timeout: 4000,
    });
    const list = screen.getByTestId("track-list");
    expect(within(list).getByText(/#77 car/)).toBeInTheDocument();
    expect(within(list).getByText(/frames 4–96/)).toBeInTheDocument();
  });

  it("surfaces a failed job's real error", async () => {
    const failed: VideoAnalysisSummary = {
      ...COMPLETED_JOB,
      status: "failed",
      error: "DataSourceError: weights are corrupt",
      frames_analysed: 0,
      detection_tally: { by_class: {}, total: 0, vehicles: 0 },
      confirmed_detection_tally: { by_class: {}, total: 0, vehicles: 0 },
      track_tally: { by_class: {}, total: 0, vehicles: 0 },
      unique_track_count: 0,
    };
    stubFetch({ "/api/vision/jobs/": jsonResponse(failed) });
    renderVideo();
    await userEvent.click(screen.getByRole("button", { name: /run detection & tracking/i }));

    await waitFor(() => expect(screen.getByTestId("job-error")).toBeInTheDocument(), {
      timeout: 4000,
    });
    expect(screen.getByTestId("job-error")).toHaveTextContent(/weights are corrupt/i);
    expect(screen.queryByTestId("track-list")).not.toBeInTheDocument();
  });

  it("disables analysis when no detector is available", () => {
    stubFetch();
    renderVideo({ ...STATUS, detection_available: false });
    expect(
      screen.getByRole("button", { name: /run detection & tracking/i }),
    ).toBeDisabled();
  });

  it("renders no traffic conclusion", async () => {
    stubFetch();
    const container = renderVideo();
    await userEvent.click(screen.getByRole("button", { name: /run detection & tracking/i }));
    await waitFor(() => expect(screen.getByTestId("track-list")).toBeInTheDocument(), {
      timeout: 4000,
    });
    expect(container.textContent ?? "").not.toMatch(FORBIDDEN);
  });
});
