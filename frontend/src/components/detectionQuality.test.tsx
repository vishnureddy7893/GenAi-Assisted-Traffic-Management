/**
 * Detection-quality rendering tests.
 *
 * The panel's job here is to stop a reader mistaking a reported box for a
 * counted one. Three things must hold on screen:
 *
 * - an uncountable detection is still listed, so a disputed verdict can be
 *   checked rather than guessed at;
 * - it is visibly marked, and says why;
 * - the headline figure is the countable one, not the raw box count.
 *
 * Every number rendered comes from the fixture response. The panel does no
 * filtering of its own — a second implementation of the quality rules living in
 * the browser is a second implementation that can disagree with the one the
 * measurement layer used.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ImageVisionAnalysis } from "./VisionAnalysis";
import type {
  DetectionOut,
  DetectorStatus,
  ImageAnalysis,
  QualityPolicy,
} from "../api/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const POLICY: QualityPolicy = {
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
};

const STATUS: DetectorStatus = {
  detection_available: true,
  selected_backend: "ultralytics",
  configured_model: "yolov8s.pt",
  configured_device: "cpu",
  model_loaded: true,
  load_error: null,
  backends: [],
  confidence_threshold: 0.2,
  iou_threshold: 0.45,
  image_size: 1280,
  max_detections: 300,
  quality_policy: POLICY,
  frame_stride: 1,
  max_frames: 900,
  tracker: "bytetrack-lite",
};

function detection(overrides: Partial<DetectionOut> = {}): DetectionOut {
  return {
    road_user_class: "car",
    confidence: 0.87,
    bbox: { x1: 100, y1: 100, x2: 160, y2: 150, width: 60, height: 50 },
    track_id: null,
    frame_index: 0,
    timestamp: "2024-03-14T08:30:00Z",
    source_id: "image:img-1",
    quality: "confirmed",
    quality_flags: [],
    quality_reason: null,
    area_fraction: 0.007,
    aspect_ratio: 1.2,
    touches_frame_edge: false,
    duplicate_of_index: null,
    ...overrides,
  };
}

/** A confirmed car, the barrier false positive, and a rejected building. */
const ANALYSIS: ImageAnalysis = {
  analysis_id: "analysis-quality-1",
  media_id: "img-1",
  source_id: "image:img-1",
  analysed_at: "2024-03-14T08:30:00Z",
  inference: {
    backend: "ultralytics",
    model: {
      name: "yolov8s.pt",
      backend: "ultralytics",
      file_name: "yolov8s.pt",
      sha256: "c".repeat(64),
      size_bytes: 22_573_120,
      task: null,
      class_count: 80,
      source: "ultralytics release asset",
    },
    device: "cpu",
    device_kind: "cpu",
    device_reason: "No CUDA device detected",
    image_size: 1280,
    confidence_threshold: 0.2,
    iou_threshold: 0.45,
    tracker: null,
  },
  detections: [
    detection(),
    detection({
      confidence: 0.37,
      bbox: { x1: 261, y1: 362, x2: 302, y2: 392, width: 41, height: 30 },
      quality: "provisional",
      quality_flags: ["below_class_confidence"],
      quality_reason:
        "Confidence 0.37 is below the 0.40 confirmation floor for car. Reported, not counted.",
      area_fraction: 0.0029,
      aspect_ratio: 1.366,
    }),
    detection({
      road_user_class: "bus",
      confidence: 0.37,
      bbox: { x1: 689, y1: 12, x2: 800, y2: 285, width: 111, height: 273 },
      quality: "rejected",
      quality_flags: ["implausible_area"],
      quality_reason:
        "Covers 7.1% of the frame, at or above the 5.0% ceiling for a single road vehicle in this view.",
      area_fraction: 0.0707,
      aspect_ratio: 0.407,
      touches_frame_edge: true,
    }),
  ],
  detection_tally: { by_class: { car: 2, bus: 1 }, total: 3, vehicles: 3 },
  confirmed_tally: { by_class: { car: 1 }, total: 1, vehicles: 1 },
  quality_policy: POLICY,
  frame_width: 800,
  frame_height: 533,
  overlay_available: true,
};

function stubFetch(analysis: ImageAnalysis = ANALYSIS): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        ({
          ok: true,
          status: 200,
          json: async () => analysis,
        }) as Response,
    ),
  );
}

async function analyse(analysis: ImageAnalysis = ANALYSIS): Promise<void> {
  stubFetch(analysis);
  render(<ImageVisionAnalysis mediaId="img-1" status={STATUS} />);
  await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
  await screen.findByTestId("detection-list");
}

describe("detection quality on the vision panel", () => {
  it("lists every detection, countable or not", async () => {
    await analyse();
    const list = screen.getByTestId("detection-list");
    expect(within(list).getAllByText(/car|bus/).length).toBeGreaterThanOrEqual(3);
  });

  it("marks an uncountable detection with its verdict", async () => {
    await analyse();
    const list = screen.getByTestId("detection-list");
    expect(within(list).getByText("provisional")).toBeVisible();
    expect(within(list).getByText("rejected")).toBeVisible();
  });

  it("shows why a detection was not counted", async () => {
    await analyse();
    const list = screen.getByTestId("detection-list");
    expect(
      within(list).getByText(/below the 0.40 confirmation floor/i),
    ).toBeVisible();
    expect(within(list).getByText(/7.1% of the frame/i)).toBeVisible();
  });

  it("leads with the countable figure rather than the box count", async () => {
    await analyse();
    expect(screen.getByText(/1 of 3 countable/i)).toBeVisible();
  });

  it("publishes both tallies", async () => {
    await analyse();
    const reported = screen.getByTestId("detection-tally");
    const countable = screen.getByTestId("confirmed-tally");
    expect(within(reported).getByText("3")).toBeVisible();
    expect(within(countable).getByText("1")).toBeVisible();
  });

  it("explains the gap between the two tallies", async () => {
    await analyse();
    const note = screen.getByTestId("excluded-note");
    expect(note).toHaveTextContent(/2 detection\(s\) are reported but not counted/i);
    expect(note).toHaveTextContent(/stay visible/i);
  });

  it("says nothing about exclusions when there are none", async () => {
    await analyse({
      ...ANALYSIS,
      detections: [detection()],
      detection_tally: { by_class: { car: 1 }, total: 1, vehicles: 1 },
      confirmed_tally: { by_class: { car: 1 }, total: 1, vehicles: 1 },
    });
    expect(screen.queryByTestId("excluded-note")).not.toBeInTheDocument();
  });

  it("does not mark a confirmed detection", async () => {
    await analyse({
      ...ANALYSIS,
      detections: [detection()],
      detection_tally: { by_class: { car: 1 }, total: 1, vehicles: 1 },
      confirmed_tally: { by_class: { car: 1 }, total: 1, vehicles: 1 },
    });
    const list = screen.getByTestId("detection-list");
    expect(within(list).queryByText("provisional")).not.toBeInTheDocument();
    expect(within(list).queryByText("rejected")).not.toBeInTheDocument();
  });

  it("still shows the confidence of an uncountable detection", async () => {
    /** The number an operator needs in order to argue with the verdict. */
    await analyse();
    const list = screen.getByTestId("detection-list");
    expect(within(list).getAllByText("37.0%")).toHaveLength(2);
  });
});
