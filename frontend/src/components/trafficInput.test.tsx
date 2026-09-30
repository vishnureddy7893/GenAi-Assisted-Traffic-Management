/**
 * Traffic input panel tests.
 *
 * Every displayed value must come from a backend response. The fixtures use
 * deliberately distinctive numbers (853 x 617, 23.97 fps) that no default or
 * placeholder would produce, so a hard-coded value could not satisfy them.
 *
 * Also asserted: the panel never renders a traffic quantity. Phase 2 ingests
 * raw input; counts, congestion, speed, queues, predictions and routes do not
 * exist yet and must not appear.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { TrafficInputPanel } from "./TrafficInputPanel";
import type { FormatsResponse, MediaAsset, PlaybackState } from "../api/types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
const FORMATS: FormatsResponse = {
  formats: [
    { name: "jpeg", kind: "image", extensions: [".jpg", ".jpeg"], media_type: "image/jpeg" },
    { name: "png", kind: "image", extensions: [".png"], media_type: "image/png" },
    { name: "webp", kind: "image", extensions: [".webp"], media_type: "image/webp" },
    { name: "mp4", kind: "video", extensions: [".mp4"], media_type: "video/mp4" },
    { name: "mov", kind: "video", extensions: [".mov"], media_type: "video/quicktime" },
    { name: "avi", kind: "video", extensions: [".avi"], media_type: "video/x-msvideo" },
  ],
  max_image_upload_bytes: 33_554_432,
  max_video_upload_bytes: 536_870_912,
};

const IMAGE_ASSET: MediaAsset = {
  media_id: "img-fixture-id",
  kind: "image",
  format: "png",
  media_type: "image/png",
  original_filename: "junction-north.png",
  size_bytes: 4096,
  ingested_at: "2024-03-14T08:30:00Z",
  image: {
    width: 853,
    height: 617,
    format: "png",
    mode: "RGBA",
    channels: 4,
    is_animated: false,
    frame_count: null,
  },
  video: null,
};

const VIDEO_ASSET: MediaAsset = {
  media_id: "vid-fixture-id",
  kind: "video",
  format: "mp4",
  media_type: "video/mp4",
  original_filename: "corridor-clip.mp4",
  size_bytes: 91_234,
  ingested_at: "2024-03-14T08:30:00Z",
  image: null,
  video: {
    width: 1442,
    height: 811,
    fps: 23.97,
    frame_count: 719,
    duration_s: 29.99,
    codec_fourcc: "avc1",
  },
};

const PLAYBACK: PlaybackState = {
  media_id: "vid-fixture-id",
  status: "ready",
  is_open: true,
  current_frame_index: 41,
  position_s: 1.71,
  frames_read: 41,
  exhausted: false,
  fps: 23.97,
  frame_count: 719,
  duration_s: 29.99,
  last_read_succeeded: true,
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

interface RouteOptions {
  formats?: Response;
  imageUpload?: Response;
  videoUpload?: Response;
  playback?: Response;
  live?: Response;
}

/** Route fetch by path so each test only overrides what it cares about. */
function stubFetch(routes: RouteOptions = {}): ReturnType<typeof vi.fn> {
  const handler = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/media/formats")) {
      return (routes.formats ?? jsonResponse(FORMATS)).clone();
    }
    if (url.includes("/api/media/images")) {
      return (routes.imageUpload ?? jsonResponse(IMAGE_ASSET, 201)).clone();
    }
    if (url.includes("/api/media/videos") && url.includes("/playback/")) {
      return (routes.playback ?? jsonResponse(PLAYBACK)).clone();
    }
    if (url.includes("/api/media/videos")) {
      return (routes.videoUpload ?? jsonResponse(VIDEO_ASSET, 201)).clone();
    }
    if (url.includes("/api/media/live/connect")) {
      return (
        routes.live ??
        jsonResponse(
          {
            error: {
              code: "not_available_yet",
              message: "Live traffic ingestion is not implemented.",
              details: { phase: 14, phase_name: "Live Traffic Extension" },
            },
          },
          501,
        )
      ).clone();
    }
    return jsonResponse({ error: { code: "http_error", message: "no route", details: {} } }, 404);
  });
  vi.stubGlobal("fetch", handler);
  return handler;
}

function pngFile(name = "junction-north.png"): File {
  return new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], name, { type: "image/png" });
}

function mp4File(name = "corridor-clip.mp4"): File {
  return new File([new Uint8Array([0, 0, 0, 24])], name, { type: "video/mp4" });
}

/**
 * Traffic-*management* conclusions that must not appear.
 *
 * Narrowed as phases land. Detection and tracking arrived in Phase 3 and
 * derived measurements in Phase 4, so words like "detected" and "queue
 * evidence" are legitimate output now rather than overclaims. What must still
 * never appear is an *interpretation*: a congestion level, a travel time, a
 * route, or a physical speed the system cannot measure without calibration.
 *
 * The raw-input notice is excluded by textExcludingNotice: its entire purpose
 * is to name the things that do not exist yet.
 */
const FORBIDDEN_OUTPUT =
  /congestion level|travel time|recommended route|\bETA\b|km\/h/i;

function textExcludingNotice(container: HTMLElement): string {
  const clone = container.cloneNode(true) as HTMLElement;
  clone
    .querySelectorAll("[data-testid='raw-input-notice']")
    .forEach((node) => node.remove());
  return clone.textContent ?? "";
}

async function switchTo(label: RegExp): Promise<void> {
  await userEvent.click(screen.getByRole("tab", { name: label }));
}

// ---------------------------------------------------------------------------
// Selector
// ---------------------------------------------------------------------------
describe("TrafficInputPanel selector", () => {
  it("offers Image, Video and Live tabs", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    expect(await screen.findByRole("tab", { name: /image/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /video/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /live/i })).toBeInTheDocument();
  });

  it("marks only Live as a future capability", () => {
    stubFetch();
    render(<TrafficInputPanel />);
    const live = screen.getByRole("tab", { name: /live/i });
    expect(within(live).getByText(/future/i)).toBeInTheDocument();
    const image = screen.getByRole("tab", { name: /image/i });
    expect(within(image).queryByText(/future/i)).not.toBeInTheDocument();
  });

  it("derives accepted extensions from the formats API", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    const input = await screen.findByTestId("upload-input-image");
    expect(input).toHaveAttribute("accept", ".jpg,.jpeg,.png,.webp");
  });

  it("derives video extensions from the same response", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await switchTo(/video/i);
    const input = await screen.findByTestId("upload-input-video");
    expect(input).toHaveAttribute("accept", ".mp4,.mov,.avi");
  });
});

// ---------------------------------------------------------------------------
// Empty states
// ---------------------------------------------------------------------------
describe("empty states", () => {
  it("shows no image loaded before anything is ingested", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    expect(await screen.findByText(/no image loaded/i)).toBeInTheDocument();
    expect(screen.queryByTestId("image-loaded")).not.toBeInTheDocument();
  });

  it("shows no video loaded before anything is ingested", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await switchTo(/video/i);
    expect(await screen.findByText(/no video loaded/i)).toBeInTheDocument();
    expect(screen.queryByTestId("video-loaded")).not.toBeInTheDocument();
  });

  it("fabricates no metadata while empty", async () => {
    stubFetch();
    const { container } = render(<TrafficInputPanel />);
    await screen.findByText(/no image loaded/i);
    const text = container.textContent ?? "";
    expect(text).not.toMatch(/\d+\s*×\s*\d+/);
  });
});

// ---------------------------------------------------------------------------
// Image ingestion
// ---------------------------------------------------------------------------
describe("image ingestion", () => {
  it("renders metadata returned by the API", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());

    const loaded = await screen.findByTestId("image-loaded");
    expect(within(loaded).getByText("853 × 617 px")).toBeInTheDocument();
    expect(within(loaded).getByText("RGBA")).toBeInTheDocument();
    expect(within(loaded).getByText("4")).toBeInTheDocument();
    expect(within(loaded).getByText("PNG")).toBeInTheDocument();
    // Filename appears in both the caption and the metadata table.
    expect(within(loaded).getAllByText("junction-north.png").length).toBeGreaterThan(0);
  });

  it("shows the actual uploaded image from the content endpoint", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());

    const img = await screen.findByRole("img", { name: /junction-north\.png/i });
    expect(img).toHaveAttribute("src", expect.stringContaining("img-fixture-id"));
    expect(img).toHaveAttribute("src", expect.stringContaining("/content"));
  });

  it("states which capabilities exist and which do not", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());

    const notice = await screen.findByTestId("raw-input-notice");
    // Phase 3: perception exists, interpretation does not. The notice must
    // claim neither more nor less than that.
    expect(notice).toHaveTextContent(/traffic interpretation not implemented yet/i);
    expect(notice).toHaveTextContent(/no congestion level/i);
  });

  it("reports a rejected upload using the backend message", async () => {
    stubFetch({
      imageUpload: jsonResponse(
        {
          error: {
            code: "validation_failed",
            message: "Unrecognised file content.",
            details: {},
          },
        },
        422,
      ),
    });
    render(<TrafficInputPanel />);
    await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/unrecognised file content/i);
    expect(screen.queryByTestId("image-loaded")).not.toBeInTheDocument();
  });

  it("does not display an image when the upload failed", async () => {
    stubFetch({
      imageUpload: jsonResponse(
        { error: { code: "validation_failed", message: "nope", details: {} } },
        422,
      ),
    });
    render(<TrafficInputPanel />);
    await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());
    await screen.findByRole("alert");
    expect(screen.queryByRole("img", { name: /ingested traffic still/i })).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Video ingestion
// ---------------------------------------------------------------------------
describe("video ingestion", () => {
  async function uploadVideo(): Promise<void> {
    render(<TrafficInputPanel />);
    await switchTo(/video/i);
    await userEvent.upload(await screen.findByTestId("upload-input-video"), mp4File());
    await screen.findByTestId("video-loaded");
  }

  it("renders container metadata returned by the API", async () => {
    stubFetch();
    await uploadVideo();
    expect(screen.getByText("1442 × 811 px")).toBeInTheDocument();
    expect(screen.getByText("23.97 fps")).toBeInTheDocument();
    expect(screen.getByText("719")).toBeInTheDocument();
    expect(screen.getByText("29.99 s")).toBeInTheDocument();
    expect(screen.getByText("avc1")).toBeInTheDocument();
  });

  it("shows a video preview pointing at the content endpoint", async () => {
    stubFetch();
    await uploadVideo();
    const video = screen.getByTestId("video-preview");
    expect(video).toHaveAttribute("src", expect.stringContaining("vid-fixture-id"));
  });

  it("marks unreported metadata rather than inventing it", async () => {
    stubFetch({
      videoUpload: jsonResponse(
        {
          ...VIDEO_ASSET,
          video: { ...VIDEO_ASSET.video, fps: null, frame_count: null, duration_s: null },
        },
        201,
      ),
    });
    await uploadVideo();
    expect(screen.getAllByText(/not reported/i).length).toBeGreaterThanOrEqual(3);
    expect(screen.queryByText(/0\.00 s/)).not.toBeInTheDocument();
    expect(screen.queryByText(/0 fps/)).not.toBeInTheDocument();
  });

  it("offers open, read, seek, reset and close controls", async () => {
    stubFetch();
    await uploadVideo();
    const controls = screen.getByTestId("playback-controls");
    for (const name of [/open/i, /read frame/i, /seek to start/i, /reset/i, /close/i]) {
      expect(within(controls).getByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("renders playback state returned by the API", async () => {
    stubFetch();
    await uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: /read frame/i }));

    await waitFor(() => {
      expect(screen.getByTestId("playback-state")).toBeInTheDocument();
    });
    const state = screen.getByTestId("playback-state");
    // 41 is both the frame index and the frames-read count in this fixture.
    expect(within(state).getAllByText("41").length).toBe(2);
    expect(within(state).getByText("1.710 s")).toBeInTheDocument();
    expect(within(state).getByText("open")).toBeInTheDocument();
  });

  it("surfaces a playback failure instead of a stale position", async () => {
    stubFetch({
      playback: jsonResponse(
        { error: { code: "data_source_error", message: "decoder gone", details: {} } },
        502,
      ),
    });
    await uploadVideo();
    await userEvent.click(screen.getByRole("button", { name: /open/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/decoder gone/i);
    expect(screen.queryByTestId("playback-state")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Live mode
// ---------------------------------------------------------------------------
describe("live mode", () => {
  it("remains visible and states it is a future capability", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await switchTo(/live/i);
    expect(screen.getByTestId("live-mode")).toBeInTheDocument();
    expect(screen.getByText(/live traffic — future capability/i)).toBeInTheDocument();
  });

  it("says plainly that no connection is simulated", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await switchTo(/live/i);
    expect(screen.getByText(/no live connection is simulated/i)).toBeInTheDocument();
  });

  it("offers no upload control", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await switchTo(/live/i);
    expect(screen.queryByTestId("upload-input-image")).not.toBeInTheDocument();
    expect(screen.queryByTestId("upload-input-video")).not.toBeInTheDocument();
  });

  it("reports the backend's 501 with its phase when probed", async () => {
    stubFetch();
    render(<TrafficInputPanel />);
    await switchTo(/live/i);
    await userEvent.click(screen.getByRole("button", { name: /check availability/i }));

    const result = await screen.findByTestId("live-probe-result");
    expect(result).toHaveTextContent(/not implemented/i);
    expect(result).toHaveTextContent(/phase 14/i);
  });
});

// ---------------------------------------------------------------------------
// Phase boundary
// ---------------------------------------------------------------------------
describe("no traffic intelligence is displayed", () => {
  it("renders no analysis output for an ingested image", async () => {
    stubFetch();
    const { container } = render(<TrafficInputPanel />);
    await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());
    await screen.findByTestId("image-loaded");

    expect(textExcludingNotice(container)).not.toMatch(FORBIDDEN_OUTPUT);
  });

  it("renders no analysis output for an ingested video", async () => {
    stubFetch();
    const { container } = render(<TrafficInputPanel />);
    await switchTo(/video/i);
    await userEvent.upload(await screen.findByTestId("upload-input-video"), mp4File());
    await screen.findByTestId("video-loaded");

    expect(textExcludingNotice(container)).not.toMatch(FORBIDDEN_OUTPUT);
  });
});
