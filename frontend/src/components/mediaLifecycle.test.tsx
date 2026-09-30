/**
 * Media asset lifecycle in the traffic input panel.
 *
 * The regression behind these tests: after a backend restart the media index
 * was empty while the files sat on disk, so "Run detection" answered
 * `No media asset with id '...'`. The backend now rebuilds its index, which
 * fixes the common case. These tests cover what a rebuild cannot fix — an asset
 * that is genuinely gone — and the property the panel must hold regardless:
 *
 * - the id sent to the CV pipeline is the one the upload returned, never
 *   invented and never hard-coded;
 * - a `not_found` from the backend clears the selection rather than leaving a
 *   dead asset on screen for an operator to keep pressing;
 * - the failure is stated, never dressed up as a successful analysis.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TrafficInputPanel } from "./TrafficInputPanel";
import type { FormatsResponse, ImageAnalysis, MediaAsset } from "../api/types";

afterEach(() => {
  vi.restoreAllMocks();
});

const UPLOADED_ID = "3c8f1e9ab7d5427aa1c0de44bb90f217";

const FORMATS: FormatsResponse = {
  formats: [
    { name: "png", kind: "image", extensions: [".png"], media_type: "image/png" },
    { name: "mp4", kind: "video", extensions: [".mp4"], media_type: "video/mp4" },
  ],
  max_image_upload_bytes: 33_554_432,
  max_video_upload_bytes: 536_870_912,
};

const ASSET: MediaAsset = {
  media_id: UPLOADED_ID,
  kind: "image",
  format: "png",
  media_type: "image/png",
  original_filename: "junction-north.png",
  size_bytes: 4096,
  ingested_at: "2026-09-22T08:30:00Z",
  image: {
    width: 853,
    height: 617,
    format: "png",
    mode: "RGB",
    channels: 3,
    is_animated: false,
    frame_count: null,
  },
  video: null,
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function analysis(): ImageAnalysis {
  return {
    media_id: UPLOADED_ID,
    source_id: `image:${UPLOADED_ID}`,
    captured_at: "2026-09-22T08:30:00Z",
    detections: [],
    detection_tally: { by_class: {}, total: 0, vehicles: 0, pedestrians: 0, cyclists: 0 },
    confirmed_tally: { by_class: {}, total: 0, vehicles: 0, pedestrians: 0, cyclists: 0 },
    inference: {
      model: { name: "yolov8s", backend: "ultralytics" },
      device: "cpu",
      duration_ms: 120,
      image_width: 853,
      image_height: 617,
      tracker: null,
    },
    quality: null,
    overlay_token: "tok-1",
  } as unknown as ImageAnalysis;
}

/**
 * Route by path. `analyzeStatus` controls what "Run detection" answers, which
 * is the only axis these tests vary.
 */
function stubApi(
  options: { analyzeResponse?: Response } = {},
): ReturnType<typeof vi.fn> {
  const handler = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/media/formats")) return jsonResponse(FORMATS);
    if (url.includes("/api/media/images")) return jsonResponse(ASSET, 201);
    if (url.includes("/analyze")) {
      return (options.analyzeResponse ?? jsonResponse(analysis())).clone();
    }
    if (url.includes("/api/vision/status")) {
      return jsonResponse({
        detection_available: true,
        selected_backend: "ultralytics",
        backends: [],
        note: "",
      });
    }
    return jsonResponse(
      { error: { code: "http_error", message: "no route", details: {} } },
      404,
    );
  });
  vi.stubGlobal("fetch", handler);
  return handler;
}

function pngFile(): File {
  return new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], "junction-north.png", {
    type: "image/png",
  });
}

function missingAssetResponse(): Response {
  return jsonResponse(
    {
      error: {
        code: "not_found",
        message: `No media asset with id '${UPLOADED_ID}'.`,
        details: { media_id: UPLOADED_ID },
      },
    },
    404,
  );
}

async function uploadImage(): Promise<void> {
  render(<TrafficInputPanel />);
  await userEvent.upload(await screen.findByTestId("upload-input-image"), pngFile());
  await screen.findByTestId("image-loaded");
}

describe("media asset lifecycle", () => {
  it("runs detection against the id the upload returned", async () => {
    const fetchMock = stubApi();
    await uploadImage();

    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some((call) => String(call[0]).includes("/analyze")),
      ).toBe(true),
    );

    const analyze = fetchMock.mock.calls.find((call) =>
      String(call[0]).includes("/analyze"),
    );
    expect(String(analyze?.[0])).toContain(UPLOADED_ID);
    // The panel invents no id and hard-codes none: the only id it can send is
    // the one the backend handed back.
    expect(String(analyze?.[0])).toMatch(
      new RegExp(`/api/vision/images/${UPLOADED_ID}/analyze$`),
    );
  });

  it("leaves Not analysed behind once detection succeeds", async () => {
    stubApi();
    await uploadImage();
    expect(screen.getByText(/not analysed/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await waitFor(() =>
      expect(screen.queryByText(/not analysed/i)).not.toBeInTheDocument(),
    );
  });

  it("shows the backend's real message when the asset is gone", async () => {
    stubApi({ analyzeResponse: missingAssetResponse() });
    await uploadImage();

    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /No media asset with id/,
      ),
    );
  });

  it("clears a selection the backend no longer has", async () => {
    stubApi({ analyzeResponse: missingAssetResponse() });
    await uploadImage();
    expect(screen.getByTestId("image-loaded")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    await waitFor(() =>
      expect(screen.queryByTestId("image-loaded")).not.toBeInTheDocument(),
    );
    expect(screen.getByTestId("image-asset-lost")).toBeInTheDocument();
    expect(screen.getByText(/no longer held by the server/i)).toBeInTheDocument();
  });

  it("stops offering Run detection for a vanished asset", async () => {
    stubApi({ analyzeResponse: missingAssetResponse() });
    await uploadImage();
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: /run detection/i }),
      ).not.toBeInTheDocument(),
    );
    // The one action that can still succeed is still offered.
    expect(screen.getByTestId("upload-input-image")).toBeInTheDocument();
  });

  it("never presents a failed analysis as a successful one", async () => {
    stubApi({ analyzeResponse: missingAssetResponse() });
    await uploadImage();
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(screen.queryByText(/countable/i)).not.toBeInTheDocument();
  });

  it("recovers when a fresh asset is uploaded after one vanished", async () => {
    // The restart story end to end: the stale selection is dropped, and the
    // next upload works normally.
    const handler = vi.fn(async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/media/formats")) return jsonResponse(FORMATS);
      if (url.includes("/api/media/images")) return jsonResponse(ASSET, 201);
      if (url.includes("/api/vision/status")) {
        return jsonResponse({
          detection_available: true,
          selected_backend: "ultralytics",
          backends: [],
          note: "",
        });
      }
      if (url.includes("/analyze")) {
        // Fails once — the stale id — then succeeds for the new upload.
        return handler.mock.calls.filter((call) =>
          String(call[0]).includes("/analyze"),
        ).length === 1
          ? missingAssetResponse()
          : jsonResponse(analysis());
      }
      return jsonResponse(
        { error: { code: "http_error", message: "no route", details: {} } },
        404,
      );
    });
    vi.stubGlobal("fetch", handler);

    await uploadImage();
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await waitFor(() =>
      expect(screen.getByTestId("image-asset-lost")).toBeInTheDocument(),
    );

    await userEvent.upload(screen.getByTestId("upload-input-image"), pngFile());
    await screen.findByTestId("image-loaded");
    expect(screen.queryByTestId("image-asset-lost")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await waitFor(() =>
      expect(screen.queryByText(/not analysed/i)).not.toBeInTheDocument(),
    );
  });

  it("does not clear the selection for an unrelated failure", async () => {
    // A 500 is a different problem: the asset is still there, and dropping the
    // operator's selection would lose their place for no reason.
    stubApi({
      analyzeResponse: jsonResponse(
        { error: { code: "internal_error", message: "boom", details: {} } },
        500,
      ),
    });
    await uploadImage();
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("boom"));
    expect(screen.getByTestId("image-loaded")).toBeInTheDocument();
    expect(screen.queryByTestId("image-asset-lost")).not.toBeInTheDocument();
  });
});
