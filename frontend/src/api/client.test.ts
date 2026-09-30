/**
 * API client tests.
 *
 * The behaviour under test is the distinction the whole dashboard rests on:
 * a 501 ("not built yet") must be classified differently from a 500 ("broken")
 * and from an unreachable backend, and no failure may ever resolve successfully
 * with an empty value.
 */

import { describe, expect, it, vi } from "vitest";
import {
  ApiError,
  apiGet,
  apiUpload,
  endpoints,
  observationAreaIdForMedia,
} from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("apiGet", () => {
  it("returns the parsed body on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "ok" })));
    await expect(apiGet<{ status: string }>(endpoints.health)).resolves.toEqual({
      status: "ok",
    });
  });

  it("classifies a 501 as not-available-yet and exposes the phase", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "not_available_yet",
              message: "No road network is loaded.",
              details: { phase: 8, phase_name: "Traffic Simulation" },
            },
          },
          501,
        ),
      ),
    );

    const error = await apiGet(endpoints.network).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).isNotAvailableYet).toBe(true);
    expect((error as ApiError).phase).toBe(8);
    expect((error as ApiError).isUnreachable).toBe(false);
  });

  it("does not classify a genuine server error as not-available-yet", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { error: { code: "internal_error", message: "boom", details: {} } },
          500,
        ),
      ),
    );

    const error = (await apiGet(endpoints.health).catch((e: unknown) => e)) as ApiError;
    expect(error.isNotAvailableYet).toBe(false);
    expect(error.status).toBe(500);
  });

  it("reports an unreachable backend rather than resolving empty", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const error = (await apiGet(endpoints.health).catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.isUnreachable).toBe(true);
    expect(error.message).toContain("Cannot reach the backend");
  });

  it("throws rather than returning undefined when the body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<html>502</html>", { status: 502 })),
    );

    const error = (await apiGet(endpoints.health).catch((e: unknown) => e)) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(502);
  });

  it("has no phase when the backend did not report one", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { error: { code: "not_available_yet", message: "later", details: {} } },
          501,
        ),
      ),
    );

    const error = (await apiGet(endpoints.network).catch((e: unknown) => e)) as ApiError;
    expect(error.phase).toBeNull();
  });
});

describe("apiUpload", () => {
  function capturedForm(fetchMock: ReturnType<typeof vi.fn>): FormData {
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    return init.body as FormData;
  }

  it("sends only the file when no location is supplied", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ media_id: "m1" }, 201));
    vi.stubGlobal("fetch", fetchMock);

    await apiUpload(endpoints.uploadImage, new File(["x"], "a.png"));

    const form = capturedForm(fetchMock);
    expect(form.get("file")).not.toBeNull();
    expect(form.get("latitude")).toBeNull();
    expect(form.get("longitude")).toBeNull();
  });

  it("sends the coordinates when a location is supplied", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ media_id: "m1" }, 201));
    vi.stubGlobal("fetch", fetchMock);

    await apiUpload(endpoints.uploadImage, new File(["x"], "a.png"), {
      latitude: 16.5062,
      longitude: 80.648,
      bearing_degrees: 135,
      site_label: "Benz Circle",
    });

    const form = capturedForm(fetchMock);
    expect(form.get("latitude")).toBe("16.5062");
    expect(form.get("longitude")).toBe("80.648");
    expect(form.get("bearing_degrees")).toBe("135");
    expect(form.get("site_label")).toBe("Benz Circle");
    // Omitted optional fields are absent, not sent as empty strings: an empty
    // accuracy would be parsed as a value rather than as "not reported".
    expect(form.get("accuracy_m")).toBeNull();
  });

  it("surfaces a rejected location as an ApiError rather than an empty success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "validation_failed",
              message: "A location needs both latitude and longitude.",
              details: { missing: "longitude" },
            },
          },
          422,
        ),
      ),
    );

    const error = (await apiUpload(endpoints.uploadImage, new File(["x"], "a.png"), {
      latitude: 16.5,
      longitude: 80.6,
    }).catch((e: unknown) => e)) as ApiError;

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(422);
    expect(error.isNotAvailableYet).toBe(false);
  });
});

describe("observation area ids", () => {
  it("matches the backend convention that joins a site to its verdicts", () => {
    expect(observationAreaIdForMedia("abc123")).toBe("observation-area:abc123");
  });

  it("encodes the colon when building the site path", () => {
    expect(endpoints.site("observation-area:abc123")).toBe(
      "/api/sites/observation-area%3Aabc123",
    );
  });
});
